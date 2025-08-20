import os
import re
import json
import urllib.parse

from fastapi.responses import RedirectResponse
from typing import List, Optional, Literal

import requests

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from functools import lru_cache

# --- 환경 변수 ---
load_dotenv()
API_BASE = "http://openapi.seoul.go.kr:8088"
API_KEY = os.getenv("const_KEY")
map_KEY = os.getenv("map_KEY")
SERVICE = "LOCALDATA_072404_GD"
DATA_TYPE = "json"

if not API_KEY:
    raise RuntimeError("const_KEY가 .env에 설정되어야 합니다.")

# --- 앱/미들웨어 ---
app = FastAPI(title="Seoul Food LocalData API")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],       # 배포 시 프론트 도메인으로 제한 권장
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 유틸 ---
OPEN_TOKENS = ("영업", "정상", "영업/정상")

def is_open(row: dict) -> bool:
    trd = (row.get("TRDSTATENM") or "").strip()
    dtl = (row.get("DTLSTATENM") or "").strip()
    return any(tok in trd for tok in OPEN_TOKENS) and (dtl in OPEN_TOKENS or dtl == "")

def pick_addr(row: dict) -> str:
    return (row.get("RDNWHLADDR") or row.get("SITEWHLADDR") or "").strip()

def normalize_row(row: dict) -> dict:
    return {
        "name":   (row.get("BPLCNM") or "").strip(),
        "open":   is_open(row),
        "addr":   pick_addr(row),
        "kind":   (row.get("UPTAENM") or row.get("SNTUPTAENM") or "").strip(),
    }

def geocode_address(addr: str):
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    params = {"address": addr, "key": map_KEY}
    res = requests.get(url, params=params).json()
    if res["status"] != "OK":
        return None
    loc = res["results"][0]["geometry"]["location"]
    return loc  # {'lat': 37.123, 'lng': 127.456}

def fetch_page(start: int, end: int) -> dict:
    url = f"{API_BASE}/{API_KEY}/{DATA_TYPE}/{SERVICE}/{start}/{end}"
    r = requests.get(url, timeout=8)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Upstream HTTP {r.status_code}")
    try:
        data = r.json()
    except Exception:
        # xml로 사유 확인이 필요한 경우가 있으나 여기선 그대로 반환
        raise HTTPException(status_code=502, detail="Upstream JSON decode error")
    # 공공API는 실패도 200으로 내려줄 수 있음
    svc = data.get(SERVICE)
    if not svc:
        # 더 이상 데이터 없음
        return {"row": []}
    return svc

# 간단 캐시: 동일 구간 조회는 짧게 메모리 캐싱
@lru_cache(maxsize=256)
def get_rows_range(start: int, end: int) -> list:
    svc = fetch_page(start, end)
    rows = svc.get("row", []) or []
    return rows

def iter_rows(pages: int = 10, size: int = 1000):
    """
    필요한 만큼만 페이지 순회 (기본 3, 성능/쿼터 고려)
    """
    for i in range(pages):
        start = 1 + size * i
        end = size * (i + 1)
        rows = get_rows_range(start, end)
        if not rows:
            break
        for row in rows:
            yield row

# --- API ---
@app.get("/health")
def health():
    return {"ok": True}

@app.get("/init/load")
def init_load():
    rows = []
    for row in iter_rows(pages=10, size=1000):
        rows.append(row)
    return rows
    
    
@app.get("/photo/street")
def photo_street(addr: str, w: int = 600, h: int = 360, fov: int = 90, heading: int = 0, pitch: int = 0):
    loc = geocode_address(addr)
    if not loc:
        raise HTTPException(400, detail=f"Geocoding failed for {addr}")
    lat, lng = loc["lat"], loc["lng"]

    url = (f"https://maps.googleapis.com/maps/api/streetview"
           f"?size={w}x{h}&location={lat},{lng}&fov={fov}&heading={heading}&pitch={pitch}&key={map_KEY}")
    return RedirectResponse(url)

@app.get("/restaurants")
def list_restaurants(
    q: Optional[str] = Query(None, description="가게명 부분검색"),
    area: Optional[str] = Query(None, description="지역 키워드들(쉼표구분). 예: 상일,고덕"),
    kind: Optional[Literal["전체", "한", "중", "일", "양", "etc"]] = Query(None),
    open_only: bool = Query(True),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
    pages: int = Query(10, ge=1, le=50),   # 얼마나 페이지를 스캔할지 (너무 크면 느려짐/쿼터 소모)
    size: int = Query(1000, ge=1, le=1000)
):
    # 필터 준비
    area_patterns: List[re.Pattern] = []
    if area:
        tokens = [t.strip() for t in area.split(",") if t.strip()]
        area_patterns = [re.compile(re.escape(tok)) for tok in tokens]

    def area_match(addr: str) -> bool:
        if not area_patterns:
            return True
        return any(p.search(addr) for p in area_patterns)

# 수집
    items = []
    for row in iter_rows(pages=pages, size=size):
        n = normalize_row(row)

        # 1) 업종: kind가 None 이거나 "전체"면 모두 허용, 아니면 한/중/일 이란 단어 포함시 허용
        if kind not in (None, "전체") and not any(k in n["kind"] for k in kind):
            continue

        # 2) 영업 여부
        if open_only and not n["open"]:
            continue

        # 3) 지역
        if not area_match(n["addr"]):
            continue

        # 4) 이름 검색 (대소문자 구분 없이 하고 싶으면 둘 다 lower())
        if q and q not in n["name"]:
            continue

        items.append(n)

    total = len(items)
    items = items[offset: offset + limit]
    return {"total": total, "count": len(items), "items": items}

@app.get("/restaurants/random")
def random_restaurants(
    area: Optional[str] = None,
    kind: Optional[Literal["전체","한", "중", "일", "양", "etc"]] = None,
    open_only: bool = True,
    pages: int = 10,
    size: int = 1000,
    count: int = Query(5, ge=1, le=20, description="랜덤으로 뽑을 개수 (기본 5개)")
):
    import random
    # /restaurants와 같은 필터 로직 재사용
    resp = list_restaurants(q=None, area=area, kind=kind, open_only=open_only,
                            limit=10000, offset=0, pages=pages, size=size)
    items = resp["items"]
    if not items:
        raise HTTPException(status_code=404, detail="조건에 맞는 가게가 없습니다.")

    # 요청한 개수보다 items가 적으면 전체 반환
    if len(items) <= count:
        return items
    return random.sample(items, count)

