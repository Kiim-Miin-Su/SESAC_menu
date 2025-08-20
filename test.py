import os
import re
import json
import urllib.parse

from fastapi.responses import RedirectResponse
from typing import List, Optional, Literal
from math import radians, sin, cos, sqrt, atan2

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

def calculate_distance(row: dict, loc_x: float, loc_y: float) -> float:
    try:
        lon2, lat2 = float(row.get("X")), float(row.get("Y"))
    except (TypeError, ValueError):
        return 99999999
    
    const_R = 6371000 #m
    
    lat_1, lon_1, lat_2, lon_2 = map(radians, [row.get("Y"), row.get("X"), loc_y, loc_x])
    
    if not lat_1 or not lon_1 or not lat_2 or not lon_2:
        return False
    else:
        dlat = lat_2 - lat_1
        dlon = lon_2 - lon_1
        
        a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
        c = 2 * atan2(sqrt(a), sqrt(1-a))

        return const_R * c

    
# 간단 캐시: 동일 구간 조회는 짧게 메모리 캐싱
@lru_cache(maxsize=256)
def get_rows_range(start: int, end: int) -> list:
    svc = fetch_page(start, end)
    rows = svc.get("row", []) or []
    return rows

def iter_rows(pages: int = 4, size: int = 1000):
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

@app.get("/init/load/{kind}/{selected_area}/{curr_loc_x}/{curr_loc_y}/{distance}")
def init_load(kind: str, selected_area: str, curr_loc_x: float, curr_loc_y: float, distance: int):
    categories = {
        "ko": [],
        "ch": [],
        "jp": [],
        "en": [],
        "etc": [],
        "all": []
    }
    
    category_keywords = {
        "ko": ["한", "냉면", "분식", "뷔페식", "식육"],
        "ch": ["중", "중국"],
        "jp": ["일", "횟집"],
        "en": ["양", "패스트푸드", "외국음식전문점"],
        "etc": ["까페", "소주방", "카페"]
    }
    if not curr_loc_x or not curr_loc_y:
        raise HTTPException(400, detail="Invalid location")

    area_map = {"gd": "고덕", "si": "상일"}
    area = area_map.get(selected_area, "")

    kind = kind.lower() if kind and kind != "all" else "all"

    # 데이터 로드
    for row in iter_rows(pages=4, size=1000):
        uptaenm = row.get("UPTAENM", "")
        addr = row.get("RDNWHLADDR", "") + row.get("SITEWHLADDR", "")
        categories["all"].append(row)

        if not is_open(row):
            continue

        if any(k in uptaenm for k in category_keywords["ko"]):
            categories["ko"].append(row)
        elif any(k in uptaenm for k in category_keywords["ch"]):
            categories["ch"].append(row)
        elif any(k in uptaenm for k in category_keywords["jp"]):
            categories["jp"].append(row)
        elif any(k in uptaenm for k in category_keywords["en"]):
            categories["en"].append(row)
        elif not any(k in uptaenm for k in category_keywords["etc"]):
            categories["etc"].append(row)
            
    results = categories.get(kind, categories["all"])
    
    # 거리 필터링
    if distance:
        results = [row for row in results if calculate_distance(row, curr_loc_x, curr_loc_y) <= distance]


    # 필터링
    if area:
        results = [row for row in results if area in row.get("RDNWHLADDR", "") or area in row.get("SITEWHLADDR", "")]
    
    return results


@app.get("/photo/street")
def photo_street(addr: str, w: int = 600, h: int = 360, fov: int = 90, heading: int = 0, pitch: int = 0):
    loc = geocode_address(addr)
    if not loc:
        raise HTTPException(400, detail=f"Geocoding failed for {addr}")
    lat, lng = loc["lat"], loc["lng"]

    url = (f"https://maps.googleapis.com/maps/api/streetview"
           f"?size={w}x{h}&location={lat},{lng}&fov={fov}&heading={heading}&pitch={pitch}&key={map_KEY}")
    return RedirectResponse(url)

@app.get("/restaurants/{params}")
def list_restaurants(params: str):
    params = params.split(",")
    kind = params[0]
    if kind not in ["ko", "ch", "jp", "en", "etc"]:
        raise HTTPException(400, detail=f"Invalid kind: {kind}")
    
    if len(params) > 1:
        area = params[1]
        if area not in ["all", "gd", "si"]:
            raise HTTPException(400, detail=f"Invalid area: {area}")
    
    return init_load(kind, area)        

@app.get("/restaurants/random")
def random_restaurants(
    area: Optional[str] = None,
    kind: Optional[Literal["전체","한", "중", "일", "양", "etc"]] = None,
    open_only: bool = True,
    pages: int = 4,
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

