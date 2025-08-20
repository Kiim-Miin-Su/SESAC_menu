import os, re, time, threading
from math import radians, sin, cos, sqrt, atan2
from typing import List, Optional, Literal, Tuple, Dict, Any
from dataclasses import dataclass
from contextlib import asynccontextmanager
from functools import lru_cache

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse, ORJSONResponse
import random as pyrandom
from pydantic import BaseModel

# ──────────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

class Config:
    API_BASE: str = "http://openapi.seoul.go.kr:8088"
    API_KEY: str = os.getenv("const_KEY")
    MAP_KEY: Optional[str] = os.getenv("map_KEY")
    SERVICE: str = "LOCALDATA_072404_GD"
    DATA_TYPE: str = "json"
    CACHE_REFRESH_SEC: int = int(os.getenv("CACHE_REFRESH_SEC", "600"))
    PREFETCH_PAGES: int = int(os.getenv("PREFETCH_PAGES", "6"))
    PREFETCH_SIZE: int = int(os.getenv("PREFETCH_SIZE", "1000"))
    MAX_EXCLUDE_PARAMS: int = 400

if not Config.API_KEY:
    raise RuntimeError("const_KEY가 .env에 설정되어야 합니다.")

# ──────────────────────────────────────────────────────────────────────────────
# 모델
# ──────────────────────────────────────────────────────────────────────────────
@dataclass
class Restaurant:
    id: str
    name: str
    open: bool
    addr: str
    kind: str
    category: str
    loc_x: Optional[float]
    loc_y: Optional[float]

class RestaurantResponse(BaseModel):
    total: int
    count: int
    items: List[Dict[str, Any]]

class RandomResponse(BaseModel):
    items: List[Dict[str, Any]]

# ──────────────────────────────────────────────────────────────────────────────
# 유틸/분류
# ──────────────────────────────────────────────────────────────────────────────
class RestaurantClassifier:
    OPEN_TOKENS = ("영업", "정상", "영업/정상")
    RE_RULES = [
        ("카페", re.compile(r"(카페|까페)")),
        ("호프통닭", re.compile(r"(호프|통닭|치킨|주점|소주방|포차|이자카야|와인바|술집)")),
        ("한", re.compile(r"(한식|냉면|분식|뷔페식|식육)")),
        ("중", re.compile(r"(중식|중국|중화요리)")),
        ("일", re.compile(r"(일식|스시|초밥|라멘|우동|돈카츠|규카츠|사케동|횟집)")),
        ("양", re.compile(r"(양식|패스트푸드|외국음식전문점|피자|파스타|스테이크|버거)")),
    ]

    @classmethod
    def is_open(cls, row: dict) -> bool:
        trd = (row.get("TRDSTATENM") or "").strip()
        dtl = (row.get("DTLSTATENM") or "").strip()
        return any(tok in trd for tok in cls.OPEN_TOKENS) and (dtl in cls.OPEN_TOKENS or dtl == "")

    @classmethod
    def pick_addr(cls, row: dict) -> str:
        return (row.get("RDNWHLADDR") or row.get("SITEWHLADDR") or "").strip()

    @classmethod
    def extract_coords(cls, row: dict) -> Tuple[Optional[float], Optional[float]]:
        for lx, ly in [("X","Y"),("LOC_X","LOC_Y"),("LON","LAT"),("longitude","latitude")]:
            lon, lat = row.get(lx), row.get(ly)
            if lon not in (None,"") and lat not in (None,""):
                try: return float(lon), float(lat)
                except: pass
        return None, None

    @classmethod
    def classify_category(cls, raw_kind: str) -> str:
        k = (raw_kind or "").replace(" ","")
        for cat, pat in cls.RE_RULES:
            if pat.search(k): return cat
        return "etc"

class Distance:
    @staticmethod
    def calc(lon1, lat1, lon2, lat2) -> float:
        try:
            lon1,lat1,lon2,lat2 = map(float,[lon1,lat1,lon2,lat2])
        except: return float('inf')
        R=6371000.0
        dlat=radians(lat2-lat1); dlon=radians(lon2-lon1)
        a=sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlon/2)**2
        return 2*atan2(sqrt(a),sqrt(1-a))*R

class Normalizer:
    @staticmethod
    def normalize(row: dict) -> Restaurant:
        lon, lat = RestaurantClassifier.extract_coords(row)
        raw_kind = (row.get("UPTAENM") or row.get("SNTUPTAENM") or "").strip()
        category = RestaurantClassifier.classify_category(raw_kind)
        ident = (row.get("MGTNO") or "").strip()
        return Restaurant(
            id=ident,
            name=(row.get("BPLCNM") or "").strip(),
            open=RestaurantClassifier.is_open(row),
            addr=RestaurantClassifier.pick_addr(row),
            kind=raw_kind,
            category=category,
            loc_x=lon,
            loc_y=lat,
        )

    @staticmethod
    def key(item: Restaurant) -> str:
        return item.id if item.id else f"{item.name}|{item.addr}"

# ──────────────────────────────────────────────────────────────────────────────
# 외부 API (동기)
# ──────────────────────────────────────────────────────────────────────────────
class SeoulAPIClient:
    def __init__(self):
        self.base = Config.API_BASE; self.key = Config.API_KEY
        self.svc = Config.SERVICE; self.dtype = Config.DATA_TYPE

    def fetch_page(self, start: int, end: int) -> dict:
        url = f"{self.base}/{self.key}/{self.dtype}/{self.svc}/{start}/{end}"
        try:
            r = requests.get(url, timeout=8); r.raise_for_status()
            data = r.json(); svc = data.get(self.svc)
            return svc if svc else {"row":[]}
        except requests.RequestException as e:
            raise HTTPException(502, detail=f"Upstream API error: {e}")
        except Exception as e:
            raise HTTPException(502, detail=f"Data processing error: {e}")

# ──────────────────────────────────────────────────────────────────────────────
# 캐시
# ──────────────────────────────────────────────────────────────────────────────
class RestaurantCache:
    def __init__(self):
        self._data: List[Restaurant] = []; self._ts: float = 0.0
        self._lock = threading.Lock(); self._cli = SeoulAPIClient()

    def get(self) -> List[Restaurant]:
        with self._lock: return list(self._data)

    def ts(self) -> float:
        with self._lock: return self._ts

    def refresh(self, pages=Config.PREFETCH_PAGES, size=Config.PREFETCH_SIZE):
        raw = []
        for i in range(pages):
            start = 1 + size*i; end = size*(i+1)
            resp = self._cli.fetch_page(start,end)
            rows = resp.get("row",[]) or []
            if not rows: break
            raw.extend(rows)
        norm = [Normalizer.normalize(r) for r in raw]
        with self._lock:
            self._data = norm; self._ts = time.time()

    def start_loop(self):
        def loop():
            while True:
                try: self.refresh()
                except Exception as e: print("Cache refresh error:", e)
                time.sleep(Config.CACHE_REFRESH_SEC)
        threading.Thread(target=loop, daemon=True).start()

# ──────────────────────────────────────────────────────────────────────────────
# 필터
# ──────────────────────────────────────────────────────────────────────────────
class Filter:
    def __init__(self, cache: RestaurantCache):
        self.cache = cache

    def _area_patterns(self, area: Optional[str]) -> List[re.Pattern]:
        if not area: return []
        return [re.compile(re.escape(t.strip())) for t in area.split(",") if t.strip()]

    def _area_match(self, addr: str, pats: List[re.Pattern]) -> bool:
        if not pats: return True
        return any(p.search(addr) for p in pats)

    def _kind_match(self, cat: str, kind: Optional[str]) -> bool:
        if kind in (None,"전체"): return True
        if kind == "etc": return cat not in {"한","중","일","양","카페","호프통닭"}
        return cat == kind

    def filter(
        self, *, query: Optional[str], area: Optional[str], kind: Optional[str], open_only: bool,
        curr_loc_x: Optional[float], curr_loc_y: Optional[float], distance: Optional[int],
        exclude: Optional[List[str]], order: str, seed: Optional[int]
    ) -> List[Restaurant]:
        data = self.cache.get()
        pats = self._area_patterns(area)
        q = (query or "").strip().lower()
        exclude_set = set(exclude or [])
        want_dist = (curr_loc_x is not None and curr_loc_y is not None and distance is not None)

        out: List[Restaurant] = []
        for r in data:
            if exclude_set and Normalizer.key(r) in exclude_set: continue
            if open_only and not r.open: continue
            if not self._kind_match(r.category, kind): continue
            if area and not self._area_match(r.addr, pats): continue
            if q and (q not in (r.name or "").lower()): continue
            if want_dist and r.loc_x is not None and r.loc_y is not None:
                if Distance.calc(curr_loc_x, curr_loc_y, r.loc_x, r.loc_y) > float(distance):
                    continue
            out.append(r)

        if order == "rand":
            rng = pyrandom.Random(seed) if seed is not None else pyrandom
            rng.shuffle(out)
        return out

# ──────────────────────────────────────────────────────────────────────────────
# App & Lifespan
# ──────────────────────────────────────────────────────────────────────────────
cache = RestaurantCache()
flt = Filter(cache)

@asynccontextmanager
async def lifespan(app: FastAPI):
    cache.start_loop()
    yield

app = FastAPI(title="Seoul Food LocalData API", default_response_class=ORJSONResponse, lifespan=lifespan)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=False, allow_methods=["*"], allow_headers=["*"])

# ──────────────────────────────────────────────────────────────────────────────
# Geocoding (간단 캐시)
# ──────────────────────────────────────────────────────────────────────────────
@lru_cache(maxsize=1024)
def geocode(addr: str):
    if not Config.MAP_KEY: return None
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    try:
        res = requests.get(url, params={"address": addr, "key": Config.MAP_KEY}, timeout=8).json()
        if res.get("status") != "OK": return None
        loc = res["results"][0]["geometry"]["location"]
        return (loc["lat"], loc["lng"])
    except Exception:
        return None

# ──────────────────────────────────────────────────────────────────────────────
# API
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/health")
async def health():
    return {"ok": True, "cache_ts": cache.ts(), "prefetched": len(cache.get())}

@app.get("/restaurants", response_model=RestaurantResponse)
async def list_restaurants(
    q: Optional[str] = Query(None),
    area: Optional[str] = Query(None),
    kind: Optional[Literal["전체","한","중","일","양","카페","호프통닭","etc"]] = Query(None),
    open_only: bool = Query(True),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    curr_loc_x: Optional[float] = Query(None),
    curr_loc_y: Optional[float] = Query(None),
    distance: Optional[int] = Query(None),
    exclude: Optional[List[str]] = Query(None),
    order: Optional[Literal["default","rand"]] = Query("default"),
    seed: Optional[int] = Query(None)
):
    if exclude and len(exclude) > Config.MAX_EXCLUDE_PARAMS:
        exclude = exclude[-Config.MAX_EXCLUDE_PARAMS:]

    items = flt.filter(
        query=q, area=area, kind=kind, open_only=open_only,
        curr_loc_x=curr_loc_x, curr_loc_y=curr_loc_y, distance=distance,
        exclude=exclude, order=order, seed=seed
    )
    total = len(items)
    page = items[offset: offset+limit]

    # 선택: 서버에서 distance(m) 계산해서 내려주기(프론트 표시용)
    if curr_loc_x is not None and curr_loc_y is not None:
        out = []
        for r in page:
            d = None
            if r.loc_x is not None and r.loc_y is not None:
                d = Distance.calc(curr_loc_x, curr_loc_y, r.loc_x, r.loc_y)
            obj = r.__dict__.copy()
            if d is not None: obj["distance"] = round(float(d), 2)
            out.append(obj)
    else:
        out = [r.__dict__ for r in page]

    return RestaurantResponse(total=total, count=len(out), items=out)

@app.get("/restaurants/random", response_model=RandomResponse)
async def random_restaurants(open_only: bool = True, count: int = Query(1, ge=1, le=20)):
    pool = [r for r in cache.get() if (r.open if open_only else True)]
    if not pool: raise HTTPException(404, detail="가져올 항목이 없습니다.")
    sel = pool if len(pool)<=count else pyrandom.sample(pool, count)
    return RandomResponse(items=[r.__dict__ for r in sel])

@app.get("/photo/street")
async def photo_street(addr: str, w: int=600, h: int=360, fov: int=90, heading: int=0, pitch: int=0):
    if not Config.MAP_KEY:
        raise HTTPException(400, detail="Google Maps API key not configured")
    latlng = geocode(addr)
    if not latlng: raise HTTPException(400, detail="Geocoding failed")
    lat,lng = latlng
    url = ("https://maps.googleapis.com/maps/api/streetview"
           f"?size={w}x{h}&location={lat},{lng}&fov={fov}&heading={heading}&pitch={pitch}&key={Config.MAP_KEY}")
    return RedirectResponse(url)
