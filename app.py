import os, re, time, threading
from dataclasses import dataclass
from math import radians, sin, cos, sqrt, atan2
from typing import Any, Dict, List, Optional, Tuple, Literal

import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import RedirectResponse, ORJSONResponse
from pydantic import BaseModel
import random as pyrandom

try:
    # 주소→좌표 캐시 (요금 폭탄 방지)
    from cachetools import TTLCache
except Exception:  # cachetools 없으면 간단 대체
    TTLCache = None  # type: ignore

# ──────────────────────────────────────────────────────────────────────────────
# 설정
# ──────────────────────────────────────────────────────────────────────────────
load_dotenv()

class Config:
    API_BASE: str = "http://openapi.seoul.go.kr:8088"
    API_KEY: str = os.getenv("const_KEY") or ""
    MAP_KEY: Optional[str] = os.getenv("map_KEY")
    SERVICE: str = "LOCALDATA_072404_GD"
    DATA_TYPE: str = "json"

    # 프리패치/캐시
    CACHE_REFRESH_SEC: int = int(os.getenv("CACHE_REFRESH_SEC", "600"))
    PREFETCH_PAGES: int = int(os.getenv("PREFETCH_PAGES", "6"))
    PREFETCH_SIZE: int = int(os.getenv("PREFETCH_SIZE", "1000"))

    # API 방어
    MAX_EXCLUDE_PARAMS: int = 400
    ALLOWED_ORIGINS: List[str] = [
        "https://sesac-menu.pe.kr",  # 프론트 도메인
        "http://localhost:5173",     # 로컬 개발(원하면 제거)
        "http://127.0.0.1:5173",
        "https://kiim-miin-su.github.io",
    ]

    # 레이트리밋 정책(필요 시 조절)
    RL_GLOBAL_CAPACITY: int = 120
    RL_GLOBAL_REFILL: float = 1.5     # 초당 토큰(≈분당 90)
    RL_ROUTE_WINDOW: int = 60         # 초
    RL_ROUTE_LIMIT: int = 80          # 라우트당 분당
    RL_DAILY_LIMIT: int = 1500        # IP 일일 총 호출
    RL_EXPENSIVE_ROUTES: Dict[str, Tuple[int, int]] = {
        "/photo/street": (30, 3600),  # 시간당 30회
    }
    RL_IP_ALLOW: List[str] = [x for x in os.getenv("IP_ALLOW", "").split(",") if x]

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
    kind: str          # 원문 업종
    category: str      # 표준 카테고리(한/중/일/양/카페/호프통닭/etc)
    loc_x: Optional[float]
    loc_y: Optional[float]

class RestaurantResponse(BaseModel):
    total: int
    count: int
    items: List[Dict[str, Any]]

class RandomResponse(BaseModel):
    items: List[Dict[str, Any]]

# ──────────────────────────────────────────────────────────────────────────────
# 분류/유틸
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
        candidates = [("X", "Y"), ("LOC_X", "LOC_Y"), ("LON", "LAT"), ("longitude", "latitude")]
        for lx, ly in candidates:
            lon, lat = row.get(lx), row.get(ly)
            if lon not in (None, "") and lat not in (None, ""):
                try:
                    return float(lon), float(lat)
                except ValueError:
                    pass
        return None, None

    @classmethod
    def classify_category(cls, raw_kind: str) -> str:
        k = (raw_kind or "").replace(" ", "")
        for cat, pat in cls.RE_RULES:
            if pat.search(k):
                return cat
        return "etc"

def calc_distance(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    try:
        lon1, lat1, lon2, lat2 = map(float, [lon1, lat1, lon2, lat2])
    except (TypeError, ValueError):
        return float("inf")
    R = 6371000.0
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat/2)**2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon/2)**2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def make_key(item: Restaurant) -> str:
    # 관리번호 있으면 고유키로 사용
    if item.id:
        return item.id
    return f"{(item.name or '').strip()}|{(item.addr or '').strip()}"

# ──────────────────────────────────────────────────────────────────────────────
# 외부 API / 프리패치 캐시
# ──────────────────────────────────────────────────────────────────────────────
def fetch_page(start: int, end: int) -> dict:
    url = f"{Config.API_BASE}/{Config.API_KEY}/{Config.DATA_TYPE}/{Config.SERVICE}/{start}/{end}"
    r = requests.get(url, timeout=8)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Upstream HTTP {r.status_code}")
    data = r.json()
    svc = data.get(Config.SERVICE)
    return svc if svc else {"row": []}

def iter_rows(pages: int, size: int):
    for i in range(pages):
        start = 1 + size * i
        end = size * (i + 1)
        rows = fetch_page(start, end).get("row", []) or []
        if not rows:
            break
        for row in rows:
            yield row

class RestaurantCache:
    def __init__(self):
        self._cache: List[Restaurant] = []
        self._ts: float = 0.0
        self._lock = threading.Lock()

    def get(self) -> List[Restaurant]:
        with self._lock:
            return self._cache.copy()

    def ts(self) -> float:
        with self._lock:
            return self._ts

    def refresh_once(self, pages=Config.PREFETCH_PAGES, size=Config.PREFETCH_SIZE):
        raw: List[dict] = list(iter_rows(pages, size))
        norm: List[Restaurant] = []
        for r in raw:
            lon, lat = RestaurantClassifier.extract_coords(r)
            kind_raw = (r.get("UPTAENM") or r.get("SNTUPTAENM") or "").strip()
            item = Restaurant(
                id=(r.get("MGTNO") or "").strip(),
                name=(r.get("BPLCNM") or "").strip(),
                open=RestaurantClassifier.is_open(r),
                addr=RestaurantClassifier.pick_addr(r),
                kind=kind_raw,
                category=RestaurantClassifier.classify_category(kind_raw),
                loc_x=lon,
                loc_y=lat,
            )
            norm.append(item)
        with self._lock:
            self._cache = norm
            self._ts = time.time()

    def start_loop(self):
        def _loop():
            while True:
                try:
                    self.refresh_once()
                except Exception as e:
                    # 조용히 재시도
                    pass
                time.sleep(Config.CACHE_REFRESH_SEC)
        threading.Thread(target=_loop, daemon=True).start()

CACHE = RestaurantCache()

# ──────────────────────────────────────────────────────────────────────────────
# geocoding 캐시 (요금 절감)
# ──────────────────────────────────────────────────────────────────────────────
if TTLCache:
    GEOCODE_CACHE = TTLCache(maxsize=5000, ttl=7*24*3600)
else:
    GEOCODE_CACHE = {}  # type: ignore

def geocode_address(addr: str):
    if not Config.MAP_KEY:
        return None
    if addr in GEOCODE_CACHE:
        return GEOCODE_CACHE[addr]
    url = "https://maps.googleapis.com/maps/api/geocode/json"
    res = requests.get(url, params={"address": addr, "key": Config.MAP_KEY}, timeout=5).json()
    if res.get("status") != "OK":
        return None
    loc = res["results"][0]["geometry"]["location"]
    GEOCODE_CACHE[addr] = loc
    return loc

# ──────────────────────────────────────────────────────────────────────────────
# FastAPI 앱/미들웨어
# ──────────────────────────────────────────────────────────────────────────────
app = FastAPI(title="Seoul Food LocalData API", default_response_class=ORJSONResponse)

# GZip
app.add_middleware(GZipMiddleware, minimum_size=1024)

# CORS — 반드시 프론트 도메인만 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=Config.ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "HEAD", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# ── Rate limit / Quota (in-memory) ─────────────────────────────────────────────
def get_client_ip(req: Request) -> str:
    xff = req.headers.get("x-forwarded-for")
    if xff:
        return xff.split(",")[0].strip()
    return req.client.host or "0.0.0.0"

class TokenBucket:
    def __init__(self, capacity: int, refill_rate_per_sec: float):
        self.capacity = capacity
        self.refill = refill_rate_per_sec
        self.tokens = capacity
        self.last = time.time()

    def take(self, cost: float = 1.0) -> bool:
        now = time.time()
        delta = now - self.last
        self.tokens = min(self.capacity, self.tokens + delta * self.refill)
        self.last = now
        if self.tokens >= cost:
            self.tokens -= cost
            return True
        return False

class RateLimiter:
    def __init__(self):
        from collections import defaultdict, deque
        self.TokenLog = deque  # type: ignore
        self.ip_buckets: Dict[str, TokenBucket] = {}
        self.per_route_logs: Dict[Tuple[str, str], Any] = defaultdict(self.TokenLog)  # deque[float]
        self.daily_quota: Dict[str, Tuple[int, int]] = {}  # ip -> (count, yyyymmdd)
        self.ip_allow = set(Config.RL_IP_ALLOW)

    def check(self, ip: str, path: str) -> Optional[Dict[str, Any]]:
        if ip in self.ip_allow:
            return None

        # 1) 글로벌 토큰 버킷
        bucket = self.ip_buckets.get(ip)
        if not bucket:
            bucket = self.ip_buckets.setdefault(
                ip, TokenBucket(Config.RL_GLOBAL_CAPACITY, Config.RL_GLOBAL_REFILL)
            )
        if not bucket.take(1.0):
            return {"code": 429, "msg": "Too Many Requests (global)", "retry": "2"}

        now = time.time()

        # 2) 라우트별 슬라이딩 윈도우
        q = self.per_route_logs[(ip, path)]
        q.append(now)
        while q and now - q[0] > Config.RL_ROUTE_WINDOW:
            q.popleft()
        if len(q) > Config.RL_ROUTE_LIMIT:
            return {"code": 429, "msg": "Too Many Requests (route)", "retry": "2"}

        # 3) 일일 쿼터
        today = int(time.strftime("%Y%m%d"))
        cnt, day = self.daily_quota.get(ip, (0, today))
        if day != today:
            cnt, day = 0, today
        cnt += 1
        self.daily_quota[ip] = (cnt, day)
        if cnt > Config.RL_DAILY_LIMIT:
            return {"code": 429, "msg": "Daily quota exceeded"}

        # 4) 비싼 라우트 개별 제한
        for prefix, (limit, win) in Config.RL_EXPENSIVE_ROUTES.items():
            if path.startswith(prefix):
                k = (ip, f"{prefix}:exp")
                dq = self.per_route_logs[k]
                dq.append(now)
                while dq and now - dq[0] > win:
                    dq.popleft()
                if len(dq) > limit:
                    return {"code": 429, "msg": "Too Many Requests (photo quota)"}
        return None

rate_limiter = RateLimiter()

@app.middleware("http")
async def rate_limit_middleware(request: Request, call_next):
    if request.method == "OPTIONS" or request.url.path in ("/health", "/robots.txt"):
        return await call_next(request)
    ip = get_client_ip(request)
    verdict = rate_limiter.check(ip, request.url.path)
    if verdict:
        return ORJSONResponse({"detail": verdict["msg"]}, status_code=verdict["code"],
                              headers={"Retry-After": verdict.get("retry", "1"),
                                       "X-RateLimit-Policy": "per-ip bucket+sliding+daily"})
    resp = await call_next(request)
    resp.headers.setdefault("X-RateLimit-Policy", "per-ip bucket+sliding+daily")
    return resp

# ──────────────────────────────────────────────────────────────────────────────
# 스타트업: 프리패치 루프 시작
# ──────────────────────────────────────────────────────────────────────────────
@app.on_event("startup")
def _startup():
    # 첫 로드가 비어있지 않게 즉시 1회
    try:
        CACHE.refresh_once()
    finally:
        pass
    CACHE.start_loop()

# ──────────────────────────────────────────────────────────────────────────────
# 라우트
# ──────────────────────────────────────────────────────────────────────────────
@app.get("/robots.txt")
def robots():
    # 얌전한 크롤러는 줄어듭니다
    return ("User-agent: *\nDisallow: /\n", 200, {"Content-Type": "text/plain"})

@app.get("/health")
def health():
    return {"ok": True, "cache_ts": CACHE.ts(), "prefetched": len(CACHE.get())}

@app.get("/photo/street")
def photo_street(
    addr: str,
    w: int = 600,
    h: int = 360,
    fov: int = 90,
    heading: int = 0,
    pitch: int = 0,
):
    if not Config.MAP_KEY:
        raise HTTPException(status_code=400, detail="Google Maps API key not configured")
    loc = geocode_address(addr)
    if not loc:
        raise HTTPException(status_code=400, detail=f"Geocoding failed for {addr}")
    url = (
        "https://maps.googleapis.com/maps/api/streetview"
        f"?size={w}x{h}&location={loc['lat']},{loc['lng']}&fov={fov}"
        f"&heading={heading}&pitch={pitch}&key={Config.MAP_KEY}"
    )
    return RedirectResponse(url)

@app.get("/restaurants", response_model=RestaurantResponse)
def list_restaurants(
    q: Optional[str] = Query(None, description="가게명 부분검색"),
    area: Optional[str] = Query(None, description="지역 키워드(쉼표구분) 예: 상일,고덕,천호"),
    kind: Optional[Literal["전체", "한", "중", "일", "양", "카페", "호프통닭", "etc"]] = Query(None),
    open_only: bool = Query(True),
    limit: int = Query(20, ge=1, le=200),
    offset: int = Query(0, ge=0),
    curr_loc_x: Optional[float] = Query(None, description="현재 경도"),
    curr_loc_y: Optional[float] = Query(None, description="현재 위도"),
    distance: Optional[int] = Query(None, description="반경(m)"),
    exclude: Optional[List[str]] = Query(None, description="이미 본 key 목록(name|addr 또는 id)"),
    order: Optional[Literal["default", "rand"]] = Query("default"),
    seed: Optional[int] = Query(None, description="order=rand 시 시드"),
):
    # exclude 너무 길면 뒤쪽 N개만 유지(쿼리 길이 보호)
    if exclude and len(exclude) > Config.MAX_EXCLUDE_PARAMS:
        exclude = exclude[-Config.MAX_EXCLUDE_PARAMS:]

    # 지역 키워드 정규식
    area_patterns: List[re.Pattern] = []
    if area:
        tokens = [t.strip() for t in area.split(",") if t.strip()]
        area_patterns = [re.compile(re.escape(tok)) for tok in tokens]

    def area_match(addr: str) -> bool:
        if not area_patterns:
            return True
        return any(p.search(addr) for p in area_patterns)

    def kind_match(category: str) -> bool:
        if kind in (None, "전체"):
            return True
        if kind == "etc":
            return category not in {"한", "중", "일", "양", "카페", "호프통닭"}
        return category == kind

    exclude_set = set(exclude or [])
    need_dist = (curr_loc_x is not None and curr_loc_y is not None and distance is not None)

    items: List[Dict[str, Any]] = []
    rows = CACHE.get()
    if order == "rand":
        rng = pyrandom.Random(seed) if seed is not None else pyrandom
        rows = rows.copy()
        rng.shuffle(rows)

    q_norm = (q or "").lower()

    for r in rows:
        if exclude_set and (make_key(r) in exclude_set or r.id in exclude_set):
            continue
        if open_only and not r.open:
            continue
        if not kind_match(r.category):
            continue
        if area and not area_match(r.addr):
            continue
        if q and (q_norm not in (r.name or "").lower()):
            continue

        # 거리 필터: 좌표 없으면 통과(데이터 누락 방지; 원하면 제외로 바꿔도 됨)
        if need_dist and r.loc_x is not None and r.loc_y is not None:
            d = calc_distance(curr_loc_x, curr_loc_y, r.loc_x, r.loc_y)
            if d > float(distance):
                continue

        items.append({
            "id": r.id,
            "name": r.name,
            "open": r.open,
            "addr": r.addr,
            "kind": r.kind,
            "category": r.category,
            "loc_x": r.loc_x,
            "loc_y": r.loc_y,
        })

    total = len(items)
    items = items[offset: offset + limit]
    return RestaurantResponse(total=total, count=len(items), items=items)

@app.get("/restaurants/random", response_model=RandomResponse)
def random_restaurants(open_only: bool = True, count: int = Query(1, ge=1, le=20)):
    pool = [r for r in CACHE.get() if (r.open if open_only else True)]
    if not pool:
        raise HTTPException(status_code=404, detail="가져올 항목이 없습니다.")
    sel = pool if len(pool) <= count else pyrandom.sample(pool, count)
    return RandomResponse(items=[{
        "id": r.id, "name": r.name, "open": r.open, "addr": r.addr,
        "kind": r.kind, "category": r.category, "loc_x": r.loc_x, "loc_y": r.loc_y
    } for r in sel])
