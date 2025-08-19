import requests, json
from pprint import pprint

const_API = "http://openAPI.seoul.go.kr:8088"
const_TYPE = "json"
const_SERVICE = "LOCALDATA_072404_GD"
start_index = 1
end_index = 1000

size = 1000
pages = 50

url = f"{const_API}/{const_KEY}/{const_TYPE}/{const_SERVICE}/{start_index}/{end_index}"

for i in range(pages):
    # index
    start_index = 1 + size * i
    end_index = size * (i + 1)
    
    url = f"{const_API}/{const_KEY}/{const_TYPE}/{const_SERVICE}/{start_index + 1000 * i}/{end_index + 1000 * i}"
    response = requests.get(url, timeout=8)
    
    if response.status_code != 200:
        print(f"[{i}] HTTP {response.status_code}")
        break

    data = response.json()
    svc = data.get(const_SERVICE)

    if not svc:
        print(f"[{i}] No data")
        break
    
    rows = svc.get("row", [])
    if not rows:
        print(f"[{i}] No rows")
        break

def normalize_row(row: dict) -> dict:
    name = (row.get("BPLCNM") or "").strip()
    trd = (row.get("TRDSTATENM") or "").strip()
    dtl = (row.get("DTLSTATENM") or "").strip()
    is_open = (trd in ("영업", "정상", "영업/정상")) and (dtl in ("영업", "정상", "영업/정상", ""))
    
    addr = (row.get("RDNWHLADDR") or row.get("SITEWHLADDR") or "").strip()
    
    kind = (row.get("UPTAENM") or row.get("SNTUPTAENM") or "").strip()
    
    return {
        "name": name,
        "영업중": "Y" if is_open else "N",
        "주소": addr,
        "종류": kind,
    }
    
result = [normalize_row(row) for row in rows]

open_only = [r for r in result if r['영업중'] == 'Y' and any(a in r['주소'] for a in ["상일동", "고덕동"])]

# print(json.dumps(open_only, indent=2, ensure_ascii=False))
def all():
    return open_only

def random():
    return random.choice(open_only)

def only_korean():
    return [r for r in open_only if r['종류'] == '한식']

def only_chinese():
    return [r for r in open_only if r['종류'] == '중식']

def only_japanese():
    return [r for r in open_only if r['종류'] == '일식']

def only_western():
    return [r for r in open_only if r['종류'] == '양식']

def only_etc():
    return [r for r in open_only if r['종류'] not in ["한식", "중식", "일식", "양식"]]

print (json.dumps(only_korean(), indent=2, ensure_ascii=False))

# sample = "http://openapi.seoul.go.kr:8088/sample/xml/LOCALDATA_072404_GD/1/5/"

# uvicorn app:app --reload --port 8000

