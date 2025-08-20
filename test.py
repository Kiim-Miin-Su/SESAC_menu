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

app = FastAPI()
# --- 환경 변수 ---
load_dotenv()
API_BASE = "http://openapi.seoul.go.kr:8088"
API_KEY = os.getenv("const_KEY")
map_KEY = os.getenv("map_KEY")
SERVICE = "LOCALDATA_072404_GD"
DATA_TYPE = "json"

url = f"{API_BASE}/{API_KEY}/{DATA_TYPE}/{SERVICE}/1/10"

@app.get("/test")
def test():
    return requests.get(url).json().get("LOCALDATA_072404_GD").get("row")[0]