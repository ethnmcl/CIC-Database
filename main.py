import os
import re
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timedelta

import httpx
import pytz
import dateparser
from fastapi import FastAPI, Header, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from dateutil.relativedelta import relativedelta

# === Environment ===
API_KEY = os.getenv("API_KEY")  # your shared secret for this API
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-en-v1.5")
LOCAL_TZ = pytz.timezone(os.getenv("TZ", "America/New_York"))
WEEK_START = (os.getenv("WEEK_START", "monday") or "monday").strip().lower()  # 'monday' or 'sunday'

if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

# Monday=0 ... Sunday=6 (Python weekday)
WEEKDAYS = {
    "monday": 0, "tuesday": 1, "wednesday": 2,
    "thursday": 3, "friday": 4, "saturday": 5, "sunday": 6
}
MONTHS = [
    "january","february","march","april","may","june",
    "july","august","september","october","november","december"
]
WEEK_START_IDX = 0 if WEEK_START == "monday" else 6  # Sunday-start vs Monday-start

app = FastAPI(title="Check-ins Search API (Supabase RPC)", version="1.1.0")

# === Auth guard ===
def require_key(authorization: Optional[str] = Header(None)):
    """
    If API_KEY is set, require 'Authorization: Bearer <API_KEY>'.
    If not set, routes are open (handy for local dev).
    """
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.split(" ", 1)[1].strip() != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

# === Startup / Shutdown ===
@app.on_event("startup")
async def on_startup():
    app.state.model = SentenceTransformer(MODEL_NAME)
    app.state.http = httpx.AsyncClient(
        base_url=f"{SUPABASE_URL}/rest/v1",
        headers={
            "apikey": SUPABASE_SERVICE_ROLE_KEY,
            "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        },
        timeout=15.0,
    )

@app.on_event("shutdown")
async def on_shutdown():
    try:
        await app.state.http.aclose()
    except Exception:
        pass

# === Helpers ===
def embed_text(texts: List[str]) -> List[List[float]]:
    vecs = app.state.model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]

def _day_start(dt: datetime) -> datetime:
    return dt.replace(hour=0, minute=0, second=0, microsecond=0)

def _week_start(dt: datetime) -> datetime:
    # Shift back to week start according to WEEK_START_IDX
    # Python weekday(): Monday=0...Sunday=6
    delta_days = (dt.weekday() - WEEK_START_IDX) % 7
    return _day_start(dt - timedelta(days=delta_days))

def _localize(naive_dt: datetime) -> datetime:
    # Localize naive datetime to LOCAL_TZ
    return LOCAL_TZ.localize(naive_dt)

def to_utc_iso(local_iso: str) -> str:
    return datetime.fromisoformat(local_iso).astimezone(pytz.UTC).isoformat()

def parse_phrase_to_range(phrase: str) -> Dict[str, str]:
    """
    Robust parser for human phrases into [start, end) in LOCAL_TZ.
    Supports:
      - last <weekday>
      - this week / last week (weekday start configurable via WEEK_START)
      - this month / last month
      - <month> [year]  e.g., "August" or "August 2024"
      - past <N> days|weeks|months
      - Q1/Q2/Q3/Q4 [year]
      - fallback to dateparser (English), returns 1-day range
    """
    s = (phrase or "").strip().lower()
    if not s:
        raise HTTPException(400, detail="Empty phrase")

    now = datetime.now(LOCAL_TZ)

    # --- last <weekday> ---
    m = re.fullmatch(r"last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", s)
    if m:
        target = WEEKDAYS[m.group(1)]
        delta = (now.weekday() - target) % 7
        delta = 7 if delta == 0 else delta  # ensure we go back at least one week if same day
        day = _day_start(now - timedelta(days=delta))
        return {"start": day.isoformat(), "end": (day + timedelta(days=1)).isoformat()}

    # --- this week / last week ---
    if s == "this week":
        start = _week_start(now)
        return {"start": start.isoformat(), "end": (start + timedelta(days=7)).isoformat()}
    if s == "last week":
        this_start = _week_start(now)
        start = this_start - timedelta(days=7)
        return {"start": start.isoformat(), "end": (start + timedelta(days=7)).isoformat()}

    # --- this month / last month ---
    if s == "this month":
        start = _localize(datetime(now.year, now.month, 1))
        end = _localize(datetime(now.year + (1 if now.month == 12 else 0),
                                  1 if now.month == 12 else now.month + 1, 1))
        return {"start": start.isoformat(), "end": end.isoformat()}
    if s == "last month":
        first_this = _localize(datetime(now.year, now.month, 1))
        start = _day_start(first_this - timedelta(days=1)).replace(day=1)  # first day of last month
        end = first_this
        return {"start": start.isoformat(), "end": end.isoformat()}

    # --- <month> [year]? e.g., August, August 2024 ---
    m = re.fullmatch(rf"({'|'.join(MONTHS)})(?:\s+(\d{{4}}))?", s)
    if m:
        month_name, year_str = m.group(1), m.group(2)
        month_idx = MONTHS.index(month_name) + 1
        year = int(year_str) if year_str else now.year
        start = _localize(datetime(year, month_idx, 1))
        if month_idx == 12:
            end = _localize(datetime(year + 1, 1, 1))
        else:
            end = _localize(datetime(year, month_idx + 1, 1))
        return {"start": start.isoformat(), "end": end.isoformat()}

    # --- past <N> (days|weeks|months) ---
    m = re.fullmatch(r"past\s+(\d+)\s*(day|days|week|weeks|month|months)", s)
    if m:
        n = int(m.group(1))
        unit = m.group(2)
        end = _day_start(now) + timedelta(days=1)  # up to end of today
        if unit.startswith("day"):
            start = end - timedelta(days=n)
        elif unit.startswith("week"):
            start = end - timedelta(weeks=n)
        else:  # months
            start = end - relativedelta(months=n)
        return {"start": start.isoformat(), "end": end.isoformat()}

    # --- quarters: Q1/Q2/Q3/Q4 [year]? ---
    m = re.fullmatch(r"q([1-4])(?:\s+(\d{4}))?", s)
    if m:
        q = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else now.year
        start_month = (q - 1) * 3 + 1
        start = _localize(datetime(year, start_month, 1))
        end = _localize(datetime(year, start_month, 1)) + relativedelta(months=3)
        return {"start": start.isoformat(), "end": end.isoformat()}

    # --- fallback to dateparser (English), produce a day-range ---
    settings = {
        "TIMEZONE": str(LOCAL_TZ),
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "past",
        "RELATIVE_BASE": now
    }
    dt = dateparser.parse(s, settings=settings, languages=["en"])
    if not dt:
        raise HTTPException(400, detail=f"Could not parse phrase: {phrase}\n")
    start = _day_start(dt.astimezone(LOCAL_TZ))
    end = start + timedelta(days=1)
    return {"start": start.isoformat(), "end": end.isoformat()}

# === Schemas ===
class IngestBody(BaseModel):
    id: str
    sender: Optional[str] = None
    username: Optional[str] = None
    slack_id: Optional[str] = None
    msg: str
    timestamp: Optional[str] = Field(None, description="ISO8601; if absent, now()")
    tags: Optional[List[str]] = []
    valid_checkin: Optional[bool] = True

class SearchFilters(BaseModel):
    phrase: Optional[str] = None
    start: Optional[str] = None
    end: Optional[str] = None
    sender: Optional[str] = None
    valid_only: Optional[bool] = None

class SearchBody(BaseModel):
    query: str
    k: int = 20
    filters: Optional[SearchFilters] = None
    return_fields: List[str] = ["id","ts","sender","username","msg","score"]

# === Routes ===
@app.get("/")
async def root():
    return {
        "ok": True,
        "hint": "Use /healthz, /ingest, /search, /phrases/resolve, /stats",
        "week_start": WEEK_START
    }

@app.get("/healthz")
async def health():
    return {"ok": True, "model": MODEL_NAME}

@app.get("/phrases/resolve")
async def resolve_phrase(phrase: str = Query(..., min_length=1), _: None = Depends(require_key)):
    r = parse_phrase_to_range(phrase)
    return {"phrase": phrase, "timezone": str(LOCAL_TZ), "range": r}

@app.post("/ingest")
async def ingest(body: IngestBody, _: None = Depends(require_key)):
    # timestamp handling
    if body.timestamp:
        ts_utc = datetime.fromisoformat(body.timestamp).astimezone(pytz.UTC).isoformat()
    else:
        ts_utc = datetime.now(pytz.UTC).isoformat()

    # embed
    vec = embed_text([body.msg])[0]

    payload = {
        "_id": body.id,
        "_sender": body.sender,
        "_username": body.username,
        "_slack_id": body.slack_id,
        "_msg": body.msg,
        "_ts": ts_utc,
        "_tags": body.tags or [],
        "_valid": True if body.valid_checkin is not False else False,
        "_embedding": vec,
    }

    # RPC: upsert_checkin
    r = await app.state.http.post("/rpc/upsert_checkin", json=payload)
    if r.status_code >= 300:
        raise HTTPException(r.status_code, detail=f"Supabase RPC error: {r.text[:300]}")
    return {"ok": True, "id": body.id}

@app.post("/search")
async def search(body: SearchBody, _: None = Depends(require_key)):
    q_vec = embed_text([body.query])[0]

    start_utc = end_utc = None
    if body.filters:
        if body.filters.phrase:
            rng = parse_phrase_to_range(body.filters.phrase)
            start_utc, end_utc = to_utc_iso(rng["start"]), to_utc_iso(rng["end"])
        if body.filters.start:
            start_utc = to_utc_iso(body.filters.start) if "T" in body.filters.start else to_utc_iso(LOCAL_TZ.localize(datetime.fromisoformat(body.filters.start)).isoformat())
        if body.filters.end:
            end_utc = to_utc_iso(body.filters.end) if "T" in body.filters.end else to_utc_iso(LOCAL_TZ.localize(datetime.fromisoformat(body.filters.end)).isoformat())

    payload = {
        "q_embedding": q_vec,
        "k": max(1, min(body.k, 100)),
        "start_ts": start_utc,
        "end_ts": end_utc,
        "sender_eq": body.filters.sender if body.filters and body.filters.sender else None,
        "valid_only": body.filters.valid_only if body.filters else None
    }

    r = await app.state.http.post("/rpc/search_checkins", json=payload)
    if r.status_code >= 300:
        raise HTTPException(r.status_code, detail=f"Supabase RPC error: {r.text[:300]}")
    rows = r.json()

    # trim to requested fields
    out = []
    for row in rows:
        item = {f: row.get(f) for f in body.return_fields if f in row or f == "score"}
        if "score" in item and item["score"] is not None:
            item["score"] = float(item["score"])
        out.append(item)

    return {"results": out, "used": {"semantic": True}}

@app.get("/stats")
async def stats(
    phrase: Optional[str] = None,
    bucket: Literal["weekly","monthly"] = "weekly",
    _: None = Depends(require_key)
):
    if phrase:
        rng = parse_phrase_to_range(phrase)
        start_utc, end_utc = to_utc_iso(rng["start"]), to_utc_iso(rng["end"])
    else:
        end = datetime.now(pytz.UTC)
        start = end - timedelta(days=30)
        start_utc, end_utc = start.isoformat(), end.isoformat()

    payload = { "bucket": bucket, "start_ts": start_utc, "end_ts": end_utc }
    r = await app.state.http.post("/rpc/stats_range", json=payload)
    if r.status_code >= 300:
        raise HTTPException(r.status_code, detail=f"Supabase RPC error: {r.text[:300]}")
    return {
        "bucket": bucket,
        "range": {"start": start_utc, "end": end_utc},
        **r.json()
    }
