import os
import re
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timedelta

import httpx
import pytz
import dateparser
from dateparser.search import search_dates
from fastapi import FastAPI, Header, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer
from dateutil.relativedelta import relativedelta
from fastapi.middleware.cors import CORSMiddleware

# === Environment ===
API_KEY = os.getenv("API_KEY")  # your shared secret for this API (optional)
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-en-v1.5")
LOCAL_TZ = pytz.timezone(os.getenv("TZ", "America/New_York"))
DEFAULT_WEEK_START = (os.getenv("WEEK_START", "monday") or "monday").strip().lower()  # 'monday' or 'sunday'

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

# regex helpers for extraction
TIME_PATTERNS = [
    r"\blast\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)\b",
    r"\b(this|last)\s+(week|month)\b",
    r"\b(past|last)\s+\d+\s+(?:day|days|week|weeks|month|months)\b",
    r"\bq[1-4](?:\s+\d{4})?\b",
    r"\b(today|yesterday)\b",
    r"\b(january|february|march|april|may|june|july|august|september|october|november|december)(?:\s+\d{4})?\b",
]

app = FastAPI(title="Check-ins Search API (Supabase RPC)", version="1.3.0")

# === CORS (allow Base44 app + local dev) ===
ALLOWED_ORIGINS = [
    "https://flow-state-d25e6b4b.base44.app",
    "http://localhost:3000",
    "http://127.0.0.1:3000",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,    # set to ["*"] temporarily if debugging
    allow_credentials=False,          # True only if you need cookies/auth
    allow_methods=["POST", "GET", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization", "Accept"],
    expose_headers=["Content-Type"],
    max_age=600,
)

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

def _week_start(dt: datetime, week_start: str) -> datetime:
    idx = 0 if week_start == "monday" else 6  # monday=0, sunday=6 baseline
    delta_days = (dt.weekday() - idx) % 7
    return _day_start(dt - timedelta(days=delta_days))

def _localize(tz: pytz.BaseTzInfo, naive_dt: datetime) -> datetime:
    return tz.localize(naive_dt)

def to_utc_iso(local_iso: str) -> str:
    return datetime.fromisoformat(local_iso).astimezone(pytz.UTC).isoformat()

def extract_time_subphrase(text: str, tz: pytz.BaseTzInfo) -> Optional[str]:
    s = (text or "").lower()
    # 1) Try explicit regexes (strong signals)
    for pat in TIME_PATTERNS:
        m = re.search(pat, s)
        if m:
            return m.group(0)
    # 2) Fall back to searching for any date/time phrase
    settings = {
        "TIMEZONE": str(tz),
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "past",
        "RELATIVE_BASE": datetime.now(tz)
    }
    found = search_dates(s, settings=settings, languages=["en"])
    if found:
        return found[0][0]  # matched substring (e.g., "july 2025")
    return None

def parse_phrase_to_range(
    phrase: str,
    *,
    tz: Optional[pytz.BaseTzInfo] = None,
    week_start: Optional[str] = None
) -> Dict[str, str]:
    """
    Robust parser for human phrases into [start, end) in tz (default LOCAL_TZ).
    Supports:
      - last <weekday>
      - this week / last week (weekday start configurable via week_start)
      - this month / last month
      - <month> [year]  e.g., "August" or "August 2024"
      - (past|last) <N> days|weeks|months
      - Q1/Q2/Q3/Q4 [year]
      - today / yesterday
      - fallback to dateparser (English), returns 1-day range

    Returns dict with keys: start, end, source
    """
    tz = tz or LOCAL_TZ
    week_start = (week_start or DEFAULT_WEEK_START).strip().lower()
    s_in = (phrase or "").strip()
    s = s_in.lower()
    if not s:
        raise HTTPException(400, detail="Empty phrase")

    now = datetime.now(tz)

    # --- last <weekday> ---
    m = re.fullmatch(r"last\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)", s)
    if m:
        target = WEEKDAYS[m.group(1)]
        delta = (now.weekday() - target) % 7
        delta = 7 if delta == 0 else delta  # at least one full week back
        day = _day_start(now - timedelta(days=delta))
        return {"start": day.isoformat(), "end": (day + timedelta(days=1)).isoformat(), "source": "weekday"}

    # --- today / yesterday ---
    if s == "today":
        start = _day_start(now)
        return {"start": start.isoformat(), "end": (start + timedelta(days=1)).isoformat(), "source": "day"}
    if s == "yesterday":
        end = _day_start(now)
        start = end - timedelta(days=1)
        return {"start": start.isoformat(), "end": end.isoformat(), "source": "day"}

    # --- this week / last week ---
    if s == "this week":
        start = _week_start(now, week_start)
        return {"start": start.isoformat(), "end": (start + timedelta(days=7)).isoformat(), "source": "week"}
    if s == "last week":
        this_start = _week_start(now, week_start)
        start = this_start - timedelta(days=7)
        return {"start": start.isoformat(), "end": (start + timedelta(days=7)).isoformat(), "source": "week"}

    # --- this month / last month ---
    if s == "this month":
        start = _localize(tz, datetime(now.year, now.month, 1))
        end = _localize(tz, datetime(now.year + (1 if now.month == 12 else 0),
                                      1 if now.month == 12 else now.month + 1, 1))
        return {"start": start.isoformat(), "end": end.isoformat(), "source": "month"}
    if s == "last month":
        first_this = _localize(tz, datetime(now.year, now.month, 1))
        start = _day_start(first_this - timedelta(days=1)).replace(day=1)  # first day of last month
        end = first_this
        return {"start": start.isoformat(), "end": end.isoformat(), "source": "month"}

    # --- <month> [year]? e.g., August, August 2024 ---
    m = re.fullmatch(rf"({'|'.join(MONTHS)})(?:\s+(\d{{4}}))?", s)
    if m:
        month_name, year_str = m.group(1), m.group(2)
        month_idx = MONTHS.index(month_name) + 1
        year = int(year_str) if year_str else now.year
        start = _localize(tz, datetime(year, month_idx, 1))
        end = _localize(tz, datetime(year + 1, 1, 1)) if month_idx == 12 else _localize(tz, datetime(year, month_idx + 1, 1))
        return {"start": start.isoformat(), "end": end.isoformat(), "source": "month"}

    # --- (past|last) <N> (days|weeks|months) ---
    m = re.fullmatch(r"(past|last)\s+(\d+)\s*(day|days|week|weeks|month|months)", s)
    if m:
        n = int(m.group(2))
        unit = m.group(3)
        end = _day_start(now) + timedelta(days=1)  # through end of today
        if unit.startswith("day"):
            start = end - timedelta(days=n)
        elif unit.startswith("week"):
            start = end - timedelta(weeks=n)
        else:  # months
            start = end - relativedelta(months=n)
        return {"start": start.isoformat(), "end": end.isoformat(), "source": "relative"}

    # --- quarters: Q1/Q2/Q3/Q4 [year]? ---
    m = re.fullmatch(r"q([1-4])(?:\s+(\d{4}))?", s)
    if m:
        q = int(m.group(1))
        year = int(m.group(2)) if m.group(2) else now.year
        start_month = (q - 1) * 3 + 1
        start = _localize(tz, datetime(year, start_month, 1))
        end = start + relativedelta(months=3)
        return {"start": start.isoformat(), "end": end.isoformat(), "source": "quarter"}

    # --- fallback to dateparser (English), produce a day-range ---
    settings = {
        "TIMEZONE": str(tz),
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "past",
        "RELATIVE_BASE": now
    }
    dt = dateparser.parse(s, settings=settings, languages=["en"])
    if not dt:
        raise HTTPException(400, detail=f"Could not parse phrase: {phrase}\n")
    start = _day_start(dt.astimezone(tz))
    end = start + timedelta(days=1)
    return {"start": start.isoformat(), "end": end.isoformat(), "source": "dateparser"}

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

# /interpret request schema
class InterpretDefaults(BaseModel):
    timezone: Optional[str] = None           # e.g., "America/New_York"
    week_start: Optional[str] = None         # "monday" | "sunday"
    fallback_range: Optional[str] = None     # e.g., "past 30 days"

class InterpretOptions(BaseModel):
    return_suggestions: bool = True
    infer_sender: Optional[str] = None
    k: int = 20
    return_fields: List[str] = ["id","ts","sender","username","msg","score"]
    run_search: bool = True   # NEW: run semantic search and include results

class InterpretBody(BaseModel):
    text: str
    defaults: Optional[InterpretDefaults] = None
    options: Optional[InterpretOptions] = None

# === Routes ===
@app.get("/")
async def root():
    return {
        "ok": True,
        "hint": "Use /healthz, /ingest, /search, /phrases/resolve, /interpret, /stats",
        "week_start": DEFAULT_WEEK_START
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

# === /interpret ===
class InterpretResponse(BaseModel):
    ok: bool

@app.post("/interpret")
async def interpret(body: InterpretBody, _: None = Depends(require_key)):
    """
    Takes free-form text like "what did I do July" and returns:
      - cleaned query text (non-time words)
      - detected time range (start/end in local tz)
      - ready-to-use /search payload
      - optionally, immediate semantic search results (run_search=True)
    """
    text = (body.text or "").strip()
    if not text:
        raise HTTPException(400, detail="Missing 'text'")

    tz = LOCAL_TZ
    week_start = DEFAULT_WEEK_START
    if body.defaults:
        if body.defaults.timezone:
            try:
                tz = pytz.timezone(body.defaults.timezone)
            except Exception:
                pass  # keep LOCAL_TZ if invalid
        if body.defaults.week_start and body.defaults.week_start.lower() in ("monday","sunday"):
            week_start = body.defaults.week_start.lower()

    # extract time sub-phrase from text
    sub = extract_time_subphrase(text, tz)
    rng = None
    time_source = None
    extracted = None
    suggestions: List[str] = []

    if sub:
        extracted = sub
        parsed = parse_phrase_to_range(sub, tz=tz, week_start=week_start)
        rng = {"start": parsed["start"], "end": parsed["end"], "tz": str(tz)}
        time_source = parsed.get("source", "detected")

        # simple suggestions: if only a month name, suggest current and previous year
        m = re.fullmatch(rf"({'|'.join(MONTHS)})", sub.strip().lower())
        if m and (not body.options or body.options.return_suggestions):
            now = datetime.now(tz)
            mon = m.group(1).capitalize()
            suggestions = [f"{mon} {now.year}", f"{mon} {now.year-1}"]

    # build the query by removing the extracted time phrase
    query = text
    if extracted:
        # case-insensitive removal of the first instance
        pattern = re.compile(re.escape(extracted), re.IGNORECASE)
        query = pattern.sub("", query, count=1).strip()
        # collapse extra spaces
        query = re.sub(r"\s+", " ", query).strip()

    # If still no time range, optionally use fallback
    used_fallback = False
    if rng is None:
        if body.defaults and body.defaults.fallback_range:
            parsed = parse_phrase_to_range(body.defaults.fallback_range, tz=tz, week_start=week_start)
            rng = {"start": parsed["start"], "end": parsed["end"], "tz": str(tz), "confidence": 0.2}
            time_source = "fallback"
            used_fallback = True
        else:
            return {
                "ok": False,
                "error": {
                    "code": "NO_TIME_FOUND",
                    "message": "No time phrase detected in input and no fallback_range provided."
                },
                "hints": [
                    "Add a time phrase like 'last week', 'August', 'past 30 days'",
                    "Or pass defaults.fallback_range"
                ],
                "query_guess": query or text
            }

    # build suggested /search payload
    opt = body.options or InterpretOptions()
    search_payload = {
        "query": query or text,  # if query becomes empty, fall back to full text
        "k": max(1, min(opt.k, 100)),
        "filters": {
            "start": rng["start"],
            "end": rng["end"],
            "sender": opt.infer_sender,
            "valid_only": None
        },
        "return_fields": opt.return_fields
    }

    # NEW: optionally run the semantic search now and include results
    results = None
    if opt.run_search:
        # embed query
        q_vec = embed_text([search_payload["query"]])[0]

        # convert local window to UTC for the RPC
        start_utc = to_utc_iso(search_payload["filters"]["start"])
        end_utc   = to_utc_iso(search_payload["filters"]["end"])

        rpc_payload = {
            "q_embedding": q_vec,
            "k": search_payload["k"],
            "start_ts": start_utc,
            "end_ts": end_utc,
            "sender_eq": search_payload["filters"]["sender"],
            "valid_only": search_payload["filters"].get("valid_only")
        }
        r2 = await app.state.http.post("/rpc/search_checkins", json=rpc_payload)
        if r2.status_code >= 300:
            raise HTTPException(r2.status_code, detail=f"Supabase RPC error: {r2.text[:300]}")
        rows = r2.json()

        # trim to requested fields
        results = []
        for row in rows:
            item = {f: row.get(f) for f in opt.return_fields if f in row or f == "score"}
            if "score" in item and item["score"] is not None:
                item["score"] = float(item["score"])
            results.append(item)

    resp: Dict[str, Any] = {
        "ok": True,
        "input": {
            "text": body.text,
            "timezone": str(tz),
            "week_start": week_start
        },
        "query": query or text,
        "time": {
            "phrase_raw": body.text,
            "phrase_extracted": extracted,
            "source": time_source,
            "start": rng["start"],
            "end": rng["end"],
            "tz": rng["tz"]
        },
        "search_payload": search_payload
    }
    if suggestions and (not used_fallback):
        resp["suggestions"] = suggestions
    if results is not None:
        resp["results"] = results

    return resp
