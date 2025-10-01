import os
from typing import List, Optional, Dict, Any, Literal
from datetime import datetime, timedelta

import httpx
import pytz
import dateparser
from fastapi import FastAPI, Header, HTTPException, Depends, Query
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

# --- Env ---
API_KEY = os.getenv("API_KEY")  # your own shared secret for this API
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_SERVICE_ROLE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY")
MODEL_NAME = os.getenv("MODEL_NAME", "BAAI/bge-small-en-v1.5")
LOCAL_TZ = pytz.timezone(os.getenv("TZ", "America/New_York"))

if not (SUPABASE_URL and SUPABASE_SERVICE_ROLE_KEY):
    raise RuntimeError("Missing SUPABASE_URL or SUPABASE_SERVICE_ROLE_KEY")

app = FastAPI(title="Check-ins Search API (Supabase RPC)", version="1.0.0")

# --- Auth guard ---
def require_key(authorization: Optional[str] = Header(None)):
    # If API_KEY is unset, routes are open (handy for local), otherwise require Bearer
    if not API_KEY:
        return
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Missing bearer token")
    if authorization.split(" ", 1)[1].strip() != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid token")

# --- Startup/shutdown ---
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

# --- Helpers ---
def embed_text(texts: List[str]) -> List[List[float]]:
    vecs = app.state.model.encode(texts, normalize_embeddings=True)
    return [v.tolist() for v in vecs]

def parse_phrase_to_range(phrase: str) -> Dict[str, str]:
    settings = {
        "TIMEZONE": str(LOCAL_TZ),
        "RETURN_AS_TIMEZONE_AWARE": True,
        "PREFER_DATES_FROM": "past",
        "RELATIVE_BASE": datetime.now(LOCAL_TZ),
    }
    dt = dateparser.parse(phrase, settings=settings)
    if not dt:
        raise HTTPException(400, detail=f"Could not parse phrase: {phrase}")

    lower = phrase.lower().strip()
    months = ["january","february","march","april","may","june","july",
              "august","september","october","november","december"]
    if lower in months:
        start = LOCAL_TZ.localize(datetime(dt.year, dt.month, 1, 0, 0, 0))
        if dt.month == 12:
            end = LOCAL_TZ.localize(datetime(dt.year + 1, 1, 1, 0, 0, 0))
        else:
            end = LOCAL_TZ.localize(datetime(dt.year, dt.month + 1, 1, 0, 0, 0))
    else:
        start = LOCAL_TZ.localize(datetime(dt.year, dt.month, dt.day, 0, 0, 0))
        end = start + timedelta(days=1)

    return {"start": start.isoformat(), "end": end.isoformat()}

def to_utc_iso(local_iso: str) -> str:
    return datetime.fromisoformat(local_iso).astimezone(pytz.UTC).isoformat()

# --- Schemas ---
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

# --- Endpoints ---
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
