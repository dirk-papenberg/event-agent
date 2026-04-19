from functools import lru_cache

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

load_dotenv()

from agent import (
    find_events,  # noqa: E402  (nach load_dotenv, damit env beim Import steht)
)
from schemas import EventList  # noqa: E402

app = FastAPI(title="Family Event Agent", version="0.1.0")


class EventRequest(BaseModel):
    ort: str = Field(description="z.B. 'München' oder 'Stuttgart West'")
    zeitraum: str = Field(
        description="Freitext, z.B. '25.-27. April 2026' oder 'nächstes Wochenende'"
    )


@app.post("/events", response_model=EventList)
def search_events(req: EventRequest) -> EventList:
    try:
        return _cached_find(req.zeitraum.strip())
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Agent-Fehler: {e}") from e


@app.get("/healthz")
def healthz() -> dict:
    return {"ok": True}


@app.middleware("http")
async def ensure_utf8_json(request: Request, call_next):
    response = await call_next(request)
    content_type = response.headers.get("content-type", "")
    if content_type.startswith("application/json"):
        response.headers["content-type"] = "application/json; charset=utf-8"
    return response


@lru_cache(maxsize=128)
def _cached_find(zeitraum: str) -> EventList:
    return find_events(zeitraum)
