"""Phonetics Lookup Service -- FastAPI app."""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config import API_KEY, PORT
from phonetics import PhoneticIndex, phonemize_name

security = HTTPBearer()

_index: PhoneticIndex | None = None
_raw_names: list[str] = []


def _load_index() -> PhoneticIndex:
    global _raw_names
    data_path = Path(__file__).parent / "data" / "names.json"

    if not data_path.exists():
        _raw_names = []
        return PhoneticIndex([])

    with open(data_path) as f:
        data = json.load(f)

    names = set()
    for emp in data.get("employees", []):
        names.add(emp["name"])
    for comp in data.get("companies", []):
        names.add(comp["name"])

    _raw_names = sorted(names)
    return PhoneticIndex(_raw_names)


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _index
    _index = _load_index()
    yield


app = FastAPI(title="Phonetics Lookup Service", lifespan=lifespan)


def _verify_auth(credentials: HTTPAuthorizationCredentials = Security(security)):
    if credentials.credentials != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


class SearchRequest(BaseModel):
    name: str = Field(..., min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)


class MatchResult(BaseModel):
    name: str
    score: float
    phonemes: str


class SearchResponse(BaseModel):
    matches: list[MatchResult]
    query_phonemes: str


@app.get("/health")
async def health():
    return {"status": "ok", "index_size": _index.size if _index else 0}


@app.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    _verify_auth(credentials)

    if _index is None:
        raise HTTPException(status_code=503, detail="Index not loaded")

    query_phonemes = phonemize_name(request.name)
    matches = _index.search(request.name, top_k=request.top_k)

    return SearchResponse(
        matches=[MatchResult(**m) for m in matches],
        query_phonemes=query_phonemes,
    )


@app.post("/reload")
async def reload(
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    _verify_auth(credentials)

    global _index
    _index = _load_index()
    return {"status": "reloaded", "index_size": _index.size}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
