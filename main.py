"""Phonetics Lookup Service -- FastAPI app."""

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import httpx
from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config import API_KEY, PORT, SUPABASE_KEY, SUPABASE_URL
from phonetics import PhoneticIndex, phonemize_name

logger = logging.getLogger("phonetics-api")

security = HTTPBearer()

_index: PhoneticIndex | None = None
_raw_names: list[str] = []
_name_metadata: dict[str, dict] = {}  # name -> {company, phone, id}


def _load_from_supabase() -> tuple[list[str], dict[str, dict]]:
    """Fetch names from Supabase medewerkers_bellijst table.

    Returns (names_list, metadata_dict) or raises on failure.
    """
    url = f"{SUPABASE_URL}/rest/v1/medewerkers_bellijst?select=voornaam,company_name,telefoonnummer,id"
    headers = {
        "apikey": SUPABASE_KEY,
        "Authorization": f"Bearer {SUPABASE_KEY}",
    }

    response = httpx.get(url, headers=headers, timeout=10)
    response.raise_for_status()
    rows = response.json()

    names = set()
    metadata = {}

    for row in rows:
        voornaam = (row.get("voornaam") or "").strip()
        company = (row.get("company_name") or "").strip()
        phone = row.get("telefoonnummer")
        row_id = row.get("id")

        if voornaam:
            names.add(voornaam)
            metadata[voornaam.lower()] = {
                "company": company,
                "phone": str(phone) if phone else "",
                "id": str(row_id) if row_id else "",
            }
        if company:
            names.add(company)
            if company.lower() not in metadata:
                metadata[company.lower()] = {"company": company, "phone": "", "id": ""}

    logger.info(f"Loaded {len(names)} names from Supabase ({len(rows)} rows)")
    return sorted(names), metadata


def _load_from_json() -> tuple[list[str], dict[str, dict]]:
    """Fallback: load from local names.json."""
    data_path = Path(__file__).parent / "data" / "names.json"
    if not data_path.exists():
        return [], {}

    with open(data_path) as f:
        data = json.load(f)

    names = set()
    for emp in data.get("employees", []):
        names.add(emp["name"])
    for comp in data.get("companies", []):
        names.add(comp["name"])

    logger.info(f"Loaded {len(names)} names from names.json (fallback)")
    return sorted(names), {}


def _load_index() -> PhoneticIndex:
    global _raw_names, _name_metadata

    # Try Supabase first, fall back to local JSON
    if SUPABASE_URL and SUPABASE_KEY:
        try:
            _raw_names, _name_metadata = _load_from_supabase()
        except Exception as e:
            logger.error(f"Failed to load from Supabase: {e}, falling back to names.json")
            _raw_names, _name_metadata = _load_from_json()
    else:
        logger.warning("No SUPABASE_URL/KEY configured, using names.json")
        _raw_names, _name_metadata = _load_from_json()

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
    company: str = ""
    phone: str = ""


class SearchResponse(BaseModel):
    matches: list[MatchResult]
    query_phonemes: str
    source: str = ""  # "supabase" or "json"


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "index_size": _index.size if _index else 0,
        "source": "supabase" if SUPABASE_URL else "json",
        "names": _raw_names,
    }


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

    # Enrich matches with metadata
    enriched = []
    for m in matches:
        meta = _name_metadata.get(m["name"].lower(), {})
        enriched.append(
            MatchResult(
                name=m["name"],
                score=m["score"],
                phonemes=m["phonemes"],
                company=meta.get("company", ""),
                phone=meta.get("phone", ""),
            )
        )

    return SearchResponse(
        matches=enriched,
        query_phonemes=query_phonemes,
        source="supabase" if SUPABASE_URL else "json",
    )


@app.post("/reload")
async def reload(
    credentials: HTTPAuthorizationCredentials = Security(security),
):
    _verify_auth(credentials)

    global _index
    _index = _load_index()
    return {"status": "reloaded", "index_size": _index.size, "names": _raw_names}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=PORT)
