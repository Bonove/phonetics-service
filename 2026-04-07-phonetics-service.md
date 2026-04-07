# Phonetics Lookup Service — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Een REST API service op Render die namen fonetisch matcht via phonemizer + FAISS, zodat spraakherkende varianten ("Wasteless" → "waysis") correct worden herkend.

**Architecture:** FastAPI service met espeak-ng backend (via phonemizer) die namen omzet naar IPA fonemen, vervolgens met FAISS nearest-neighbor search de beste matches vindt uit een vooraf geladen dataset. De service draait als Docker container op Render (makerstreet workspace) en wordt aangeroepen door N8N.

**Tech Stack:** Python 3.11, FastAPI, phonemizer (espeak-ng), FAISS (faiss-cpu), numpy, uvicorn, Docker

---

## Project Structuur

```
phonetics-service/
├── Dockerfile
├── requirements.txt
├── main.py                 # FastAPI app + endpoints
├── phonetics.py            # Phonemizer wrapper + FAISS index
├── config.py               # Settings en env vars
├── data/
│   └── names.json          # Seed data (namen + bedrijven)
└── tests/
    ├── test_phonetics.py   # Unit tests phonemizer + FAISS
    └── test_api.py         # API integration tests
```

---

### Task 1: Project Scaffolding

**Files:**
- Create: `phonetics-service/requirements.txt`
- Create: `phonetics-service/config.py`
- Create: `phonetics-service/data/names.json`

**Step 1: Create requirements.txt**

```
fastapi==0.115.12
uvicorn[standard]==0.34.2
phonemizer==3.3.0
faiss-cpu==1.11.0
numpy>=1.24,<2.0
python-dotenv==1.1.0
```

**Step 2: Create config.py**

```python
import os

API_KEY = os.getenv("API_KEY", "dev-key")
PORT = int(os.getenv("PORT", "10000"))
PHONEMIZER_LANGUAGE = "nl"
PHONEMIZER_BACKEND = "espeak"
FAISS_TOP_K = 5
SIMILARITY_THRESHOLD = 0.3
```

**Step 3: Create seed data**

```json
{
  "employees": [
    {"name": "Steven", "company": "xpots"},
    {"name": "Tristan", "company": "xpots"},
    {"name": "Henk", "company": "waysis"},
    {"name": "Jan", "company": "tmc"},
    {"name": "Piet", "company": "unplugged"}
  ],
  "companies": [
    {"name": "xpots"},
    {"name": "waysis"},
    {"name": "unplugged"},
    {"name": "tmc"}
  ]
}
```

**Step 4: Commit**

```bash
git add phonetics-service/
git commit -m "feat: scaffold phonetics-service project structure"
```

---

### Task 2: Phonetics Core — Phonemizer Wrapper

**Files:**
- Create: `phonetics-service/tests/test_phonetics.py`
- Create: `phonetics-service/phonetics.py`

**Step 1: Write the failing tests**

```python
import pytest


class TestPhonemize:
    """Test dat namen correct naar IPA fonemen worden omgezet."""

    def test_dutch_name_produces_phonemes(self):
        from phonetics import phonemize_name
        result = phonemize_name("Steven")
        assert isinstance(result, str)
        assert len(result) > 0

    def test_batch_phonemize(self):
        from phonetics import phonemize_batch
        names = ["Steven", "Henk", "Tristan"]
        results = phonemize_batch(names)
        assert len(results) == 3
        assert all(isinstance(r, str) and len(r) > 0 for r in results)

    def test_empty_string(self):
        from phonetics import phonemize_name
        result = phonemize_name("")
        assert result == ""

    def test_phonetic_similarity_concept(self):
        """Varianten van dezelfde naam moeten vergelijkbare fonemen opleveren."""
        from phonetics import phonemize_name
        p1 = phonemize_name("waysis")
        p2 = phonemize_name("Wasteless")
        # Beide moeten fonemen bevatten (we testen similarity later via FAISS)
        assert len(p1) > 0
        assert len(p2) > 0
```

**Step 2: Run tests to verify they fail**

Run: `cd phonetics-service && python -m pytest tests/test_phonetics.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'phonetics'`

**Step 3: Write minimal implementation**

```python
"""Phonemizer wrapper — zet namen om naar IPA fonemen."""

from phonemizer import phonemize
from phonemizer.separator import Separator

from config import PHONEMIZER_LANGUAGE, PHONEMIZER_BACKEND

# Separator: spaties tussen fonemen, geen woord/syllabe scheiding
_SEPARATOR = Separator(phone=" ", word="", syllable="")


def phonemize_name(name: str) -> str:
    """Zet een enkele naam om naar IPA fonemen.

    Args:
        name: De naam om te phonemiseren.

    Returns:
        IPA string met spaties tussen fonemen, of "" bij lege input.
    """
    if not name or not name.strip():
        return ""
    result = phonemize(
        name.strip(),
        backend=PHONEMIZER_BACKEND,
        language=PHONEMIZER_LANGUAGE,
        separator=_SEPARATOR,
        strip=True,
    )
    return result.strip()


def phonemize_batch(names: list[str]) -> list[str]:
    """Zet een lijst namen om naar IPA fonemen (efficienter dan per stuk).

    Args:
        names: Lijst van namen.

    Returns:
        Lijst van IPA strings, zelfde volgorde als input.
    """
    cleaned = [n.strip() if n else "" for n in names]
    non_empty = [n for n in cleaned if n]

    if not non_empty:
        return [""] * len(names)

    results = phonemize(
        non_empty,
        backend=PHONEMIZER_BACKEND,
        language=PHONEMIZER_LANGUAGE,
        separator=_SEPARATOR,
        strip=True,
    )

    # Map results terug naar originele posities
    output = []
    result_iter = iter(results if isinstance(results, list) else [results])
    for n in cleaned:
        if n:
            output.append(next(result_iter).strip())
        else:
            output.append("")
    return output
```

**Step 4: Run tests to verify they pass**

Run: `cd phonetics-service && python -m pytest tests/test_phonetics.py -v`
Expected: PASS (4 tests)

> **Vereiste**: espeak-ng moet geinstalleerd zijn op de dev machine.
> macOS: `brew install espeak`
> Linux: `apt-get install espeak-ng`

**Step 5: Commit**

```bash
git add phonetics-service/phonetics.py phonetics-service/tests/test_phonetics.py
git commit -m "feat: phonemizer wrapper for Dutch name-to-phoneme conversion"
```

---

### Task 3: FAISS Index — Similarity Search

**Files:**
- Modify: `phonetics-service/tests/test_phonetics.py` (toevoegen)
- Modify: `phonetics-service/phonetics.py` (toevoegen)

**Step 1: Write the failing tests**

Voeg toe aan `tests/test_phonetics.py`:

```python
class TestPhoneticIndex:
    """Test de FAISS-gebaseerde fonetische zoekindex."""

    def test_build_index(self):
        from phonetics import PhoneticIndex
        names = ["Steven", "Henk", "Tristan", "waysis", "xpots"]
        index = PhoneticIndex(names)
        assert index.size == 5

    def test_search_exact_match(self):
        from phonetics import PhoneticIndex
        names = ["Steven", "Henk", "Tristan"]
        index = PhoneticIndex(names)
        results = index.search("Steven", top_k=3)
        assert len(results) > 0
        assert results[0]["name"] == "Steven"
        assert results[0]["score"] >= 0.9

    def test_search_phonetic_variant(self):
        """Kerntest: 'Exports' moet 'xpots' matchen."""
        from phonetics import PhoneticIndex
        names = ["xpots", "waysis", "unplugged", "tmc"]
        index = PhoneticIndex(names)
        results = index.search("Exports", top_k=3)
        # xpots moet in de top-3 zitten
        matched_names = [r["name"] for r in results]
        assert "xpots" in matched_names

    def test_search_returns_scores(self):
        from phonetics import PhoneticIndex
        names = ["Steven", "Henk"]
        index = PhoneticIndex(names)
        results = index.search("Steven", top_k=2)
        for r in results:
            assert "name" in r
            assert "score" in r
            assert "phonemes" in r
            assert 0.0 <= r["score"] <= 1.0

    def test_search_top_k(self):
        from phonetics import PhoneticIndex
        names = ["a", "b", "c", "d", "e"]
        index = PhoneticIndex(names)
        results = index.search("a", top_k=2)
        assert len(results) <= 2

    def test_empty_index(self):
        from phonetics import PhoneticIndex
        index = PhoneticIndex([])
        results = index.search("test", top_k=3)
        assert results == []
```

**Step 2: Run tests to verify they fail**

Run: `cd phonetics-service && python -m pytest tests/test_phonetics.py::TestPhoneticIndex -v`
Expected: FAIL with `ImportError: cannot import name 'PhoneticIndex'`

**Step 3: Write implementation**

Voeg toe aan `phonetics.py`:

```python
import numpy as np
import faiss

from config import FAISS_TOP_K, SIMILARITY_THRESHOLD


def _phonemes_to_vector(phonemes: str, dim: int = 128) -> np.ndarray:
    """Zet een IPA string om naar een numerieke vector via character n-gram hashing.

    Gebruikt bigram + trigram hashing voor robuuste fonetische representatie.
    Genormaliseerd naar unit length voor cosine similarity via inner product.

    Args:
        phonemes: IPA string met spaties tussen fonemen.
        dim: Dimensie van de output vector.

    Returns:
        Genormaliseerde numpy vector van shape (dim,).
    """
    vec = np.zeros(dim, dtype=np.float32)
    chars = phonemes.replace(" ", "")
    if not chars:
        return vec

    # Bigrams en trigrams hashen naar vector dimensies
    for n in (2, 3):
        for i in range(len(chars) - n + 1):
            ngram = chars[i : i + n]
            h = hash(ngram) % dim
            vec[h] += 1.0

    # Normaliseer naar unit vector (voor cosine similarity)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec /= norm
    return vec


class PhoneticIndex:
    """FAISS-gebaseerde index voor fonetische naam-matching.

    Bouwt een inner-product index van foneem-vectoren.
    Search retourneert de meest gelijkende namen met scores.
    """

    def __init__(self, names: list[str]):
        self._names = list(names)
        self._phonemes: list[str] = []
        self._index = None

        if not names:
            return

        # Batch phonemize voor efficientie
        self._phonemes = phonemize_batch(names)

        # Bouw FAISS index
        dim = 128
        vectors = np.array(
            [_phonemes_to_vector(p, dim) for p in self._phonemes],
            dtype=np.float32,
        )
        self._index = faiss.IndexFlatIP(dim)  # Inner product = cosine sim op unit vectors
        self._index.add(vectors)

    @property
    def size(self) -> int:
        return len(self._names)

    def search(self, query: str, top_k: int = FAISS_TOP_K) -> list[dict]:
        """Zoek de meest gelijkende namen op basis van fonetische similarity.

        Args:
            query: De naam om te zoeken.
            top_k: Maximaal aantal resultaten.

        Returns:
            Lijst van dicts met 'name', 'score' en 'phonemes', gesorteerd op score.
        """
        if not self._index or self.size == 0:
            return []

        query_phonemes = phonemize_name(query)
        if not query_phonemes:
            return []

        query_vec = _phonemes_to_vector(query_phonemes).reshape(1, -1)
        k = min(top_k, self.size)
        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0:
                continue
            # Clamp score to [0, 1] (inner product kan licht > 1 door float precision)
            clamped_score = float(max(0.0, min(1.0, score)))
            if clamped_score >= SIMILARITY_THRESHOLD:
                results.append({
                    "name": self._names[idx],
                    "score": round(clamped_score, 4),
                    "phonemes": self._phonemes[idx],
                })
        return results
```

**Step 4: Run tests to verify they pass**

Run: `cd phonetics-service && python -m pytest tests/test_phonetics.py -v`
Expected: PASS (10 tests)

**Step 5: Commit**

```bash
git add phonetics-service/phonetics.py phonetics-service/tests/test_phonetics.py
git commit -m "feat: FAISS phonetic index with similarity search"
```

---

### Task 4: FastAPI Service

**Files:**
- Create: `phonetics-service/main.py`
- Create: `phonetics-service/tests/test_api.py`

**Step 1: Write the failing tests**

```python
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    # Set env before import
    import os
    os.environ["API_KEY"] = "test-key"
    from main import app
    return TestClient(app)


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestSearchEndpoint:
    def test_search_requires_auth(self, client):
        response = client.post("/search", json={"name": "test"})
        assert response.status_code == 401

    def test_search_with_valid_auth(self, client):
        response = client.post(
            "/search",
            json={"name": "Steven"},
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
        data = response.json()
        assert "matches" in data
        assert "query_phonemes" in data

    def test_search_empty_name(self, client):
        response = client.post(
            "/search",
            json={"name": ""},
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 422  # Validation error

    def test_search_top_k_parameter(self, client):
        response = client.post(
            "/search",
            json={"name": "Steven", "top_k": 2},
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
        assert len(response.json()["matches"]) <= 2


class TestReloadEndpoint:
    def test_reload_requires_auth(self, client):
        response = client.post("/reload")
        assert response.status_code == 401

    def test_reload_with_auth(self, client):
        response = client.post(
            "/reload",
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
```

**Step 2: Run tests to verify they fail**

Run: `cd phonetics-service && python -m pytest tests/test_api.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'main'`

**Step 3: Write implementation**

```python
"""Phonetics Lookup Service — FastAPI app.

Fonetische naam-matching via phonemizer + FAISS.
Wordt aangeroepen door N8N voor spraak-gebaseerde naam-lookups.
"""

import json
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel, Field

from config import API_KEY, PORT
from phonetics import PhoneticIndex, phonemize_name

security = HTTPBearer()

# Global index — geladen bij startup, herlaadbaar via /reload
_index: PhoneticIndex | None = None
_raw_names: list[str] = []


def _load_index() -> PhoneticIndex:
    """Laad namen uit data/names.json en bouw de FAISS index."""
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
```

**Step 4: Run tests to verify they pass**

Run: `cd phonetics-service && python -m pytest tests/test_api.py -v`
Expected: PASS (6 tests)

**Step 5: Commit**

```bash
git add phonetics-service/main.py phonetics-service/tests/test_api.py
git commit -m "feat: FastAPI service with /search, /health, /reload endpoints"
```

---

### Task 5: Dockerfile + Render Config

**Files:**
- Create: `phonetics-service/Dockerfile`
- Create: `phonetics-service/render.yaml`
- Create: `phonetics-service/.dockerignore`

**Step 1: Create Dockerfile**

```dockerfile
FROM python:3.11-slim

# Systeem dependencies voor phonemizer
RUN apt-get update && \
    apt-get install -y --no-install-recommends espeak-ng libespeak-ng-dev && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 10000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "10000"]
```

**Step 2: Create .dockerignore**

```
__pycache__/
*.pyc
.env
tests/
.git/
.pytest_cache/
```

**Step 3: Create render.yaml**

```yaml
services:
  - type: web
    name: phonetics-api
    runtime: docker
    dockerfilePath: ./Dockerfile
    dockerContext: ./phonetics-service
    plan: starter
    region: frankfurt
    healthCheckPath: /health
    envVars:
      - key: API_KEY
        sync: false
      - key: PORT
        value: "10000"
```

**Step 4: Build en test Docker image lokaal**

Run:
```bash
cd phonetics-service
docker build -t phonetics-api .
docker run --rm -p 10000:10000 -e API_KEY=test phonetics-api
```

Test in een andere terminal:
```bash
curl http://localhost:10000/health
curl -X POST http://localhost:10000/search \
  -H "Authorization: Bearer test" \
  -H "Content-Type: application/json" \
  -d '{"name": "Exports", "top_k": 3}'
```

Expected: Health returns `{"status":"ok","index_size":...}`, search returns matches met xpots bovenaan.

**Step 5: Commit**

```bash
git add phonetics-service/Dockerfile phonetics-service/.dockerignore phonetics-service/render.yaml
git commit -m "feat: Dockerfile + Render config for phonetics-api deployment"
```

---

### Task 6: Deploy naar Render (makerstreet)

**Step 1: Push branch naar GitHub**

```bash
git push -u origin feature/phonetics-service
```

**Step 2: Deploy via Render dashboard of CLI**

- Ga naar Render dashboard (Unplugged workspace)
- New → Web Service → Connect GitHub repo
- Selecteer branch `feature/phonetics-service`
- Stel Dockerfile path in: `phonetics-service/Dockerfile`
- Docker context: `phonetics-service`
- Plan: Starter ($7/mo)
- Region: Frankfurt
- Environment variables:
  - `API_KEY`: genereer een sterk random token
  - `PORT`: 10000
- Health check path: `/health`

**Step 3: Verify deployment**

```bash
curl https://phonetics-api.onrender.com/health
curl -X POST https://phonetics-api.onrender.com/search \
  -H "Authorization: Bearer <API_KEY>" \
  -H "Content-Type: application/json" \
  -d '{"name": "Wasteless", "top_k": 3}'
```

Expected: `waysis` in de matches.

**Step 4: Commit render config indien nodig**

```bash
git commit -m "chore: finalize Render deployment config"
```

---

### Task 7: N8N Integratie

> Dit is de latere stap — documenteer hier de verwachte N8N setup.

**N8N HTTP Request Node configuratie:**

```
Method: POST
URL: https://phonetics-api.onrender.com/search
Headers:
  Authorization: Bearer {{$env.PHONETICS_API_KEY}}
  Content-Type: application/json
Body:
  {
    "name": "{{$json.medewerker}}",
    "top_k": 3
  }
```

**Verwacht response:**
```json
{
  "matches": [
    {"name": "waysis", "score": 0.92, "phonemes": "ʋaːzɪs"}
  ],
  "query_phonemes": "ʋeɪstlɛs"
}
```

De N8N workflow kan dan de top match gebruiken in het `call-employee` proces.

---

## Samenvatting

| Task | Wat | Geschatte tijd |
|------|-----|---------------|
| 1 | Project scaffolding | 5 min |
| 2 | Phonemizer wrapper + tests | 15 min |
| 3 | FAISS index + tests | 15 min |
| 4 | FastAPI endpoints + tests | 15 min |
| 5 | Dockerfile + Render config | 10 min |
| 6 | Deploy naar Render | 10 min |
| 7 | N8N integratie (later) | 10 min |

**Totaal: ~80 minuten**
