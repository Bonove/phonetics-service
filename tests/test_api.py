import os
import sys

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def client():
    os.environ["API_KEY"] = "test-key"
    # Need to reimport to pick up env var
    import importlib

    import config

    importlib.reload(config)
    from main import app

    with TestClient(app) as c:
        yield c


class TestHealthEndpoint:
    def test_health_returns_ok(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"


class TestSearchEndpoint:
    def test_search_requires_auth(self, client):
        response = client.post("/search", json={"name": "test"})
        assert response.status_code in (401, 403)

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
        assert response.status_code == 422

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
        assert response.status_code in (401, 403)

    def test_reload_with_auth(self, client):
        response = client.post(
            "/reload",
            headers={"Authorization": "Bearer test-key"},
        )
        assert response.status_code == 200
