from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_root_page():
    resp = client.get('/')
    assert resp.status_code in (200, 404)
