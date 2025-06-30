from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_match_empty():
    # send a dummy base64 string
    resp = client.post('/match', json={'image_base64': ''})
    assert resp.status_code == 200
    data = resp.json()
    assert 'matches' in data

