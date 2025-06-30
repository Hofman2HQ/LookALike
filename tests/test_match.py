from fastapi.testclient import TestClient
from backend.app.main import app

client = TestClient(app)

def test_match_basic():
    import base64
    import cv2
    import numpy as np

    img = np.zeros((10, 10, 3), dtype=np.uint8)
    _, buf = cv2.imencode('.jpg', img)
    img_b64 = base64.b64encode(buf.tobytes()).decode('utf-8')

    resp = client.post('/match', json={'image_base64': img_b64})
    assert resp.status_code == 200
    data = resp.json()
    assert 'matches' in data

