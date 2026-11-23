import base64
from unittest.mock import patch

import cv2
import numpy as np
from fastapi.testclient import TestClient

from backend.app.main import app

client = TestClient(app)


def _encode_dummy_img() -> str:
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return base64.b64encode(buf.tobytes()).decode("utf-8")


def test_match_no_face_returns_422():
    resp = client.post("/match", json={"image_base64": _encode_dummy_img()})
    assert resp.status_code == 422


def test_match_returns_multi_celebrity_carousel():
    fake_emb = np.ones(512, dtype=np.float32)
    fake_emb /= np.linalg.norm(fake_emb)

    fake_candidates = [
        {"name": "Zendaya", "score": 0.42, "photo_url": "/static/Zendaya/1.jpg"},
        {"name": "Ana de Armas", "score": 0.36, "photo_url": "/static/Ana_de_Armas/1.jpg"},
        {"name": "Margot Robbie", "score": 0.31, "photo_url": "/static/Margot_Robbie/1.jpg"},
        {"name": "Natalie Portman", "score": 0.27, "photo_url": "/static/Natalie_Portman/1.jpg"},
        {"name": "Emma Stone", "score": 0.24, "photo_url": "/static/Emma_Stone/1.jpg"},
    ]

    with patch("backend.app.main.get_pipeline") as mock_pipe, patch(
        "backend.app.main.get_index"
    ) as mock_idx:
        mock_pipe.return_value.get_embedding.return_value = fake_emb
        mock_idx.return_value.search.return_value = fake_candidates

        resp = client.post("/match", json={"image_base64": _encode_dummy_img()})
        assert resp.status_code == 200
        data = resp.json()
        assert "matches" in data
        assert len(data["matches"]) == 5
        names = [m["name"] for m in data["matches"]]
        assert len(set(names)) == 5
        assert data["best_guess"] is False


def test_match_real_index_end_to_end():
    from pathlib import Path

    static_dir = Path(__file__).resolve().parents[1] / "static"
    samples = list(static_dir.rglob("*.jpg"))
    if not samples:
        return

    sample_img = samples[0]
    b64 = base64.b64encode(sample_img.read_bytes()).decode("utf-8")
    resp = client.post("/match", json={"image_base64": b64})
    assert resp.status_code == 200
    data = resp.json()
    assert len(data["matches"]) >= 5
    names = [m["name"] for m in data["matches"]]
    assert len(set(names)) == len(names)
    expected_name = sample_img.parent.name.replace("_", " ")
    assert data["matches"][0]["name"] == expected_name
    assert data["matches"][0]["score"] > 0.70
    assert "details" in data and data["details"] is not None
    assert data["details"]["total_ms"] > 0
    assert data["details"]["comparison_rate_per_sec"] > 0
    assert data["details"]["celebrities_compared"] > 0


def test_celebrities_sample_endpoint():
    resp = client.get("/celebrities/sample")
    assert resp.status_code == 200
    data = resp.json()
    assert "total_celebrities" in data
    assert "sample" in data


def test_match_full_head_mode_with_hair_and_traits():
    from pathlib import Path

    static_dir = Path(__file__).resolve().parents[1] / "static"
    samples = list(static_dir.rglob("*.jpg"))
    if not samples:
        return

    sample_img = samples[0]
    b64 = base64.b64encode(sample_img.read_bytes()).decode("utf-8")
    resp = client.post(
        "/match",
        json={"image_base64": b64, "mode": "full_head"},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["details"]["mode"] == "full_head"
    assert data["details"]["subject_traits"] is not None
    assert "hair_color" in data["details"]["subject_traits"]
    assert "hair_style" in data["details"]["subject_traits"]
    top = data["matches"][0]
    assert top["hair_style_score"] is not None
    assert top["hair_color_score"] is not None
    assert top["head_vibe_score"] is not None
    assert top["traits"] is not None


def test_session_lifecycle_and_mobile_upload():
    from pathlib import Path

    create_resp = client.post("/session/create?mode=full_head")
    assert create_resp.status_code == 200
    c_data = create_resp.json()
    sid = c_data["session_id"]
    poll_token = c_data["poll_token"]

    # Unauthorized poll without poll_token or with wrong token must be rejected (403)
    unauth_resp = client.get(f"/session/{sid}/status")
    assert unauth_resp.status_code == 403
    wrong_resp = client.get(
        f"/session/{sid}/status", headers={"X-Poll-Token": "wrong-token"}
    )
    assert wrong_resp.status_code == 403

    status_resp = client.get(
        f"/session/{sid}/status", headers={"X-Poll-Token": poll_token}
    )
    assert status_resp.status_code == 200
    assert status_resp.json()["status"] == "waiting"

    static_dir = Path(__file__).resolve().parents[1] / "static"
    samples = list(static_dir.rglob("*.jpg"))
    if samples:
        b64 = base64.b64encode(samples[0].read_bytes()).decode("utf-8")
        up_resp = client.post(f"/mobile/{sid}/upload", json={"image_base64": b64})
        assert up_resp.status_code == 200

        done_resp = client.get(
            f"/session/{sid}/status", headers={"X-Poll-Token": poll_token}
        )
        assert done_resp.status_code == 200
        assert done_resp.json()["status"] == "done"
        assert done_resp.json()["user_image_base64"].startswith("data:image/jpeg;base64,")

        # Burn-after-reading: once retrieved, session & selfie are wiped from RAM (404)
        after_resp = client.get(
            f"/session/{sid}/status", headers={"X-Poll-Token": poll_token}
        )
        assert after_resp.status_code == 404


def test_security_headers_and_input_validation():
    # 1. Security headers present
    h_resp = client.get("/health")
    assert h_resp.headers.get("X-Content-Type-Options") == "nosniff"
    assert h_resp.headers.get("X-Frame-Options") == "DENY"
    assert h_resp.headers.get("Referrer-Policy") == "no-referrer"
    assert "default-src 'self'" in h_resp.headers.get("Content-Security-Policy", "")

    # 2. Host header injection blocked on /session/create
    s_resp = client.post(
        "/session/create",
        headers={
            "X-Forwarded-Proto": "javascript",
            "X-Forwarded-Host": "evil.example.com/phish?x=1",
        },
    )
    assert s_resp.status_code == 200
    mobile_url = s_resp.json()["mobile_url"]
    assert not mobile_url.startswith("javascript:")
    assert "evil.example.com/phish" not in mobile_url

    # 3. Unsupported mode & arbitrary weight keys rejected by Pydantic
    bad_mode = client.post(
        "/match", json={"image_base64": _encode_dummy_img(), "mode": "<script>"}
    )
    assert bad_mode.status_code == 422

    bad_weights = client.post(
        "/match",
        json={
            "image_base64": _encode_dummy_img(),
            "mode": "custom",
            "weights": {"malicious_key": 100.0},
        },
    )
    assert bad_weights.status_code == 422
