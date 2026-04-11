const upload  = document.getElementById('upload');
const phonBtn = document.getElementById('phone-btn');
const results = document.getElementById('results');

const ITEM_W = 220; // must match .carousel-item width in CSS

// ── Helpers ───────────────────────────────────────────────────────────────

function showStatus(msg, isError = false) {
    results.innerHTML = '';
    const el = document.createElement('p');
    el.className = 'status-msg' + (isError ? ' error' : '');
    el.textContent = msg;
    results.appendChild(el);
}

function showSpinner() {
    results.innerHTML =
        '<div class="status-msg"><div class="spinner"></div>Analysing your photo…</div>';
}

// ── File upload ───────────────────────────────────────────────────────────

upload.addEventListener('change', () => {
    const file = upload.files[0];
    // Reset so the same file can be re-selected without a hard-refresh.
    upload.value = '';
    if (!file) return;

    // Cancel any active phone-upload poll.
    if (_pollInterval) { clearInterval(_pollInterval); _pollInterval = null; }

    showSpinner();

    const reader = new FileReader();
    reader.onload = async () => {
        const b64 = reader.result.split(',')[1];
        try {
            const resp = await fetch('/match', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({image_base64: b64}),
            });
            if (resp.status === 422) {
                showStatus('No face detected. Please upload a clear, front-facing photo.', true);
                return;
            }
            if (!resp.ok) {
                showStatus('Error: ' + resp.statusText, true);
                return;
            }
            const data = await resp.json();
            showMatches(data.matches, data.best_guess);
        } catch (e) {
            showStatus('Error: ' + e.message, true);
        }
    };
    reader.readAsDataURL(file);
});

// ── Results carousel ──────────────────────────────────────────────────────

function showMatches(matches, bestGuess = false) {
    results.innerHTML = '';

    if (!matches || matches.length === 0) {
        showStatus('No close matches found. Try a clearer, front-facing photo.');
        return;
    }

    // Header
    const header = document.createElement('div');
    header.className = 'results-header';
    const headerLabel = document.createElement('p');
    if (bestGuess) {
        headerLabel.textContent = 'No strong match found — closest resemblance:';
    } else {
        headerLabel.textContent = `Top ${matches.length} match${matches.length > 1 ? 'es' : ''}`;
    }
    header.appendChild(headerLabel);
    results.appendChild(header);

    // Carousel shell
    const carousel = document.createElement('div');
    carousel.className = 'carousel';

    const prev = document.createElement('button');
    prev.className = 'carousel-btn';
    prev.setAttribute('aria-label', 'Previous');
    prev.innerHTML = '&#8249;';

    const next = document.createElement('button');
    next.className = 'carousel-btn';
    next.setAttribute('aria-label', 'Next');
    next.innerHTML = '&#8250;';

    const viewport = document.createElement('div');
    viewport.className = 'carousel-viewport';

    const track = document.createElement('div');
    track.className = 'carousel-track';

    // Score display: ArcFace cosine similarity in [0,1].
    // Showing raw score * 100 gives honest percentages — with the 0.35 backend
    // threshold, results range from 35–~65 %, which is more accurate than the
    // previous "97.4%" from the mismatch between index and query model.
    const barFills = [];
    matches.forEach(m => {
        const pct = Math.round(m.score * 100);

        const item = document.createElement('div');
        item.className = 'carousel-item';

        const img = document.createElement('img');
        img.src = m.photo_url;
        img.alt = m.name;
        img.loading = 'lazy';

        const info = document.createElement('div');
        info.className = 'carousel-item-info';

        const name = document.createElement('div');
        name.className = 'celeb-name';
        name.textContent = m.name;

        const score = document.createElement('div');
        score.className = 'match-pct';
        score.textContent = `${pct}% match`;

        const bar = document.createElement('div');
        bar.className = 'match-bar';
        const fill = document.createElement('div');
        fill.className = 'match-bar-fill';
        bar.appendChild(fill);
        barFills.push({ el: fill, pct });

        info.appendChild(name);
        info.appendChild(score);
        info.appendChild(bar);
        item.appendChild(img);
        item.appendChild(info);
        track.appendChild(item);
    });

    viewport.appendChild(track);

    let index = 0;

    const counter = document.createElement('div');
    counter.className = 'carousel-counter';

    const updateCounter = () => {
        counter.textContent = `${index + 1} / ${matches.length}`;
    };

    const update = () => {
        track.style.transform = `translateX(-${index * ITEM_W}px)`;
        updateCounter();
    };

    prev.addEventListener('click', () => {
        index = (index - 1 + matches.length) % matches.length;
        update();
    });
    next.addEventListener('click', () => {
        index = (index + 1) % matches.length;
        update();
    });

    carousel.appendChild(prev);
    carousel.appendChild(viewport);
    carousel.appendChild(next);
    results.appendChild(carousel);
    results.appendChild(counter);
    update();

    // Animate the fill bars after a short delay so CSS transition triggers
    requestAnimationFrame(() => requestAnimationFrame(() => {
        barFills.forEach(({ el, pct }) => { el.style.width = pct + '%'; });
    }));
}

// ── Phone upload via QR code ──────────────────────────────────────────────

let _pollInterval = null;

phonBtn.addEventListener('click', async () => {
    if (_pollInterval) { clearInterval(_pollInterval); _pollInterval = null; }

    showStatus('Creating session…');

    let sessionData;
    try {
        const resp = await fetch('/session/create', {method: 'POST'});
        if (!resp.ok) throw new Error(resp.statusText);
        sessionData = await resp.json();
    } catch (e) {
        showStatus('Could not create session: ' + e.message, true);
        return;
    }

    const {session_id, mobile_url} = sessionData;

    results.innerHTML = '';

    const panel = document.createElement('div');
    panel.className = 'qr-panel';

    const title = document.createElement('h3');
    title.textContent = 'Scan with your phone';
    const sub = document.createElement('p');
    sub.className = 'sub';
    sub.textContent = 'Open the camera app and point it at the QR code.';

    const qrDiv = document.createElement('div');
    new QRCode(qrDiv, {
        text: mobile_url,
        width: 180,
        height: 180,
        correctLevel: QRCode.CorrectLevel.M,
    });

    const urlNote = document.createElement('div');
    urlNote.className = 'qr-url';
    urlNote.textContent = mobile_url;

    const waiting = document.createElement('p');
    waiting.style.cssText = 'margin-top:1.2rem;font-size:0.85rem;color:var(--muted)';
    waiting.innerHTML = '<span class="waiting-text">Waiting for photo</span>';

    panel.appendChild(title);
    panel.appendChild(sub);
    panel.appendChild(qrDiv);
    panel.appendChild(urlNote);
    panel.appendChild(waiting);

    if (mobile_url.includes('127.0.0.1') || mobile_url.includes('localhost')) {
        const warn = document.createElement('div');
        warn.className = 'warn-banner';
        warn.textContent =
            '⚠ Server only reachable from this machine. '
            + 'Restart with: uvicorn backend.app.main:app --host 0.0.0.0 --reload';
        panel.appendChild(warn);
    }

    results.appendChild(panel);

    _pollInterval = setInterval(async () => {
        try {
            const pollResp = await fetch(`/session/${session_id}/status`);
            if (!pollResp.ok) { clearInterval(_pollInterval); return; }
            const data = await pollResp.json();
            if (data.status === 'done') {
                clearInterval(_pollInterval);
                _pollInterval = null;
                showMatches(data.matches, data.best_guess);
            }
        } catch (_) { /* keep polling on transient errors */ }
    }, 2000);
});
