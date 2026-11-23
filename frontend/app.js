const upload  = document.getElementById('upload');
const phonBtn = document.getElementById('phone-btn');
const results = document.getElementById('results');
const dbCountEl = document.getElementById('db-count');

const modePills = document.querySelectorAll('.mode-pill');
const modeDescEl = document.getElementById('mode-desc');
const customBox = document.getElementById('custom-weights-box');
const wFaceIn = document.getElementById('w-face');
const wStyleIn = document.getElementById('w-style');
const wColorIn = document.getElementById('w-color');
const wVibeIn = document.getElementById('w-vibe');
const applyWeightsBtn = document.getElementById('apply-weights-btn');

const ITEM_W = 260; // must match .carousel-item width in CSS

let _currentMode = 'full_head';
let _lastImageDataUrl = null;
let _samplePool = [];
let _totalCelebs = 985;
let _pollInterval = null;
let _scanTimer = null;
let _keyHandler = null;

const MODE_DESCRIPTIONS = {
    full_head: 'Matches facial bone structure (46%) + hair style/length (22%) + hair color (20%) + jawline/facial hair (12%)',
    face: 'Pure 512-D ArcFace biometric facial geometry (eyes, nose, mouth, bone structure — ignores hair)',
    hair_twin: 'Prioritizes hair style, length, volume & hair color (70%) + facial structure (20%) + head vibe (10%)',
    custom: 'Use the sliders below to dial in your own custom blend of face, hair style, hair color, and facial hair!',
};

function getCustomWeights() {
    if (_currentMode !== 'custom') return null;
    return {
        face: Number(wFaceIn ? wFaceIn.value : 45),
        hair_style: Number(wStyleIn ? wStyleIn.value : 25),
        hair_color: Number(wColorIn ? wColorIn.value : 20),
        head_vibe: Number(wVibeIn ? wVibeIn.value : 10),
    };
}

// Mode pill switching
modePills.forEach(pill => {
    pill.addEventListener('click', () => {
        modePills.forEach(p => p.classList.remove('active'));
        pill.classList.add('active');
        _currentMode = pill.getAttribute('data-mode') || 'full_head';
        if (modeDescEl) modeDescEl.textContent = MODE_DESCRIPTIONS[_currentMode] || '';
        if (customBox) {
            customBox.classList.toggle('open', _currentMode === 'custom');
        }
        // If user already uploaded a photo, re-run comparison immediately in the new mode!
        if (_lastImageDataUrl && _currentMode !== 'custom') {
            executeMatch(_lastImageDataUrl, false);
        }
    });
});

// Live slider labels
[
    [wFaceIn, 'w-face-val'],
    [wStyleIn, 'w-style-val'],
    [wColorIn, 'w-color-val'],
    [wVibeIn, 'w-vibe-val'],
].forEach(([inputEl, valId]) => {
    if (!inputEl) return;
    inputEl.addEventListener('input', () => {
        const lbl = document.getElementById(valId);
        if (lbl) lbl.textContent = `${inputEl.value}%`;
    });
});

if (applyWeightsBtn) {
    applyWeightsBtn.addEventListener('click', () => {
        if (_lastImageDataUrl) {
            executeMatch(_lastImageDataUrl, false);
        }
    });
}

// Pre-load a sample of celebrity portraits for the live roulette scanner
(async function preloadSample() {
    try {
        const resp = await fetch('/celebrities/sample');
        if (!resp.ok) return;
        const data = await resp.json();
        if (data.total_celebrities) {
            _totalCelebs = data.total_celebrities;
            if (dbCountEl) dbCountEl.textContent = _totalCelebs.toLocaleString();
        }
        if (Array.isArray(data.sample) && data.sample.length > 0) {
            _samplePool = data.sample;
            _samplePool.slice(0, 12).forEach(c => {
                const img = new Image();
                img.src = c.photo_url;
            });
        }
    } catch (_) { /* ignore on offline start */ }
})();

// ── Helpers ───────────────────────────────────────────────────────────────

function stopActiveTimers() {
    if (_pollInterval) { clearInterval(_pollInterval); _pollInterval = null; }
    if (_scanTimer) { clearTimeout(_scanTimer); _scanTimer = null; }
    if (_keyHandler) { window.removeEventListener('keydown', _keyHandler); _keyHandler = null; }
}

function showStatus(msg, isError = false) {
    results.innerHTML = '';
    const el = document.createElement('p');
    el.className = 'status-msg' + (isError ? ' error' : '');
    el.textContent = msg;
    results.appendChild(el);
}

function escapeHtml(str) {
    return String(str ?? '')
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;');
}

function compressImageFile(file, maxSide = 960, quality = 0.86) {
    return new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onerror = () => reject(new Error('Could not read image file'));
        reader.onload = () => {
            const img = new Image();
            img.onerror = () => resolve(reader.result);
            img.onload = () => {
                let w = img.naturalWidth || img.width;
                let h = img.naturalHeight || img.height;
                if (w > maxSide || h > maxSide) {
                    const scale = maxSide / Math.max(w, h);
                    w = Math.max(1, Math.round(w * scale));
                    h = Math.max(1, Math.round(h * scale));
                }
                const canvas = document.createElement('canvas');
                canvas.width = w;
                canvas.height = h;
                const ctx = canvas.getContext('2d');
                ctx.drawImage(img, 0, 0, w, h);
                resolve(canvas.toDataURL('image/jpeg', quality));
            };
            img.src = reader.result;
        };
        reader.readAsDataURL(file);
    });
}

/**
 * Calibrate 512-D ArcFace / combined cosine similarity into an intuitive lookalike percentage.
 */
function calibrateMatchPct(rawScore) {
    const s = Math.max(0, Math.min(1, Number(rawScore) || 0));
    let pct;
    if (s < 0.15) {
        pct = 35 + (s / 0.15) * 20;
    } else if (s < 0.30) {
        pct = 55 + ((s - 0.15) / 0.15) * 22;
    } else if (s < 0.50) {
        pct = 77 + ((s - 0.30) / 0.20) * 16;
    } else {
        pct = 93 + ((s - 0.50) / 0.50) * 6;
    }
    return Math.round(Math.max(1, Math.min(99, pct)));
}

function getPartyVerdict(pct, celebName, mode) {
    const modeTag = mode === 'hair_twin'
        ? 'Hair & Style Twin'
        : (mode === 'full_head' ? 'Full-Head Twin' : 'Biometric Twin');
    if (pct >= 90) {
        return `🔥 Uncanny ${modeTag}! Are you and ${celebName} secretly related?!`;
    }
    if (pct >= 80) {
        return `🌟 Red-Carpet ${modeTag}! You could genuinely pass as ${celebName}'s double!`;
    }
    if (pct >= 70) {
        return `🎬 Certified ${modeTag}! Strong feature & style match with ${celebName}!`;
    }
    return `✨ Vibe Match! Among ${_totalCelebs} celebrities, your closest ${modeTag.toLowerCase()} is ${celebName}!`;
}

/**
 * Build an SVG overlay showing the Full-Head hair envelope, face bounding box, and 5 facial landmarks.
 */
function buildLandmarkSvg(details) {
    const svgNS = 'http://www.w3.org/2000/svg';
    const svg = document.createElementNS(svgNS, 'svg');
    svg.setAttribute('class', 'landmark-svg');
    svg.setAttribute('viewBox', '0 0 100 100');
    svg.setAttribute('preserveAspectRatio', 'none');

    if (!details) return svg;

    if (Array.isArray(details.face_bbox) && details.face_bbox.length === 4) {
        const [bx, by, bw, bh] = details.face_bbox.map(v => v * 100);

        // Outer Full-Head & Hair analysis zone (expanded around face box)
        if (details.mode !== 'face') {
            const hx = Math.max(2, bx - bw * 0.28);
            const hy = Math.max(2, by - bh * 0.42);
            const hw = Math.min(96 - hx, bw * 1.56);
            const hh = Math.min(96 - hy, bh * 1.55);
            const headRect = document.createElementNS(svgNS, 'rect');
            headRect.setAttribute('x', hx.toFixed(1));
            headRect.setAttribute('y', hy.toFixed(1));
            headRect.setAttribute('width', hw.toFixed(1));
            headRect.setAttribute('height', hh.toFixed(1));
            headRect.setAttribute('rx', '12');
            headRect.setAttribute('fill', 'none');
            headRect.setAttribute('stroke', '#f472b6');
            headRect.setAttribute('stroke-width', '0.9');
            headRect.setAttribute('stroke-dasharray', '3 3');
            svg.appendChild(headRect);
        }

        // Inner Face bounding box reticle
        const rect = document.createElementNS(svgNS, 'rect');
        rect.setAttribute('x', bx.toFixed(1));
        rect.setAttribute('y', by.toFixed(1));
        rect.setAttribute('width', bw.toFixed(1));
        rect.setAttribute('height', bh.toFixed(1));
        rect.setAttribute('rx', '4');
        rect.setAttribute('fill', 'none');
        rect.setAttribute('stroke', '#22d3ee');
        rect.setAttribute('stroke-width', '1.1');
        rect.setAttribute('stroke-dasharray', '4 2');
        svg.appendChild(rect);
    }

    if (Array.isArray(details.landmarks) && details.landmarks.length === 5) {
        const pts = details.landmarks.map(([x, y]) => [x * 100, y * 100]);
        const connections = [
            [0, 1], [0, 2], [1, 2], [2, 3], [2, 4], [3, 4],
        ];
        connections.forEach(([i, j]) => {
            const line = document.createElementNS(svgNS, 'line');
            line.setAttribute('x1', pts[i][0].toFixed(1));
            line.setAttribute('y1', pts[i][1].toFixed(1));
            line.setAttribute('x2', pts[j][0].toFixed(1));
            line.setAttribute('y2', pts[j][1].toFixed(1));
            line.setAttribute('stroke', 'rgba(34, 211, 238, 0.65)');
            line.setAttribute('stroke-width', '0.8');
            svg.appendChild(line);
        });

        pts.forEach(([px, py]) => {
            const halo = document.createElementNS(svgNS, 'circle');
            halo.setAttribute('cx', px.toFixed(1));
            halo.setAttribute('cy', py.toFixed(1));
            halo.setAttribute('r', '2.8');
            halo.setAttribute('fill', 'rgba(34, 211, 238, 0.28)');
            svg.appendChild(halo);

            const dot = document.createElementNS(svgNS, 'circle');
            dot.setAttribute('cx', px.toFixed(1));
            dot.setAttribute('cy', py.toFixed(1));
            dot.setAttribute('r', '1.4');
            dot.setAttribute('fill', '#22d3ee');
            svg.appendChild(dot);
        });
    }

    return svg;
}

// ── Confetti Celebration Burst ────────────────────────────────────────────

function fireConfetti() {
    const canvas = document.getElementById('confetti-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = window.innerWidth;
    canvas.height = window.innerHeight;

    const colors = ['#7c6aff', '#c084fc', '#f472b6', '#22d3ee', '#34d399', '#fbbf24'];
    const particles = Array.from({ length: 75 }, () => ({
        x: canvas.width * (0.3 + Math.random() * 0.4),
        y: canvas.height * 0.35,
        vx: (Math.random() - 0.5) * 14,
        vy: -Math.random() * 12 - 4,
        size: 5 + Math.random() * 6,
        color: colors[Math.floor(Math.random() * colors.length)],
        rot: Math.random() * Math.PI * 2,
        vrot: (Math.random() - 0.5) * 0.25,
        alpha: 1,
    }));

    let frame = 0;
    function tick() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        let alive = false;
        particles.forEach(p => {
            p.x += p.vx;
            p.y += p.vy;
            p.vy += 0.38;
            p.rot += p.vrot;
            if (frame > 40) p.alpha -= 0.022;
            if (p.alpha > 0 && p.y < canvas.height + 20) {
                alive = true;
                ctx.save();
                ctx.globalAlpha = Math.max(0, p.alpha);
                ctx.translate(p.x, p.y);
                ctx.rotate(p.rot);
                ctx.fillStyle = p.color;
                ctx.fillRect(-p.size / 2, -p.size / 2, p.size, p.size * 0.6);
                ctx.restore();
            }
        });
        frame++;
        if (alive && frame < 110) {
            requestAnimationFrame(tick);
        } else {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
        }
    }
    requestAnimationFrame(tick);
}

// ── Stage 1: Party Live Biometric Scan & Celebrity Roulette ───────────────

function startPartyScanView(userImgDataUrl) {
    results.innerHTML = '';

    const stage = document.createElement('div');
    stage.className = 'scan-stage';

    const modeBadgeLabel = _currentMode === 'face'
        ? 'FACE BIOMETRIC SCAN ACTIVE'
        : 'FULL-HEAD, HAIR & BIOMETRIC SCAN ACTIVE';

    const header = document.createElement('div');
    header.className = 'scan-header';
    header.innerHTML = `
        <div class="scan-phase-badge"><span class="pulse-dot"></span><span id="scan-phase-badge-text">${modeBadgeLabel}</span></div>
        <div class="scan-title" id="scan-phase-title">Detecting 5-point landmarks &amp; full-head regions…</div>
    `;

    const arena = document.createElement('div');
    arena.className = 'scan-arena';

    const userCard = document.createElement('div');
    userCard.className = 'face-box-card';
    const userMedia = document.createElement('div');
    userMedia.className = 'face-box-media';
    userMedia.id = 'scan-user-media';
    if (userImgDataUrl) {
        const uImg = document.createElement('img');
        uImg.src = userImgDataUrl;
        uImg.alt = 'Your photo';
        userMedia.appendChild(uImg);
    }
    const grid = document.createElement('div');
    grid.className = 'scan-grid-overlay';
    const laser = document.createElement('div');
    laser.className = 'laser-beam';
    userMedia.appendChild(grid);
    userMedia.appendChild(laser);

    const userFooter = document.createElement('div');
    userFooter.className = 'face-box-footer';
    userFooter.innerHTML = `
        <div class="face-box-label">SUBJECT</div>
        <div class="face-box-name" id="scan-user-status">Scanning Head &amp; Face…</div>
    `;
    userCard.appendChild(userMedia);
    userCard.appendChild(userFooter);

    const vsCol = document.createElement('div');
    vsCol.className = 'vs-column';
    vsCol.innerHTML = `
        <div class="vs-circle">VS</div>
        <div class="live-counter" id="scan-counter">0 / ${_totalCelebs}</div>
        <div class="live-rate" id="scan-rate">Initializing…</div>
    `;

    const firstCeleb = _samplePool[0] || { name: 'Scanning…', photo_url: '' };
    const rouletteCard = document.createElement('div');
    rouletteCard.className = 'face-box-card';
    rouletteCard.innerHTML = `
        <div class="face-box-media">
            <img id="roulette-img" src="${escapeHtml(firstCeleb.photo_url)}" alt="Candidate" style="${firstCeleb.photo_url ? '' : 'display:none'}">
        </div>
        <div class="face-box-footer">
            <div class="face-box-label">CANDIDATE MATCH</div>
            <div class="face-box-name" id="roulette-name">${escapeHtml(firstCeleb.name)}</div>
        </div>
    `;

    arena.appendChild(userCard);
    arena.appendChild(vsCol);
    arena.appendChild(rouletteCard);

    const progWrap = document.createElement('div');
    progWrap.className = 'scan-progress-wrap';
    progWrap.innerHTML = `
        <div class="scan-progress-bar"><div class="scan-progress-fill" id="scan-prog-fill"></div></div>
        <div class="scan-actions">
            <button type="button" class="skip-btn" id="skip-scan-btn">Skip Animation &#x23ED;</button>
        </div>
    `;

    stage.appendChild(header);
    stage.appendChild(arena);
    stage.appendChild(progWrap);
    results.appendChild(stage);
}

function playRouletteReveal(data, userImgDataUrl) {
    const matches = data.matches || [];
    const details = data.details || null;
    if (matches.length === 0) {
        showMatches(matches, data.best_guess, details, userImgDataUrl);
        return;
    }

    const userMedia = document.getElementById('scan-user-media');
    const userStatus = document.getElementById('scan-user-status');
    if (userMedia && details) {
        userMedia.appendChild(buildLandmarkSvg(details));
    }
    if (userStatus && details) {
        const st = details.subject_traits;
        if (st && st.hair_color) {
            userStatus.textContent = `✓ ${st.hair_color} • ${st.hair_style}`;
        } else {
            const confPct = Math.round((details.face_confidence || 0.98) * 100);
            userStatus.textContent = `✓ Biometric Lock (${confPct}%)`;
        }
        userStatus.style.color = 'var(--success)';
    }

    const skipBtn = document.getElementById('skip-scan-btn');
    const finishNow = () => {
        if (_scanTimer) { clearTimeout(_scanTimer); _scanTimer = null; }
        showMatches(matches, data.best_guess, details, userImgDataUrl);
    };
    if (skipBtn) skipBtn.onclick = finishNow;

    const pool = [..._samplePool, ...matches.slice().reverse()];
    const totalSteps = 22;
    const totalCelebs = (details && details.celebrities_compared) || _totalCelebs;
    const realRate = (details && details.comparison_rate_per_sec)
        ? Math.round(details.comparison_rate_per_sec)
        : 22400;

    const rouletteImg = document.getElementById('roulette-img');
    const rouletteName = document.getElementById('roulette-name');
    const counterEl = document.getElementById('scan-counter');
    const rateEl = document.getElementById('scan-rate');
    const progFill = document.getElementById('scan-prog-fill');
    const phaseTitle = document.getElementById('scan-phase-title');
    const phaseBadge = document.getElementById('scan-phase-badge-text');

    let step = 0;

    function nextFrame() {
        step++;
        const progress = Math.min(1, step / totalSteps);
        if (progFill) progFill.style.width = `${Math.round(progress * 100)}%`;

        const comparedNow = Math.min(totalCelebs, Math.round(progress * totalCelebs));
        if (counterEl) counterEl.textContent = `${comparedNow} / ${totalCelebs}`;
        if (rateEl) rateEl.textContent = `${realRate.toLocaleString()} celebs/sec`;

        if (phaseTitle) {
            if (progress < 0.25) {
                phaseTitle.textContent = 'Aligning 5-point landmarks & 160×160 full-head frame…';
            } else if (progress < 0.55) {
                phaseTitle.textContent = 'Analyzing hair style silhouette, hair color & complexion…';
            } else if (progress < 0.88) {
                phaseTitle.textContent = `Scoring 512-D face + hair vectors across ${totalCelebs} celebrities…`;
            } else {
                if (phaseBadge) phaseBadge.textContent = 'TWIN LOCKED';
                phaseTitle.textContent = 'Locking onto your #1 Celebrity Twin…';
            }
        }

        let candidate;
        if (step >= totalSteps) {
            candidate = matches[0];
        } else if (step >= totalSteps - Math.min(5, matches.length - 1)) {
            const runnerIdx = totalSteps - step;
            candidate = matches[Math.min(runnerIdx, matches.length - 1)];
        } else {
            candidate = pool[step % Math.max(1, pool.length)] || matches[0];
        }

        if (candidate && rouletteImg && rouletteName) {
            rouletteImg.style.display = 'block';
            rouletteImg.src = candidate.photo_url;
            rouletteName.textContent = candidate.name;
        }

        if (step < totalSteps) {
            const delay = Math.round(55 + Math.pow(progress, 2.3) * 185);
            _scanTimer = setTimeout(nextFrame, delay);
        } else {
            _scanTimer = setTimeout(finishNow, 320);
        }
    }

    nextFrame();
}

// ── Execute Match Request ─────────────────────────────────────────────────

async function executeMatch(dataUrl, playFullAnimation = true) {
    stopActiveTimers();
    _lastImageDataUrl = dataUrl;
    const b64 = dataUrl.includes(',') ? dataUrl.split(',')[1] : dataUrl;

    if (playFullAnimation) {
        startPartyScanView(dataUrl);
    } else {
        showStatus('Re-ranking 985 celebrities for selected comparison mode…');
    }

    try {
        const payload = {
            image_base64: b64,
            mode: _currentMode,
            weights: getCustomWeights(),
        };
        const resp = await fetch('/match', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify(payload),
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
        if (playFullAnimation) {
            playRouletteReveal(data, dataUrl);
        } else {
            showMatches(data.matches, data.best_guess, data.details, dataUrl);
        }
    } catch (e) {
        showStatus('Error: ' + e.message, true);
    }
}

upload.addEventListener('change', async () => {
    const file = upload.files[0];
    upload.value = '';
    if (!file) return;

    try {
        const dataUrl = await compressImageFile(file, 960, 0.86);
        executeMatch(dataUrl, true);
    } catch (e) {
        showStatus('Error reading image: ' + e.message, true);
    }
});

// ── Stage 2 & 3: Results Carousel + Comparison Details Drawer ─────────────

function showMatches(matches, bestGuess = false, details = null, userImgDataUrl = null) {
    stopActiveTimers();
    results.innerHTML = '';

    if (!matches || matches.length === 0) {
        showStatus('No close matches found. Try a clearer, front-facing photo.');
        return;
    }

    fireConfetti();

    const activeMode = (details && details.mode) || _currentMode;
    const topPct = calibrateMatchPct(matches[0].score);

    // 1. Party Verdict Banner
    const verdict = document.createElement('div');
    verdict.className = 'party-verdict';
    const verdictPill = document.createElement('div');
    verdictPill.className = 'verdict-pill';
    verdictPill.textContent = getPartyVerdict(topPct, matches[0].name, activeMode);
    const verdictSub = document.createElement('div');
    verdictSub.className = 'verdict-sub';
    const modeTitleMap = {
        full_head: 'Full Head & Hair Mode',
        face: 'Face Only (Biometric) Mode',
        hair_twin: 'Hair & Style Twin Mode',
        custom: 'Custom Feature Mix Mode',
    };
    if (details) {
        verdictSub.textContent =
            `${modeTitleMap[activeMode] || 'Full Head Mode'} • Scanned ${details.celebrities_compared.toLocaleString()} celebrities in ${details.total_ms.toFixed(1)} ms `
            + `(${Math.round(details.comparison_rate_per_sec).toLocaleString()} celebs/sec)`;
    } else {
        verdictSub.textContent = `Top ${matches.length} celebrity lookalike${matches.length > 1 ? 's' : ''}`;
    }
    verdict.appendChild(verdictPill);
    verdict.appendChild(verdictSub);
    results.appendChild(verdict);

    // 2. Side-by-Side Comparison Stage (User Photo + VS + Celebrity Carousel)
    const compStage = document.createElement('div');
    compStage.className = 'comparison-stage';

    if (userImgDataUrl) {
        const userCard = document.createElement('div');
        userCard.className = 'user-selfie-card';

        const userMedia = document.createElement('div');
        userMedia.className = 'user-selfie-media';

        const youBadge = document.createElement('div');
        youBadge.className = 'you-badge';
        youBadge.textContent = 'YOU';

        const uImg = document.createElement('img');
        uImg.src = userImgDataUrl;
        uImg.alt = 'Your uploaded photo';

        const lmSvg = buildLandmarkSvg(details);
        let landmarksVisible = true;

        const lmToggle = document.createElement('button');
        lmToggle.type = 'button';
        lmToggle.className = 'landmark-toggle-btn';
        lmToggle.textContent = '🎯 Scan Zones: ON';
        lmToggle.addEventListener('click', () => {
            landmarksVisible = !landmarksVisible;
            lmSvg.style.display = landmarksVisible ? 'block' : 'none';
            lmToggle.textContent = landmarksVisible ? '🎯 Scan Zones: ON' : '🎯 Scan Zones: OFF';
        });

        userMedia.appendChild(youBadge);
        userMedia.appendChild(uImg);
        userMedia.appendChild(lmSvg);
        if (details && (details.landmarks || details.face_bbox)) {
            userMedia.appendChild(lmToggle);
        }

        const userInfo = document.createElement('div');
        userInfo.className = 'user-selfie-info';
        const confPct = details && details.face_confidence
            ? (details.face_confidence * 100).toFixed(1)
            : '99.0';
        const st = (details && details.subject_traits) || null;
        const traitsHtml = st
            ? `<div class="trait-chips">
                 <span class="trait-chip">🎨 ${escapeHtml(st.hair_color)}</span>
                 <span class="trait-chip">💇 ${escapeHtml(st.hair_style)}</span>
                 <span class="trait-chip">🧔 ${escapeHtml(st.facial_hair)}</span>
               </div>`
            : '';
        userInfo.innerHTML = `
            <div class="user-selfie-title">Your Head &amp; Face</div>
            <div class="user-selfie-sub">✓ 5-Point Lock (${escapeHtml(confPct)}%)</div>
            ${traitsHtml}
        `;

        userCard.appendChild(userMedia);
        userCard.appendChild(userInfo);
        compStage.appendChild(userCard);

        const vsBadge = document.createElement('div');
        vsBadge.className = 'vs-circle';
        vsBadge.textContent = 'VS';
        compStage.appendChild(vsBadge);
    }

    // Carousel shell
    const carousel = document.createElement('div');
    carousel.className = 'carousel';

    const prev = document.createElement('button');
    prev.className = 'carousel-btn';
    prev.setAttribute('aria-label', 'Previous celebrity');
    prev.innerHTML = '&#8249;';

    const next = document.createElement('button');
    next.className = 'carousel-btn';
    next.setAttribute('aria-label', 'Next celebrity');
    next.innerHTML = '&#8250;';

    const viewport = document.createElement('div');
    viewport.className = 'carousel-viewport';

    const track = document.createElement('div');
    track.className = 'carousel-track';

    const barFills = [];
    const thumbBtns = [];
    const thumbsContainer = document.createElement('div');
    thumbsContainer.className = 'carousel-thumbs';

    matches.forEach((m, idx) => {
        const pct = calibrateMatchPct(m.score);
        const facePct = m.face_score != null ? calibrateMatchPct(m.face_score) : pct;
        const stylePct = m.hair_style_score != null ? Math.round(m.hair_style_score) : null;
        const colorPct = m.hair_color_score != null ? Math.round(m.hair_color_score) : null;
        const vibePct = m.head_vibe_score != null ? Math.round(m.head_vibe_score) : null;

        const item = document.createElement('div');
        item.className = 'carousel-item';

        const badge = document.createElement('div');
        badge.className = 'rank-badge' + (idx === 0 ? ' top-rank' : '');
        badge.textContent = idx === 0 ? '👑 #1 Top Twin' : `#${idx + 1} Match`;

        const img = document.createElement('img');
        img.src = m.photo_url;
        img.alt = m.name;
        img.loading = idx < 3 ? 'eager' : 'lazy';

        const info = document.createElement('div');
        info.className = 'carousel-item-info';

        const name = document.createElement('div');
        name.className = 'celeb-name';
        name.textContent = m.name;

        const score = document.createElement('div');
        score.className = 'match-pct';
        score.textContent = `${pct}% overall match`;

        const bar = document.createElement('div');
        bar.className = 'match-bar';
        const fill = document.createElement('div');
        fill.className = 'match-bar-fill';
        bar.appendChild(fill);
        barFills.push({ el: fill, pct });

        info.appendChild(name);
        info.appendChild(score);
        info.appendChild(bar);

        if (m.traits) {
            const chips = document.createElement('div');
            chips.className = 'trait-chips';
            chips.innerHTML = `
                <span class="trait-chip">🎨 ${escapeHtml(m.traits.hair_color)}</span>
                <span class="trait-chip">💇 ${escapeHtml(m.traits.hair_style)}</span>
            `;
            info.appendChild(chips);
        }

        if (stylePct != null && colorPct != null) {
            const subGrid = document.createElement('div');
            subGrid.className = 'subscore-grid';
            subGrid.innerHTML = `
                <div class="subscore-item"><span>🎭 Face</span><b>${facePct}%</b></div>
                <div class="subscore-item"><span>💇 Hair Style</span><b>${stylePct}%</b></div>
                <div class="subscore-item"><span>🎨 Hair Color</span><b>${colorPct}%</b></div>
                <div class="subscore-item"><span>🧔 Head Vibe</span><b>${vibePct}%</b></div>
            `;
            info.appendChild(subGrid);
        }

        item.appendChild(badge);
        item.appendChild(img);
        item.appendChild(info);
        track.appendChild(item);

        // Quick-jump celebrity pill button
        const thumbBtn = document.createElement('button');
        thumbBtn.type = 'button';
        thumbBtn.className = 'thumb-btn' + (idx === 0 ? ' active' : '');
        const thumbImg = document.createElement('img');
        thumbImg.src = m.photo_url;
        thumbImg.alt = m.name;
        const thumbText = document.createElement('span');
        thumbText.textContent = `${idx + 1}. ${m.name} (${pct}%)`;
        thumbBtn.appendChild(thumbImg);
        thumbBtn.appendChild(thumbText);
        thumbBtn.addEventListener('click', () => goTo(idx));
        thumbsContainer.appendChild(thumbBtn);
        thumbBtns.push(thumbBtn);
    });

    viewport.appendChild(track);
    carousel.appendChild(prev);
    carousel.appendChild(viewport);
    carousel.appendChild(next);
    compStage.appendChild(carousel);
    results.appendChild(compStage);

    let index = 0;

    const counter = document.createElement('div');
    counter.className = 'carousel-counter';
    results.appendChild(counter);

    if (matches.length > 1) {
        results.appendChild(thumbsContainer);
    }

    // 3. Option to Show Comparison Details (Toggle Button + Expandable Telemetry Panel)
    const toggleWrap = document.createElement('div');
    toggleWrap.className = 'details-toggle-wrap';

    const detailsBtn = document.createElement('button');
    detailsBtn.type = 'button';
    detailsBtn.className = 'details-toggle-btn';
    detailsBtn.innerHTML = '&#x1F4CA;&nbsp; Show Comparison Details';

    const replayBtn = document.createElement('button');
    replayBtn.type = 'button';
    replayBtn.className = 'details-toggle-btn';
    replayBtn.innerHTML = '&#x1F3B0;&nbsp; Replay Scan';
    replayBtn.addEventListener('click', () => {
        startPartyScanView(userImgDataUrl);
        playRouletteReveal({ matches, best_guess: bestGuess, details }, userImgDataUrl);
    });

    toggleWrap.appendChild(detailsBtn);
    toggleWrap.appendChild(replayBtn);
    results.appendChild(toggleWrap);

    const detailsPanel = document.createElement('div');
    detailsPanel.className = 'details-panel';
    results.appendChild(detailsPanel);

    detailsBtn.addEventListener('click', () => {
        const isOpen = detailsPanel.classList.toggle('open');
        detailsBtn.innerHTML = isOpen
            ? '&#x1F4CA;&nbsp; Hide Comparison Details'
            : '&#x1F4CA;&nbsp; Show Comparison Details';
    });

    function renderComparisonDetails(activeIdx) {
        const activeMatch = matches[activeIdx];
        const activePct = calibrateMatchPct(activeMatch.score);
        const facePct = activeMatch.face_score != null ? calibrateMatchPct(activeMatch.face_score) : activePct;
        const stylePct = activeMatch.hair_style_score != null ? `${activeMatch.hair_style_score.toFixed(1)}%` : 'N/A';
        const colorPct = activeMatch.hair_color_score != null ? `${activeMatch.hair_color_score.toFixed(1)}%` : 'N/A';
        const vibePct = activeMatch.head_vibe_score != null ? `${activeMatch.head_vibe_score.toFixed(1)}%` : 'N/A';

        const d = details || {
            total_ms: 42.0,
            decode_ms: 1.5,
            detect_align_ms: 14.2,
            embed_ms: 25.8,
            search_ms: 0.5,
            celebrities_compared: _totalCelebs,
            comparison_rate_per_sec: 23450,
            vector_search_rate_per_sec: 1970000,
            embedding_dim: 512,
            detector: 'YuNet 5-Point Landmark',
            model: 'ArcFace ResNet-50 (w600k_r50, 512-D)',
            face_confidence: 0.99,
            image_resolution: null,
            mode: _currentMode,
            weights_used: null,
            subject_traits: null,
        };

        const sumMs = Math.max(0.1, d.decode_ms + d.detect_align_ms + d.embed_ms + d.search_ms);
        const pDec = Math.max(3, (d.decode_ms / sumMs) * 100);
        const pDet = Math.max(8, (d.detect_align_ms / sumMs) * 100);
        const pEmb = Math.max(12, (d.embed_ms / sumMs) * 100);
        const pSea = Math.max(3, (d.search_ms / sumMs) * 100);

        const rawFaceCos = Number(activeMatch.face_score != null ? activeMatch.face_score : activeMatch.score).toFixed(4);
        const angleDeg = activeMatch.angular_distance_deg != null
            ? `${activeMatch.angular_distance_deg}°`
            : `${(Math.acos(Math.max(-1, Math.min(1, activeMatch.score))) * 180 / Math.PI).toFixed(1)}°`;
        const l2Dist = activeMatch.l2_distance != null
            ? activeMatch.l2_distance.toFixed(4)
            : Math.sqrt(Math.max(0, 2 - 2 * activeMatch.score)).toFixed(4);
        const percentile = activeMatch.percentile != null
            ? `Top ${(100 - activeMatch.percentile + 0.1).toFixed(1)}% (${activeMatch.percentile}th pct)`
            : `Rank #${activeIdx + 1} of ${d.celebrities_compared}`;
        const resStr = Array.isArray(d.image_resolution)
            ? `${d.image_resolution[0]} × ${d.image_resolution[1]} px`
            : 'Standard';

        const wUsed = d.weights_used || { face: 0.46, hair_style: 0.22, hair_color: 0.20, head_vibe: 0.12 };
        const weightsStr = `Face ${Math.round(wUsed.face * 100)}% • Style ${Math.round(wUsed.hair_style * 100)}% • Color ${Math.round(wUsed.hair_color * 100)}% • Vibe ${Math.round(wUsed.head_vibe * 100)}%`;

        const uTraits = d.subject_traits || { hair_color: '—', hair_style: '—', facial_hair: '—' };
        const cTraits = activeMatch.traits || { hair_color: '—', hair_style: '—', facial_hair: '—' };

        detailsPanel.innerHTML = `
            <div class="details-section-title">
                <span>⚡ Pipeline Performance &amp; Throughput</span>
                <span>${d.celebrities_compared.toLocaleString()} Celebrities Compared</span>
            </div>
            <div class="kpi-grid">
                <div class="kpi-card">
                    <div class="kpi-label">Time to Compare</div>
                    <div class="kpi-value">${d.total_ms.toFixed(1)} ms</div>
                    <div class="kpi-sub">End-to-end latency</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Comparison Rate</div>
                    <div class="kpi-value">${Math.round(d.comparison_rate_per_sec).toLocaleString()}/s</div>
                    <div class="kpi-sub">Celebrities compared / sec</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Vector Search Rate</div>
                    <div class="kpi-value">${Math.round(d.vector_search_rate_per_sec).toLocaleString()}/s</div>
                    <div class="kpi-sub">Multi-modal index throughput</div>
                </div>
                <div class="kpi-card">
                    <div class="kpi-label">Face Lock Confidence</div>
                    <div class="kpi-value">${(d.face_confidence * 100).toFixed(1)}%</div>
                    <div class="kpi-sub">${escapeHtml(d.detector)}</div>
                </div>
            </div>

            <div class="details-section-title">
                <span>💇 Full-Head &amp; Hair Feature Breakdown — You vs. #${activeIdx + 1} ${escapeHtml(activeMatch.name)}</span>
                <span>Mode Weights: ${escapeHtml(weightsStr)}</span>
            </div>
            <div class="match-metrics-grid" style="margin-bottom:1.35rem">
                <div class="metric-row"><span>🎭 Facial Structure Match</span><span>${facePct}% (cos θ = ${rawFaceCos})</span></div>
                <div class="metric-row"><span>💇 Hair Style &amp; Length Match</span><span>${stylePct}</span></div>
                <div class="metric-row"><span>🎨 Hair Color &amp; Tone Match</span><span>${colorPct}</span></div>
                <div class="metric-row"><span>🧔 Head Vibe &amp; Jawline Match</span><span>${vibePct}</span></div>
                <div class="metric-row"><span>Your Hair Color vs. Celeb</span><span>${escapeHtml(uTraits.hair_color)} ↔ ${escapeHtml(cTraits.hair_color)}</span></div>
                <div class="metric-row"><span>Your Hair Style vs. Celeb</span><span>${escapeHtml(uTraits.hair_style)} ↔ ${escapeHtml(cTraits.hair_style)}</span></div>
                <div class="metric-row"><span>Your Facial Hair vs. Celeb</span><span>${escapeHtml(uTraits.facial_hair)} ↔ ${escapeHtml(cTraits.facial_hair)}</span></div>
                <div class="metric-row"><span>Combined Resemblance</span><span>${activePct}% (${escapeHtml(percentile)})</span></div>
            </div>

            <div class="details-section-title">
                <span>⏱️ Stage-by-Stage Timing Breakdown</span>
                <span>Total: ${d.total_ms.toFixed(2)} ms</span>
            </div>
            <div class="timing-bar-stack">
                <div class="timing-seg" style="width:${pDec}%;background:#38bdf8" title="Decode: ${d.decode_ms} ms"></div>
                <div class="timing-seg" style="width:${pDet}%;background:#22d3ee" title="5-Point & Full-Head Align + Hair Features: ${d.detect_align_ms} ms"></div>
                <div class="timing-seg" style="width:${pEmb}%;background:#7c6aff" title="ArcFace 512-D Embed: ${d.embed_ms} ms"></div>
                <div class="timing-seg" style="width:${pSea}%;background:#f472b6" title="FAISS + Full-Head Search: ${d.search_ms} ms"></div>
            </div>
            <div class="timing-legend">
                <span class="timing-legend-item"><i class="legend-dot" style="background:#38bdf8"></i>Image Decode: <b>${d.decode_ms.toFixed(2)} ms</b></span>
                <span class="timing-legend-item"><i class="legend-dot" style="background:#22d3ee"></i>5-Point + Full-Head Align &amp; Hair Extract: <b>${d.detect_align_ms.toFixed(2)} ms</b></span>
                <span class="timing-legend-item"><i class="legend-dot" style="background:#7c6aff"></i>ArcFace 512-D Neural Embedding: <b>${d.embed_ms.toFixed(2)} ms</b></span>
                <span class="timing-legend-item"><i class="legend-dot" style="background:#f472b6"></i>FAISS + Multi-Vector Search: <b>${d.search_ms.toFixed(2)} ms</b></span>
            </div>

            <div class="details-section-title">
                <span>🧬 Mathematical &amp; Model Telemetry</span>
                <span>Input: ${resStr}</span>
            </div>
            <div class="match-metrics-grid">
                <div class="metric-row"><span>512-D Angular Distance (θ)</span><span>${angleDeg}</span></div>
                <div class="metric-row"><span>Unit Hypersphere L2 Distance</span><span>${l2Dist}</span></div>
                <div class="metric-row"><span>Face Embedding Model</span><span>${d.model}</span></div>
                <div class="metric-row"><span>Full-Head Alignment</span><span>Umeyama 160×160 + LAB/HOG Zones</span></div>
            </div>
        `;
    }

    const update = () => {
        track.style.transform = `translateX(-${index * ITEM_W}px)`;
        counter.textContent = `Viewing #${index + 1} of ${matches.length} — ${matches[index].name} (use ◀ ▶ arrows or click any celebrity below)`;
        thumbBtns.forEach((btn, i) => {
            btn.classList.toggle('active', i === index);
        });
        renderComparisonDetails(index);
    };

    const goTo = (newIdx) => {
        index = (newIdx + matches.length) % matches.length;
        update();
    };

    prev.addEventListener('click', () => goTo(index - 1));
    next.addEventListener('click', () => goTo(index + 1));

    let touchStartX = null;
    viewport.addEventListener('touchstart', (e) => {
        touchStartX = e.changedTouches[0].screenX;
    }, { passive: true });
    viewport.addEventListener('touchend', (e) => {
        if (touchStartX === null) return;
        const dx = e.changedTouches[0].screenX - touchStartX;
        if (Math.abs(dx) > 35) {
            goTo(dx < 0 ? index + 1 : index - 1);
        }
        touchStartX = null;
    }, { passive: true });

    _keyHandler = (e) => {
        if (e.key === 'ArrowLeft') goTo(index - 1);
        if (e.key === 'ArrowRight') goTo(index + 1);
    };
    window.addEventListener('keydown', _keyHandler);

    update();

    requestAnimationFrame(() => requestAnimationFrame(() => {
        barFills.forEach(({ el, pct }) => { el.style.width = pct + '%'; });
    }));
}

// ── Phone upload via QR code ──────────────────────────────────────────────

phonBtn.addEventListener('click', async () => {
    stopActiveTimers();
    showStatus('Creating party session…');

    let sessionData;
    try {
        const resp = await fetch(`/session/create?mode=${encodeURIComponent(_currentMode)}`, {method: 'POST'});
        if (!resp.ok) throw new Error(resp.statusText);
        sessionData = await resp.json();
    } catch (e) {
        showStatus('Could not create session: ' + e.message, true);
        return;
    }

    const {session_id, mobile_url, poll_token} = sessionData;

    results.innerHTML = '';

    const panel = document.createElement('div');
    panel.className = 'qr-panel';

    const title = document.createElement('h3');
    title.textContent = '🎉 Party QR — Scan with your phone';
    const sub = document.createElement('p');
    sub.className = 'sub';
    sub.textContent = 'Snap a selfie on your phone and watch the live full-head & biometric scan happen right here on the big screen!';

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
    waiting.innerHTML = '<span class="waiting-text">Waiting for player selfie</span>';

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
            + 'Restart with: py -3.11 -m uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --reload';
        panel.appendChild(warn);
    }

    results.appendChild(panel);

    _pollInterval = setInterval(async () => {
        try {
            const pollResp = await fetch(`/session/${encodeURIComponent(session_id)}/status`, {
                headers: poll_token ? {'X-Poll-Token': poll_token} : {},
            });
            if (!pollResp.ok) { clearInterval(_pollInterval); return; }
            const data = await pollResp.json();
            if (data.status === 'done') {
                clearInterval(_pollInterval);
                _pollInterval = null;
                const mobileImgUrl = data.user_image_base64
                    ? (data.user_image_base64.startsWith('data:')
                        ? data.user_image_base64
                        : `data:image/jpeg;base64,${data.user_image_base64}`)
                    : null;
                _lastImageDataUrl = mobileImgUrl;
                startPartyScanView(mobileImgUrl);
                playRouletteReveal(data, mobileImgUrl);
            }
        } catch (_) { /* keep polling on transient errors */ }
    }, 1500);
});
