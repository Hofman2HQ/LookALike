const upload = document.getElementById('upload');
const results = document.getElementById('results');

upload.addEventListener('change', () => {
    const file = upload.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = async () => {
        const b64 = reader.result.split(',')[1];
        const resp = await fetch('/match', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({image_base64: b64})
        });
        if (!resp.ok) {
            results.textContent = 'Error: ' + resp.statusText;
            return;
        }
        const data = await resp.json();
        results.innerHTML = '';

        const matches = data.matches.filter(m => m.score >= 0.88);
        if (matches.length === 0) {
            results.textContent = 'No close matches found.';
            return;
        }

        const carousel = document.createElement('div');
        carousel.className = 'carousel';

        const prev = document.createElement('button');
        prev.className = 'carousel-btn';
        prev.textContent = '<';

        const next = document.createElement('button');
        next.className = 'carousel-btn';
        next.textContent = '>';

        const inner = document.createElement('div');
        inner.className = 'carousel-inner';
        inner.style.width = `${matches.length * 150}px`;

        matches.forEach(m => {
            const item = document.createElement('div');
            item.className = 'carousel-item';
            item.style.width = '150px';
            item.innerHTML = `<h3>${m.name} (${(m.score * 100).toFixed(1)}%)</h3>` +
                `<img src="${m.photo_url}" width="112" />`;
            inner.appendChild(item);
        });

        let index = 0;
        const update = () => {
            inner.style.transform = `translateX(-${index * 150}px)`;
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
        carousel.appendChild(inner);
        carousel.appendChild(next);
        results.appendChild(carousel);
        update();
    };
    reader.readAsDataURL(file);
});
