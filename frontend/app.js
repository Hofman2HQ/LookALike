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
        data.matches.forEach(m => {
            const div = document.createElement('div');
            div.innerHTML = `<h3>${m.name} (${(m.score * 100).toFixed(1)}%)</h3>` +
                `<img src="${m.photo_url}" width="112" />`;
            results.appendChild(div);
        });
    };
    reader.readAsDataURL(file);
});
