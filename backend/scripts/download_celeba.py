from pathlib import Path
import requests
import zipfile

DATA_URL = "https://www.dropbox.com/s/d1kjpkqklf0uw77/celeba.zip?dl=1"
IDENTITY_URL = "https://raw.githubusercontent.com/mireshghallah/CelebA/master/identity_CelebA.txt"


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    with requests.get(url, stream=True) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)


def download_dataset(root: Path = Path("data")):
    zip_path = root / "celeba.zip"
    _download(DATA_URL, zip_path)

    print("Extracting dataset...")
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(root)
    zip_path.unlink()

    id_path = root / "identity_CelebA.txt"
    _download(IDENTITY_URL, id_path)
    print("Dataset downloaded to", root)


if __name__ == "__main__":
    download_dataset()
