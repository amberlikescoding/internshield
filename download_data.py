"""Download the public Kaggle dataset ZIP; extract only the expected CSV."""
from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

URL = "https://www.kaggle.com/api/v1/datasets/download/shivamb/real-or-fake-fake-jobposting-prediction"
DESTINATION = Path(__file__).resolve().parent / "data" / "fake_job_postings.csv"

if __name__ == "__main__":
    try:
        with urlopen(URL, timeout=60) as response:
            archive = response.read()
        with ZipFile(BytesIO(archive)) as zipped:
            content = zipped.read("fake_job_postings.csv")
        DESTINATION.parent.mkdir(exist_ok=True)
        DESTINATION.write_bytes(content)
        print(f"Saved {DESTINATION}")
    except Exception as error:
        raise SystemExit(f"Download failed: {error}. Follow data/README.md for manual download.")
