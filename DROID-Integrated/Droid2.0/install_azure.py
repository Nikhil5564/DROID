
from pathlib import Path
from urllib.request import Request, urlopen
from concurrent.futures import ThreadPoolExecutor, as_completed

BASE_URL = (
    "https://github.com/Azure/AzurePublicDataset/releases/"
    "download/dataset-v2/"
    "trace_data_vm_cpu_readings_vm_cpu_readings-file-"
)

OUTPUT = Path("data/azure")
OUTPUT.mkdir(parents=True, exist_ok=True)


def download(i):
    name = f"trace_data_vm_cpu_readings_vm_cpu_readings-file-{i}-of-195.csv.gz"
    path = OUTPUT / name

    if path.exists() and path.stat().st_size > 0:
        return f"SKIP {i}/195 (already exists)"

    url = f"{BASE_URL}{i}-of-195.csv.gz"

    try:
        request = Request(url, headers={"User-Agent": "DROID"})
        with urlopen(request, timeout=120) as response:
            data = response.read()

        if not data:
            return f"FAILED {i}/195 (empty file)"

        path.write_bytes(data)
        return f"OK {i}/195 ({len(data) / 1e6:.1f} MB)"

    except Exception as e:
        return f"FAILED {i}/195: {e}"


if __name__ == "__main__":
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(download, i) for i in range(1, 196)]
        for future in as_completed(futures):
            print(future.result())

    files = list(OUTPUT.glob("*.csv.gz"))
    print(f"\nFound {len(files)} .csv.gz files in {OUTPUT}")
    print("Check FAILED entries and rerun if needed.")
