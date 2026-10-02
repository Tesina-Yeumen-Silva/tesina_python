"""
download_validation_dataset_ddg.py

Descarga N imágenes por cada búsqueda del CSV (categoria,query,nombre_sugerido),
usando DuckDuckGo image search (paquete "ddgs") en vez de icrawler+Bing,
que dejó de funcionar por cambios en el HTML de Bing.

Requisitos:
    pip install ddgs requests

Uso:
    python download_validation_dataset_ddg.py --csv validation_queries.csv --out validation_dataset/real_images --n 35

Resultado: igual que antes, una carpeta por categoría con todas las
imágenes de todas sus búsquedas numeradas sin pisarse.
"""

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import requests

try:
    from ddgs import DDGS
except ImportError:
    # nombre del paquete antes de renombrarse
    from duckduckgo_search import DDGS

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
}


def read_csv(csv_path: Path):
    rows = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            categoria = row["categoria"].strip()
            query = row["query"].strip()
            if categoria and query:
                rows.append((categoria, query))
    return rows


def is_real_jpeg(content: bytes) -> bool:
    return content[:2] == b"\xff\xd8"


def download_image(url: str, dest: Path, strict_jpeg: bool, timeout: int = 10) -> bool:
    try:
        resp = requests.get(url, headers=HEADERS, timeout=timeout)
        resp.raise_for_status()
        content = resp.content
        if strict_jpeg and not is_real_jpeg(content):
            return False
        if len(content) < 2000:  # descarta thumbnails/errores muy chicos
            return False
        dest.write_bytes(content)
        return True
    except Exception:
        return False


def download_all(csv_path: Path, out_dir: Path, n: int, strict_jpeg: bool):
    rows = read_csv(csv_path)

    by_category = defaultdict(list)
    for categoria, query in rows:
        by_category[categoria].append(query)

    for categoria, queries in by_category.items():
        cat_dir = out_dir / categoria
        cat_dir.mkdir(parents=True, exist_ok=True)
        counter = len(list(cat_dir.glob("*")))

        for query in queries:
            print(f"\n=== [{categoria}] '{query}' (empezando en {counter:06d}) ===")
            saved = 0
            try:
                with DDGS() as ddgs:
                    results = ddgs.images(query, max_results=n * 2)  # margen por si algunas fallan
            except Exception as e:
                print(f"  Error buscando '{query}': {e}")
                continue

            for r in results:
                if saved >= n:
                    break
                url = r.get("image")
                if not url:
                    continue
                counter += 1
                dest = cat_dir / f"{counter:06d}.jpg"
                if download_image(url, dest, strict_jpeg):
                    saved += 1
                else:
                    counter -= 1  # no gastar número si falló
                time.sleep(0.2)

            print(f"  Guardadas {saved}/{n} para esta búsqueda")

        total = len(list(cat_dir.iterdir()))
        print(f"  [{categoria}] Total final: {total} imágenes")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--out", default="validation_dataset/real_images")
    parser.add_argument("--n", type=int, default=35)
    parser.add_argument("--strict-jpeg", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encontró {csv_path}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    download_all(csv_path, out_dir, args.n, args.strict_jpeg)

    print("\nListo. Dataset generado en:", out_dir.resolve())


if __name__ == "__main__":
    main()
