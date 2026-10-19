"""Cache the official Moving MNIST file over HTTPS, then verify it before use.

torchvision remains the dataset loader. A separate .part download is never exposed
as a complete dataset, and interrupted transfers can be continued safely.
"""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import subprocess

URL = "https://www.cs.toronto.edu/~nitish/unsupervised_video/mnist_test_seq.npy"
SIZE = 819200096
MD5 = "be083ec986bfe91a449d63653c411eb2"


def digest(path):
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    target = Path("data/MovingMNIST/mnist_test_seq.npy")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        partial = target.with_suffix(".npy.part")
        offset = partial.stat().st_size if partial.exists() else 0
        print(f"Downloading official Moving MNIST ({SIZE:,} bytes), retaining {offset:,} bytes", flush=True)
        if offset > SIZE:
            raise RuntimeError("Partial dataset is larger than the official file")
        step = (SIZE - offset + 3) // 4 if offset < SIZE else 1
        ranges = [(start, min(start + step, SIZE) - 1) for start in range(offset, SIZE, step)]

        def fetch(bounds):
            start, end = bounds
            piece = target.with_suffix(f".npy.range-{start}-{end}")
            length = end - start + 1
            if not piece.exists() or piece.stat().st_size != length:
                subprocess.run(["curl", "--fail", "--location", "--retry", "3",
                                "--range", f"{start}-{end}", "--max-filesize", str(length),
                                "--silent", "--show-error", "--output", str(piece), URL], check=True)
            if piece.stat().st_size != length:
                raise RuntimeError(f"Incomplete dataset range {start}-{end}")
            print(f"Dataset range {start}-{end} complete", flush=True)
            return piece

        with ThreadPoolExecutor(max_workers=4) as pool:
            pieces = list(pool.map(fetch, ranges))
        assembled = target.with_suffix(".npy.assembling")
        import shutil
        with assembled.open("wb") as out:
            if offset:
                with partial.open("rb") as prefix:
                    shutil.copyfileobj(prefix, out)
            for piece in pieces:
                with piece.open("rb") as stream:
                    shutil.copyfileobj(stream, out)
        if assembled.stat().st_size != SIZE or digest(assembled) != MD5:
            raise RuntimeError("Moving MNIST size/checksum mismatch; partial file retained for inspection")
        assembled.replace(target)
        partial.unlink(missing_ok=True)
        for piece in pieces:
            piece.unlink()
    if target.stat().st_size != SIZE or digest(target) != MD5:
        raise RuntimeError("Existing Moving MNIST file does not match the official dataset")
    Path("results").mkdir(exist_ok=True)
    info = {"source": URL, "size_bytes": SIZE, "md5": MD5, "path": str(target)}
    Path("results/dataset.json").write_text(json.dumps(info, indent=2) + "\n")
    print("Official dataset checksum verified. Ready for torchvision MovingMNIST(split=None).", flush=True)


if __name__ == "__main__":
    main()
