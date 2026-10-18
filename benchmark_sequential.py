"""
ARTI 503 - Project Stage 1
Sequential Benchmarking Script

Project:
Parallel Medical Image Compression on Multicore CPUs

What this script does:
- Finds the medical images recursively inside the dataset folder.
- Builds deterministic small/medium/large workloads.
- Runs the ORIGINAL sequential algorithms only (no multiprocessing/threads).
- Repeats each measurement 3 times.
- Uses time.perf_counter() for timing.
- Records CPU model, physical/logical cores, RAM, Python version.
- Saves raw and summarized CSV files for the Stage 1 report.

Expected repository layout:
project/
|-- brain-tumor-mri-dataset/
|-- code/
|   |-- benchmark_sequential.py   <-- put this file here
|   |-- scripts/
|   |-- main_test.py

Default dataset path:
../brain-tumor-mri-dataset

The script does not modify the dataset and does not save compressed images.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import io
import json
import os
import platform
import random
import re
import statistics
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.huffman_coding import huffman_encoding, huffman_decoding
from scripts.rle_compression import RLE_encoding, RLE_decoding
from scripts.kmeans_compression import kmeans
from scripts.roi_compression import roi_kmeans_compression


SUPPORTED = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
DEFAULT_SIZES = (20, 50, 100)
DEFAULT_REPEATS = 3
SEED = 42


class _NullWriter:
    def write(self, text):
        return len(text)

    def flush(self):
        pass


def _powershell(command: str) -> str:
    try:
        completed = subprocess.run(
            ["powershell", "-NoProfile", "-Command", command],
            capture_output=True,
            text=True,
            timeout=12,
            check=False,
        )
        return completed.stdout.strip()
    except Exception:
        return ""


def _linux_cpu_model() -> str:
    try:
        for line in Path("/proc/cpuinfo").read_text(errors="ignore").splitlines():
            if "model name" in line.lower():
                return line.split(":", 1)[1].strip()
    except Exception:
        pass
    return ""


def get_system_info() -> dict:
    system = platform.system()

    cpu_model = ""
    physical_cores = None
    logical_threads = os.cpu_count()

    if system == "Windows":
        cpu_model = _powershell(
            "(Get-CimInstance Win32_Processor | Select-Object -First 1 "
            "-ExpandProperty Name)"
        )
        p = _powershell(
            "(Get-CimInstance Win32_Processor | "
            "Measure-Object -Property NumberOfCores -Sum).Sum"
        )
        l = _powershell(
            "(Get-CimInstance Win32_Processor | "
            "Measure-Object -Property NumberOfLogicalProcessors -Sum).Sum"
        )
        try:
            physical_cores = int(float(p))
        except Exception:
            physical_cores = None
        try:
            logical_threads = int(float(l))
        except Exception:
            pass

        ram_bytes = _powershell(
            "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory"
        )
        try:
            ram_gb = round(int(ram_bytes) / (1024 ** 3), 2)
        except Exception:
            ram_gb = None
    else:
        cpu_model = _linux_cpu_model() or platform.processor() or "Unknown"
        ram_gb = None
        try:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            ram_gb = round((pages * page_size) / (1024 ** 3), 2)
        except Exception:
            pass

    if not cpu_model:
        cpu_model = platform.processor() or "Unknown"

    return {
        "operating_system": f"{platform.system()} {platform.release()}",
        "cpu_model": cpu_model.strip(),
        "physical_cores": physical_cores if physical_cores is not None else "Unknown",
        "logical_threads": logical_threads if logical_threads is not None else "Unknown",
        "ram_gb": ram_gb if ram_gb is not None else "Unknown",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
    }


def discover_images(dataset_root: Path) -> list[Path]:
    images = [
        p for p in dataset_root.rglob("*")
        if p.is_file() and p.suffix.lower() in SUPPORTED
        and "__MACOSX" not in p.relative_to(dataset_root).parts
        and not p.name.startswith("._")
    ]
    return sorted(images, key=lambda p: p.relative_to(dataset_root).as_posix().lower())


def validate_images(images: list[Path]) -> None:
    for p in images:
        try:
            with Image.open(p) as im:
                im.verify()
        except Exception as exc:
            raise RuntimeError(f"Unreadable image: {p}\n{exc}") from exc


def balanced_select(all_images: list[Path], dataset_root: Path, n: int) -> list[Path]:
    """Deterministic round-robin selection across relative parent folders."""
    if n > len(all_images):
        raise ValueError(
            f"Requested {n} images, but only {len(all_images)} supported images were found."
        )

    groups: dict[str, list[Path]] = {}
    for p in all_images:
        rel_parent = p.relative_to(dataset_root).parent.as_posix()
        groups.setdefault(rel_parent, []).append(p)

    for key in groups:
        groups[key] = sorted(
            groups[key],
            key=lambda p: p.relative_to(dataset_root).as_posix().lower()
        )

    keys = sorted(groups.keys(), key=str.lower)
    indices = {k: 0 for k in keys}
    selected: list[Path] = []

    while len(selected) < n:
        progress = False
        for key in keys:
            idx = indices[key]
            if idx < len(groups[key]):
                selected.append(groups[key][idx])
                indices[key] += 1
                progress = True
                if len(selected) == n:
                    break
        if not progress:
            break

    return selected


def manifest_hash(paths: list[Path], root: Path) -> str:
    payload = "\n".join(
        p.relative_to(root).as_posix() for p in paths
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def reset_seed(offset: int = 0) -> None:
    random.seed(SEED + offset)
    np.random.seed(SEED + offset)


def run_huffman(path: Path) -> None:
    encoded, codes, _ = huffman_encoding(str(path), bit_depth=8)
    with Image.open(path) as img:
        gray = img.convert("L")
        arr = np.array(gray)
    huffman_decoding(encoded, codes, arr.shape)


def run_rle(path: Path) -> None:
    encoded_list, shape, _, _ = RLE_encoding(str(path))
    RLE_decoding(encoded_list, shape)


def run_kmeans(path: Path, seed_offset: int) -> None:
    reset_seed(seed_offset)
    kmeans(
        str(path),
        k=8,
        iterations=10,
        show_result=False,
        savePath=None,
        medical_warning=False,
    )


def run_roi_kmeans(path: Path, seed_offset: int) -> None:
    reset_seed(seed_offset)
    roi_kmeans_compression(
        str(path),
        k_roi=16,
        k_background=4,
        iterations=10,
        savePath=None,
        show_result=False,
    )


ALGORITHMS = (
    ("Huffman", run_huffman),
    ("RLE", run_rle),
    ("K-Means", run_kmeans),
    ("ROI K-Means", run_roi_kmeans),
)


def time_algorithm(name: str, func, images: list[Path]) -> float:
    """Run one algorithm sequentially over a fixed image list."""
    null = _NullWriter()
    start = time.perf_counter()

    with contextlib.redirect_stdout(null):
        for idx, image_path in enumerate(images):
            if name in {"K-Means", "ROI K-Means"}:
                func(image_path, idx)
            else:
                func(image_path)

    return time.perf_counter() - start


def slugify(value: str) -> str:
    value = value.strip()
    value = re.sub(r"[^A-Za-z0-9_-]+", "_", value)
    return value.strip("_") or "member"


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="ARTI 503 Stage 1 sequential benchmark"
    )
    parser.add_argument(
        "--dataset",
        default="../brain-tumor-mri-dataset",
        help="Path to the dataset folder (default: ../brain-tumor-mri-dataset)",
    )
    parser.add_argument(
        "--member",
        default="",
        help='Member full name, e.g. --member "Abdullah Ibrahim Amri"',
    )
    parser.add_argument(
        "--sizes",
        default="20,50,100",
        help="Comma-separated workload sizes (default: 20,50,100)",
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=DEFAULT_REPEATS,
        help="Number of timing repeats (default: 3)",
    )
    args = parser.parse_args()

    member = args.member.strip() or input("Enter your full name: ").strip()
    if not member:
        print("ERROR: Member name is required.")
        return 2

    try:
        sizes = tuple(int(x.strip()) for x in args.sizes.split(",") if x.strip())
        if not sizes or any(x <= 0 for x in sizes):
            raise ValueError
    except ValueError:
        print("ERROR: --sizes must look like 20,50,100")
        return 2

    if args.repeats < 2:
        print("ERROR: Use at least 2 repeats; Stage 1 default is 3.")
        return 2

    dataset_root = Path(args.dataset).expanduser().resolve()
    if not dataset_root.exists():
        print(f"ERROR: Dataset folder does not exist:\n{dataset_root}")
        print("\nIf needed, pass the correct path using:")
        print('python benchmark_sequential.py --member "Your Name" --dataset "FULL_PATH"')
        return 2

    all_images = discover_images(dataset_root)
    if not all_images:
        print(f"ERROR: No supported images found under:\n{dataset_root}")
        print("Supported:", ", ".join(sorted(SUPPORTED)))
        return 2

    max_size = max(sizes)
    if max_size > len(all_images):
        print(
            f"ERROR: Largest workload is {max_size}, but only "
            f"{len(all_images)} supported images were found."
        )
        return 2

    print("=" * 72)
    print("ARTI 503 - STAGE 1 SEQUENTIAL BENCHMARK")
    print("=" * 72)
    print(f"Member:          {member}")
    print(f"Dataset:         {dataset_root}")
    print(f"Images found:    {len(all_images)}")
    print(f"Workload sizes:  {sizes}")
    print(f"Repeats:         {args.repeats}")
    print("Parallelism:     NONE (sequential baseline)")
    print("=" * 72)

    # Build all workloads before timing.
    workloads: dict[int, list[Path]] = {}
    for size in sizes:
        workloads[size] = balanced_select(all_images, dataset_root, size)

    print("Validating selected images...")
    validate_images(workloads[max_size])
    print("Validation complete.\n")

    system_info = get_system_info()
    print("System information:")
    for k, v in system_info.items():
        print(f"  {k}: {v}")
    print()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = Path("benchmark_results") / f"{slugify(member)}_{timestamp}"
    result_dir.mkdir(parents=True, exist_ok=True)

    raw_rows: list[dict] = []
    workload_meta: dict[int, dict] = {}

    # Save deterministic manifest for reproducibility.
    manifest_lines = []
    for size in sizes:
        selected = workloads[size]
        sha = manifest_hash(selected, dataset_root)
        workload_meta[size] = {
            "images": len(selected),
            "manifest_sha256": sha,
        }
        manifest_lines.append(
            f"\n### WORKLOAD {size} IMAGES | SHA256 {sha}\n"
        )
        manifest_lines.extend(
            p.relative_to(dataset_root).as_posix() + "\n"
            for p in selected
        )

    (result_dir / "benchmark_manifest.txt").write_text(
        "".join(manifest_lines), encoding="utf-8"
    )

    for size in sizes:
        images = workloads[size]
        print(f"\n{'='*72}")
        print(f"WORKLOAD: {size} images")
        print(f"Manifest SHA256: {workload_meta[size]['manifest_sha256']}")
        print(f"{'='*72}")

        for repeat in range(1, args.repeats + 1):
            print(f"\nRepeat {repeat}/{args.repeats}")
            for algo_name, algo_func in ALGORITHMS:
                elapsed = time_algorithm(algo_name, algo_func, images)
                print(f"  {algo_name:<14} {elapsed:>10.4f} s")
                raw_rows.append({
                    "member": member,
                    "workload_images": size,
                    "manifest_sha256": workload_meta[size]["manifest_sha256"],
                    "repeat": repeat,
                    "algorithm": algo_name,
                    "runtime_seconds": round(elapsed, 6),
                })

    raw_fields = [
        "member", "workload_images", "manifest_sha256",
        "repeat", "algorithm", "runtime_seconds"
    ]
    write_csv(result_dir / "raw_timings.csv", raw_rows, raw_fields)

    summary_rows: list[dict] = []
    for size in sizes:
        for algo_name, _ in ALGORITHMS:
            vals = [
                float(r["runtime_seconds"])
                for r in raw_rows
                if r["workload_images"] == size and r["algorithm"] == algo_name
            ]
            row = {
                "member": member,
                "cpu_model": system_info["cpu_model"],
                "physical_cores": system_info["physical_cores"],
                "logical_threads": system_info["logical_threads"],
                "ram_gb": system_info["ram_gb"],
                "python_version": system_info["python_version"],
                "workload_images": size,
                "manifest_sha256": workload_meta[size]["manifest_sha256"],
                "algorithm": algo_name,
            }
            for i, value in enumerate(vals, start=1):
                row[f"run_{i}_seconds"] = round(value, 6)
            row["mean_seconds"] = round(statistics.mean(vals), 6)
            row["stdev_seconds"] = round(
                statistics.stdev(vals) if len(vals) > 1 else 0.0, 6
            )
            summary_rows.append(row)

    summary_fields = [
        "member", "cpu_model", "physical_cores", "logical_threads",
        "ram_gb", "python_version", "workload_images", "manifest_sha256",
        "algorithm",
    ] + [f"run_{i}_seconds" for i in range(1, args.repeats + 1)] + [
        "mean_seconds", "stdev_seconds"
    ]
    write_csv(result_dir / "summary.csv", summary_rows, summary_fields)

    # One wide row for the group leader's Stage 1 all-members table.
    largest = max(sizes)
    wide = {
        "member": member,
        "cpu_model": system_info["cpu_model"],
        "physical_cores": system_info["physical_cores"],
        "logical_threads": system_info["logical_threads"],
        "ram_gb": system_info["ram_gb"],
        "workload_images": largest,
        "manifest_sha256": workload_meta[largest]["manifest_sha256"],
    }
    total = 0.0
    for algo_name, _ in ALGORITHMS:
        mean_value = next(
            r["mean_seconds"]
            for r in summary_rows
            if r["workload_images"] == largest and r["algorithm"] == algo_name
        )
        key = {
            "Huffman": "huffman_mean_s",
            "RLE": "rle_mean_s",
            "K-Means": "kmeans_mean_s",
            "ROI K-Means": "roi_kmeans_mean_s",
        }[algo_name]
        wide[key] = mean_value
        total += float(mean_value)
    wide["total_mean_s"] = round(total, 6)

    group_fields = [
        "member", "cpu_model", "physical_cores", "logical_threads", "ram_gb",
        "workload_images", "manifest_sha256",
        "huffman_mean_s", "rle_mean_s", "kmeans_mean_s",
        "roi_kmeans_mean_s", "total_mean_s",
    ]
    write_csv(
        result_dir / "member_result_for_group.csv",
        [wide],
        group_fields,
    )

    with (result_dir / "system_info.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "member": member,
                "dataset_root": str(dataset_root),
                "images_discovered": len(all_images),
                "sizes": list(sizes),
                "repeats": args.repeats,
                **system_info,
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    print("\n" + "=" * 72)
    print("DONE")
    print("=" * 72)
    print(f"Results folder: {result_dir.resolve()}")
    print("\nSEND THESE TWO FILES TO THE GROUP LEADER:")
    print(f"  1) {result_dir / 'member_result_for_group.csv'}")
    print(f"  2) {result_dir / 'summary.csv'}")
    print("\nKeep benchmark_manifest.txt as reproducibility evidence.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
