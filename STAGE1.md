# Stage 1 — sequential benchmark

## Team procedure

1. Follow the setup command in README and run from the repository root.
2. Use your full name and an explicit `--dataset` path. Use the same dataset folder structure and image contents as the rest of the team.
3. Confirm the banner shows workloads `(20, 50, 100)` and 3 repeats. The older four-image runner is not the final Stage 1 benchmark.
4. Close unnecessary applications and keep the machine's power settings consistent during measurement.
5. Allow all four algorithms and all repeats to finish. Record actual measurements only.
6. Enter the results into `ARTI503_Shared_Team_Benchmark.xlsx` and provide `summary.csv` and `member_result_for_group.csv` to the team leader.

## Measurement contract

- Sequential Python execution; no multiprocessing or thread pool is used. This runner does not explicitly constrain native library thread counts.
- Deterministic round-robin selection across relative parent folders, sorted by path.
- Workloads: 20, 50 and 100 images; default repeats: 3.
- Timer: `time.perf_counter()`. Timed work includes image loading and each algorithm's processing; discovery, validation and CSV output are outside the timer.
- Huffman: 8-bit grayscale encode/decode. RLE: encode/decode. K-Means: K=8, 10 iterations. ROI K-Means: ROI K=16, background K=4, 10 iterations.
- Python and NumPy random seeds are reset to 42 + image index for K-Means variants.
- Images are validated before timing. The runner does not verify decoded pixel equality or diagnostic image quality.
- No explicit warm-up or file-cache clearing is performed.

## Outputs

Each completed run writes to `benchmark_results/<member>_<timestamp>/` under the working directory:

| File | Contents |
|---|---|
| `raw_timings.csv` | Every workload, algorithm and repetition time |
| `summary.csv` | Per-workload means and sample standard deviations, plus machine information |
| `member_result_for_group.csv` | One team-summary row for the largest workload |
| `benchmark_manifest.txt` | Selected relative paths and workload hashes |
| `system_info.json` | Machine, Python, NumPy and run settings |

Compare manifest hashes between members. These hashes cover relative paths, not image bytes, so also ensure everyone uses the same dataset copy. Keep outputs locally; they are excluded from Git.

## Troubleshooting

- Missing `scripts`: retain the repository layout; do not copy only the runner.
- Missing packages: install dependencies with the same Python executable used for the run.
- Dataset not found: supply its actual path using `--dataset`.
- macOS metadata (`__MACOSX` and `._*`) is ignored by image discovery.
- Any other unreadable image: inspect the reported file and restore a valid dataset copy before comparing results.
- The default dataset path is relative to the working directory; the README command avoids ambiguity by passing an explicit path.

The benchmark runs the original algorithms without changing their compression behavior. Existing algorithm errors may stop a run; partial or failed runs must not be entered as completed measurements.
