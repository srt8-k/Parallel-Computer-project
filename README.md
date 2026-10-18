# Parallel Computer project — ARTI503 Stage 1

Sequential baseline for medical image compression on multicore CPUs. Stage 1 runs Huffman, RLE, K-Means and ROI K-Means one algorithm and one image at a time, with workloads of **20, 50 and 100 images**, each repeated **3 times**.

## Setup and run

Use Python 3.10 or newer. Open PowerShell in this repository's root:

```powershell
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install numpy pillow matplotlib
& .\.venv\Scripts\python.exe .\benchmark_sequential.py --member "Your Full Name" --dataset "C:\path\to\brain-tumor-mri-dataset"
```

Environment activation is unnecessary. Download/extract the team's MRI dataset separately; it is not included in this repository. Every member must use the same dataset contents and directory layout. Pass the dataset path explicitly as shown above. Keep `benchmark_sequential.py` beside `scripts/`.

The original compression algorithms come from [FaisalBinJahlan/medical-image-project](https://github.com/FaisalBinJahlan/medical-image-project). Their implementations are preserved here as runtime dependencies.

## Files

- `benchmark_sequential.py`: sequential timing runner.
- `scripts/`: four compression algorithm modules required by the runner.
- `STAGE1.md`: measurement procedure and team submission instructions.
- `ARTI503_Shared_Team_Benchmark.xlsx`: shared team workbook.
- `.gitignore`: excludes environments, datasets, archive metadata and local outputs.

See [STAGE1.md](STAGE1.md) for result files and comparison rules. The updated Word report has not yet been supplied and is not included.
