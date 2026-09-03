# TPU Microbenchmarks

## Quickstart

From the root directory of the XLA project, run:
```bash
# Run setup script to create the venv and install dependencies
./xla/benchmarks/setup.sh

# Activate the venv
source xla/benchmarks/.venv/bin/activate

# Run an individual benchmark
python3 xla/benchmarks/pallas_microbenchmarks/dense_matmul.py  --dim=1,2048,2048,2048 --fmt=f8e4m3fn,f8e4m3fn,f32

# Run dense matmul benchmark suite and write results to a CSV file
python3 xla/benchmarks/run_benchmarks.py --benchmarks=dense_matmul --csv_path=<path_to_dir>

# Run full benchmark suite and write results to multiple CSV files
python3 xla/benchmarks/run_benchmarks.py --csv_path=<path_to_dir>
```