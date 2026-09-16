# TPU Microbenchmarks

## Prerequisites

- Python >= 3.12 (required by JAX >= 0.11)

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
Some platforms, like TPU v5e and v5p, do not have enough scoped VMEM by default
to run some of the benchmarks, so you need to increase the limit. It may also be
necessary to turn on large 2nd minor for bf16 inputs in some cases, or the cost
model will underestimate the required VMEM. Both of these options can be
controlled using `LIBTPU_INIT_ARGS`:
```bash
LIBTPU_INIT_ARGS="--xla_tpu_scoped_vmem_limit_kib=32768 --xla_tpu_control_large_2nd_minor_layout_for_x16=true" python3 xla/benchmarks/run_benchmarks.py --benchmarks=dense_matmul --csv_path=<path_to_dir>
```