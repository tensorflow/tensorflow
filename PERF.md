# CPU kernel speedups on `perf/cpu`

Fork of tensorflow/tensorflow at `60cd513b`. Every number below was measured on
this branch. Nothing is projected.

## How it was measured

- Machine: shared cloud VM, 4 vCPU "Intel Xeon Processor" (1 thread per core), about 15 GB RAM, no GPU.
  The box is shared and noisy. Compare the ratios, not the absolute times.
  (An earlier session on this PR used an 8-vCPU box; absolute times and speedups
  differ with core count, as expected for the sharded kernels.)
- Build: `bazel build -c opt --config=avx_linux` with TF's hermetic Clang toolchain.
  Only the three kernel test binaries were built:
  `//tensorflow/core/kernels:{bucketize_op_test,in_topk_op_test,unique_op_test}`.
  The full pip package was not built.
- Baseline: the same tests and benchmarks with the kernel sources from
  `60cd513b` (Bucketize / InTopK / Unique). New: this branch head. Both use the
  same flags, toolchain, and test harness.
- Method: each benchmark binary runs with `--benchmark_min_time=0.3s`. Base and new
  alternate order every round, for 7 rounds. The tables show the median of the 7 runs
  (us per op, real time), and the speedup is base median divided by new median.
  The script is `tools/tf_bench/run3.py`.
- The TF Python tests (`*_op_test.py`) were not run, because they need the full pip build.
  The C++ correctness tests were run and passed on the new binaries. They include
  reference comparisons on large inputs that span many shards, plus NaN, signed-zero,
  tie, Inf, empty-input, and out-of-range cases.

## 1. Bucketize: shard the per-element binary search (`bucketize_op.cc`)

The CPU functor did one `std::upper_bound` per element on the calling thread. It now
uses the device `ParallelFor` for inputs with at least 4096 elements. Output is
bit-identical (including historical IEEE NaN placement at the last bucket). The
element count is `int64`. Cost hint follows searchsorted's `Log2Ceiling` shape.

| shape (elements / boundaries) | base us | new us | speedup |
|---|---:|---:|---:|
| 1024 / 100 (inline path) | 17.4 | 17.6 | 0.99x (noise) |
| 8192 / 100 | 254.2 | 148.9 | 1.71x |
| 65536 / 100 | 2161.4 | 625.4 | 3.46x |
| 1M / 10 | 15373.9 | 4104.5 | 3.75x |
| 1M / 100 | 31222.4 | 8878.4 | 3.52x |
| 1M / 1000 | 49280.6 | 12668.9 | 3.89x |
| 16M / 100 | 496004.2 | 136985.3 | 3.62x |

## 2. Unique (1-D): `absl::flat_hash_map` for float/double + ordered fill (`unique_op.cc`)

`float` and `double` fell back to `std::unordered_map` only because NaN keys can't live
in `flat_hash_map`. NaNs now skip the map and each gets a fresh output slot, which is
what `std::unordered_map` produced. Output values are collected into a vector in
first-occurrence order and copied contiguously (instead of scattering through the
hash map), which also helps int types. `-0.0` and `+0.0` still collapse to the first
one seen. `Eigen::half` / `bfloat16` keep `std::unordered_map`.

| benchmark (N / value range) | base us | new us | speedup |
|---|---:|---:|---:|
| FLOAT 4096 / 1024 | 2427.4 | 19.5 | 124.6x |
| FLOAT 1M / 1024 | 28674.8 | 6506.1 | 4.41x |
| FLOAT 1M / 65536 | 136481.0 | 21548.9 | 6.33x |
| FLOAT 1M / 64M | 105323.4 | 26903.6 | 3.91x |
| FLOAT 4M / 4M | 678110.6 | 157266.7 | 4.31x |
| INT32 1024 / 1M (map unchanged; fill path) | 13.8 | 6.7 | 2.06x |
| INT32 1M / 1M (map unchanged; fill path) | 32406.6 | 28036.0 | 1.16x |

The 124x at 4096/1024 is larger than the others. The `std::unordered_map` baseline is
especially slow at that size; the result was consistent across the 7 rounds.

## 3. InTopK: adaptive row sharding (`in_topk_op.h`)

Rows are independent but ran serially. The kernel now samples up to 32 rows
**strided across the batch**, measures how many predictions they actually read
(rows stop early once more than k larger values are seen), and shards all
remaining rows with `ParallelFor` only if the estimated total work is at least
16K predictions. The row loop reads through raw pointers.

These numbers come from the same 7-round run, with k=5.
`hit=1` means the target is the row maximum, so every row is a full scan; that is the
common case for a trained classifier. `hit=0` means random predictions, so rows exit early.

| rows x classes, hit | base us | new us | speedup |
|---|---:|---:|---:|
| 64 x 1000, early exit | 14.6 | 14.4 | 1.01x |
| 64 x 1000, full scan | 64.5 | 59.5 | 1.09x |
| 1024 x 1000, early exit | 77.4 | 60.4 | 1.28x |
| 1024 x 1000, full scan | 913.9 | 233.7 | 3.91x |
| 4096 x 1000, full scan | 3514.7 | 779.5 | 4.51x |
| 1024 x 10000, full scan | 8377.1 | 2004.0 | 4.18x |
| 256 x 100000, full scan | 20773.3 | 6226.2 | 3.34x |

## Negative results and caveats

- Small Bucketize inputs (below 4096 elements) gain nothing. They stay on the caller
  thread on purpose. Measured 0.99x at 1024 elements (noise).
- InTopK, first attempt (not kept): always calling `ParallelFor` with a full-row cost
  hint gave large wins on full scans, but made small and early-exit inputs much
  slower because dispatch overhead dominated rows that stop after a few dozen reads.
  Adaptive sampling replaced it.
- InTopK previously sampled only the leading 32 rows and gated on remaining work.
  A cheap prefix could hide expensive later rows and stay serial; that is why this
  revision samples strided rows and gates on total estimated work. On this 4-vCPU
  box the early-exit 1024x1000 case is 1.28x (no longer a regression).
- Multithreaded speedups scale with idle cores. On a loaded machine, or inside an
  inter-op pool that is already busy, the gains will be smaller. Absolute times here
  are lower than an earlier 8-vCPU session of this PR; ratios for the sharded kernels
  are correspondingly closer to ~4x than ~7x.
