# CPU kernel speedups on `perf/cpu`

Fork of tensorflow/tensorflow at `60cd513b`. Every number below was measured on
this branch. Nothing is projected.

## How it was measured

- Machine: shared cloud VM, 8 vCPU "Intel Xeon Processor" (1 thread per core), about 15 GB RAM, no GPU.
  The box is shared and noisy: absolute times moved by about 30% between sessions.
  Compare the ratios, not the absolute times.
- Build: `bazel build -c opt --config=avx_linux` with TF's hermetic Clang toolchain.
  Only the three kernel test binaries were built:
  `//tensorflow/core/kernels:{bucketize_op_test,in_topk_op_test,unique_op_test}`.
  The full pip package was not built.
- Baseline: the same tests and benchmarks (commit `d257e1f3`) with the kernel sources from
  `60cd513b`. New: the branch head. Both use the same flags and toolchain.
- Method: each benchmark binary runs with `--benchmark_min_time=0.3s`. Base and new
  alternate order every round, for 7 rounds. The tables show the median of the 7 runs
  (us per op, real time), and the speedup is base median divided by new median.
  The script is in `tools/tf_bench/run3.py` on the build box (not committed).
- The TF Python tests (`*_op_test.py`) were not run, because they need the full pip build.
  The C++ correctness tests added in `d257e1f3` were run and passed on base and new,
  10 repeats for new. They include reference comparisons on large inputs that span
  many shards, plus NaN, signed-zero, tie, Inf and out-of-range cases.

## 1. Bucketize: shard the per-element binary search (`bucketize_op.cc`)

The CPU functor did one `std::upper_bound` per element on the calling thread. It now
uses the device `ParallelFor` for inputs with at least 4096 elements. Output is
bit-identical, and the element count is now `int64`.

| shape (elements / boundaries) | base us | new us | speedup |
|---|---:|---:|---:|
| 1024 / 100 (inline path) | 31.8 | 32.9 | 0.97x (noise) |
| 8192 / 100 | 364.1 | 96.9 | 3.76x |
| 65536 / 100 | 2790.6 | 434.5 | 6.42x |
| 1M / 10 | 18808.8 | 2844.2 | 6.61x |
| 1M / 100 | 42671.5 | 5975.4 | 7.14x |
| 1M / 1000 | 62242.8 | 9481.8 | 6.56x |
| 16M / 100 | 656978.6 | 92348.8 | 7.11x |

## 2. Unique (float/double, 1-D): `absl::flat_hash_map` instead of `std::unordered_map` (`unique_op.cc`)

`float` and `double` fell back to `std::unordered_map` only because NaN keys can't live
in `flat_hash_map`. NaNs now skip the map and each gets a fresh output slot, which is
what `std::unordered_map` produced. Output values, order, indices and counts are
unchanged. `-0.0` and `+0.0` still collapse to the first one seen. Int types were
already on `flat_hash_map` and are unchanged.

| benchmark (N / value range) | base us | new us | speedup |
|---|---:|---:|---:|
| FLOAT 4096 / 1024 | 2973.2 | 42.4 | 70.0x |
| FLOAT 1M / 1024 | 35639.3 | 10536.5 | 3.38x |
| FLOAT 1M / 65536 | 162785.0 | 39072.8 | 4.17x |
| FLOAT 1M / 64M | 185597.9 | 59992.3 | 3.09x |
| FLOAT 4M / 4M | 1513570.2 | 309862.7 | 4.88x |
| INT32 1024 / 1M (control) | 16.7 | 17.0 | 0.98x |
| INT32 1M / 1M (control) | 50486.1 | 57400.0 | 0.88x median, 0.99x min (noise; this code path did not change) |

The 70x at 4096/1024 is larger than the others. The `std::unordered_map` baseline is
slow at that size, and the result was consistent across sessions (73-75x in an
earlier 7-round run).

## 3. InTopK: adaptive row sharding (`in_topk_op.h`)

Rows are independent but ran serially. The kernel now runs the first 32 rows inline,
measures how many predictions they actually read (rows stop early once more than k
larger values are seen), and shards the remaining rows with `ParallelFor` only if the
estimated remaining work is at least 32K predictions. The row loop now reads through
raw pointers.

These numbers come from a separate 7-round run (base vs. the branch head), with k=5.
`hit=1` means the target is the row maximum, so every row is a full scan; that is the
common case for a trained classifier. `hit=0` means random predictions, so rows exit early.

| rows x classes, hit | base us | new us | speedup |
|---|---:|---:|---:|
| 64 x 1000, early exit | 16.4 | 15.8 | 1.04x |
| 64 x 1000, full scan | 76.2 | 64.9 | 1.17x |
| 1024 x 1000, early exit | 77.9 | 81.4 | **0.96x** |
| 1024 x 1000, full scan | 1080.8 | 192.4 | 5.62x |
| 4096 x 1000, full scan | 4124.3 | 625.2 | 6.60x |
| 1024 x 10000, full scan | 10147.4 | 1733.2 | 5.85x |
| 256 x 100000, full scan | 28234.5 | 6080.5 | 4.64x |
| BM_InTopK int64 64 x 1000, k=10 | 19.1 | 18.9 | 1.01x |
| BM_InTopK int64 64 x 10000, k=10 | 19.6 | 19.2 | 1.02x |

## Negative results and caveats

- Small Bucketize inputs (below 4096 elements) gain nothing. They stay on the caller
  thread on purpose. An earlier session measured 0.85-0.89x at 1024 elements for the
  inline path, and this session measured 0.97x. The loop is the same, so treat it as
  noise or code layout.
- InTopK, first attempt (not kept): always calling `ParallelFor` with a full-row cost
  hint gave 1.5-7.6x on full scans, but made small and early-exit inputs 1.5-2.9x
  slower (for example 64x10000 early exit went from 12.8 to 36.5 us), because dispatch
  overhead dominated rows that stop after a few dozen reads. The adaptive sampling
  above replaced it.
- InTopK cost-hint tuning made no difference. A ParallelFor cost hint of 2x, 8x, or a
  full row of `num_classes` per row, and a 128K instead of 32K threshold, all landed
  within noise of each other on these benchmarks, so the simplest hint was kept.
- InTopK 1024x1000 early exit is about 4-10% slower across runs (0.88-0.96x). It stays
  on the serial path (about 22 reads per sampled row, below the threshold), and that
  loop is a bit slower than the original for very short rows. I didn't find a fix that
  holds up beyond noise.
- The multithreaded speedups depend on idle cores. On a loaded machine, or inside an
  inter-op pool that is already busy, the gains will be smaller.
