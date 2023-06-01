# Performance benchmarks for MLIR based code generation

These benchmarks compare performance of Tensorflow -> LLVM code generation
with Eigen. These benchmarks are based on the Google Benchmark library and
can be integrated with performance monitoring tools.

## Running benchmarks

```
bazel run -c opt --cpu=haswell \
  :cwise_op_tanh_benchmark -- --benchmark_filter="f32/10k"
```

## Using perf and pprof with these benchmarks

1. Record perf profile
```
perf record -k 1 -o /tmp/perf.data --        \
  bazel run -c opt --cpu=haswell -copt=-gmlt \
  :cwise_op_tanh_benchmark -- --benchmark_filter="f32/10k"
```

2. Inject data from the JIT compiled functions
```
perf inject -j -v -i /tmp/perf.data -o /tmp/perf.data.jit
```

3. Report perf data

```
perf report -i /tmp/perf.data.jit
```

or

```
pprof -flame -nodecount=10000 /tmp/perf.data.jit
```

<!-- BEGIN GOOGLE-INTERNAL -->
## Running benchmarks using perflab and benchy

1. go/benchy
2. go/perflab

```
benchy                                                                        \
  --reference=${reference} --cpu=haswell --runs=20 --benchmark_filter=all     \
  --perflab --borg_constraints="platform_family_genus_cpu=indus-skylake-2000" \
  third_party/tensorflow/compiler/mlir/tfrt/benchmarks:cwise_op_tanh_benchmark
```

As of Q1 2021 `indus-skylake-2000` is the machine of the day, and roughly 60% of
the fleet cycles are executed on Skylakes.

Reference can be: 1. Cl number to test agains another pending change 2. `srcfs`
to test agains the g3 head 3. Another client number to test local changes
without exporting them <!-- END GOOGLE-INTERNAL -->
