# XLA Tooling

The XLA development workflow is usually centered around
[HLO](./operation_semantics) IR, which represents isolated functional
computation given to the compiler. XLA comes with multiple command line tools
(described below) which consume HLO and either run it, or provide an
intermediate compilation stage. Using such tools is invaluable for a fast
`compile->modify->run` iteration cycle, as HLO is both visualizable and
hackable, and iteratively changing and running it is often the fastest way to
understand and to fix an XLA performance or behavior.

The easiest way to obtain the HLO for a program being compiled with XLA is
usually to use the `XLA_FLAGS` environment variable:

```
$ XLA_FLAGS=--xla_dump_to=/tmp/myfolder ./myprogram-entry-point
```

which stores all before-optimization HLO files in the folder specified, along
with many other useful artifacts.

## [`run_hlo_module`] Run HLO Modules

```
$ bazel run //xla/tools:run_hlo_module -- [flags] <filename>
```

The tool `run_hlo_module` operates on pre-optimization HLO, and by default
bundles compilation, running and comparison with the reference interpreter
implementation. For example, the usual invocation to run an input file
`computation.hlo` on an NVIDIA GPU and to check it for correctness is:

```
$ run_hlo_module --platform=CUDA --reference_platform=Interpreter computation.hlo
```

### Run Multiple HLO Modules
Invocation with multiple HLO modules is supported for `run_hlo_module`. To run
all hlo modules from a directory:

```
$ bazel run //xla/tools:run_hlo_module -- [flags] /dump/*before_optimizations*
```

## [`multihost_hlo_runner`] Run HLO Modules With SPMD Support

```
Note: Binary name is `hlo_runner_main`.
$ bazel run //xla/tools/multihost_hlo_runner:hlo_runner_main -- [flags] <filename>
```

Multihost HLO runner is a very similar tool, with the caveat that it supports
SPMD, including cross host communication. See
[Multi-Host HLO Runner](./tools_multihost_hlo_runner) for details.

### Run Multiple HLO Modules With SPMD Support

Similar to `run_hlo_module`, `multihost_hlo_runner` also supports invocation
with multiple modules.

```
$ bazel run //xla/tools/multihost_hlo_runner:hlo_runner_main -- [flags] /dump/*before_optimizations*
```

## [`hlo-opt`] Compile HLO Module

```
$ bazel run //xla/tools:hlo-opt -- --platform=[gpu|cpu|...] [more flags] <filename>
```

When debugging or understanding the workings of the compiler, it is often useful
to get the expansion for a particular hardware at a particular point in the
pipeline (be it HLO, optimized HLO, TritonIR or LLVM), for a given HLO or
StableHLO input.

`hlo-opt` supports multiple output stages: be it PTX, HLO after optimizations,
LLVM IR before optimizations, or TritonIR. The exact set of stages supported
depends on the platform (as e.g. PTX is NVIDIA-specific), and can be seen using
the --list-stages command:

```
$ hlo-opt --platform=CUDA --list-stages
buffer-assignment
hlo
hlo-backend
html
llvm
llvm-after-optimizations
llvm-before-optimizations
ptx
```

After selecting a stage, the user can write the result of the conversion for a
given platform to a given stream:

```
$ hlo-opt --platform=cpu --stage=hlo input.hlo
```

which would print the dump to stdout (or to a given file if `-o` was specified).

### Deviceless Compilation for GPU

Deviceless compilation do not need access to a GPU. The Deviceless Compilation
provides a way to specify GPU spec on the command line
(`--xla_gpu_target_config_filename`) for stages where access to GPU is required.
eliminating a need for GPU device.

Example: PTX output without access to a gpu device:

```
$ hlo-opt  --platform=CUDA --stage=llvm  --xla_gpu_target_config_filename=/xla/tools/hlo_opt/gpu_specs/a100_pcie_80.txtpb input.hlo
```

Specs for popular GPUs are shipped with the compiler, and the provided file is
string serialization of `device_description.proto`:

```
gpu_device_info {
  cuda_compute_capability {
    major: 8
    minor: 0
  }
  threads_per_block_limit: 1024
  threads_per_warp: 32
  shared_memory_per_block: 127152
  shared_memory_per_core: 65536
  threads_per_core_limit: 2048
  core_count: 6192
  fpus_per_core: 64
  block_dim_limit_x: 2147483647
  block_dim_limit_y: 65535
  block_dim_limit_z: 65535
  memory_bandwidth: 2039000000000
  l2_cache_size: 4194304
  clock_rate_ghz: 1.1105
  device_memory_size: 79050250240
}
platform_name: "CUDA"
```
More GPU specs are located at `/xla/tools/hlo_opt/gpu_specs`

Note: **Autotuning**\
Sometimes compilation may involve autotuning based on a compilation `--stage`.
For the deviceless compilation to work, the user either need to\
**disable** autotuning with `--xla_gpu_autotune_level=0`\
or\
**load a pre-existing
autotuning results** with `--xla_gpu_load_autotune_results_from=<filename>`
(obtained with `--xla_gpu_dump_autotune_results_to=<filename>`).

Deviceless compilation might run into issues if autotuning is required. Luckily,
we can also provide those on the command line:

```
$ hlo-opt  --platform=CUDA --stage=llvm  --xla_gpu_target_config_filename=gpu_specs/a100_pcie_80.txtpb --xla_gpu_load_autotune_results_from=results.textpb input.hlo
```

The autotune file is text serialization of `autotune_results.proto`, with
example looking like:

```
version: 3
results {
  device: "CUDA: 8.0, Cores: 108, GPU clock: 1.41 GHz, Memory bandwidth: 1555 GB/s, L2 cache: 40 MB"
  hlo: "{\n  tmp_0 = f16[1,16,17,3]{3,2,1,0} parameter(0)\n  tmp_1 = f16[16,51]{1,0} bitcast(f16[1,16,17,3]{3,2,1,0} tmp_0)\n  tmp_2 = s8[16,17,3]{2,1,0} parameter(1)\n  tmp_3 = s8[51,16]{0,1} bitcast(s8[16,17,3]{2,1,0} tmp_2)\n  tmp_4 = f16[51,16]{0,1} convert(s8[51,16]{0,1} tmp_3)\n  tmp_5 = f16[16,16]{1,0} dot(f16[16,51]{1,0} tmp_1, f16[51,16]{0,1} tmp_4), lhs_contracting_dims={1}, rhs_contracting_dims={0}\n  ROOT tmp_6 = f16[1,16,16]{2,1,0} bitcast(f16[16,16]{1,0} tmp_5)\n}"
  result {
    run_time {
      nanos: 31744
    }
    triton {
      block_m: 32
      block_n: 32
      block_k: 32
      split_k: 1
      num_stages: 1
      num_warps: 4
    }
  }
}
```

The autotuning database can be serialized using
`XLA_FLAGS=--xla_gpu_dump_autotune_results_t=<myfile.pbtxt>`

## [`hlo-opt`] HLO Pass Development And Debugging

```
If you are working with hardware independent passes from the
`xla/hlo/transforms/` directory, prefer light-weight version
of the `hlo-opt` tool with fewer dependencies:

$ bazel run //xla/hlo/tools:hlo-opt -- [flags] <filename>

Otherwise, for hardware independent and CPU, GPU passes use
the same binary from "Compile HLO Modules" section above:

$ bazel run //xla/tools:hlo-opt -- [flags] <filename>
```

The `hlo-opt` tool allows execution of an individual passes
independent of the given platform compilation stages. This isolation helps to
quickly run passes on input hlo module and pinpoint the root cause of failures.

```
$ hlo-opt --passes=schedule-aware-collective-cse input.hlo
```

Note: `--platform` option is not required.

`hlo-opt` tool also supports [`DebugOptions XLA_FLAGS`](https://github.com/openxla/xla/blob/5bf1e6420d250dce5eb840889096bdf8aad6f432/xla/xla.proto#L40-L1197).

```
$ hlo-opt --passes=schedule-aware-collective-cse
--xla_gpu_experimental_collective_cse_distance_threshold=20 input.hlo
```

Use`--list-passes` option to get the pass name string.

```
$ hlo-opt --list-passes
```

Users can create their own custom pipeline by specifying more than one passes
to `--passes` option.

```
$ hlo-opt --passes=pass1,pass2,pass3 input.hlo
```

### Assist New HLO Pass Development

1. First, write your pass.
1. Register the new pass to the `hlo-opt` tool pass registry.

    ```
    RegisterPass<FooPass>(FooPassInputOptions)
    ```

    Based on the pass type, choose one of the following locations for
    registration:\
    [`opt_lib.cc`](https://github.com/openxla/xla/blob/5d015a2ddfcf4f40934a33891dc63471704f221d/xla/hlo/tools/hlo_opt/opt_lib.cc) Hardware-independent passes.\
    [`cpu_opt.cc`](https://github.com/openxla/xla/blob/5d015a2ddfcf4f40934a33891dc63471704f221d/xla/tools/hlo_opt/cpu_opt.cc) CPU specific passes.\
    [`gpu_opt.cc`](https://github.com/openxla/xla/blob/5d015a2ddfcf4f40934a33891dc63471704f221d/xla/tools/hlo_opt/gpu_opt.cc) GPU specific passes.\
    [`compiled_opt.cc`](https://github.com/openxla/xla/blob/5d015a2ddfcf4f40934a33891dc63471704f221d/xla/tools/hlo_opt/compiled_opt_lib.cc) Passes common to CPU, GPU, XPU.\
    Don't forget to add build dependency.

    Include pass registration as part of your PR([example](https://github.com/openxla/xla/pull/22968/files#diff-e37a0ea999dfc5764d624240cd2edebb8b7ee4e6d91686be89c632dd7203b823)) so that the pass will be
    available to use for all `hlo-opt` users.

1. Rebuild the `hlo-opt` tool, validate successful pass registration using
   `--list-passes` option and then use `--passes` option to run the pass.

    ```
    $ hlo-opt --passes=foo-pass input.hlo
    ```

1. Writing unit tests for the pass? refer https://openxla.org/xla/test_hlo_passes for more details.

### Pass Runtime Measurement

For large models, full compilation runs can take upto few minutes, making it
challenging to detect subtle performance regressions. In contrast, individual
pass runs using `hlo-opt` allow for precise
performance measurement and the easy detection of even small increases in
execution time caused by new code changes.

```
$ time hlo-opt --passes=reduce-window-rewriter,scatter_simplifier
--xla_reduce_window_rewrite_base_length=128 input.hlo
```

## [`hlo-opt`] Convert HLO Module Formats

```
Use the light weight version of the `hlo-opt` tool.

$ bazel run //xla/hlo/tools:hlo-opt -- [flags] <filename>
```

#### Convert `HLO Text` -> `HLO Proto`

```
$ hlo-opt --emit-proto input.hlo
```

#### Convert `HLO Proto` or `HLO Proto Binary` -> `HLO Text`

```
$ hlo-opt input.pbtxt or input.pb
```
