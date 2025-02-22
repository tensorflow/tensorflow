# Using XLA tooling

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

## Running HLO snippets: `run_hlo_module`

The tool `run_hlo_module` operates on pre-optimization HLO, and by default
bundles compilation, running and comparison with the reference interpreter
implementation. For example, the usual invocation to run an input file
`computation.hlo` on an NVIDIA GPU and to check it for correctness is:

```
$ run_hlo_module --platform=CUDA --reference_platform=Interpreter computation.hlo
```

As with all the tools, `--help` can be used to obtain the full list of options.

## Running HLO snippets with SPMD support: `multihost_hlo_runner`

Multihost HLO runner is a very similar tool, with the caveat that it supports
SPMD, including cross host communication. See
[Multi-Host HLO Runner](./tools_multihost_hlo_runner) for details.

## Multi-HLO replay

Invocation with multiple modules is supported for both `run_hlo_module` and
`hlo_runner_main`, which is often convenient to replay all modules in a dump
directory:

```shell
$ hlo_runner_main /dump/*before_optimizations*
```

## Running passes/stages of HLO compilation: `hlo-opt`

When debugging or understanding the workings of the compiler, it is often useful
to get the expansion for a particular hardware at a particular point in the
pipeline (be it HLO, optimized HLO, TritonIR or LLVM), for a given (Stable) HLO
input.

`hlo-opt` supports multiple output stages: be it PTX, HLO after optimizations,
LLVM IR before optimizations, or TritonIR. The exact set of stages supported
depends on the platform (as e.g. PTX is NVIDIA-specific), and can be seen using
the --list-stages command:

```
$ hlo-opt --platform=CUDA --list-stages
hlo
llvm
ptx
```

After selecting a stage, the user can write the result of the conversion for a
given platform to a given stream:

```
$ hlo-opt myinput.hlo --platform=CUDA --stage=llvm
```

which would print the dump to stdout (or to a given file if `-o` was specified).

### Deviceless Compilation

Access to a GPU is not needed for most of the compilation, and by specifying a
GPU spec on the command line we can get e.g. PTX output without access to an
accelerator:

```
$ hlo-opt  --platform=CUDA --stage=llvm  --xla_gpu_target_config_filename=(pwd)/tools/data/gpu_specs/a100_pcie_80.txtpb input.hlo
```

Note: For the above invocation to work, the user would usually either need to
disable autotuning with `--xla_gpu_autotune_level=0` or load a pre-existing
autotuning results with `--xla_gpu_load_autotune_results_from=<filename>`
(obtained with `--xla_gpu_dump_autotune_results_to=<filename>`).

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

### `hlo-opt` and HLO Passes

#### Run Single/Multiple passes
The `hlo-opt` tool allows execution of an individual passes
independent of the full compilation pipeline. This isolation helps to quickly
run passes on input hlo module and pinpoint the root cause of failures,
eliminating need to run full compilation pipeline wait for it to complete.

```
$ hlo-opt --passes=reduce-window-rewriter,scatter_expander input.hlo
```
Note: `--platform` option is not required.

`hlo-opt` tool also supports XLA_FLAGS.

```
$ hlo-opt --passes=reduce-window-rewriter,scatter_expander --xla_reduce_window_rewrite_base_length=128 input.hlo
```
Use`list-passes` option to get the pass name string.

```
$ hlo-opt --list-passes
```

#### Assist New HLO Pass Development
If you are writing a new hlo pass, `hlo-opt` tool provides an easier way to
validate your pass functionality and write unit tests.

* First, write your pass.
* Register the new pass to the `hlo-opt` tool at `hlo/tools/opt_lib.cc`. Don't forget to add build dependency.

```
RegisterPass<FooPass>(FooPassInputOptions)
```
Include pass registration as part of your PR so that the pass will be
available to use for all hlo-opt users.

* Rebuild the `hlo-opt` tool and use `--passes=` option to run the pass.

```
$ hlo-opt --passes=foo-pass input.hlo
```

* Writing unit tests for the pass? refer https://openxla.org/xla/test_hlo_passes. for more details.

### Miscellaneous Uses

* Pass Runtime Measurement: For large models, full
compilation runs can take minutes and making it challenging to detect subtle
performance regressions. In contrast, individual pass runs allow for precise
performance measurement and the easy detection of even small increases in
execution time caused by new code changes.

* Convert HLO `HLO text` -> `pbtxt`
```
$ hlo-opt --emit-proto input.hlo
```

* Convert HLO `pb` or `pbtxt` -> `HLO text`
```
$ hlo-opt input.pb or input.pbtxt
```
