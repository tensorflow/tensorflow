# XLA:GPU ahead-of-time (AOT) compilation

Ahead-of-time (AOT) compilation compiles and serializes an XLA computation in
one environment, allowing the resulting binary artifact to be loaded and
executed later by a separately deployed runtime.

Unlike just-in-time (JIT) compilation, which compiles computations in the
running process upon first execution, AOT removes compilation from the serving
path, which reduces startup latency and lets compilation run in a dedicated
build environment.

> **Note:** If you are here because a reviewer flagged a backward or forward
> compatibility problem with your change, go to the
> [GPU AOT compatibility guide](gpu_aot_compatibility.md).

## Workflow and APIs

The standard AOT pipeline consists of compile, serialize, and load/deserialize
phases:

*   **PjRt entry points:** In the C++ PjRt interface
    ([`xla/pjrt/pjrt_client.h`](https://github.com/openxla/xla/blob/main/xla/pjrt/pjrt_client.h)),
    compilation is initiated via `PjRtClient::Compile` (or `CompileAndLoad`).
    The resulting executable is serialized into a string via
    `PjRtExecutable::SerializeExecutable`. On the serving host, the runtime
    reconstructs the executable using `PjRtClient::DeserializeExecutable` or
    loads it directly for execution with `PjRtClient::LoadSerializedExecutable`.
*   **Compiler-level artifacts:** At the compiler layer, ahead-of-time
    compilation produces a
    [`GpuAotCompilationResult`](https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_aot_compilation_result.h)
    (a `CompiledModule`;
    [`AotCompilationResult`](https://github.com/openxla/xla/blob/main/xla/service/compiler.h)
    is a deprecated alias). This object encapsulates the serialized
    [`GpuExecutableProto`](https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_executable.proto)
    and exposes `SerializeAsString()` and `LoadExecutable()`.
*   **Command-line tooling:** The
    [`xla_compile`](https://github.com/openxla/xla/blob/main/xla/service/xla_compile_main.cc)
    tool compiles HLO or StableHLO modules into serialized executables:

    ```bash
    xla_compile --module_file=module.hlo --output_file=output.bin \
        --platform=gpu --gpu_target_config=gpu_target_config.pbtxt
    ```

    Compilation can run on a host without a GPU by providing a target device
    configuration (see the
    [`xla/backends/gpu/target_config` README](https://github.com/openxla/xla/blob/main/xla/backends/gpu/target_config/README.md)).

## What is serialized

The top-level container for serialized GPU programs is
[`GpuExecutableProto`](https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_executable.proto):

*   `thunks`: The sequence of execution thunks. Kernel thunks (for example
    `CustomKernelThunk`) embed their own compiled kernel binary (`cubin` for
    NVIDIA, `hsaco` for ROCm), so most of the device code lives here.
*   `binary`: A module-level device binary. Today it carries only the module's
    constant buffers; kernels are no longer stored here.
*   `buffer_allocations`: Buffer sizes and allocation slices.
*   `gpu_compute_capability`: Hardware architecture capability of the compile
    target.
*   `executable_abi_version`: Toolchain and runtime ABI versions.
*   `hlo_module_with_config`: Embedded serialized `HloModuleProto` and debug
    metadata.

Every proto message and field transitively reachable from `GpuExecutableProto`
is part of the compatibility contract described in the
[GPU AOT compatibility guide](gpu_aot_compatibility.md).

## How an artifact is loaded and executed

1.  **Reconstruct the thunk graph:** Deserialize `ThunkProto` entries into the
    in-memory thunk sequence, resolving custom call, FFI, and kernel symbols
    along the way. The embedded HLO module is parsed as well, so an HLO opcode
    unknown to the loading binary fails the load.
2.  **Run thunk graph conversion passes:** Execute load-time optimization and
    conversion passes on the in-memory thunks (for example, constructing runtime
    [`CommandBufferThunk`](https://github.com/openxla/xla/blob/main/xla/backends/gpu/runtime/command_buffer_thunk.cc)s
    to submit work via CUDA Graphs). `CommandBufferThunk` is reconstructed
    dynamically at load time and is never directly serialized.
3.  **Allocate buffers:** Initialize memory spaces and buffer allocations
    according to the serialized buffer assignment slices.
4.  **Execute:** Dispatch the prepared thunk sequence on device streams.

Pass, fusion, codegen, and autotuning changes that keep the serialized schema
and runtime semantics intact do not affect AOT compatibility.

## Troubleshooting: load failures that are not versioning bugs

Not all executable load failures are caused by serialization regressions. Two
common environment mismatches produce load-time errors:

*   **Compute capability mismatch:**
    [`GpuExecutable::FromProto`](https://github.com/openxla/xla/blob/main/xla/service/gpu/gpu_executable.cc)
    validates that the serialized `gpu_compute_capability` matches the target
    device. An executable compiled for NVIDIA Hopper cannot execute on NVIDIA
    Ampere.
*   **CUDA toolchain ABI mismatch:**
    [`CudaRuntimeAbiVersion::IsCompatibleWith`](https://github.com/openxla/xla/blob/main/xla/stream_executor/cuda/cuda_runtime_abi_version.cc)
    checks that the runtime host's CUDA toolkit, cuDNN, and CUB versions satisfy
    or exceed the minimum versions recorded in `ExecutableAbiVersionProto`. If
    the host driver or library environment is older than the compile target,
    loading fails with `FailedPreconditionError`.
