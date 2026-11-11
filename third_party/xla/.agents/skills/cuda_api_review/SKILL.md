---
name: cuda-api-review
description: >-
  Verifies that new or changed CUDA Toolkit and cuDNN API usage in XLA (Driver
  API, Runtime API, cuBLAS/cuBLASLt, cuFFT, cuSPARSE, cuSOLVER,
  NVRTC/nvJitLink/nvPTXCompiler, and cuDNN) satisfies compile-time header
  floors, runtime major-version compatibility rules, and dynamic symbol stub
  registration, and produces a CUDA API compatibility summary table. Use when
  authoring or reviewing any change touching xla/stream_executor/cuda/... that
  adds, removes, or changes calls to cu* / cuda* / cublas* / cufft* / cusparse*
  / cusolver* / nvrtc* / nvJitLink* / nvPTXCompiler* / cudnn* functions, enum
  values, struct fields, or CUDA_VERSION / CUDNN_VERSION guards. Don't use for
  NCCL, CUPTI, or NVML (which have separate versioning schemes), ROCm/HIP, TPU,
  or GPU AOT proto compatibility (use gpu-aot-compatibility-review).
---

# XLA CUDA API Version & Runtime Compatibility Review

Use this skill when reviewing a change or validating local modifications that
touch any file under `xla/stream_executor/cuda/...` (or for on-demand CUDA API
reviews across XLA).

## Context & Compatibility Policy

XLA enforces a minimum supported CUDA and cuDNN version at compile time (defined
in `xla/stream_executor/cuda/cuda_dnn.cc` and `tensorflow.bazelrc`; see
**Determine Current Version Floors from the Codebase** below), while binaries
compiled against a given CUDA major version `X` must run against any CUDA `X.0+`
driver or toolkit library at execution time:

1. **Why compile-time guards are not enough**:
   - `libcuda.so` is loaded dynamically (`xla/tsl/cuda/cuda_stub.cc`). Calling a
     `cu*` entry point missing from an older runtime `libcuda.so` logs a warning
     and returns `CUDA_ERROR_SHARED_OBJECT_INIT_FAILED` (or
     `CUDA_ERROR_SHARED_OBJECT_SYMBOL_NOT_FOUND`).
   - Toolkit libraries (`libcudart`, `cublas`, `cublasLt`, `cufft`, `cusparse`,
     `cusolver`, `nvjitlink`, `nvrtc`, etc.) are linked via dynamic trampoline
     stubs (`xla/tsl/cuda/*_stub.cc`). Calling a symbol absent from an older
     `.so` returns a generic symbol-not-found status (such as
     `CUBLAS_STATUS_INTERNAL_ERROR`) or crashes with `LOG(FATAL)`
     (`FatalErrorSymbols` in `cublas_stub.cc`).
   - Passing a newly introduced enum value, attribute, or struct field to an
     older function (such as `cuDeviceGetAttribute`, `cuGraphAddNode_v2`,
     `cuGraphKernelNodeSetAttribute`, or `cublasLtMatmulDescSetAttribute`)
     fails at runtime with `CUDA_ERROR_INVALID_VALUE` or
     `CUDA_ERROR_NOT_SUPPORTED`.
   - Ignoring error statuses (such as `.value_or(0)`) is **not** a substitute
     for an explicit runtime version check because it still triggers stub
     warnings and obscures whether the failure was an unsupported driver
     version or a genuine runtime error.
2. **Runtime floor (`X.0`)**:
   - XLA binaries do not guarantee that forward-compatibility driver packages
     (`cuda_compat`) are present in all deployment environments and must support
     running on stock system drivers and toolkit libraries back to `X.0` of the
     compiled CUDA major version `X` (`12.0` / R525 for CUDA 12.x builds; `13.0`
     / R580 for CUDA 13.x-only builds).
3. **In-scope vs. out-of-scope libraries**:
   - **In scope**: CUDA Driver API (`libcuda.so`, `cu*`, `cuda.h`), CUDA
     Runtime API (`libcudart.so`, `cuda*`, `cuda_runtime_api.h`), `cuBLAS` /
     `cuBLASLt` (`cublas*`, `cublasLt*`), `cuFFT`, `cuSPARSE`, `cuSOLVER`,
     `NVRTC`, `nvJitLink`, `nvPTXCompiler`, and `cuDNN` (`cudnn*`,
     `cudnn_frontend`).
   - **Out of scope**: `NCCL`, `CUPTI`, and `NVML` (which have separate
     versioning policies, though `xla/tsl/cuda/*.symbols` registration still
     applies if new symbols are called).

### Determine Current Version Floors from the Codebase

Do not assume static version numbers when the codebase defines them directly:

1. **Minimum & maximum compile-time CUDA versions (`min_cuda` / `max_cuda`)**:
   - Read `common:cuda_version --repo_env=HERMETIC_CUDA_VERSION=...` and the
     per-major release overrides (`build:pjrt_cuda12` / `build:pjrt_cuda13
     --repo_env=HERMETIC_CUDA_VERSION=...`) in `tensorflow.bazelrc` (e.g.,
     `"13.2.0"` default and `"12.9.1"` for `build:pjrt_cuda12`).
   - Note that XLA supports compiling against CUDA `>= 12.8` (`CUDA_VERSION >=
     12080`), so treat `min_cuda` as `12.8` (`12080`) unless the minimum
     supported CUDA floor across XLA has been raised higher.
2. **Minimum compile-time cuDNN version (`min_cudnn`)**:
   - Read the `static_assert(CUDNN_VERSION >= ...)` check in
     `xla/stream_executor/cuda/cuda_dnn.cc` (e.g., `90000` for cuDNN `9.0.0`)
     and `HERMETIC_CUDNN_VERSION` in `tensorflow.bazelrc`.
   - Take the minimum across those definitions (e.g., `min_cudnn = 9.0.0` /
     `90000`).
3. **Runtime floor (`X.0`)**: For any binary compiled against CUDA major version
   `X`, the runtime driver and toolkit library floor is `X.0` (`12.0` for CUDA
   12.x builds; `13.0` for CUDA 13.x-only builds).

### Version Compatibility Matrix

The table below illustrates how `min_cuda` (shown for `min_cuda = 12.8`) and the
`X.0` runtime rule interact:

| Library & Version Where Identifier Was Introduced | Compile-Time Guard Required? | Runtime Version Check Required? | Notes |
| :--- | :--- | :--- | :--- |
| **CUDA Toolkit `<= 12.0`** | **No** (flag `<= min_cuda` guards on touched lines as stale nit) | **No** | Guaranteed present at compile time (`>= min_cuda`) and runtime (`>= 12.0`). |
| **CUDA Toolkit `12.1` - `min_cuda` (`12.8`)** | **No** (flag `<= min_cuda` guards on touched lines as stale nit) | **Yes** (`>= 12.minor`) | Present in `min_cuda` headers at compile time, but absent at runtime on `12.0` - `12.(minor-1)` drivers/libraries. |
| **CUDA Toolkit `> min_cuda` in `12.x` (`12.9`)** | **Yes** (`>= 12.9` / `12090`) | **Yes** (`>= 12.9`) | Absent from `min_cuda` headers AND absent from `12.0` - `12.8` drivers/libraries at runtime. |
| **CUDA Toolkit `13.0`** | **Yes** (`>= 13.0` / `13000`) | **No** (once compile-time guarded for `>= 13.0`) | Only compiled into CUDA 13+ builds, where the runtime floor is already `13.0`. |
| **CUDA Toolkit `>= 13.1`** (e.g. `13.2`, `13.4`) | **Yes** (`>= 13.minor` / `130X0`) | **Yes** (`>= 13.minor`) | A binary compiled with CUDA 13.1+ headers must still run against a CUDA 13.0 driver/runtime. |
| **cuDNN `<= min_cudnn` (`9.0.0`)** | **No** (flag `<= min_cudnn` guards on touched lines as stale nit) | **No** | Guaranteed present at compile time (`>= min_cudnn`) and runtime (`loaded_version >= kSourceVersion`). |
| **cuDNN `> min_cudnn` (`> 9.0.0`)** | **Yes** (`#if CUDNN_VERSION >= N`) | **No** | cuDNN has no binary backward compatibility; `CudnnSupport::Init()` (`cuda_dnn.cc`) enforces `loaded_version >= kSourceVersion` at startup. |

## Step 1: Scope the Diff & Check Guard Removals

1. **Obtain only the target change's diff**:
   - Inspect only the delta introduced by the change or commit under review
     (e.g., `git diff` or `git diff main...HEAD`).
2. **Collect all touched CUDA/cuDNN identifiers and guards**:
   - Identify every function call, enum enumerator, attribute constant, struct
     or union type, struct field, or macro (`cu*`, `CU_*`, `cuda*`, `cublas*`,
     `cublasLt*`, `cufft*`, `cusparse*`, `cusolver*`, `nvrtc*`, `nvJitLink*`,
     `nvPTXCompiler*`, `cudnn*`) added, modified, or moved in the diff.
   - Inspect any `BUILD` hunks in the same change that add or modify
     `if_cuda_newer_than(...)` selects.
3. **Check guard-removal changes**:
   - When a change removes a stale compile-time guard `<= min_cuda` (`<= 12.8`),
     verify that the corresponding **runtime** version check (`> X.0`) is
     **preserved**. Flag any removal of a still-needed runtime check (`12.1+`
     in CUDA 12.x builds or `13.1+` in CUDA 13.x builds) as a **blocking**
     violation.
   - *Example*: Removing `#if CUDA_VERSION >= 12030` is valid for compilation
     because `min_cuda` is `12.8`, but the `driver_version >= 12.3` runtime
     check must remain because the CUDA 12.x runtime floor is `12.0`.

## Step 2: Lookup of Symbol Introduction Versions

Never guess from memory which CUDA or cuDNN version introduced an identifier.
Verify the introducing version using the CUDA/cuDNN headers and documentation:

### Tier 1: CUDA Driver API Functions (`cu*`) via `cudaTypedefs.h`

In the CUDA Toolkit headers, `cudaTypedefs.h` defines `PFN_cu<Name>_v<NNNNN>`
typedefs encoding the exact CUDA version in which each CUDA Driver API symbol or
ABI revision was introduced:

Examples:
- `PFN_cuGraphConditionalHandleCreate_v12030` -> introduced in **CUDA 12.3**
  (`12030`).
- `PFN_cuGraphAddNode_v12020` and `PFN_cuGraphAddNode_v12030`
  (`cuGraphAddNode_v2`) -> `cuGraphAddNode_v2` was introduced in **CUDA 12.3**
  (`12030`).
- `PFN_cuLaunchHostFunc_v2_v13020` -> introduced in **CUDA 13.2** (`13020`).

### Tier 2: CUDA / cuDNN Headers and Official Documentation

For Runtime API (`cuda*`), library symbols (`cublas*`, `cufft*`, `cusparse*`,
`cusolver*`, `nvrtc*`, `nvJitLink*`, `nvPTXCompiler*`, `cudnn*`), and enum
values or struct fields:
- Inspect the CUDA or cuDNN headers fetched in the Bazel output base (under
  `$(bazel info output_base)/external/` for `@cuda_cudart`, `@cuda_nvcc`,
  `@cuda_cublas`, `@cuda_cudnn`, etc.) or installed under
  `/usr/local/cuda/include/`.
- Check version comments, `#if` guards, or deprecation/introduction annotations
  in those headers, or consult the official NVIDIA CUDA Toolkit and cuDNN
  release notes.

## Step 3: Verify Compile-Time Guards (`> min_cuda` CUDA, `> min_cudnn` cuDNN)

1. **Missing compile-time guards (Blocking)**:
   - Every function, enum constant, struct/union type, struct field, or macro
     introduced after `min_cuda` (e.g., `CUDA_VERSION >= 12090` when `min_cuda`
     is `12.8`) or after `min_cudnn` (e.g., `CUDNN_VERSION > 90000` when
     `min_cudnn` is `9.0.0`) must be guarded at compile time with a version
     threshold matching at least its introducing version.
2. **Accepted compile-time guard patterns**:
   - **Pattern A (Recommended for helpers and multi-function features):
     Separate `.cc` file (`*_X_Y.cc`) gated via `if_cuda_newer_than("X_Y", ...)`
     in `BUILD` + `if constexpr (kHasCudaXXXXX)`**:
     - See `docs/avoiding_ifdefs.md` and
       `xla/stream_executor/cuda/cuda_command_buffer_12_9.cc`.
     - **Important C++17 caveat**: In non-templated functions, writing
       `if constexpr (CUDA_VERSION >= XXXXX)` directly around undeclared CUDA
       functions, types, or enum constants inside a single `.cc` file **fails
       to compile** against `min_cuda` headers because both branches must be
       valid syntax and pass name lookup. Undeclared CUDA identifiers must
       either live in a separate `if_cuda_newer_than`-selected `.cc` file
       behind a header-declared wrapper, or use `#if CUDA_VERSION >= XXXXX`.
   - **Pattern B (Allowed and preferred for small localized snippets): Inline
     `#if CUDA_VERSION >= XXXXX` / `#if CUDNN_VERSION >= XXXXX`**:
     - See `xla/stream_executor/cuda/cuda_stream.cc`.
     - When `#if CUDA_VERSION` (or `CUDA_VERSION` in C++ expressions) is used
       in a `.cc`/`.h` file, verify that the file explicitly includes
       `#include "third_party/gpus/cuda/include/cuda.h"`.
   - Verify that fallback `#else` / `if constexpr` branches return a clean
     `absl::UnimplementedError` naming the required version rather than
     silently doing nothing (unless falling back to an equivalent legacy API).
3. **Stale compile-time guards on touched lines (Non-blocking nit)**:
   - If a change adds or modifies a compile-time guard with threshold
     `<= min_cuda` (e.g., `<= 12080` / `12_8`) or `CUDNN_VERSION <= min_cudnn`
     (e.g., `<= 90000`), suggest removing the redundant compile-time guard as a
     cleanup nit, while reminding the author to keep any required runtime
     version check (`> X.0`).

## Step 4: Verify Runtime Checks (`> 12.0` for 12.x, `> 13.0` for 13.x)

1. **When a runtime check is required (Blocking if missing)**:
   - Required before first use on the code path for any CUDA Toolkit function,
     enum constant, attribute, or struct field introduced in:
     - **`12.1` - `12.9`** when reachable in CUDA 12.x builds (runtime floor is
       `SemanticVersion{12, 0, 0}`).
     - **`13.1+`** when compiled only under CUDA 13.x
       (`#if CUDA_VERSION >= 13000` or `if_cuda_newer_than("13_0", ...)`, where
       the runtime floor is `SemanticVersion{13, 0, 0}`).
   - **Not required for `13.0` APIs already guarded at compile time for
     `>= 13.0`**, and **not required for `cuDNN`** (`CudnnSupport::Init()`
     enforces `loaded_version >= kSourceVersion`).
2. **Dominating entry-point checks satisfy downstream calls**:
   - A single runtime check at object construction or function entry covers all
     downstream calls on that object or execution path up to the checked
     version.
   - For example, `CudaCommandBuffer::Create` in
     `xla/stream_executor/cuda/cuda_command_buffer.cc` checks
     `device_description.driver_version() < SemanticVersion{12, 3, 0}`, which
     satisfies the runtime requirement for all `<= 12.3` graph APIs called on
     `CudaCommandBuffer` (such as `cuGraphAddNode_v2` and
     `cuGraphExecNodeSetParams`), whereas `12.4+` APIs (such as `12.9`
     `CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE`) require their own `>= 12.9` check.
3. **New enum values, attributes, or struct fields passed to older functions**:
   - Passing a newer enum constant or struct member (such as
     `CU_DEVICE_ATTRIBUTE_MAX_OVERSIZED_SHARED_MEMORY_PER_BLOCK` from CUDA 13.4
     or `CU_GRAPH_CHILD_GRAPH_OWNERSHIP_MOVE` from CUDA 12.9) to an older
     function (`cuDeviceGetAttribute`, `cuGraphAddNode_v2`,
     `cuGraphKernelNodeSetAttribute`, `cublasLtMatmulDescSetAttribute`)
     requires an explicit runtime version check.
   - Relying on `.value_or(0)` or ignoring the return status when the call
     fails on older drivers is **not** sufficient.
4. **GPU compute-capability gates do not waive runtime checks**:
   - Always require an explicit runtime version check even when a code path is
     gated on GPU compute capability (`IsAtLeastHopper()`,
     `IsAtLeastBlackwell()`, etc.), so the CUDA version dependency is
     self-documenting in code. You may note in the review comment when a
     compute-capability gate makes the check redundant in practice.
5. **Idiomatic runtime version checks in `xla/stream_executor/cuda`**:
   - **Driver API (`cu*`)**:
     `device_description.driver_version() >= SemanticVersion{major, minor, 0}`
     (or `cuDriverGetVersion(&v)` / `ParseCudaVersion(v)`, as in
     `cuda_stream.cc`).
   - **Runtime API (`cuda*`)**:
     `device_description.runtime_version() >= SemanticVersion{major, minor, 0}`
     (or `cudaRuntimeGetVersion(&v)`).
   - **Other CUDA Toolkit libraries**: Query the library's version API
     (`cublasGetVersion`, `cublasLtGetVersion`, `cufftGetVersion`,
     `GetLibNvPtxCompilerVersion`, `nvJitLinkVersion`, etc.).

## Step 5: Verify Dynamic Stub Symbol Registration (`xla/tsl/cuda/*.symbols`)

Whenever a change introduces a call to a CUDA or cuDNN function that was not
previously called in XLA:

1. Check that the symbol name is listed in the corresponding symbol file under
   `xla/tsl/cuda/`:
   - `cuda.symbols` (`cu*`), `cudart.symbols` (`cuda*`), `cublas.symbols`,
     `cublasLt.symbols`, `cudnn.symbols`, `cufft.symbols`, `cusolver.symbols`,
     `cusparse.symbols`, `nvjitlink.symbols`, `nvrtc.symbols`.
2. Check `cuda.h` / `cuda_runtime_api.h` for per-thread default stream macro
   aliases (`__CUDA_API_PTSZ` or `__CUDA_API_PTDS`, e.g.,
   `#define cuLaunchHostFunc_v2 __CUDA_API_PTSZ(cuLaunchHostFunc_v2)`). When a
   header aliases the symbol to a `_ptsz` or `_ptds` variant, verify that
   **both** the base symbol and the `_ptsz` / `_ptds` variant are present in
   `*.symbols`.

## Step 6: Report CUDA API Compatibility Summary & Findings

Provide a `### CUDA API Version Compatibility Review` summary containing a
concise Markdown table of every CUDA/cuDNN identifier analyzed in the diff (or
state explicitly if the diff touches no CUDA/cuDNN APIs), alongside any inline
findings (`file:line` with concrete fix suggestions naming the introducing
CUDA/cuDNN version and the required compile-time guard, runtime check, or
`.symbols` entry):

```markdown
### CUDA API Version Compatibility Review

| Symbol / Identifier | Introduced In | Compile-Time Guard | Runtime Guard | `.symbols` Status | Verdict |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `cuLaunchHostFunc_v2` | CUDA 13.2 (`13020`) | `#if CUDA_VERSION >= 13020` | `cuDriverGetVersion >= 13020` | `cuLaunchHostFunc_v2` + `_ptsz` in `cuda.symbols` | Compliant |
```
