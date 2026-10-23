---
name: avoid-ifdefs-review
description: >-
  Reviews XLA C++ changes under xla/ that add or extend preprocessor
  conditionals (#if, #ifdef, #ifndef, #elif, #else) against
  docs/avoiding_ifdefs.md. Classifies each conditional as exempt,
  acceptable-if-minimal, compile-required, or avoidable (Category I test
  skipping, II runtime-if, III low-level runtime access), flags avoidable
  conditionals as major findings, and proposes a concrete StreamExecutor /
  runtime-if / if constexpr (e.g. tsl::kIsDebugBuild) / separate-target rewrite.
  Use when reviewing or authoring XLA changes that touch #if directives. Don't
  use for xla/pjrt/ or xla/python/ (out of scope), non-C++ files, or changes
  that do not touch a preprocessor conditional.
---

# XLA: Avoiding `#ifdef`s - Review Skill

Encodes the guidelines in `docs/avoiding_ifdefs.md`
(https://openxla.org/xla/avoiding_ifdefs). Preprocessor conditionals multiply
build configurations, hide untested branches from the compiler and from static
analysis tools (`clang-tidy`, `include-cleaner`), and make refactorings fail
late in CI. This skill catches *new or extended* conditionals and steers them to
a runtime decision or a proper abstraction.

When in doubt between "minor" and "major", read the surrounding code and headers
before deciding.

## Step 0: Gate - Decide Whether This Skill Applies

1. Obtain **only the target change's diff** (e.g., `git diff` or
   `git diff main...HEAD`), avoiding changes from parent or unrelated commits in
   a stack.
2. Collect the **review-worthy hunks** - anything that increases or widens
   preprocessor branching:
   - Added lines matching `^\s*#\s*(if|ifdef|ifndef|elif|else)\b`.
   - Added or changed code lines that sit *inside* an existing `#if ... #endif`
     region (extending an existing branch). Open the file at the current
     revision and look upwards for the enclosing directive; the diff context
     alone is often too short to see it.
   - Modified condition expressions (e.g. `#ifdef __APPLE__` ->
     `#if defined(__APPLE__) || defined(__s390x__)`, or a version bound change).
   - **Not** review-worthy: pure code motion or re-indentation of an existing
     block, comment-only edits, and removed conditionals (see Step 4).
3. If there are no review-worthy hunks, stop. Do not post "nothing to report".

## Step 1: Classify Every Review-Worthy Conditional

| Class | Typical conditions | Verdict |
| --- | --- | --- |
| **A. Exempt** | Include guards (`XLA_..._H_`, `_TD_`), `#pragma once`; compiler feature detection (`__has_include`, `__has_builtin`, `__has_attribute`, `__has_feature`, `__clang__`, `__GNUC__`, `_MSC_VER`, `__cplusplus`); device-code guards (`__CUDA_ARCH__`, `__CUDACC__`, `__HIP_DEVICE_COMPILE__`, `__HIPCC__`) inside kernel source | No finding. |
| **B. Acceptable if minimal** | `NDEBUG`; sanitizer macros (`ABSL_HAVE_ADDRESS_SANITIZER`, `ABSL_HAVE_MEMORY_SANITIZER`, `ABSL_HAVE_THREAD_SANITIZER`, `__SANITIZE_*`); OS/arch detection (`_WIN32`, `__linux__`, `__APPLE__`, `__x86_64__`, `__aarch64__`, `__AVX2__`, `__ARM_NEON`); Highway `HWY_ONCE`/`HWY_TARGET` | No finding if the branches are minimal (Step 2); otherwise a **minor, non-blocking** suggestion. For `NDEBUG` always suggest `if constexpr (tsl::kIsDebugBuild)` (Step 2a) unless one of its listed exceptions applies. |
| **C. Compile-required** | Library version guards (`CUDA_VERSION`, `CUDART_VERSION`, `CUDNN_VERSION`, `CUDNN_FRONTEND_VERSION`, `NCCL_VERSION_CODE`, `TF_ROCM_VERSION`, `HIP_VERSION`), and backend guards (`GOOGLE_CUDA`, `TENSORFLOW_USE_ROCM`) **inside platform-leaf targets** (`xla/stream_executor/cuda/*.cc`, `xla/stream_executor/rocm/*.cc`, `*_cuda.cc`, `*_rocm.cc`) | Acceptable **only if** the guarded code references a symbol, enum value, header, or overload that does not exist in the other configuration. Verify it by inspecting the relevant library header or release notes. If the code would compile either way, treat it as class D. |
| **D. Avoidable** | `GOOGLE_CUDA`, `TENSORFLOW_USE_ROCM`, `TENSORFLOW_USE_SYCL`, `XLA_*`/`TF_*` feature toggles (`ENABLE_ONEDNN_ASYNC`, `INTEL_MKL`, ...) outside platform-leaf targets; any backend/version conditional in a `_test.cc`; conditionals in public headers that pull in platform headers | **Major** Category I / II / III finding (Step 3). |

Classification is about the *condition and location*, not about who wrote it.
A `CUDA_VERSION` guard in `xla/backends/gpu/transforms/` is class C only if the
compile-required test passes; a tiny `NDEBUG` block in a test is class B.

## Step 2: Minimality Check (Class B, and Class C That Passed Verification)

A conditional is minimal when the two branches differ by a constant or a single
call, not by logic. The preferred shape confines the preprocessor to **one
constant definition** and writes ordinary C++ everywhere else:

```cpp
// Compute the configuration once, then write ordinary code.
#if defined(ABSL_HAVE_MEMORY_SANITIZER) || defined(ABSL_HAVE_ADDRESS_SANITIZER)
constexpr int kSanitizerMultiplier = 10;
#else
constexpr int kSanitizerMultiplier = 1;
#endif
// ... later, no preprocessor involved:
absl::Duration timeout = kBaseTimeout * kSanitizerMultiplier;
```

### Step 2a: `if constexpr` on a Compile-Time Constant

When the condition is a build-mode or configuration constant and **both
branches compile in every configuration**, the right tool is `if constexpr`
on a `constexpr bool`, not `#if`. Both branches are then parsed and
type-checked in every build, so a typo in the debug-only branch fails the
optimized build too and static analysis tools (`clang-tidy`, `include-cleaner`)
see all of the code.

```cpp
#include "xla/tsl/platform/platform.h"  // tsl::kIsDebugBuild

// Before:
#ifndef NDEBUG
  GTEST_SKIP() << "Debug builds only.";
#endif
// After:
if constexpr (tsl::kIsDebugBuild) {
  GTEST_SKIP() << "Debug builds only.";
}

// Before:
#ifdef NDEBUG
  EXPECT_TRUE(Convert(thunks).ok());
#else
  EXPECT_DEATH(Convert(thunks).IgnoreError(), "started twice");
#endif
// After (both branches are now type-checked in every build):
if constexpr (tsl::kIsDebugBuild) {
  EXPECT_DEATH(Convert(thunks).IgnoreError(), "started twice");
} else {
  EXPECT_TRUE(Convert(thunks).ok());
}

// A pure boolean result needs no branch at all:
static constexpr bool ShouldCheckBufferSlices() { return tsl::kIsDebugBuild; }
```

Constants that exist in `xla/tsl/platform/platform.h`: `tsl::kIsOpenSource` and
`tsl::kIsDebugBuild`. Name them directly. If a change needs a constant that does
not exist yet (e.g. a sanitizer flag), suggest a file-local `constexpr bool`
defined with a single `#if` (as in the sanitizer example above) and
`if constexpr` at the use sites; do not block on the missing constant.

`if constexpr` does **not** apply - keep the `#if` and say why - when:

- the inactive branch would not compile (missing symbol, header, or overload;
  that is class C and must be verified as such);
- the conditional defines a macro differently per configuration
  (`XLA_INTERNAL_DCHECK_IMPL`), adds or removes a struct/class member
  (`AsyncValue::TypeInfo::has_data`), or selects `#include`s;
- the branch is itself a preprocessor construct (`#pragma`, attribute).

Raise a **minor** suggestion when:

- The conditional is `NDEBUG`-based and none of the exceptions above apply -
  suggest `if constexpr (tsl::kIsDebugBuild)` with the snippet, even if the
  block is tiny; the rewrite is mechanical.
- The branches contain different control flow, different test assertions, or
  duplicate more than a handful of lines - suggest `constexpr bool kFoo = ...;`
  plus `if constexpr` (Step 2a) or a runtime `if`, or two separate tests.
- A version guard wraps a whole function body - suggest guarding only the
  symbol that does not exist and keeping the rest unconditional.
- A block has grown beyond roughly ten lines - suggest separate source files
  selected in `BUILD` (`if_cuda_is_configured`, `if_rocm_is_configured`,
  `select()`), see Category III below.
- A verified compile-required library version guard (`CUDNN_VERSION`,
  `CUDNN_FRONTEND_VERSION`, `NCCL_VERSION_CODE`, ...) sits **outside
  `xla/stream_executor/`** - e.g. in `xla/backends/gpu/transforms/`,
  `xla/service/gpu/`, or `xla/backends/gpu/collectives/`. Accept it, but suggest
  confining the version-dependent piece to `xla/stream_executor/` (or the
  backend's leaf target) behind a backend-agnostic API, so compiler passes never
  see library version macros. Say explicitly that this is a non-blocking
  suggestion.
- A **version bound moved** (`>= 22900` -> `>= 22902`). State what changed in
  the library between the two versions that forces the new bound (renamed or
  added symbol, changed signature - cite the header or release note you
  checked). That sentence is the compile-required evidence; without it a
  reviewer cannot tell a necessary bump from an arbitrary one.

## Step 3: Recommend the Mitigation (Class D, and Class C That Failed Verification)

Always name the category, explain *why* the conditional is avoidable here, and
include a concrete rewrite snippet (these are **major** findings, see Step 4).
Pick the lightest fix that works:

- **Category I - skipping a test case.** The conditional decides whether a test
  runs. Ask `StreamExecutor` at runtime instead; the test fixture already has
  the device description:
  ```cpp
  const se::DeviceDescription& device =
      backend().default_stream_executor()->GetDeviceDescription();
  if (device.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }
  if (device.runtime_version() < se::SemanticVersion{12, 3, 0}) {
    GTEST_SKIP() << "Requires CUDA 12.3+.";
  }
  ```
  Available today in `xla/stream_executor/device_description.h`:
  `gpu_compute_capability().IsCuda()/IsRocm()`, `runtime_version()`,
  `driver_version()`, `compile_time_toolkit_version()`, `dnn_version()`,
  `cub_version()`, all `se::SemanticVersion`. If the needed property is
  missing, ask the author to add it to `DeviceDescription` rather than accept
  the conditional.
- **Category II - the code compiles in every configuration.** Replace the
  conditional with a runtime `if` on `DeviceDescription` (or a flag); if the
  decision is genuinely a build-time constant, use `if constexpr` on a
  `constexpr bool` instead (Step 2a). Point out explicitly that the guarded
  code has no configuration-specific symbols, so the preprocessor buys nothing
  but an untested branch.
- **Category III - the code needs backend-specific headers.** Two options, in
  order of preference:
  1. Push the backend-specific part into `xla/stream_executor/` behind a
     backend-agnostic API that exists for every backend (possibly returning
     `absl::UnimplementedError`), then call it unconditionally.
  2. Split into separate compilation units selected in `BUILD`:
     `feature.h` (no platform includes) + `feature_<backend>.cc` +
     `feature_stub.cc`, with
     `deps = if_rocm_is_configured([":feature_rocm"], [":feature_stub"])` (or
     `if_cuda_newer_than(...)` for CUDA version gates).
  Headers must never include CUDA/ROCm/SYCL headers conditionally; that is the
  pattern that leaks `CUDA_VERSION` checks all over XLA.
- **Widened conditions** (adding `|| defined(TENSORFLOW_USE_SYCL)` to an
  existing `GOOGLE_CUDA || TENSORFLOW_USE_ROCM` guard) count as the category
  of the code they guard. Note that the widening is the moment to introduce
  the proper abstraction, because a third backend makes the branch count grow
  again.

## Step 4: Severity & Tone

- **Major vs. minor**: Class D findings, and class C conditionals whose
  compile-required claim did not hold, are **major** findings; link to
  https://openxla.org/xla/avoiding_ifdefs (`docs/avoiding_ifdefs.md`). Class B
  minimality suggestions are **minor** and non-blocking.
- **Removals**: A change that net-removes conditionals (e.g. replacing an
  `#if CUDNN_VERSION` skip with `dnn_version()`) deserves one short positive
  note and no further findings from this skill.
- Group repeated instances of the same pattern into one finding with a bulleted
  list of the other locations; do not post the same rewrite ten times.
