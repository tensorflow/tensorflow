# The Hitchhiker's Guide to fewer `#ifdef`s in XLA

**TL;DR:** `#ifdef`s are bad for maintainability. Please try to avoid them.
Scroll down for details on how.

[TOC]

## Introduction

Preprocessor conditionals (`#if`, `#ifdef`, `#ifndef`, etc.) are a convenient
way to make existing code work in a different environment than it was originally
intended for. Examples include:

*   Making existing CUDA-specific code work in a ROCm or SYCL environment.
*   Adding new library features (for example, from a newer cuDNN or hipDNN
    release) while maintaining support for older versions.

However, they come with a high maintenance cost (more details in the next
section), especially when refactoring existing code.

This guide explains the downsides of preprocessor conditionals and offers
alternatives for the most common use cases in a cookbook style. It is intended
to help contributors design maintainable, platform-portable changes from the
start and to serve as a shared reference during code reviews.

## Motivation

The C preprocessor is the first stage in a C++ compilation pipeline.
Preprocessor conditionals manipulate the stream of preprocessor tokens that is
read from the source code and header files and handed over to the C++ compiler.
The fact that it runs before the compiler and operates on the preprocessor token
level is the main reason why it results in a hard-to-maintain codebase:

*   **Code Complexity:** `#ifdef`s are convenient to use because they can be
    inserted almost anywhere in the code. Since they operate on tokens, there
    are almost no syntactic or semantic limits. Developers don't need to think
    about a proper abstraction; they can just insert the conditionals right
    where they are needed. Not being forced to design a proper abstraction makes
    the code harder to read and harder to reason about. Having even a simple
    abstraction like a free function allows easier testing of this function or
    mocking its behavior—either now or later.

    It also renders a subset of compiler warnings meaningless because the
    compiler will only ever see one evaluated token stream. A common example is
    the need for the `[[maybe_unused]]` attribute on a function parameter that
    is only used in one branch of a preprocessor conditional.

*   **Testability (Build):** Since preprocessor conditionals operate on token
    streams, all the untaken branches neither need to be syntactically nor
    semantically correct and are therefore not checked for correctness. This
    severely hinders larger refactorings where a developer applies a
    find-and-replace and relies on the compiler to tell them where to make
    manual fixups. Incorrect changes in uncompiled preprocessor branches either
    go in unnoticed or get detected late in the process (if CI happens to build
    that particular preprocessor branch). The latter is also relevant for
    day-to-day development (see the next point).

*   **Testability (Coverage):** Having more preprocessor conditionals means
    having more build configurations to test, and the number of configurations
    increases exponentially with every new condition introduced. It is already
    impossible to test all of them. For example, XLA has quite a few
    conditionals based on cuDNN version numbers that are not all exercised in
    CI. One could argue that this is less relevant as long as we test the
    configurations we care about—which is probably true. But unnecessarily
    broken code leads to bug reports that someone needs to address, even if it
    is just saying that a configuration is not supported.

    More importantly, potentially broken build configurations increase the
    developer's iteration dead time. A change might work fine when running
    `bazel test` locally, but CI might build a slightly different configuration
    and fail. Fixes are often benign—like adding a `[[maybe_unused]]`
    attribute—but this additional round trip costs the developer extra time on
    every pull request.

    **Therefore, decreasing the number of preprocessor conditionals decreases
    the number of build configurations, which in turn decreases the likelihood
    of additional CI round trips on a PR.**

*   **Tooling:** C++ parsing is such a complex task that most tools nowadays
    rely on a compiler frontend for parsing and semantic analysis and then
    operate directly on the AST. Notable examples include `clang-tidy`,
    `include-cleaner`, and language servers (`clangd`) that provide code
    completion, navigation, and syntax highlighting in your IDE. Another class
    of tools relies on compiler instrumentation, including sanitizers and code
    coverage tools.

    All of these tools only see a single build configuration, so extensive use
    of preprocessor conditionals hinders their usability. `include-cleaner`, for
    example, suggests removing `#include`s that are only used in an unevaluated
    preprocessor branch. Similarly, your IDE won't show syntax highlighting or
    code navigation for ROCm code when it is configured for a CUDA or CPU build,
    making it a pain to edit.

## Mitigations

### Category I - Skipping test cases in unit tests

It is very common that a certain test case is only supported:

*   On a certain backend.
*   With a certain GPU model.
*   When library X is at least of version Y.

Previously, it was common to skip tests using preprocessor conditionals:

```cpp
TEST(Foo, Bar) {
#ifdef TENSORFLOW_USE_ROCM
  GTEST_SKIP();
#endif
  // ...
}
```

#### Alternative: Ask StreamExecutor

`StreamExecutor` is XLA's hardware abstraction layer, and its
[`stream_executor::DeviceDescription`](https://github.com/openxla/xla/blob/main/xla/stream_executor/device_description.h)
has the information needed to make the same decision at runtime:

```cpp
TEST_F(FooTest, Bar) {
  // `device_description()` is provided by `HloPjRtGpuTestBase`. Other test
  // fixtures expose it via `executor->GetDeviceDescription()`.
  const se::DeviceDescription& device = device_description();

  // Skip based on the backend platform.
  if (device.gpu_compute_capability().IsRocm()) {
    GTEST_SKIP() << "Not supported on ROCm.";
  }

  // Skip based on the GPU model / compute capability.
  if (const auto* cc =
          device.gpu_compute_capability().cuda_compute_capability();
      cc != nullptr && !cc->IsAtLeastHopper()) {
    GTEST_SKIP() << "Requires Hopper or newer.";
  }

  // Skip based on the runtime or library version.
  if (device.runtime_version() < se::SemanticVersion{12, 2, 0}) {
    GTEST_SKIP() << "Requires CUDA runtime >= 12.2.";
  }
  if (device.dnn_version() < se::SemanticVersion{9, 0, 0}) {
    GTEST_SKIP() << "Requires cuDNN >= 9.0.";
  }
  // ...
}
```

`DeviceDescription` exposes both compute capabilities and structured version
numbers via
[`se::SemanticVersion`](https://github.com/openxla/xla/blob/main/xla/stream_executor/semantic_version.h):

*   **Backend & architecture:** `device.gpu_compute_capability().IsCuda()`,
    `device.gpu_compute_capability().IsRocm()`,
    `device.gpu_compute_capability().IsOneAPI()`, and accessors for
    `CudaComputeCapability`, `RocmComputeCapability`, and
    `OneAPIComputeCapability`.
*   **Runtime & driver versions:** `device.runtime_version()`,
    `device.driver_version()`, `device.kernel_mode_driver_version()`,
    `device.compile_time_toolkit_version()`.
*   **Library versions:** `device.dnn_version()`, `device.cub_version()`.

If a specific version or hardware property is not yet available in
`DeviceDescription`, please add it (or ask the contributor to add it) to
`DeviceDescription` instead of falling back to `#ifdef`.

#### Long term (author's opinion)

In the long term, none of our higher-level tests should need to make decisions
based on backends, hardware variants, or runtime/driver versions. Instead, tests
should ask `StreamExecutor` whether a certain feature is available, and
`StreamExecutor` determines that based on all the necessary details. All this
logic should live in one place, though this does not yet have a concrete design.

### Category II - Could be a runtime `if` statement

Another class of preprocessor conditionals guards code that would also compile
fine without the conditional:

```cpp
void Foo::Bar() {
#if TENSORFLOW_USE_ROCM
  // Do something that would also compile in CUDA/CPU mode
#endif
  // ...
}
```

#### Alternative: Use a runtime `if`

Replace the preprocessor conditional with a runtime conditional:

```cpp
void Foo::Bar() {
  if (stream_executor_.GetDeviceDescription()
          .gpu_compute_capability()
          .IsRocm()) {
    // Do something that compiles fine everywhere
  }
  // ...
}
```

Whether this should all be factored out into a separate function is up to the
reviewer's discretion. At this point it is just "normal" code and normal code
review practices apply.

**This class of preprocessor conditionals appears most often when checking a
condition that is not readily available in StreamExecutor's
`DeviceDescription`. Everyone is encouraged to ask contributors to add the
relevant pieces of information to `DeviceDescription` instead of accepting the
preprocessor conditional.**

### Category III - Requires access to the low-level runtime

The most common (and also the best justified) class of preprocessor conditionals
are those that guard code that wouldn't compile otherwise. Most often this is
because it uses something from a low-level backend-specific header (such as the
CUDA or ROCm API):

```cpp
#if TENSORFLOW_USE_ROCM
#include <something/something/rocm.h>
#endif

void Foo::Bar() {
#if TENSORFLOW_USE_ROCM
  // Do something that would *NOT* compile in CUDA/CPU mode
#endif
  // ...
}
```

The same applies to version guards on library headers (for example,
`#if CUDNN_VERSION >= 90000`) when the code references symbols that do not exist
in older versions of the header.

#### Alternative I: Let StreamExecutor deal with it

The first question to ask is whether the changed component should depend on
low-level runtime specifics at all. Ideally, only our hardware abstraction layer
`StreamExecutor` should. So if this code is not part of `StreamExecutor`, the
first option is to see if it can be pushed down into `xla/stream_executor/`.

That means creating an API (a function, a class, or whatever makes sense) in
`StreamExecutor` that exists for *all* backends. This API can have different
implementations based on the backend. It might also allow querying whether this
particular feature is available, which enables the runtime-`if` solution
described above.

Of course, that means `StreamExecutor` now has to deal with accessing the
low-level backend APIs. Alternative II describes how this can be modeled without
(too many) `#ifdef`s.

#### Alternative II: Code in separate targets

If the usage of backend-specific headers can't be pushed into `StreamExecutor`
or the change is inside `StreamExecutor`, the best option is to split the code
into separate compilation units. In our example, we have one build target with a
single C++ source file containing the ROCm code and another build target with a
single C++ source file containing the non-ROCm code—typically a stub
implementation that returns `absl::UnimplementedError` on all the defined
functions.

This has the following advantages:

*   Splitting code across multiple files means there needs to be some layer of
    abstraction in between—at the very least a free function. This encourages
    contributors and reviewers alike to think about the right level of
    abstraction, leading to higher-quality code in the long term.
*   Both files only contain a single build configuration—giving you full syntax
    highlighting and IDE tooling in both of them.
*   Both build targets can be built separately in the same build graph. No need
    to change your Bazel build flags and evict your build cache.
*   Both build targets can have their own tests. For example, if a certain API
    only exists for ROCm, we can have unit tests only testing the ROCm
    implementation instead of higher-level tests that get skipped in all other
    cases, which can speed up testing. (To be fair: in many cases there are good
    reasons for having the higher-level tests as well.)

A concrete design could look like this:

```cpp
// feature.h
// Do NOT include any platform-specific headers (CUDA/ROCm) here.
#include "absl/status/status.h"

absl::Status Foo();
```

```cpp
// feature_rocm.cc
#include "feature.h"
#include "something/something/rocm.h"

absl::Status Foo() {
  // Do something ROCm-specific.
}
```

```cpp
// feature_stub.cc
#include "feature.h"
#include "absl/status/status.h"

absl::Status Foo() {
  return absl::UnimplementedError("This is a ROCm-only feature.");
}
```

```python
# BUILD
load("@local_config_rocm//rocm:build_defs.bzl", "if_rocm_is_configured")

cc_library(
    name = "feature_rocm",
    srcs = [
        "feature.h",
        "feature_rocm.cc",
    ],
    tags = ["manual"],  # Exclude this from wildcard builds when ROCm is not enabled
    deps = [
        "@com_google_absl//absl/status",
        "@local_config_rocm//rocm:rocm_headers",
    ],
)

cc_library(
    name = "feature_stub",
    srcs = [
        "feature.h",
        "feature_stub.cc",
    ],
    deps = ["@com_google_absl//absl/status"],
)

cc_library(
    name = "feature",
    hdrs = ["feature.h"],
    deps = if_rocm_is_configured(
        [":feature_rocm"],
        [":feature_stub"],
    ) + [
        "@com_google_absl//absl/status",
    ],
)
```

Note that `feature.h` is only in the `hdrs` attribute of the `:feature` target
(and in `srcs` of the two implementation targets). This ensures that consumers
including `feature.h` (and automated dependency tools) depend on `:feature`
rather than directly on `:feature_rocm` or `:feature_stub`.

It is also important that `feature.h` does **not** include any platform-specific
headers (such as CUDA, ROCm, or cuDNN headers); only `.cc` files (and private
headers) should include those. The same build target pattern works for CUDA
(`if_cuda_is_configured` from `//xla/tsl/platform/default:cuda_build_defs.bzl`)
and SYCL.

#### Long term (author's opinion)

In the previous example, the targets `:feature_rocm` and `:feature_stub` define
the same symbol `Foo`, so they can't be linked into the same binary without
causing an ODR violation. Long term, XLA could have a plugin infrastructure
where multiple backends can be loaded at the same time. In that case, the design
above needs to be changed slightly so that `:feature_rocm` and `:feature_stub`
define symbols with different names and `:feature` has a dispatch function with
the original name—though it is currently unclear whether this will become a high
enough priority.
