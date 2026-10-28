# AI Assistant Guidelines for OpenXLA Development

This document provides guidelines for AI code assistants when generating,
suggesting, or modifying code within the OpenXLA codebase (everything under
`xla/` in this repository).

## General Context

*   **Impact:** OpenXLA is a core compiler for machine learning acceleration. Changes here affect the open-source community and various hardware backends.
*   **Code Quality:** Adhere to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and OpenXLA-specific conventions.
*   **Portability:** This code is open-sourced and runs on a number of host platforms (e.g. Linux, Windows, etc.)

## Repository Layout and Build

*   Source lives under `xla/`. Include paths are relative to the repository
    root, e.g. `#include "xla/hlo/ir/hlo_module.h"`.
*   Bazel labels use the same prefix, e.g. `//xla/hlo/ir:hlo_module`.
*   Build and test with Bazel, e.g. `bazel test //xla/service:hlo_verifier_test`.
*   Common external dependencies and their labels:
    `@com_google_absl//absl/...`, `@llvm-project//mlir/...`,
    `@llvm-project//llvm/...`, `@com_google_googletest//:gtest_main`,
    `@com_google_protobuf//:protobuf`. TSL (the platform layer) lives in-tree
    under `//xla/tsl/...`.
*   Test macros: `xla_cc_test` from `//xla:xla.default.bzl` (hardware-
    independent unit tests) and `xla_test` from `//xla/tests:build_defs.bzl`
    (tests that run on configured hardware backends).

## Coding Guidelines for AI Assistance

1.  **Error Handling (`absl::Status`, `absl::StatusOr`)**:
    *   **Always** use `absl::Status` or `absl::StatusOr<T>` for functions that can encounter recoverable errors.
    *   **Macros**:
        *   Use header `absl/status/status_macros.h`.
        *   Use `ABSL_RETURN_IF_ERROR` for error propagation.
        *   Use `ABSL_ASSIGN_OR_RETURN` for `StatusOr` assignments.
    *   **Safely access `StatusOr<T>` values**: Check `.ok()` before accessing,
        or unwrap via `ABSL_ASSIGN_OR_RETURN` (never call `.value()` or
        `*status_or` unchecked).

2.  **Assertions & Invariant Checks (`TF_RET_CHECK`)**:
    *   **Avoid `DCHECK` / `LOG(DFATAL)`** for checking returnable errors.
    *   **Prefer `TF_RET_CHECK`**:
        *   Located in `xla/status_macros.h`.
        *   Use strict internal invariant checks inside functions returning `absl::Status` / `StatusOr`.
        *   Example:
            ```cpp
            #include "xla/status_macros.h"

            absl::Status Process(const Thing* t) {
              TF_RET_CHECK(t != nullptr) << "Thing cannot be null";
              // ...
              return absl::OkStatus();
            }
            ```

3.  **Decision making**:
    *   **Avoid `bool`** for returning a decision to do or not to do something.
    *   **Prefer `Decision`**:
        *   Located in `xla/service/decision.h`
        *   Example:
            ```cpp
            #include "xla/service/decision.h"

            using AutotunerDecision = Decision;

            AutotunerDecision ShouldAutotuneCublasCall(HloInstruction* instr) {
                // ...
                return AutotunerDecision::Forbid("Cublas autotuning was explicitly disabled");
            }

            void AutotuneCublas(const AutotunerDecision& decision) {
                if (decision.IsForbidden()) {
                    return;
                }
                ...
            ```

4.  **Performance Sensitivity, Containers, and Determinism**:
    *   OpenXLA is a compiler; patterns should be efficient.
    *   Avoid unnecessary string copies or expensive allocations in hot paths (e.g., HLO passes).
    *   Prefer Abseil container algorithms (`absl::c_find`, `absl::c_any_of`,
        `absl::c_all_of`, `absl::c_sort`, `absl::c_transform` from
        `absl/algorithm/container.h`) over iterator-pair `std::*` algorithms on
        full containers.
    *   Prefer `absl::flat_hash_map` and `absl::flat_hash_set` over
        `std::unordered_map` and `std::unordered_set`, and `std::vector` indexed
        by dense ID over `std::map` for dense integer keys.
    *   **Deterministic iteration**: Never iterate directly over pointer-keyed
        or unordered hash containers (`absl::flat_hash_map`,
        `absl::flat_hash_set`) when creating instructions, assigning IDs, or
        emitting code. Sort by a deterministic key (such as
        `HloInstruction::unique_id()` or name) or use `HloInstructionMap` so
        compilation output is reproducible across runs.
    *   Use `int64_t` when computing element counts, byte sizes, or strides to
        prevent 32-bit signed integer overflow.

5.  **Testing**:
    *   Write unit tests using `EXPECT_EQ`, `EXPECT_TRUE`, etc., and GoogleMock
        container matchers (`ElementsAre`, `UnorderedElementsAre`, `IsEmpty`,
        `FieldsAre`, `Optional`).
    *   Use the status testing macros `ASSERT_OK_AND_ASSIGN`, `ASSERT_OK`, and
        `EXPECT_OK` from `<gmock/gmock.h>` and the status matchers
        (`absl_testing::IsOk`, `absl_testing::IsOkAndHolds`,
        `absl_testing::StatusIs`) from `absl/status/status_matchers.h` instead
        of their legacy `TF_*` or `tsl::testing::*` counterparts:
        *   DO NOT USE `TF_ASSERT_OK_AND_ASSIGN`, `TF_ASSERT_OK`, and
            `TF_EXPECT_OK`.
        *   When you refactor code that uses the `TF_*` macros, replace them,
            but do not touch unrelated code.
        *   Never write weak boolean error checks like
            `EXPECT_FALSE(status.ok())`; verify both the error code and message
            substring with
            `absl_testing::StatusIs(absl::StatusCode::k..., HasSubstr("..."))`.
        *   When testing functions returning `xla::Decision`, assert on
            `decision.Explain()` whenever a case is forbidden.
    *   Put tests and `using ::testing::...;` declarations into an anonymous
        `namespace {` block.
    *   Use `HloInterpreterReferenceMixin<HloTestBase>` or
        `HloHardwareIndependentTestBase` for compiler pass tests locally where
        possible (do not use `HloTestBaseLegacy`), and configure flags by
        overriding `GetDebugOptionsForTest()`.
    *   Embed HLO modules directly in C++ unit tests using `R"hlo(...)hlo"` raw
        string literals rather than imperative `HloInstruction::Create*` builder
        chains or standalone `.hlo` files.
    *   Test helper functions must return `absl::Status` or `absl::StatusOr<T>`
        rather than calling `CHECK(...)` inside helpers.
    *   When running GPU tests, pass `--config=cuda` to the test command.
    *   Ensure tests are deterministic and do not flake (never use
        `absl::SleepFor` or unseeded RNGs).

6.  **BUILD targets**:
    *   When defining BUILD targets prefer these XLA specific rules:
        *   Instead of `proto_library` use `tf_proto_library`. There is no need
            to define language specific targets with `tf_proto_library`.
        *   Instead of `cc_test` use `xla_cc_test` (or `xla_test` when the test
            requires a real hardware backend).

7.  **Strict C++17 (`-std=c++17`), Control Flow, and Explicit Typing**:
    *   **C++17 only**: The codebase compiles with `-std=c++17` and
        `-Wc++17-compat-pedantic`. Do not use C++20/23 features:
        *   Use `absl::Span` (`absl/types/span.h`), never `std::span`.
        *   Do not call `.contains()` on `std::map`, `std::set`,
            `std::unordered_map`, or `std::unordered_set` (use
            `absl::flat_hash_map` / `absl::flat_hash_set` or
            `.find(k) != .end()`).
        *   Use `absl::StartsWith` / `absl::EndsWith` (`absl/strings/match.h`),
            never `std::string::starts_with` / `ends_with`.
        *   Use `absl::StrCat` / `absl::StrFormat`, never `std::format`.
        *   Use `absl::bit_cast` (`absl/base/casts.h`), never `std::bit_cast`.
    *   **Braces and early returns**: Always enclose `if`, `for`, and `while`
        bodies in `{}` braces, even for single-line statements (preventing macro
        expansion bugs with `ABSL_RETURN_IF_ERROR`), and prefer early returns or
        `continue` over deep nesting.
    *   **Avoid `auto`** in public headers or complex logic chains; spell out
        types explicitly when unwrapping `absl::StatusOr<T>` or where integer
        width matters (`int64_t` vs. `int32_t`).

8.  **Compiler Phases & Invariants**:
    *   **Phase Ordering**: Understand where your pass or change sits in the pipeline (e.g., Optimizations, Layout Assignment, Fusion).
    *   **Invariants**: Respect the invariants of the current phase.
        *   *Example*: Do not generate `kCustomCall` instructions before the relevant expansion pass if they are not supported by the HLO verifier at that stage.
        *   *Example*: Do not rely on layout information before Layout
            Assignment. Passes running *after* Layout Assignment or Memory Space
            Assignment must preserve physical layouts (`minor_to_major`) and
            memory spaces (`Layout::memory_space()`).
    *   **Metadata and `OriginalValue` preservation**: Return an accurate
        `bool changed` from `HloPassInterface::RunImpl`, and preserve
        `sharding()`, `metadata()`, `frontend_attributes()`, `original_value()`
        (`OriginalValue`), and control dependencies when replacing or
        decomposing HLO instructions.
    *   **Corner-case shapes and semantics**: Handle dynamic dimensions
        (`shape.is_dynamic()`), zero-element arrays
        (`ShapeUtil::IsZeroElementArray(shape)`), rank-0 scalars, tuple/token
        shapes, sub-byte element types (`s4`, `u4`, `s2`, `u2`), and
        `DynamicSlice` start-index clamping
        (`[0, max(0, operand_dim - slice_dim)]`).

9.  **Namespaces**:
    *   Prefer `xla::gpu` over nested namespaces.
    *   Never put namespace-level `using` declarations in `.h` files, and always
        qualify `std::` and `absl::` explicitly at call sites.

10. **MLIR and Triton Codegen**:
    *   **Always** use the static `OpTy::create(rewriter, ...)` method when creating MLIR operations.
    *   **Avoid** using `rewriter.create<OpTy>(...)`. This syntax is deprecated.
    *   Create top-level MLIR modules with
        `xla::llvm_ir::CreateMlirModuleOp(loc)`
        (`xla/service/llvm_ir/llvm_util.h`) rather than
        `mlir::ModuleOp::create(loc)`, and pass a cached
        `mlir::SymbolTableCollection` instead of constructing
        `mlir::SymbolTable(module_op)` inside per-operation rewrite patterns
        ($O(N^2)$).
    *   Validate Triton tile element counts (`<= 1,048,576`) and device shared
        memory limits from `se::DeviceDescription` before invoking the emitter.

11. **License Headers**:
    *   Any new source or build files created in this codebase must include the
        OpenXLA Apache 2.0 copyright header at the top of the file.
    *   Replace `<YEAR>` in the template below with the current 4-digit calendar
        year (e.g., `2026`—do not leave `<YEAR>` literally or copy an outdated
        year):
        ```cpp
        /* Copyright <YEAR> The OpenXLA Authors.

        Licensed under the Apache License, Version 2.0 (the "License");
        you may not use this file except in compliance with the License.
        You may obtain a copy of the License at

            http://www.apache.org/licenses/LICENSE-2.0

        Unless required by applicable law or agreed to in writing, software
        distributed under the License is distributed on an "AS IS" BASIS,
        WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
        See the License for the specific language governing permissions and
        limitations under the License.
        ==============================================================================*/
        ```
        (Use `#` comment prefixes for Python and `BUILD` files.)

12. **Portability and Compatibility Policies**:
    *   **Avoiding `#ifdef`s** (see `docs/avoiding_ifdefs.md`): Never use
        `#if` / `#ifdef` to skip unit tests (query `se::DeviceDescription` at
        runtime and call `GTEST_SKIP()`); use
        `if constexpr (tsl::kIsDebugBuild)` instead of `#ifndef NDEBUG`; and
        never conditionally `#include` CUDA, ROCm, or cuDNN headers in public
        `.h` files.
    *   **GPU AOT Compatibility** (see `docs/gpu_aot_compatibility.md`):
        Maintain 6-month backward and 2-week forward compatibility for
        `GpuExecutableProto`, `ThunkProto`, and transitive protos; use a 2-phase
        rollout (runtime support first, compiler emission after 2 weeks) for new
        thunks or custom-call/FFI targets; and register runtime-read
        `DebugOptions` field numbers in `kRuntimeFields` in
        `xla/debug_options_runtime_fields.cc`.
    *   **CUDA/cuDNN Dynamic Version Checks**: Guard any CUDA/cuDNN API, enum,
        or struct field introduced after major version `X.0` with an explicit
        runtime version check (`driver_version()`, `runtime_version()`, or
        library version) and register newly called C API symbols in
        `xla/tsl/cuda/*.symbols`.
    *   **CPU Runtime ABI**: Preserve bidirectional compatibility across
        `CompilationResultProto`, `ThunkProto`, and `XLA_CPU_KernelCallFrame`
        within the same `kHostExecutionAbiVersion`
        (`xla/backends/cpu/runtime/host_execution_abi.h`).

## Editing This File

This file is shared between the open-source repository and Google-internal
development. Keep it in open-source terms only: repository-relative paths
(`xla/...`), OSS macro names (`ABSL_RETURN_IF_ERROR`, `ABSL_ASSIGN_OR_RETURN`),
Bazel labels (`//xla/...`, `@repo//...`), and `bazel` commands.
