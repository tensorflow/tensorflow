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
    *   **Safely access `StatusOr<T>` values**: Check `.ok()` before accessing.

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

4.  **Performance Sensitivity**:
    *   OpenXLA is a compiler; patterns should be efficient.
    *   Avoid unnecessary string copies or expensive allocations in hot paths (e.g., HLO passes).

5.  **Testing**:
    *   Write unit tests using `EXPECT_EQ`, `EXPECT_TRUE`, etc.
    *   Use the status testing macros `ASSERT_OK_AND_ASSIGN`, `ASSERT_OK`, and
        `EXPECT_OK` from `<gmock/gmock.h>` and the status matchers
        (`absl_testing::IsOk`, `absl_testing::IsOkAndHolds`,
        `absl_testing::StatusIs`) from `absl/status/status_matchers.h` instead
        of their legacy `TF_*` or `tsl::testing::*` counterparts:
        *   DO NOT USE `TF_ASSERT_OK_AND_ASSIGN`, `TF_ASSERT_OK`, and
            `TF_EXPECT_OK`.
        *   When you refactor code that uses the `TF_*` macros, replace them,
            but do not touch unrelated code.
    *   Put tests into an anonymous namespace.
    *   Use `HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>` or
        `HloHardwareIndependentTestBase` for compiler pass tests locally where
        possible.
    *   When running GPU tests, pass `--config=cuda` to the test command.
    *   Ensure tests are deterministic and do not flake.

6.  **BUILD targets**:
    *   When defining BUILD targets prefer these XLA specific rules:
        *   Instead of `proto_library` use `tf_proto_library`. There is no need
            to define language specific targets with `tf_proto_library`.
        *   Instead of `cc_test` use `xla_cc_test` (or `xla_test` when the test
            requires a real hardware backend).

7.  **Explicit Typing**:
    *   **Avoid `auto`** in public headers or complex logic chains.

8.  **Compiler Phases & Invariants**:
    *   **Phase Ordering**: Understand where your pass or change sits in the pipeline (e.g., Optimizations, Layout Assignment, Fusion).
    *   **Invariants**: Respect the invariants of the current phase.
        *   *Example*: Do not generate `kCustomCall` instructions before the relevant expansion pass if they are not supported by the HLO verifier at that stage.
        *   *Example*: Do not rely on layout information before Layout Assignment.

9.  **Namespaces**:
    *   Prefer `xla::gpu` over nested namespaces.

10. **MLIR Operation Creation**:
    *   **Always** use the static `OpTy::create(rewriter, ...)` method when creating MLIR operations.
    *   **Avoid** using `rewriter.create<OpTy>(...)`. This syntax is deprecated.

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

## Editing This File

This file is shared between the open-source repository and Google-internal
development. Keep it in open-source terms only: repository-relative paths
(`xla/...`), OSS macro names (`ABSL_RETURN_IF_ERROR`, `ABSL_ASSIGN_OR_RETURN`),
Bazel labels (`//xla/...`, `@repo//...`), and `bazel` commands.
