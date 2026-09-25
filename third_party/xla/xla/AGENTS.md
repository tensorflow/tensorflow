# AI Assistant Guidelines for OpenXLA Development

This document provides guidelines for AI code assistants when generating,
suggesting, or modifying code within the
`third_party/tensorflow/compiler/xla` (OpenXLA) codebase.

## General Context

*   **Impact:** OpenXLA is a core compiler for machine learning acceleration. Changes here affect the open-source community and various hardware backends.
*   **Code Quality:** Adhere to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and OpenXLA-specific conventions.
*   **Portability:** This code is open-sourced and runs on a number of host platforms (e.g. Linux, Windows, etc.)

## Coding Guidelines for AI Assistance

1.  **Error Handling (`absl::Status`, `absl::StatusOr`)**:
    *   **Always** use `absl::Status` or `absl::StatusOr<T>` for functions that can encounter recoverable errors.
    *   **Macros**:
        *   Use header `tsl/platform/status_macros.h`.
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
        *   Located in `third_party/tensorflow/compiler/xla/service/decision.h`
        *   Example:
            ```cpp
            #include "third_party/tensorflow/compiler/xla/service/decision.h"

            using AutotunerDecision = Decision;

            AutotunerDecision ShouldAutotuneCublasCall(HloInstruction* instr) {
                // ...
                return AutotunerDecision::Forbid("Cublas autotuning was explicitly disabled");
            }

            voud AutotuneCublas(const AutotunerDecision& decision) {
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
    *   Use macros in `tsl/platform/status_matchers.h` instead of their TF_*
        counterparts. For example:
        *   Use `ASSERT_OK_AND_ASSIGN`, `ASSERT_OK`, and `EXPECT_OK`.
        *   DO NOT USE `TF_ASSERT_OK_AND_ASSIGN`, `TF_ASSERT_OK`, and
            `TF_EXPECT_OK`.
        *   When you refactor code that uses the TF_* macros., replace them, but
            do not touch unrelated code.
    *   When using gMock status matchers (`IsOk`, `IsOkAndHolds`, `StatusIs`),
        use the OSS-compatible matchers from `absl_testing` (`#include
        "third_party/absl/status/status_matchers.h"`):
        *   Prefer `::absl_testing::IsOk`, `::absl_testing::IsOkAndHolds`, and
            `::absl_testing::StatusIs`.
        *   DO NOT USE Google-internal `::testing::status::IsOk`,
            `::testing::status::IsOkAndHolds`, or `::testing::status::StatusIs`
            (which compile internally via `testing/base/public/gmock.h` but
            fail in OSS when rewritten to `<gmock/gmock.h>`).
    *   Put tests into an anonymous namespace
    *   Use `HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>` or
        `HloHardwareIndependentTestBase` for compiler pass tests locally where
        possible.
    *   When running GPU tests, pass `--config=cuda` to the test command.
    *   Ensure tests are deterministic and do not flake.

6.  **BUILD targets**:
    * When defining BUILD targets prefer these XLA specific rules:
        *   Instead of `proto_library` use `tf_proto_library`. There is no need
            to define language specific targets with `tf_proto_library`.
        *   Instead of `cc_test` use `xla_cc_test`.

7.  **Explicit Typing**:
    *   **Avoid `auto`** in public headers or complex logic chains.

8.  **Compiler Phases & Invariants**:
    *   **Phase Ordering**: Understand where your pass or change sits in the pipeline (e.g., Optimizations, Layout Assignment, Fusion).
    *   **Invariants**: Respect the invariants of the current phase.
        *   *Example*: Do not generate `kCustomCall` instructions before the relevant expansion pass if they are not supported by the HLO verifier at that stage.
        *   *Example*: Do not rely on layout information before Layout Assignment.

9.  **Namespaces**:
    *   Prefer xla::gpu over nested namespaces.

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
