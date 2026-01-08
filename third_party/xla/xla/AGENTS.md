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
        *   Use `TF_RETURN_IF_ERROR` (from `tsl/platform/errors.h` or `tsl/platform/status_macros.h`) for error propagation.
        *   Use `TF_ASSIGN_OR_RETURN` (if available) or `ASSIGN_OR_RETURN` (from `tsl/platform/status_macros.h`) for `StatusOr` assignments.
        *   Ensure the corresponding headers are included.
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

3.  **Performance Sensitivity**:
    *   OpenXLA is a compiler; patterns should be efficient.
    *   Avoid unnecessary string copies or expensive allocations in hot paths (e.g., HLO passes).

4.  **Testing**:
    *   Write unit tests using `EXPECT_EQ`, `EXPECT_TRUE`, etc.
    *   Use `HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>` or
        `HloHardwareIndependentTestBase` for compiler pass tests locally where
        possible.
    *   Ensure tests are deterministic and do not flake.

5.  **BUILD targets**:
    * When defining BUILD targets prefer these XLA specific rules:
        *   Instead of `proto_library` use `tf_proto_library`. There is no need
            to define language specific targets with `tf_proto_library`.
        *   Instead of `cc_test` use `xla_cc_test`.

6.  **Explicit Typing**:
    *   **Avoid `auto`** in public headers or complex logic chains.

7.  **Compiler Phases & Invariants**:
    *   **Phase Ordering**: Understand where your pass or change sits in the pipeline (e.g., Optimizations, Layout Assignment, Fusion).
    *   **Invariants**: Respect the invariants of the current phase.
        *   *Example*: Do not generate `kCustomCall` instructions before the relevant expansion pass if they are not supported by the HLO verifier at that stage.
        *   *Example*: Do not rely on layout information before Layout Assignment.
