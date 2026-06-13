# AI Assistant Guidelines for OpenXLA Development

This document provides guidelines for AI code assistants when generating,
suggesting, or modifying code within the
`third_party/tensorflow/compiler/xla` (OpenXLA) codebase.

> [!CAUTION]
> **CRITICAL BOUNDARY RULE**: Double-check which file path you are modifying!
> - **OSS (`third_party/tensorflow/compiler/xla/...`)**: Follow *this* document.
>   You must adhere to strict open-source portability. **NO** `LOG(DFATAL)`,
>   **NO** `DCHECK`, and no Google-internal libraries are permitted.
> - **Internal**: If modifying internal-only files, internal-specific
>   guidelines and overrides (such as permitting `LOG(DFATAL)` or depending on
>   internal libraries) may apply. Consult the internal guidelines directory if
>   working within the internal tree.

## General Context

*   **Impact:** OpenXLA is a core compiler for machine learning acceleration. Changes here affect the open-source community and various hardware backends.
*   **Code Quality:** Adhere to [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html) and OpenXLA-specific conventions.
*   **Portability:** This code is open-sourced and runs on a number of host platforms (e.g. Linux, Windows, etc.)

## Coding Guidelines for AI Assistance

1.  **Error Handling (`absl::Status`, `absl::StatusOr`)**:
    *   **Always** use `absl::Status` or `absl::StatusOr<T>` for functions that can encounter recoverable errors.
    *   **Macros**:
        *   Use header `tsl/platform/status_macros.h`.
        *   Use `RETURN_IF_ERROR` for error propagation.
        *   Use `ASSIGN_OR_RETURN` for `StatusOr` assignments.
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
    *   Put tests into an anonymous namespace
    *   Use `HloPjRtInterpreterReferenceMixin<HloPjRtTestBase>` or
        `HloHardwareIndependentTestBase` for compiler pass tests locally where
        possible.
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

10. **Tooling & Hygiene Check**:
    *   **Dependency Hygiene**: Run `build_cleaner` on any modified `BUILD`
        files to ensure all imports and dependencies conform to the strict
        OpenXLA target policy.
    *   **Formatting & Style**: Apply C++ formatting and style checks (e.g.,
        `hg lint` or clang-format) to modified files. Keep changes highly
        focused and minimize unrelated modifications.
    *   **Modernization & Standard Replacements**: Where applicable, replace
        verbose `std::` algorithms with Abseil container algorithms (e.g.,
        prefer `absl::c_any_of` over iterator-based `std::any_of`).
