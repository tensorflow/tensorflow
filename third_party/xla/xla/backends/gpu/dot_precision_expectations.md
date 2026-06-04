# Dot Precision Expectations in Tests

## Why do we change default precision?

TPUs are generally set up to use `bf16` precision for matrix multiplications
(dots) by default, even when operands are `f32`. This provides significantly
higher throughput at the cost of some precision loss.

To align GPU behavior with TPU and provide better performance in default
configurations, we are moving towards enabling `ALG_DOT_BF16_BF16_F32` by
default for `f32` dots on GPU (controlled by flags like
`xla_gpu_match_tpu_precision`).

This change means that tests expecting exact pure `f32` or `tf32` precision
results by default will fail. This document outlines how to handle such failures
and when to adjust expectations versus fixing test setups.

## Criteria for Adjusting Test Expectations vs. Fixing Tests

When changing default precision algorithms, many tests may fail. Use the
following criteria to determine whether to adjust the test expectations (relax
`ErrorSpec` or update `FileCheck`) or fix the test/code.

### 1. Do NOT Relax Expectations (Fix Setup or Code Instead)

*   **Explicit Precision/Algorithm is Set:** If the test HLO or code explicitly
    requests `PrecisionConfig::HIGHEST` or a specific algorithm (e.g.,
    `algorithm=dot_bf16_bf16_f32_x6`), it **should not** be affected by default
    flag flips.
    *   *If it fails:* It usually indicates a bug where explicit settings are
        overridden, or the **Reference Module** used for comparison is degraded
        because it lacks explicit settings.
*   **Reference Modules as Ground Truth:** Reference modules (e.g., used in
    correctness comparisons) should **explicitly set high precision** (e.g.,
    `algorithm=dot_f32_f32_f32`) to serve as a true ground truth baseline.
    Comparisons of other algorithms against this baseline are declarative,
    though they might require tuning of expected error bounds (`ErrorSpec`) due
    to comparing emulate algorithms against pure F32.

### 2. Explicit Algorithms for Specific Precision Verification (Declarative Tests)

*   **Verifying Specific F32/TF32 Paths:** If the core purpose of the test is to
    verify the numerical accuracy or invariants of traditional pure F32 or TF32
    paths, **explicitly set the expected algorithm** (e.g.,
    `algorithm=dot_tf32_tf32_f32` or `algorithm=dot_f32_f32_f32`) in the test
    HLO or code. Do not rely on default flag values or disable flags in the
    fixture to achieve this, as explicit settings are more robust and describe
    intent better.

### 3. Adjust Expectations for Default Precision Changes

*   **Default Precision Validation:** If the test is verifying standard
    operations running with **default precision**, and we change that default
    (e.g., to emulate TPU's reduced precision on GPU), the **error expectations
    should be adjusted (relaxed)** to adapt to the new default, rather than
    disabling the flag to keep it strict.
*   **Neural Network / Training Convergence:** Tests verifying model training or
    inference. Reduced precision is standard practice here, and GPU expectations
    should match TPU tolerances for default configurations.

### 4. Update FileCheck Expectations

*   **E2E Compiler Tests:** If the test is validating that the compiler pipeline
    functions correctly (e.g., MLIR/HLO IR dumps), expectations should be
    updated to look for the new nodes or types (e.g., `stablehlo.convert` to
    `bf16`) induced by the new default algorithm, rather than fighting the
    default by disabling the flag.
