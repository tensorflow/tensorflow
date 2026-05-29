# TensorFlow PR Review Guidelines (Gemini Code Assist)

## Objective

Provide high-signal, technically rigorous, and actionable feedback on pull
requests. Prioritize correctness, API stability, performance, security, and
long-term maintainability while minimizing unnecessary or low-value comments.

## Alignment with TensorFlow Contribution Guidelines

This style guide is aligned with TensorFlow’s official contribution guidelines:
https://github.com/tensorflow/tensorflow/blob/master/CONTRIBUTING.md

The following rules are derived from TensorFlow contribution requirements and
are used by Gemini Code Assist to guide PR review feedback.

These include requirements such as mandatory test coverage, adherence to coding
standards, API stability, and consistent behavior across supported environments.

## Core Review Mindset

### 1. Evaluate Necessity and Scope

*   Is this change essential?
*   Does it solve a real problem for TensorFlow users?
*   Is it aligned with TensorFlow’s scope and design goals?
*   Does it introduce unnecessary scope or complexity?
*   Does the benefit justify the long-term maintenance cost?
*   **Action:** If not, clearly question the need for the change.

### 2. Challenge the Implementation

*   Actively identify edge cases, failure scenarios, and incorrect assumptions.
*   Validate tensor shapes, dtypes, and execution paths.
*   Do not assume correctness based solely on passing tests.

### 3. Ensure Robustness

*   Flag fragile or environment-dependent logic.
*   Identify risks across different hardware targets (CPU, GPU, TPU).
*   Ensure proper error handling and defensive checks.

### 4. Detect Low-Quality or Non-Idiomatic Code

*   Identify overly generic, verbose, or context-insensitive code.
*   Flag patterns that do not align with established TensorFlow practices.
*   Highlight inconsistencies with existing repository patterns.

### 5. Respect Existing Design Patterns

*   Follow established TensorFlow APIs and architectural conventions.
*   Avoid suggesting unnecessary abstractions or structural changes.
*   Maintain consistency with similar modules.

## Review Priorities

### 1. Test Coverage and Reliability (Mandatory)

*   Verify that unit tests are included for any new logic, feature, or bug fix.
*   Flag PRs where source code is modified without adding or updating
    corresponding tests.
*   Ensure tests cover edge cases and expected behavior.
*   Tests must be small, fast, and deterministic.
*   Flag flaky tests or reliance on external systems (e.g., network, file
    system).
*   Ensure tests run reliably across supported platforms (Linux, macOS,
    Windows).

### 2. Code Quality and Formatting

*   Enforce standard Python and C++ formatting conventions.
*   Flag issues such as inconsistent indentation, poor variable naming, and
    missing docstrings.
*   Ensure code readability and consistency with repository standards.
*   Formatting issues should be flagged, but feedback should remain concise and
    avoid overwhelming developers with low-value comments.

### 3. API Stability (High Priority)

*   Flag breaking changes to public APIs (`tf.*`).
*   Ensure backward compatibility is preserved.
*   Verify adherence to the official deprecation lifecycle.
*   Maintain consistency in naming and argument structure.

### 4. Correctness and Numerical Behavior

*   Validate tensor operations, shapes, and broadcasting logic.
*   Identify potential numerical instability (e.g., overflow, underflow,
    division by zero).

### 5. Performance and Efficiency

*   Flag Python-side loops over tensors and suggest vectorized TensorFlow
    operations.
*   Ensure compatibility with `tf.function` and avoid retracing issues.
*   Identify unnecessary memory usage or redundant computations.
*   Flag hardcoded device placement (e.g., `/gpu:0`).

### 6. Security and Safe Execution

*   Flag unsafe data handling, deserialization, or file operations.
*   Highlight potential memory safety issues, especially in C++ kernels or
    Python–C++ boundaries.
*   Ensure code avoids patterns that could introduce security risks.

### 7. Dependencies and Scope

*   Flag the introduction of unnecessary or heavy external dependencies.
*   Ensure the change aligns with TensorFlow’s scope and does not increase
    maintenance burden unnecessarily.
*   Flag large pull requests that combine multiple unrelated changes (e.g., bug
    fix + refactor) and recommend splitting them into smaller, focused PRs for
    easier review and maintainability.

### 8. Readability and Maintainability

*   Flag issues only when they affect clarity or long-term maintainability.
*   Avoid suggesting purely subjective stylistic preferences that do not impact
    readability or consistency.

### 9. Idiomatic TensorFlow & Model Training

*   **Data Pipelines:** Flag raw in-memory tensor training when dataset size or
    scalability is a concern. \
    Prefer `tf.data.Dataset` with `.batch()`, `.prefetch(tf.data.AUTOTUNE)`, and
    optional `.cache()` where it fits in memory to improve performance.

*   **Batching:** Ensure training uses appropriate batching (`batch_size` or
    dataset batching). \
    Avoid feeding the entire dataset at once unless it is trivially small.

*   **Vectorization:** Avoid Python loops over tensors (training or inference).
    \
    Prefer batched/model-level operations (`model(x)` instead of per-sample
    calls).

*   **Reproducibility:** Require explicit seeds in data creation and model
    initialization:

    ```python
    np.random.seed(0)
    tf.random.set_seed(0)
    ```

*   **Observability:** Require validation data (`validation_split` or validation
    dataset) and meaningful metrics (e.g., `mae`, `accuracy`) to monitor
    training and detect overfitting.

*   **Callbacks:** Encourage use of `tf.keras.callbacks.EarlyStopping` and
    `tf.keras.callbacks.ModelCheckpoint` for stable and efficient training.

## Out of Scope (Do Not Comment)

*   Minor subjective stylistic preferences (e.g., personal formatting opinions)
    that do not impact readability, consistency, or correctness.
*   Trivial or non-impactful differences.

## Feedback Standards

*   Be clear, concise, and actionable.
*   Provide concrete suggestions where possible.
*   Avoid vague or non-specific feedback.
*   **Bad:** "This might be slow."
*   **Good:** "This loop introduces Python overhead; consider vectorized
    TensorFlow operations to improve performance and enable better graph
    optimization."

## Review Guardrails

*   Avoid speculative or uncertain feedback.
*   Do not comment if no meaningful issue is identified.
*   Avoid duplicate or redundant comments.
*   Prioritize high-impact issues over minor observations.

## Behavior

*   Provide suggestions only (do not block pull requests).
*   Focus on correctness, performance, security, and API stability.
*   Maintain a high signal-to-noise ratio in all feedback.
