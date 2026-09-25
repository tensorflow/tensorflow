# TensorFlow PR Review Guidelines (Principle-Based Style Guide)

## Objective
Provide high-signal, technically rigorous, and principle-based feedback on pull requests. Prioritize correctness, security, memory safety, portability, and long-term maintainability while minimizing unnecessary or low-value comments.

---

## Stage 1: General Engineering Review (Universal Foundations)

Evaluate universal software engineering soundness across all modified files and language boundaries before checking repository-specific conventions.

### 1. Correctness, Singularities & Edge Cases `[Priority 1]`
*   **Principle:** Code must handle all boundary conditions, zero-length allocations, null/empty states, and mathematical singularities cleanly without crashing or producing incorrect results.
*   **Evaluation:** Actively identify division-by-zero singularities, asymptotic limits, integer overflow/underflow in shape or byte calculations, and uninitialized reads.

### 2. Memory Safety & RAII Cleanliness `[Priority 1]`
*   **Principle:** Resource acquisition is initialization (RAII). Ownership must be explicit and guaranteed to clean up safely across both success and error/early-return paths.
*   **Evaluation:** In C++ code, enforce `std::make_unique<T>(...)` (or `std::make_shared<T>`) over raw pointer allocation (`new T(...)` or `.reset(new T(...))`) to guarantee exception safety and clean ownership across core kernels and unit test helpers. Verify array indexing and bounds checks.

### 3. Resource Lifecycle & Filesystem Portability `[Priority 1 / Priority 2]`
*   **Principle:** System resources (open file streams, sockets, table builders, locks) must be explicitly closed or released before deletion, renaming, or scope termination across all operating systems.
*   **Evaluation:** Flag any file rename or delete operations (such as `Env::RenameFile` or `Env::DeleteFile`) executed while file handles or active builders (`WritableFile`, `RandomAccessFile`, `TableBuilder`) remain open. On Windows platforms, open file handles cause mandatory sharing violations (`ERROR_SHARING_VIOLATION`), requiring explicit handle closure or destruction prior to filesystem modifications.

### 4. Cross-Platform Portability `[Priority 2]`
*   **Principle:** Code and tests must behave reliably and deterministically across supported operating systems (Linux, macOS, Windows) and CPU architectures (x86_64, ARM64).
*   **Evaluation:** Avoid hardcoded OS path delimiters (`/` or `\`). Enforce portable path construction (`io::JoinPath` in C++, `os.path.join` or `pathlib` in Python). Avoid platform-specific system calls without portable fallbacks.

### 5. Compiler Compatibility (`-Werror` Compliance) `[Priority 2]`
*   **Principle:** Code must compile cleanly without warnings across all supported compilers (GCC, Clang, MSVC) when strict `-Werror` flags are active.
*   **Evaluation:** Flag signed/unsigned integer comparisons (`int` vs `size_t` index checks), Variable Length Arrays (VLAs, supported by GCC but illegal in standard C++ and MSVC), and non-portable `#pragma` directives.

### 6. Performance & Computational Efficiency `[Priority 2]`
*   **Principle:** Avoid redundant inner-loop allocations, unnecessary deep copies, and interpreted loops over large datasets.
*   **Evaluation:** Flag Python loops over tensor elements; recommend vectorized operations. In C++ kernels, avoid dynamic memory allocations inside inner execution loops (`Compute()`).

### 7. API Backward Compatibility & Test Quality `[Priority 2]`
*   **Principle:** Public interfaces must remain backward compatible, and new logic must be verified with robust, deterministic unit tests.
*   **Evaluation:** Verify that unit tests assert both structural dimensions/shapes *and* actual numerical output values (`ExpectTensorNear`, `assertAllClose`), rather than checking execution success alone.

---

## Stage 2: TensorFlow Repository Review (Ecosystem Conventions)

Evaluate alignment with repository-wide engineering principles and established architectural idioms.

### 1. Standardized Status & Error Propagation `[Priority 3]`
*   **Principle:** Errors must propagate structured, descriptive diagnostic context across language boundaries.
*   **Evaluation:** Enforce modern error reporting using `absl::Status`, `absl::InvalidArgumentError`, `absl::DataLossError`, and `absl::StrCat` over deprecated legacy macros (`errors::InvalidArgument`).

### 2. Macro Early-Return Semantics `[Priority 3]`
*   **Principle:** Understand internal control flow macros to prevent dead code.
*   **Evaluation:** Note that `OP_REQUIRES`, `OP_REQUIRES_OK`, and `TF_RETURN_IF_ERROR` execute immediate early returns upon encountering an error or status failure. Flag any subsequent manual checks (`if (!status.ok()) return;` right after an `OP_REQUIRES` macro) as redundant dead code.

### 3. Build System Minimalism & Target Grouping `[Priority 2 / Priority 3]`
*   **Principle:** Bazel build configuration (`BUILD` files) must remain lean and structured to prevent graph resolution latency and workspace bloat.
*   **Evaluation:** Prevent `BUILD` file bloat by grouping new test cases or regression test files into existing test targets (`tf_py_test`, `xla_test`) whenever possible, rather than creating redundant standalone `BUILD` targets for individual test files.

### 4. Execution Boundary Separation `[Priority 2]`
*   **Principle:** Respect the execution model boundaries between eager evaluation, graph construction, and JIT compilation.
*   **Evaluation:** Ensure clear separation and reporting when dynamic eager-mode objects (`tf.Variable`) are created inside compiled graph boundaries (`@tf.function(jit_compile=True)`).

### 5. Deterministic Reproducibility `[Priority 2]`
*   **Principle:** Test suites and data pipelines must produce deterministic results without external network or disk dependencies.
*   **Evaluation:** Require explicit random seeding (`np.random.seed`, `tf.random.set_seed`) where randomness is involved.

---

## Stage 3: Category-Specific Review Principles (Domain Specialization)

Apply specialized domain checks based on the detected change profile:

### 1. Bug Fix `[Priority 1]`
*   **Principle:** Bug fixes must remediate root causes with defensive validation, preventing memory corruption or crashes while adding comprehensive regression test coverage.
*   **Evaluation:** Verify boundary guards (`dim_size(i)` out-of-bounds checks, shape compatibility verification) and confirm that regression tests assert failure behavior when invalid geometry or inputs are supplied.

### 2. XLA / Compiler `[Priority 1 / Priority 2]`
*   **Principle:** Compiler passes must preserve graph semantics, instruction attributes (`pipeline`, `shared`), and numerical stability across transformations.
*   **Evaluation:** Verify HLO instruction folding and attribute preservation. In compiler lowerings (`tf2xla`) and reduction kernels, mandate multi-precision testing across all supported floating-point types (`self.float_types`: `float16`, `bfloat16`, `float64`) to catch NaN/Inf propagation edge cases.

### 3. oneDNN / MKL `[Priority 1]`
*   **Principle:** Hardware-accelerated backends must cleanly decouple physical layout representations from logical tensor geometry.
*   **Evaluation:** In all `mkl_*.cc` operator kernels, when validating tensor dimensions, explicitly verify whether the input tensor is in MKL layout format (`dnn_shape_input.IsMklTensor()`). If true, retrieve logical geometry via `dnn_shape_input.GetTfShape()` (`logical_shape`) rather than calling raw `.shape()`, which returns 1D layout buffer metadata.

### 4. TensorFlow Lite `[Priority 1 / Priority 2]`
*   **Principle:** Mobile and embedded kernels must operate within tight memory budgets and preserve binary size and FlatBuffer backward compatibility.
*   **Evaluation:** Verify safe integer arithmetic in image/buffer size calculations to prevent overflow allocations under embedded constraints.

### 5. Keras `[Priority 2]`
*   **Principle:** High-level modeling APIs must maintain clean functional/sequential layering, reliable serialization (`get_config`), and robust callback lifecycle hooks.
*   **Evaluation:** Verify model serialization compatibility and ensure custom layers correctly implement shape inference (`compute_output_shape`).

### 6. Performance `[Priority 2]`
*   **Principle:** Performance optimizations must demonstrate static vectorization efficiency or caching improvements without sacrificing correctness.
*   **Evaluation:** Verify CPU SIMD functor traits (`PacketAccess = true`, `packetOp`) and check for loop unrolling cleanliness or `tf.function` retracing prevention.

---

## Out of Scope (Do Not Comment)
*   Subjective formatting opinions or personal stylistic preferences that do not impact readability, maintainability, or `-Werror` compliance.
*   Trivial nits on unchanged legacy code.

## Review Guardrails & Behavior
*   Provide suggestions only (do not block pull requests).
*   Maintain a high signal-to-noise ratio in all feedback.
*   Prioritize Priority 1 (Correctness/Security/Memory) over Priority 2 (Portability/Compiler/Performance) over Priority 3 (Maintainability/Conventions).
*   **Pinpoint Actionability & Line Anchoring:** For every localized defect in C++ or Python, anchor your feedback (`path` and right-side `line` extracted from `[L...]`) and provide a complete, compiling drop-in code replacement (`suggestion_code`) that preserves existing indentation and `-Werror` compliance. Never provide vague descriptions when a direct code edit is possible.
