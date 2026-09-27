# Determinism (GPU)

Non-determinism on GPU in XLA stems from **two distinct sources**:

1.  **Compilation-Time Non-Determinism (Autotuning)**
2.  **Execution-Time Non-Determinism (Non-Deterministic Ops)**

--------------------------------------------------------------------------------

## Source 1: Compilation-Time Non-Determinism (Autotuning)

### Cause

During compilation, XLA's autotuner profiles multiple candidate kernel
implementations (e.g., cuBLAS, cuDNN, Triton, native emitters) live on the
host's GPU to find the fastest algorithm for instructions like GEMMs and
convolutions.

Because live profiling relies on microsecond-level timing measurements, subtle
environmental hardware fluctuations (GPU thermal throttling, dynamic clock
frequency scaling, host OS scheduling) can cause **different kernel binaries to
be selected/generated as the winner across separate compilation runs**.

Because different kernel algorithms use distinct tile sizes, memory access
patterns, and reduction tree structures, they accumulate floating-point numbers
in different orders. Due to floating-point non-associativity, compiling the same
HLO graph twice can produce binaries that output slightly different numerical
calculation results.

### How to Achieve Compilation-time Determinism

*   **Persisting & Caching Autotuning Decisions**: Run autotuning once and save
    the choices to a persistent cache (see
    [persisted autotuning](./persisted_autotuning)). Using the flag
    **`--xla_gpu_require_complete_aot_autotune_results`** enforces that XLA
    strictly reuses cached decisions and immediately fails compilation with a
    `NotFound` error if an entry is missing, completely avoiding live hardware
    benchmarking.
*   **Disabling Autotuning**: Setting **`--xla_gpu_autotune_level=0`** disables
    autotuning benchmarks at the compile time altogether and selects default /
    first valid configurations deterministically.

--------------------------------------------------------------------------------

## Source 2: Execution-Time Non-Determinism (Non-Deterministic Ops)

### Cause

Even when the compiled binary is 100% identical across runs, running the *exact
same executable* multiple times can naturally yield different numerical outputs.

On GPUs, thousands of threads execute in parallel. Operations such as `scatter`,
accumulate values into memory buffers using atomic operations across
uncoordinated threads. Because thread scheduling order varies naturally from run
to run, floating-point additions occur in non-deterministic order. Because
floating-point addition is non-associative (`(a + b) + c != a + (b + c)`),
execution outputs vary between runs.

### How to Achieve Determinism

*   **`--xla_gpu_exclude_nondeterministic_ops`**: Enforces execution determinism
    by instructing XLA to:
    1.  Filter out non-deterministic algorithm/kernel/backend variants during
        config selection and enforce deterministic lowerings (for scatter, GEMM,
        convolution).
    2.  Rewrite operations like `select-and-scatter` into a sequence of
        deterministic ops to eliminate non-deterministic atomic scatter
        operations.
    3.  Disable live autotuning benchmarks (picking the first valid
        deterministic configuration).

--------------------------------------------------------------------------------

## Side Effects & Trade-Offs

Enforcing determinism on GPU involves notable trade-offs:

*   **Runtime Performance Degradation (Slowdowns)**:
    *   Deterministic implementations of operations like `scatter` (and expanded
        `select-and-scatter`) replace uncoordinated atomic updates with
        additional index sorting passes or deterministic reduction steps, which
        can lead to substantial runtime throughput loss compared to
        non-deterministic atomic fast paths.
    *   Bypassing live autotuning (`--xla_gpu_autotune_level=0`) prevents XLA
        from discovering the fastest kernel for the specific GPU architecture,
        resulting in sub-optimal kernel choices.
*   **Compilation Hard-Failures**:
    *   If an HLO graph contains instructions without any deterministic
        implementation, enabling `--xla_gpu_exclude_nondeterministic_ops` causes
        compilation to halt with an error.
    *   When using `--xla_gpu_require_complete_aot_autotune_results`,
        compilation will fail immediately with a `NotFound` error if any
        instruction lacks an entry in the pre-populated autotune cache.
*   **Disabled Optimizations & Emitters**:
    *   Certain high-performance kernel emitters and fusion transformations
        (such as Triton fusion autotuning) are automatically excluded from
        consideration when determinism is required.
