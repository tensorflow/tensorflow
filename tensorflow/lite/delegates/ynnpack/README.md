# YNNPACK Delegate for LiteRT

> [!WARNING]
> The YNNPACK delegate is **experimental** and under active development. Expect
> bugs and performance issues when using it.

The YNNPACK delegate allows LiteRT (formerly TensorFlow Lite) to offload
supported operators to YNNPACK.

YNNPACK aims to provide great flexibility with good performance.

## Delegate Provider Options

When using LiteRT tooling (e.g., benchmarks, evaluation tools) that link the
`ynnpack_delegate_provider`, the following command-line flags are exposed to
configure the YNNPACK delegate:

### Core Options

*   **`--use_ynnpack=true|false`** (default: `false`):
    Explicitly apply the YNNPACK delegate to the model.

*   **`--num_threads=N`** (default: `0` or `1` depending on tool):
    The number of threads to use for execution. Note that YNNPACK will only use
    a thread pool for `num_threads > 1`. A value of `0` or `1` disables the
    thread pool (single-threaded execution).

### YNNPACK Specific Options

*   **`--ynnpack_static_shape=true|false`** (default: `false`):
    Make input shapes static instead of dynamic. Enabling this may improve
    execution (`Invoke`) performance by allowing YNNPACK to optimize for fixed
    shapes, but it makes model reshaping (`ResizeInputTensor`) much more
    expensive.

*   **`--ynnpack_fast_math=true|false`** (default: `false`):
    Enable `YNN_FLAG_FAST_MATH`. This allows YNNPACK to use faster but
    potentially less precise mathematical approximations.

*   **`--ynnpack_consistent_arithmetic=true|false`** (default: `false`):
    Enable `YNN_FLAG_CONSISTENT_ARITHMETIC`. YNNPACK will attempt to provide
    numerically consistent results for all hardware the **same build** of
    YNNPACK runs on. It does not guarantee consistency across different builds
    (which means it does not guarantee consistency across different platforms,
    which are necessarily different builds).

*   **`--ynnpack_no_excess_precision=true|false`** (default: `false`):
    Enable `YNN_FLAG_NO_EXCESS_PRECISION`. YNNPACK will not promote tensors to
    higher precision as a performance optimization.
