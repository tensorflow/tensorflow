# tfprof: TensorFlow Profiler and Beyond

<h1>Full Document in
<a href="https://github.com/tensorflow/tensorflow/blob/master/tensorflow/core/profiler/README.md">tensorflow/core/profiler/README.md</a><h1>

### Features

*   Profile model architectures
    *   parameters, tensor shapes, float operations, device placement, etc.
*   Profile model performance
    *   execution time, memory consumption
    *   Profile multiple steps.
*   Auto profile and advise.
    *   accelerator utilization check
    *   expensive operation check
    *   operation configuration check
    *   distributed runtime check (Not OSS)

### Interfaces

*   Python API
*   Command Line
*   Visualization
