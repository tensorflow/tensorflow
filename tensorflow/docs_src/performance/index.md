# Performance

Performance is an important consideration when training machine learning
models. Performance speeds up and scales research while
also providing end users with near instant predictions. This section provides
details on the high level APIs to use along with best practices to build
and train high performance models, and quantize models for the least latency
and highest throughput for inference.

  * [Performance Guide](../performance/performance_guide.md) contains a collection of best
    practices for optimizing your TensorFlow code.

  * [Data input pipeline guide](../performance/datasets_performance.md) describes the tf.data
    API for building efficient data input pipelines for TensorFlow.

  * [Benchmarks](../performance/benchmarks.md) contains a collection of
    benchmark results for a variety of hardware configurations.

  * For improving inference efficiency on mobile and
    embedded hardware, see
    [How to Quantize Neural Networks with TensorFlow](../performance/quantization.md), which
    explains how to use quantization to reduce model size, both in storage
    and at runtime.

  * For optimizing inference on GPUs, refer to [NVIDIA TensorRT™
  integration with TensorFlow.](
    https://medium.com/tensorflow/speed-up-tensorflow-inference-on-gpus-with-tensorrt-13b49f3db3fa)


XLA (Accelerated Linear Algebra) is an experimental compiler for linear
algebra that optimizes TensorFlow computations. The following guides explore
XLA:

  * [XLA Overview](../performance/xla/index.md), which introduces XLA.
  * [Broadcasting Semantics](../performance/xla/broadcasting.md), which describes XLA's
    broadcasting semantics.
  * [Developing a new back end for XLA](../performance/xla/developing_new_backend.md), which
    explains how to re-target TensorFlow in order to optimize the performance
    of the computational graph for particular hardware.
  * [Using JIT Compilation](../performance/xla/jit.md), which describes the XLA JIT compiler that
    compiles and runs parts of TensorFlow graphs via XLA in order to optimize
    performance.
  * [Operation Semantics](../performance/xla/operation_semantics.md), which is a reference manual
    describing the semantics of operations in the `ComputationBuilder`
    interface.
  * [Shapes and Layout](../performance/xla/shapes.md), which details the `Shape` protocol buffer.
  * [Using AOT compilation](../performance/xla/tfcompile.md), which explains `tfcompile`, a
    standalone tool that compiles TensorFlow graphs into executable code in
    order to optimize performance.



