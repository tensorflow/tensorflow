# Performance

Performance is an important consideration when training machine learning
models. Performance speeds up and scales research while
also providing end users with near instant predictions. This section provides
details on the high level APIs to use along with best practices to build
and train high performance models, and quantize models for the least latency
and highest throughput for inference.

  * @{$performance_guide$Performance Guide} contains a collection of best
    practices for optimizing your TensorFlow code.

  * @{$datasets_performance$Data input pipeline guide} describes the tf.data
    API for building efficient data input pipelines for TensorFlow.

  * @{$performance/benchmarks$Benchmarks} contains a collection of
    benchmark results for a variety of hardware configurations.

  * For improving inference efficiency on mobile and
    embedded hardware, see
    @{$quantization$How to Quantize Neural Networks with TensorFlow}, which
    explains how to use quantization to reduce model size, both in storage
    and at runtime.

  * For optimizing inference on GPUs, refer to [NVIDIA TensorRT™
  integration with TensorFlow.](
    https://medium.com/tensorflow/speed-up-tensorflow-inference-on-gpus-with-tensorrt-13b49f3db3fa)


XLA (Accelerated Linear Algebra) is an experimental compiler for linear
algebra that optimizes TensorFlow computations. The following guides explore
XLA:

  * @{$xla$XLA Overview}, which introduces XLA.
  * @{$broadcasting$Broadcasting Semantics}, which describes XLA's
    broadcasting semantics.
  * @{$developing_new_backend$Developing a new back end for XLA}, which
    explains how to re-target TensorFlow in order to optimize the performance
    of the computational graph for particular hardware.
  * @{$jit$Using JIT Compilation}, which describes the XLA JIT compiler that
    compiles and runs parts of TensorFlow graphs via XLA in order to optimize
    performance.
  * @{$operation_semantics$Operation Semantics}, which is a reference manual
    describing the semantics of operations in the `ComputationBuilder`
    interface.
  * @{$shapes$Shapes and Layout}, which details the `Shape` protocol buffer.
  * @{$tfcompile$Using AOT compilation}, which explains `tfcompile`, a
    standalone tool that compiles TensorFlow graphs into executable code in
    order to optimize performance.



