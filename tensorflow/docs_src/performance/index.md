# Performance

Performance is often a significant issue when training a machine learning
model.  This section explains various ways to optimize performance.  Start
your investigation with the @{$performance_guide$Performance Guide} and then go
deeper with techniques detailed in @{$performance_models$High-Performance Models}:

  * @{$performance_guide$Performance Guide}, which contains a collection of best
    practices for optimizing your TensorFlow code.

  * @{$performance_models$High-Performance Models}, which contains a collection
    of advanced techniques to build highly scalable models targeting different
    system types and network topologies.

  * @{$benchmarks$Benchmarks}, which contains a collection of benchmark
    results.

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

And finally, we offer the following guide:

  * @{$quantization$How to Quantize Neural Networks with TensorFlow}, which
    can explains how to use quantization to reduce model size, both in storage
    and at runtime. Quantization can improve performance, especially on
    mobile hardware.

