<!-- mdformat off(b/169948621#comment2) -->
# TensorFlow Lite for Microcontrollers Port of 16x8 Quantized Operators

| Status        | Proposed                                                    |
:-------------- |:----------------------------------------------------------- |
| **RFC #2**    | [46767](https://github.com/tensorflow/tensorflow/pull/46767)|
| **Author(s)** | Daniel Situnayake (dan@edgeimpulse.com)                     |
| **Sponsor**   | Pete Warden (petewarden@google.com)                         |
| **Updated**   | 2021-01-28                                                  |

## Objective

TensorFlow Lite has kernel implementations that support 8 bit quantized weights
but use 16 bit activations. We wish to port these implementations to TensorFlow
Lite for Microcontrollers. The increased precision available for activations can
improve performance for some quantized models.

Arm have agreed to support the initiative by adding the necessary 16x8 APIs to
CMSIS-NN and porting the CMSIS-NN kernels.

### Goals
- Port a subset of 16x8 reference kernels from TensorFlow Lite to TensorFlow Lite Micro
- Avoid increasing default code size or arena size of TensorFlow Lite Micro
- Lay the groundwork for creating a CMSIS-NN port of the 16x8 kernels

### Non-goals
- Port every single operator to 16x8; we only plan to port a subset of those with existing reference implementations

## Motivation

Some networks that suffer unacceptable degradation when quantized with 8 bit weights
and 8 bit activations perform adequately when quantized with 8 bit weights and 16
bit activations. The [TensorFlow Lite documentation](https://www.tensorflow.org/lite/performance/post_training_integer_quant_16x8) states the following:

> [16x8 quantization] mode can improve accuracy of the quantized model significantly, when activations are sensitive to the quantization, while still achieving almost 3-4x reduction in model size. Moreover, this fully quantized model can be consumed by integer-only hardware accelerators.

Edge Impulse, a company that deploys TensorFlow Lite for Microcontrollers as part of its embedded
machine learning pipeline, has gathered feedback from customers with production models for which 8 bit
quantization results in unacceptable degradation but for whom 16x8 is fine.

While 16x8 quantization is well supported within TensorFlow Lite, it is not currently supported
within TensorFlow Lite for Microcontrollers. Porting the TensorFlow Lite reference kernels is
relatively straightforward and will improve adoption of TensorFlow Lite for Microcontrollers with users
for whom degradation is too severe with full 8 bit quantization.

## User Benefit

The headline would be "16x8 kernels improve accuracy for quantized models on microcontrollers without
increasing model size".

Users would benefit in the following ways:

- Improved accuracy for quantized models without increasing model size (in exchange for additional
  runtime memory usage)
- Improved performance under certain conditions (for example, 16x8 CMSIS-NN kernels will run faster)
  than 8 bit kernels since less unpacking is required)

## Design Proposal

We propose that the 16x8 kernels are ported from the TensorFlow Lite reference kernels to
TensorFlow Lite for Microcontrollers following the process in the [Porting TensorFlow Lite Ops to Micro](https://docs.google.com/document/d/1KLJTPWm4TUKB9YyIqFJl9VCP0ZMJDt_P8RNpRmwqMxw/edit#heading=h.5x0d5h95i329)
guide.

We wish to ensure that the following kernels are compatible with 16x8 mode:

- Conv2D
- MaxPool2D
- DepthwiseConv2D
- FullyConnected
- Relu
- Relu6
- Tanh
- Softmax
- Pad
- Reshape
- Pack
- Unpack
- Add
- Mul

Adding the 16x8 kernels directly to TFLM alongside the existing kernels would increase the default code size by an unacceptable amount. Instead, we will make use of the kernel registration API currently under development by the TFLM team. The use of this is demonstrated in the
[Keyword benchmark code](https://github.com/tensorflow/tensorflow/blob/a30d20b632b4ffbfd437ccf8ee205fef0917a3eb/tensorflow/lite/micro/benchmarks/keyword_benchmark.cc#L56).
By doing this, the end user can decide which kernels and dependencies they want to include (e.g. 8 bit, 16x8,
or float32).

For example, the following could be registered:

```
// Support for all datatypes
op_resolver->AddFullyConnected(tflite::Register_FULLY_CONNECTED);
// Support for 8 bit quantized models
op_resolver->AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT8);
// Support for 16x8 quantized models
op_resolver->AddFullyConnected(tflite::Register_FULLY_CONNECTED_INT16X8());
```

This means that kernels not currently using this registration API will need to be refactored to use it. Currently only **FullyConnected** uses the API.

The following associated tasks will be required to support this work:

- Build or port unit tests for the new kernels
- Prove that code memory is not impacted by running benchmarks before and after the port

### Alternatives Considered
* An alternative would be to add the 16x8 kernels without using the new kernel registration API, but this would
  result in a major increase in code size.

### Performance Implications
- Impact on memory usage for current modes (int8 and float32) will be minimal. This will be confirmed by
  benchmarking of current performance against performance of the submitted changes.
- When 16x8 mode is used, RAM usage will be approximately 2x. Latency may change depending on the target
  platform.
- End to end and unit tests will be updated to prove that the new implementations are operating correctly.

### Dependencies
- No additional dependencies will be added to TensorFlow
- No other parts of TensorFlow will be affected

### Engineering Impact
- Impact on binary size should be minimal
- Test times may increase due to additional kernel unit tests
- The reference kernels already exist within TensorFlow Lite so there will be minimal additional maintenance

### Platforms and Environments
- The proposed changes will work on all currently supported platforms

### Best Practices
- TensorFlow Lite for Microcontrollers should be updated to indicate that 16x8 kernels are now available

### Tutorials and Examples
- A benchmark will be added to [`tensorflow/lite/micro/benchmarks`](https://github.com/tensorflow/tensorflow/tree/975335bc83bf3cb80a71a04ed407725508709808/tensorflow/lite/micro/benchmarks) that demonstrates the use of the ops that provide a 16x8 kernel.
- A Colab will be created that demonstrates quantizing a model in 16x8 mode and exporting it as a C header file for use with TensorFlow Lite for Microcontrollers

### Compatibility
- This work will improve compatibility and feature parity between TensorFlow Lite and TensorFlow Lite for Microcontrollers

### User Impact
- Since TFLM does not have a versioning system the feature can be rolled out as any other commit

## Implementation plan

The work will be broken down into a series of pull requests, some for the benchmarks and some for each kernel.

Benchmark pull requests:
- PR1: Create a new benchmark in [`tensorflow/lite/micro/benchmarks`](https://github.com/tensorflow/tensorflow/tree/975335bc83bf3cb80a71a04ed407725508709808/tensorflow/lite/micro/benchmarks) that attempts to run a 16x8 model that includes the kernels mentioned in this RFC. The model’s weights and biases can be random. The benchmark should use the MicroMutableOpResolver. The PR should include the Colab used to generate the model.
- PR2: Port the person_detection and keyword benchmarks to use the MicroMutableOpResolver.
- PR3: Add code to both benchmarks that prints the arena size using the [`RecordingMemoryAllocator`](https://github.com/tensorflow/tensorflow/blob/ee87d58a6504375c28f21ea303f0eefa29118c38/tensorflow/lite/micro/docs/memory_management.md#recording-memory-apis).

For each kernel:
- PR1: Refactor the implementation to support the new kernel variant registration API.
- PR2: Add 16x8 support and make sure that the benchmark binary and arena sizes are unchanged.

Note that @njeffrie from the TF Lite Micro team also plans to prepare PR(s) for the kernels that are of interest internally
(without using the kernel variant registation API for binary size). This will provide some quick examples of porting the kernels.

## Questions and Discussion Topics
