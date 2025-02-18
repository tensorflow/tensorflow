# Performance best practices

Mobile and embedded devices have limited computational resources, so it is
important to keep your application resource efficient. We have compiled a list
of best practices and strategies that you can use to improve your TensorFlow
Lite model performance.

## Choose the best model for the task

Depending on the task, you will need to make a tradeoff between model complexity
and size. If your task requires high accuracy, then you may need a large and
complex model. For tasks that require less precision, it is better to use a
smaller model because they not only use less disk space and memory, but they are
also generally faster and more energy efficient. For example, graphs below show
accuracy and latency tradeoffs for some common image classification models.

![Graph of model size vs accuracy](../images/performance/model_size_vs_accuracy.png "Model Size vs Accuracy")

![Graph of accuracy vs latency](../images/performance/accuracy_vs_latency.png "Accuracy vs Latency")

One example of models optimized for mobile devices are
[MobileNets](https://arxiv.org/abs/1704.04861), which are optimized for mobile
vision applications.
[TensorFlow Hub](https://tfhub.dev/s?deployment-format=lite) lists several other
models that have been optimized specifically for mobile and embedded devices.

You can retrain the listed models on your own dataset by using transfer
learning. Check out the transfer learning tutorials using TensorFlow Lite
[Model Maker](../models/modify/model_maker/).

## Profile your model

Once you have selected a candidate model that is right for your task, it is a
good practice to profile and benchmark your model. TensorFlow Lite
[benchmarking tool](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/tools/benchmark)
has a built-in profiler that shows per operator profiling statistics. This can
help in understanding performance bottlenecks and which operators dominate the
computation time.

You can also use
[TensorFlow Lite tracing](measurement.md#trace-tensorflow-lite-internals-in-android)
to profile the model in your Android application, using standard Android system
tracing, and to visualize the operator invocations by time with GUI based
profiling tools.

## Profile and optimize operators in the graph

If a particular operator appears frequently in the model and, based on
profiling, you find that the operator consumes the most amount of time, you can
look into optimizing that operator. This scenario should be rare as TensorFlow
Lite has optimized versions for most operators. However, you may be able to
write a faster version of a custom op if you know the constraints in which the
operator is executed. Check out the
[custom operators guide](../guide/ops_custom.md).

## Optimize your model

Model optimization aims to create smaller models that are generally faster and
more energy efficient, so that they can be deployed on mobile devices.
TensorFlow Lite supports multiple optimization techniques, such as quantization.

Check out the
[model optimization docs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/model_optimization.md)
for details.

## Tweak the number of threads

TensorFlow Lite supports multi-threaded kernels for many operators. You can
increase the number of threads and speed up execution of operators. Increasing
the number of threads will, however, make your model use more resources and
power.

For some applications, latency may be more important than energy efficiency. You
can increase the number of threads by setting the number of interpreter
[threads](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/interpreter.h#L346).
Multi-threaded execution, however, comes at the cost of increased performance
variability depending on what else is executed concurrently. This is
particularly the case for mobile apps. For example, isolated tests may show 2x
speed-up vs single-threaded, but, if another app is executing at the same time,
it may result in worse performance than single-threaded.

## Eliminate redundant copies

If your application is not carefully designed, there can be redundant copies
when feeding the input to and reading the output from the model. Make sure to
eliminate redundant copies. If you are using higher level APIs, like Java, make
sure to carefully check the documentation for performance caveats. For example,
the Java API is a lot faster if `ByteBuffers` are used as
[inputs](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/java/src/main/java/org/tensorflow/lite/Interpreter.java#L175).

## Profile your application with platform specific tools

Platform specific tools like
[Android profiler](https://developer.android.com/studio/profile/android-profiler)
and [Instruments](https://help.apple.com/instruments/mac/current/) provide a
wealth of profiling information that can be used to debug your app. Sometimes
the performance bug may be not in the model but in parts of application code
that interact with the model. Make sure to familiarize yourself with platform
specific profiling tools and best practices for your platform.

## Evaluate whether your model benefits from using hardware accelerators available on the device

TensorFlow Lite has added new ways to accelerate models with faster hardware
like GPUs, DSPs, and neural accelerators. Typically, these accelerators are
exposed through
[delegate](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/delegates.md)
submodules that take over parts of the interpreter execution. TensorFlow Lite
can use delegates by:

*   Using Android's
    [Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks/).
    You can utilize these hardware accelerator backends to improve the speed and
    efficiency of your model. To enable the Neural Networks API, check out the
    [NNAPI delegate](https://www.tensorflow.org/lite/android/delegates/nnapi)
    guide.
*   GPU delegate is available on Android and iOS, using OpenGL/OpenCL and Metal,
    respectively. To try them out, see the
    [GPU delegate tutorial](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/gpu.md)
    and
    [documentation](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/gpu.md#advanced-gpu-support).
*   Hexagon delegate is available on Android. It leverages the Qualcomm Hexagon
    DSP if it is available on the device. See the
    [Hexagon delegate tutorial](https://www.tensorflow.org/lite/android/delegates/hexagon)
    for more information.
*   It is possible to create your own delegate if you have access to
    non-standard hardware. See
    [TensorFlow Lite delegates](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/delegates.md)
    for more information.

Be aware that some accelerators work better for different types of models. Some
delegates only support float models or models optimized in a specific way. It is
important to
[benchmark](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/g3doc/performance/measurement.md)
each delegate to see if it is a good choice for your application. For example,
if you have a very small model, it may not be worth delegating the model to
either the NN API or the GPU. Conversely, accelerators are a great choice for
large models that have high arithmetic intensity.

## Need more help

The TensorFlow team is happy to help diagnose and address specific performance
issues you may be facing. Please file an issue on
[GitHub](https://github.com/tensorflow/tensorflow/issues) with details of the
issue.
