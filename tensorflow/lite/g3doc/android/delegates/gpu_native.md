# GPU acceleration delegate with C/C++ API

Using graphics processing units (GPUs) to run your machine learning (ML) models
can dramatically improve the performance and the user experience
of your ML-enabled applications. On Android devices, you can enable
GPU-accelerated execution of your models using a
[*delegate*](../../performance/delegates) and one of the following APIs:

- Interpreter API - [guide](./gpu)
- Task library API - [guide](./gpu_task)
- Native (C/C++) API - this guide

This guide covers advanced
uses of the GPU delegate for the C API, C++ API, and use of quantized models.
For more information about using the GPU delegate for TensorFlow Lite,
including best practices and advanced techniques, see the
[GPU delegates](../../performance/gpu) page.

## Enable GPU acceleration

Use the TensorFlow Lite GPU delegate for Android in C or C++ by creating the
delegate with `TfLiteGpuDelegateV2Create()` and destroying it with
`TfLiteGpuDelegateV2Delete()`, as shown in the following example code:

```c++
// Set up interpreter.
auto model = FlatBufferModel::BuildFromFile(model_path);
if (!model) return false;
ops::builtin::BuiltinOpResolver op_resolver;
std::unique_ptr<Interpreter> interpreter;
InterpreterBuilder(*model, op_resolver)(&interpreter);

// NEW: Prepare GPU delegate.
auto* delegate = TfLiteGpuDelegateV2Create(/*default options=*/nullptr);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;

// Run inference.
WriteToInputTensor(interpreter->typed_input_tensor<float>(0));
if (interpreter->Invoke() != kTfLiteOk) return false;
ReadFromOutputTensor(interpreter->typed_output_tensor<float>(0));

// NEW: Clean up.
TfLiteGpuDelegateV2Delete(delegate);
```

Review the `TfLiteGpuDelegateOptionsV2` object code to build a delegate instance
with custom options. You can initialize the default options with
`TfLiteGpuDelegateOptionsV2Default()` and then modify them as necessary.

The TensorFlow Lite GPU delegate for Android in C or C++ uses the
[Bazel](https://bazel.io) build system. You can build the delegate using the
following command:

```sh
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:delegate                           # for static library
bazel build -c opt --config android_arm64 tensorflow/lite/delegates/gpu:libtensorflowlite_gpu_delegate.so  # for dynamic library
```

When calling `Interpreter::ModifyGraphWithDelegate()` or
`Interpreter::Invoke()`, the caller must have an `EGLContext` in the current
thread and `Interpreter::Invoke()` must be called from the same `EGLContext`. If
an `EGLContext` does not exist, the delegate creates one internally, but then
you must ensure that `Interpreter::Invoke()` is always called from the same
thread in which `Interpreter::ModifyGraphWithDelegate()` was called.

## Quantized models {:#quantized-models}

Android GPU delegate libraries support quantized models by default. You do not
have to make any code changes to use quantized models with the GPU delegate. The
following section explains how to disable quantized support for testing or
experimental purposes.

#### Disable quantized model support

The following code shows how to ***disable*** support for quantized models.

<div>
  <devsite-selector>
    <section>
      <h3>C++</h3>
      <p><pre class="prettyprint lang-c++">
TfLiteGpuDelegateOptionsV2 options = TfLiteGpuDelegateOptionsV2Default();
options.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_NONE;

auto* delegate = TfLiteGpuDelegateV2Create(options);
if (interpreter->ModifyGraphWithDelegate(delegate) != kTfLiteOk) return false;
      </pre></p>
    </section>
  </devsite-selector>
</div>

For more information about running quantized models with GPU acceleration,
see [GPU delegate](../../performance/gpu#quantized-models) overview.