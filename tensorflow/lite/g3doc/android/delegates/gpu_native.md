# GPU acceleration delegate with C/C++ API

Using graphics processing units (GPUs) to run your machine learning (ML) models
can dramatically improve the performance and the user experience of your
ML-enabled applications. On Android devices, you can enable GPU-accelerated
execution of your models using a
[*delegate*](https://ai.google.dev/edge/litert/performance/delegates) and one of
the following APIs:

-   Interpreter API - [guide](./gpu)
-   Task library API - [guide](./gpu_task.md)
-   Native (C/C++) API - this guide

This guide covers advanced uses of the GPU delegate for the C API, C++ API, and
use of quantized models. For more information about using the GPU delegate for
TensorFlow Lite, including best practices and advanced techniques, see the
[GPU delegates](https://ai.google.dev/edge/litert/performance/gpu) page.

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

#### With TensorFlow Lite in Google Play Services:

If you are using TensorFlow Lite in Google Play Services
[C API](https://ai.google.dev/edge/litert/android/native), you’ll need to use
the Java/Kotlin API to check if a GPU delegate is available for your device
before initializing the TensorFlow Lite runtime.

Add the GPU delegate gradle dependencies to your application:

```
implementation 'com.google.android.gms:play-services-tflite-gpu:16.2.0'
```

Then, check the GPU availability and initialize TfLiteNative if the check is
successful:

<div>
  <devsite-selector>
    <section>
      <h3>Java</h3>
      <pre class="prettyprint">
Task<Void> tfLiteHandleTask =
TfLiteGpu.isGpuDelegateAvailable(this)
   .onSuccessTask(gpuAvailable -> {
      TfLiteInitializationOptions options =
        TfLiteInitializationOptions.builder()
          .setEnableGpuDelegateSupport(gpuAvailable).build();
        return TfLiteNative.initialize(this, options);
      }
    );
      </pre>
      </section>
      <section>
      <h3>Kotlin</h3>
        <pre class="prettyprint">
val tfLiteHandleTask = TfLiteGpu.isGpuDelegateAvailable(this)
    .onSuccessTask { gpuAvailable ->
        val options = TfLiteInitializationOptions.Builder()
            .setEnableGpuDelegateSupport(gpuAvailable)
            .build()
        TfLiteNative.initialize(this, options)
    }
        </pre>
      </section>
  </devsite-selector>
</div>

You also need to update your CMake configuration to include the
`TFLITE_USE_OPAQUE_DELEGATE` compiler flag:

```
add_compile_definitions(TFLITE_USE_OPAQUE_DELEGATE)
```

The [FlatBuffers](https://flatbuffers.dev/) library is used to configure
delegate plugins, so you need to add it to the dependencies of your native code.
You can use the official `CMake` project configuration as follow:

```
target_include_directories(tflite-jni PUBLIC
        third_party/headers # flatbuffers
     ...)
```

You can also just bundle the headers to your app.

Finally to use GPU inference in your C code, create the GPU delegate using
`TFLiteSettings`:

```
#include "flatbuffers/flatbuffers.h"
#include "tensorflow/lite/acceleration/configuration/configuration_generated.h"

flatbuffers::FlatBufferBuilder fbb;
tflite::TFLiteSettingsBuilder builder(fbb);
const tflite::TFLiteSettings* tflite_settings =
    flatbuffers::GetTemporaryPointer(fbb, builder.Finish());

const TfLiteOpaqueDelegatePlugin* pluginCApi = TfLiteGpuDelegatePluginCApi();
TfLiteOpaqueDelegate* gpu_delegate = pluginCApi->create(tflite_settings);
```

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

For more information about running quantized models with GPU acceleration, see
[GPU delegate](https://ai.google.dev/edge/litert/performance/gpu#quantized_models)
overview.
