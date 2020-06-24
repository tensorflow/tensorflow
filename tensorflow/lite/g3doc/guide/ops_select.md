# Select TensorFlow operators to use in TensorFlow Lite

Caution: This feature is experimental.

The TensorFlow Lite builtin op library has grown rapidly and will continue to
grow, but there remains a long tail of TensorFlow ops that are not yet natively
supported by TensorFlow Lite. These unsupported ops can be a point of friction
in the TensorFlow Lite model conversion process. To that end, the team has
recently been working on an experimental mechanism for reducing this friction.

This document outlines how to use TensorFlow Lite with select TensorFlow ops.
*Note that this feature is experimental and is under active development.* As you
use this feature, keep in mind the [known limitations](#known-limitations), and
please send feedback about models that work and issues you are facing to
tflite@tensorflow.org.

TensorFlow Lite will continue to have
[TensorFlow Lite builtin ops](ops_compatibility.md) optimized for mobile and
embedded devices. However, TensorFlow Lite models can now use a subset of
TensorFlow ops when TFLite builtin ops are not sufficient.

Models converted with TensorFlow ops will require a TensorFlow Lite interpreter
that has a larger binary size than the interpreter with only TFLite builtin ops.
Additionally, performance optimizations will not be available for any TensorFlow
ops in the TensorFlow Lite model.

This document outlines how to [convert](#converting-the-model) and
[run](#running-the-model) a TFLite model with TensorFlow ops on your platform of
choice. It also discusses some [known limitations](#known-limitations), the
[future plans](#future-plans) for this feature, and basic
[performance and size metrics](#metrics).

## Converting the model

To convert a TensorFlow model to a TensorFlow Lite model with TensorFlow ops,
use the `target_spec.supported_ops` argument in the
[TensorFlow Lite converter](../convert/). The following values are valid options
for `target_spec.supported_ops`:

*   `TFLITE_BUILTINS` - Converts models using TensorFlow Lite builtin ops.
*   `SELECT_TF_OPS` - Converts models using TensorFlow ops. The exact subset of
    supported ops can be found in the whitelist at
    `lite/delegates/flex/whitelisted_flex_ops.cc`.

Note: `target_spec.supported_ops` was previously `target_ops` in the Python API.

The recommended approach is to convert the model with `TFLITE_BUILTINS`, then
with both `TFLITE_BUILTINS,SELECT_TF_OPS`, and finally with only
`SELECT_TF_OPS`. Using both options (i.e. `TFLITE_BUILTINS,SELECT_TF_OPS`)
creates models with TensorFlow Lite ops where possible. Using only
`SELECT_TF_OPS` is useful when the model contains TensorFlow ops that are only
partially supported by TensorFlow Lite, and one would like to avoid those
limitations.

The following example shows how to use this feature in the
[`TFLiteConverter`](../convert/python_api.md) Python API.

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS,
                                       tf.lite.OpsSet.SELECT_TF_OPS]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

The following example shows how to use this feature in the
[`tflite_convert`](../convert/cmdline_examples.md) command line tool using the
command line flag `target_ops`.

```sh
tflite_convert \
  --output_file=/tmp/foo.tflite \
  --graph_def_file=/tmp/foo.pb \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1 \
  --target_ops=TFLITE_BUILTINS,SELECT_TF_OPS
```

When building and running `tflite_convert` directly with `bazel`, please pass
`--define=tflite_convert_with_select_tf_ops=true` as an additional argument.

```sh
bazel run --define=tflite_convert_with_select_tf_ops=true tflite_convert -- \
  --output_file=/tmp/foo.tflite \
  --graph_def_file=/tmp/foo.pb \
  --input_arrays=input \
  --output_arrays=MobilenetV1/Predictions/Reshape_1 \
  --target_ops=TFLITE_BUILTINS,SELECT_TF_OPS
```

## Running the model

When using a TensorFlow Lite model that has been converted with support for
select TensorFlow ops, the client must also use a TensorFlow Lite runtime that
includes the necessary library of TensorFlow ops.

### Android AAR

For Android, we recommend using the prebuilt [AAR with TensorFlow ops hosted at
JCenter](https://bintray.com/google/tensorflow/tensorflow-lite-select-tf-ops).

You can specify this in your `build.gradle` dependencies by adding it alongside
the standard TensorFlow Lite AAR as follows:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly'
    // This dependency adds the necessary TF op support.
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.0.0-nightly'
}
```

Once you've added the dependency, the necessary delegate for handling
the graph's TensorFlow ops should be automatically installed for
graphs that require them.

*Note*: The TensorFlow ops dependency is relatively large, so you'll probably
want to filter out unnecessary x86 ABIs in your `.gradle` file by setting up
your `abiFilters`.

```build
android {
    defaultConfig {
        ndk {
            abiFilters 'armeabi-v7a', 'arm64-v8a'
        }
    }
}
```

#### Building the Android AAR

For more advanced cases, you can also build the library manually. Assuming a
<a href="android.md">working TensorFlow Lite build environment</a>, build the
Android AAR with select TensorFlow ops as follows:

```sh
bazel build --cxxopt='--std=c++14' -c opt   \
  --config=android_arm --config=monolithic  \
  //tensorflow/lite/java:tensorflow-lite-select-tf-ops
```

This will generate an AAR file in `bazel-bin/tensorflow/lite/java/`. From there,
you can either import the AAR directly into your project, or publish the custom
AAR to your local Maven repository:

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tensorflow/lite/java/tensorflow-lite-select-tf-ops.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite-select-tf-ops -Dversion=0.1.100 -Dpackaging=aar
```

Finally, in your app's `build.gradle`, ensure you have the `mavenLocal()`
dependency and replace the standard TensorFlow Lite dependency with the one that
has support for select TensorFlow ops:

```build
allprojects {
    repositories {
        jcenter()
        mavenLocal()
    }
}

dependencies {
    implementation 'org.tensorflow:tensorflow-lite-with-select-tf-ops:0.1.100'
}
```

### iOS

#### Using CocoaPods

We provide nightly prebuilt select TF ops CocoaPods, which you can depend on
alongside the `TensorFlowLiteSwift` or `TensorFlowLiteObjC` CocoaPods.

```ruby
# In your Podfile target:
  pod 'TensorFlowLiteSwift'   # or 'TensorFlowLiteObjC'
  pod 'TensorFlowLiteSelectTfOps', '~> 0.0.1-nightly'
```

After running `pod install`, you need to provide an additional linker flag to
force load the select TF ops framework into your project. In your Xcode project,
go to `Build Settings` -> `Other Linker Flags`, and add:

```text
-force_load $(SRCROOT)/Pods/TensorFlowLiteSelectTfOps/Frameworks/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps
```

You should then be able to run any models converted with the `SELECT_TF_OPS` in
your iOS app. For example, you can modify the
[Image Classification iOS app](https://github.com/tensorflow/examples/tree/master/lite/examples/image_classification/ios)
to test the select TF ops feature.

*   Replace the model file with the one converted with `SELECT_TF_OPS` enabled.
*   Add `TensorFlowLiteSelectTfOps` dependency to the `Podfile` as instructed.
*   Add the additional linker flag as above.
*   Run the example app and see if the model works correctly.

#### Using Bazel + Xcode

TensorFlow Lite with select TensorFlow ops for iOS can be built using Bazel.
First, follow the [iOS build instructions](build_ios.md) to configure your Bazel
workspace and `.bazelrc` file correctly.

Once you have configured the workspace with iOS support enabled, you can use the
following command to build the select TF ops addon framework, which can be added
on top of the regular `TensorFlowLiteC.framework`. Note that the select TF ops
framework cannot be built for `i386` architecture, so you need to explicitly
provide the list of target architectures excluding `i386`.

```sh
bazel build -c opt --config=ios --ios_multi_cpus=armv7,arm64,x86_64 \
  //tensorflow/lite/experimental/ios:TensorFlowLiteSelectTfOps_framework
```

This will generate the framework under
`bazel-bin/tensorflow/lite/experimental/ios/` directory. You can add this new
framework to your Xcode project by following similar steps described under the
[Xcode project settings](./build_ios.md#modify_xcode_project_settings_directly)
section in the iOS build guide.

After adding the framework into your app project, an additional linker flag
should be specified in your app project to force load the select TF ops
framework. In your Xcode project, go to `Build Settings` -> `Other Linker
Flags`, and add:

```text
-force_load <path/to/your/TensorFlowLiteSelectTfOps.framework/TensorFlowLiteSelectTfOps>
```

### C++

When building TensorFlow Lite libraries using the bazel pipeline, the additional
TensorFlow ops library can be included and enabled as follows:

*   Enable monolithic builds if necessary by adding the `--config=monolithic`
    build flag.
*   Add the TensorFlow ops delegate library dependency to the build
    dependencies: `tensorflow/lite/delegates/flex:delegate`.

Note that the necessary `TfLiteDelegate` will be installed automatically when
creating the interpreter at runtime as long as the delegate is linked into the
client library. It is not necessary to explicitly install the delegate instance
as is typically required with other delegate types.

### Python pip package

Python support is actively under development.

## Metrics

### Performance

When using a mixture of both builtin and select TensorFlow ops, all of the same
TensorFlow Lite optimizations and optimized builtin kernels will be be available
and usable with the converted model.

The following table describes the average time taken to run inference on
MobileNet on a Pixel 2. The listed times are an average of 100 runs. These
targets were built for Android using the flags: `--config=android_arm64 -c opt`.

Build                                | Time (milliseconds)
------------------------------------ | -------------------
Only built-in ops (`TFLITE_BUILTIN`) | 260.7
Using only TF ops (`SELECT_TF_OPS`)  | 264.5

### Binary size

The following table describes the binary size of TensorFlow Lite for each build.
These targets were built for Android using `--config=android_arm -c opt`.

Build                 | C++ Binary Size | Android APK Size
--------------------- | --------------- | ----------------
Only built-in ops     | 796 KB          | 561 KB
Built-in ops + TF ops | 23.0 MB         | 8.0 MB

## Known limitations

The following is a list of some of the known limitations:

*   Control flow ops are not yet supported.
*   The
    [`post_training_quantization`](https://www.tensorflow.org/performance/post_training_quantization)
    flag is currently not supported for TensorFlow ops, so it will not quantize
    weights for any TensorFlow ops. In models with both TensorFlow Lite builtin
    ops and TensorFlow ops, the weights for the builtin ops will be quantized.
*   Ops that require explicit initialization from resources, like `HashTableV2`,
    are not yet supported.
*   Certain TensorFlow ops may not support the full set of input/output types
    that are typically available on stock TensorFlow.

## Future plans

The following is a list of improvements to this pipeline that are in progress:

*   *Selective registration* - There is work being done to make it simple to
    generate TFLite interpreter binaries that only contain the TensorFlow ops
    required for a particular set of models.
*   *Improved usability* - The conversion process will be simplified to only
    require a single pass through the converter.
*   *Improved performance* - Work is being done to ensure TensorFlow Lite with
    TensorFlow ops has performance parity to TensorFlow Mobile.
