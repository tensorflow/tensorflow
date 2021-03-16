# Select TensorFlow operators

Since the TensorFlow Lite builtin operator library only supports a limited
number of TensorFlow operators, not every model is convertible. For details,
refer to [operator compatibility](ops_compatibility.md).

To allow conversion, users can enable the usage of
[certain TensorFlow ops](op_select_allowlist.md) in their TensorFlow Lite model.
However, running TensorFlow Lite models with TensorFlow ops requires pulling in
the core TensorFlow runtime, which increases the TensorFlow Lite interpreter
binary size. For Android, you can avoid this by selectively building only
required Tensorflow ops. For the details, refer to
[reduce binary size](../guide/reduce_binary_size.md).

This document outlines how to [convert](#convert_a_model) and
[run](#run_inference) a TensorFlow Lite model containing TensorFlow ops on a
platform of your choice. It also discusses
[performance and size metrics](#metrics) and
[known limitations](#known_limitations).

## Convert a model

The following example shows how to generate a TensorFlow Lite model with select
TensorFlow ops.

```python
import tensorflow as tf

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
converter.target_spec.supported_ops = [
  tf.lite.OpsSet.TFLITE_BUILTINS, # enable TensorFlow Lite ops.
  tf.lite.OpsSet.SELECT_TF_OPS # enable TensorFlow ops.
]
tflite_model = converter.convert()
open("converted_model.tflite", "wb").write(tflite_model)
```

## Run Inference

When using a TensorFlow Lite model that has been converted with support for
select TensorFlow ops, the client must also use a TensorFlow Lite runtime that
includes the necessary library of TensorFlow ops.

### Android AAR

To reduce the binary size, please build your own custom AAR files as guided in
the [next section](#building-the-android-aar). If the binary size is not a
considerable concern, we recommend using the prebuilt
[AAR with TensorFlow ops hosted at JCenter](https://bintray.com/google/tensorflow/tensorflow-lite-select-tf-ops).

You can specify this in your `build.gradle` dependencies by adding it alongside
the standard TensorFlow Lite AAR as follows:

```build
dependencies {
    implementation 'org.tensorflow:tensorflow-lite:0.0.0-nightly-SNAPSHOT'
    // This dependency adds the necessary TF op support.
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.0.0-nightly-SNAPSHOT'
}
```

Once you've added the dependency, the necessary delegate for handling the
graph's TensorFlow ops should be automatically installed for graphs that require
them.

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

For reducing the binary size or other advanced cases, you can also build the
library manually. Assuming a <a href="android.md">working TensorFlow Lite build
environment</a>, build the Android AAR with select TensorFlow ops as follows:

```sh
sh tensorflow/lite/tools/build_aar.sh \
  --input_models=/a/b/model_one.tflite,/c/d/model_two.tflite \
  --target_archs=x86,x86_64,arm64-v8a,armeabi-v7a
```

This will generate the AAR file `bazel-bin/tmp/tensorflow-lite.aar` for
TensorFlow Lite built-in and custom ops; and generate the AAR file
`bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar` for TensorFlow ops. If you
don't have a working build environment, You can also
[build above files with docker](../guide/reduce_binary_size.md#selectively_build_tensorflow_lite_with_docker).

From there, you can either import the AAR files directly into your project, or
publish the custom AAR files to your local Maven repository:

```sh
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite.aar \
  -DgroupId=org.tensorflow \
  -DartifactId=tensorflow-lite -Dversion=0.1.100 -Dpackaging=aar
mvn install:install-file \
  -Dfile=bazel-bin/tmp/tensorflow-lite-select-tf-ops.aar \
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
    implementation 'org.tensorflow:tensorflow-lite:0.1.100'
    implementation 'org.tensorflow:tensorflow-lite-select-tf-ops:0.1.100'
}
```

### iOS

#### Using CocoaPods

We provide nightly prebuilt select TF ops CocoaPods for `armv7` and `arm64`,
which you can depend on alongside the `TensorFlowLiteSwift` or
`TensorFlowLiteObjC` CocoaPods.

*Note*: If you need to use select TF ops in an `x86_64` simulator, you can build
the select ops framework yourself. See [Using Bazel + Xcode](#using_bazel_xcode)
section for more details.

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
  //tensorflow/lite/ios:TensorFlowLiteSelectTfOps_framework
```

This will generate the framework under `bazel-bin/tensorflow/lite/ios/`
directory. You can add this new framework to your Xcode project by following
similar steps described under the
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

### Python

TensorFlow Lite with select TensorFlow ops will be installed automatically with
the [TensorFlow pip package](https://www.tensorflow.org/install/pip). You can
also choose to only install the
[TensorFlow Lite Interpreter pip package](https://www.tensorflow.org/lite/guide/python#install_just_the_tensorflow_lite_interpreter).

Note: TensorFlow Lite with select TensorFlow ops are available in the TensorFlow
pip package version since 2.3 for Linux and 2.4 for other environments.

## Metrics

### Performance

When using a mixture of both builtin and select TensorFlow ops, all of the same
TensorFlow Lite optimizations and optimized builtin ops will be available and
usable with the converted model.

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

Build                     | C++ Binary Size | Android APK Size
------------------------- | --------------- | ----------------
Only built-in ops         | 796 KB          | 561 KB
Built-in ops + TF ops     | 23.0 MB         | 8.0 MB
Built-in ops + TF ops (1) | 4.1 MB          | 1.8 MB

(1) These libraries are selectively built for
[i3d-kinetics-400 model](https://tfhub.dev/deepmind/i3d-kinetics-400/1) with 8
TFLite builtin ops and 3 Tensorflow ops. For more details, please see the
[Reduce TensorFlow Lite binary size](../guide/reduce_binary_size.md) section.

## Known limitations

*   Unsupported types: Certain TensorFlow ops may not support the full set of
    input/output types that are typically available in TensorFlow.
*   Unsupported ops: Control flow ops and ops that require explicit
    initialization from resources, like `HashTableV2`, are not yet supported.

## Updates

*   Version 2.5 (not yet officially released)
    -   You can apply an optimization known as
        [post training quantization](../performance/post_training_quantization.md)
*   Version 2.4
    -   Compatibility with hardware accelerated delegates has improved
