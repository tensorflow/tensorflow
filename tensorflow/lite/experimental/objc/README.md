# TensorFlow Lite for Objective-C

[TensorFlow Lite](https://www.tensorflow.org/lite/) is TensorFlow's lightweight
solution for Objective-C developers. It enables low-latency inference of
on-device machine learning models with a small binary size and fast performance
supporting hardware acceleration.

## Getting Started

### Bazel

In your `BUILD` file, add the `TensorFlowLite` dependency:

```python
objc_library(
  deps = [
      "//tensorflow/lite/experimental/objc:TensorFlowLite",
  ],
)
```

If you would like to build the Objective-C TensorFlow Lite library using Bazel on Apple
platforms, clone or download the [TensorFlow GitHub repo](https://github.com/tensorflow/tensorflow),
then navigate to the root `tensorflow` directory and execute the `configure.py` script:

```shell
python configure.py
```

Follow the prompts and when asked to configure the Bazel rules for Apple
platforms, enter `y`.

Build the `TensorFlowLite` Objective-C library target:

```shell
bazel build tensorflow/lite/experimental/objc:TensorFlowLite
```

Build the `TensorFlowLiteTests` target:

```shell
bazel test tensorflow/lite/experimental/objc:TensorFlowLiteTests
```

### Tulsi

Open the `TensorFlowLite.tulsiproj` using the
[TulsiApp](https://github.com/bazelbuild/tulsi) or by running the
[`generate_xcodeproj.sh`](https://github.com/bazelbuild/tulsi/blob/master/src/tools/generate_xcodeproj.sh)
script from the root `tensorflow` directory:

```shell
generate_xcodeproj.sh --genconfig tensorflow/lite/experimental/objc/TensorFlowLite.tulsiproj:TensorFlowLite --outputfolder ~/path/to/generated/TensorFlowLite.xcodeproj
```
