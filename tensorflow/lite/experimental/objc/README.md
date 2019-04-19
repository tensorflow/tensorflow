# TensorFlow Lite for Objective-C

[TensorFlow Lite](https://www.tensorflow.org/lite/) is TensorFlow's lightweight
solution for Objective-C developers. It enables low-latency inference of
on-device machine learning models with a small binary size and fast performance
supporting hardware acceleration.

## Getting Started

To build the Objective-C TensorFlow Lite library on Apple platforms,
[install from source](https://www.tensorflow.org/install/source#setup_for_linux_and_macos)
or [clone the GitHub repo](https://github.com/tensorflow/tensorflow).
Then, configure TensorFlow by navigating to the root directory and executing the
`configure.py` script:

```shell
python configure.py
```

Follow the prompts and when asked to build TensorFlow with iOS support, enter `y`.

### Bazel

In your `BUILD` file, add the `TensorFlowLite` dependency:

```python
objc_library(
  deps = [
      "//tensorflow/lite/experimental/objc:TensorFlowLite",
  ],
)
```

In your Objective-C files, import the umbrella header:

```objectivec
#import "TFLTensorFlowLite.h"
```

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

### CocoaPods

Add the following to your `Podfile`:

```ruby
pod 'TensorFlowLiteObjC'
```

Then, run `pod install`.

In your Objective-C files, import the umbrella header:

```objectivec
#import "TFLTensorFlowLite.h"
```

Or, the module if `CLANG_ENABLE_MODULES = YES` and `use_frameworks!` is
specified in your `Podfile`:

```objectivec
@import TFLTensorFlowLite;
```
