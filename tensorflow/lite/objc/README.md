# TensorFlow Lite for Objective-C

[TensorFlow Lite](https://www.tensorflow.org/lite/) is TensorFlow's lightweight
solution for Objective-C developers. It enables low-latency inference of
on-device machine learning models with a small binary size and fast performance
supporting hardware acceleration.

## Build TensorFlow with iOS support

To build the Objective-C TensorFlow Lite library on Apple platforms,
[install from source](https://www.tensorflow.org/install/source#setup_for_linux_and_macos)
or [clone the GitHub repo](https://github.com/tensorflow/tensorflow).
Then, configure TensorFlow by navigating to the root directory and executing the
`configure.py` script:

```shell
python configure.py
```

Follow the prompts and when asked to build TensorFlow with iOS support, enter `y`.

### CocoaPods developers

Add the TensorFlow Lite pod to your `Podfile`:

```ruby
pod 'TensorFlowLiteObjC'
```

Then, run `pod install`.

In your Objective-C files, import the umbrella header:

```objectivec
#import "TFLTensorFlowLite.h"
```

Or, the module if you set `CLANG_ENABLE_MODULES = YES` in your Xcode project:

```objectivec
@import TFLTensorFlowLite;
```

Note: To import the TensorFlow Lite module in your Objective-C files, you must
also include `use_frameworks!` in your `Podfile`.

### Bazel developers

In your `BUILD` file, add the `TensorFlowLite` dependency to your target:

```python
objc_library(
    deps=[
        "//tensorflow/lite/objc:TensorFlowLite",
    ],)
```

In your Objective-C files, import the umbrella header:

```objectivec
#import "TFLTensorFlowLite.h"
```

Or, the module if you set `CLANG_ENABLE_MODULES = YES` in your Xcode project:

```objectivec
@import TFLTensorFlowLite;
```

Build the `TensorFlowLite` Objective-C library target:

```shell
bazel build tensorflow/lite/objc:TensorFlowLite
```

Build the `tests` target:

```shell
bazel test tensorflow/lite/objc:tests
```

#### Generate the Xcode project using Tulsi

Open the `//tensorflow/lite/objc/TensorFlowLite.tulsiproj` using
the [TulsiApp](https://github.com/bazelbuild/tulsi)
or by running the
[`generate_xcodeproj.sh`](https://github.com/bazelbuild/tulsi/blob/master/src/tools/generate_xcodeproj.sh)
script from the root `tensorflow` directory:

```shell
generate_xcodeproj.sh --genconfig tensorflow/lite/objc/TensorFlowLite.tulsiproj:TensorFlowLite --outputfolder ~/path/to/generated/TensorFlowLite.xcodeproj
```
