# TensorFlow Lite for Swift

[TensorFlow Lite](https://www.tensorflow.org/lite/) is TensorFlow's lightweight
solution for Swift developers. It enables low-latency inference of on-device
machine learning models with a small binary size and fast performance supporting
hardware acceleration.

## Getting Started

To build the Swift TensorFlow Lite library on Apple platforms,
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
swift_library(
  deps = [
      "//tensorflow/lite/experimental/swift:TensorFlowLite",
  ],
)
```

In your Swift files, import the module:

```swift
import TensorFlowLite
```

Build the `TensorFlowLite` Swift library target:

```shell
bazel build tensorflow/lite/experimental/swift:TensorFlowLite
```

Build the `TensorFlowLiteTests` target:

```shell
bazel test tensorflow/lite/experimental/swift:TensorFlowLiteTests --swiftcopt=-enable-testing
```

Note that `--swiftcopt=-enable-testing` is required for optimized builds (`-c opt`).

### Tulsi

Open the `TensorFlowLite.tulsiproj` using the
[TulsiApp](https://github.com/bazelbuild/tulsi)
or by running the
[`generate_xcodeproj.sh`](https://github.com/bazelbuild/tulsi/blob/master/src/tools/generate_xcodeproj.sh)
script from the root `tensorflow` directory:

```shell
generate_xcodeproj.sh --genconfig tensorflow/lite/experimental/swift/TensorFlowLite.tulsiproj:TensorFlowLite --outputfolder ~/path/to/generated/TensorFlowLite.xcodeproj
```

### CocoaPods

Add the following to your `Podfile`:

```ruby
pod 'TensorFlowLiteSwift'
```

Then, run `pod install`.

In your Swift files, import the module:

```swift
import TensorFlowLite
```
