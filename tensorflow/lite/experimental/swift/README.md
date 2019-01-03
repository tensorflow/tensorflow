# TensorFlow Lite for Swift

[TensorFlow Lite](https://www.tensorflow.org/lite/) is TensorFlow's lightweight
solution for Swift developers. It enables low-latency inference of on-device
machine learning models with a small binary size and fast performance supporting
hardware acceleration.

## Getting Started

### Bazel

In your `BUILD` file, add the `TensorFlowLite` dependency:

```python
swift_library(
  # ...
  deps = [
      "//tensorflow/lite/swift:TensorFlowLite",
  ],
)
```

In your Swift files, import the module:

```swift
import TensorFlowLite
```

### Tulsi

Open the `TensorFlowLite.tulsiproj` using the [TulsiApp](https://github.com/bazelbuild/tulsi) or by
running the [`generate_xcodeproj.sh`](https://github.com/bazelbuild/tulsi/blob/master/src/tools/generate_xcodeproj.sh)
script:

```shell
generate_xcodeproj.sh --genconfig tensorflow/lite/swift/TensorFlowLite.tulsiproj:TensorFlowLite --outputfolder ~/path/to/generated/TensorFlowLite.xcodeproj
```

### CocoaPods

Add the following to your `Podfile`:

```ruby
use_frameworks!
pod 'TensorFlowLiteSwift'
```

Then, run `pod install`.

In your Swift files, import the module:

```swift
import TensorFlowLite
```
