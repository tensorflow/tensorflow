# Tensorflow Lite Core ML Delegate

TensorFlow Lite Core ML Delegate enables running TensorFlow Lite models on
[Core ML framework](https://developer.apple.com/documentation/coreml),
which results in faster model inference on iOS devices.

Note: This delegate is in experimental (beta) phase.

Note: Core ML delegate is using Core ML version 2.1.

**Supported iOS versions and devices:**

*   iOS 12 and later. In the older iOS versions, Core ML delegate will
    automatically fallback to CPU.
*   When running on iPhone Xs and later, it will use Neural Engine for faster
    inference.

**Supported models**

The Core ML delegate currently supports float32 models.

## Trying the Core ML delegate on your own model

The Core ML delegate is already included in nightly release of TensorFlow lite
CocoaPods. To use Core ML delegate, change your TensorFlow lite pod
(`TensorflowLiteC` for C++ API, and `TensorFlowLiteSwift` for Swift) version to
`0.0.1-nightly` in your `Podfile`.

```
target 'YourProjectName'
  # pod 'TensorFlowLiteSwift'
  pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly'
```

Note: After updating `Podfile`, you should run `pod cache clean` and `pod
update` to reflect changes.

### Swift

Initialize TensorFlow Lite interpreter with the Core ML delegate.

```swift
let coreMlDelegate = CoreMLDelegate()
let interpreter = try Interpreter(modelPath: modelPath,
                                  delegates: [coreMlDelegate])
```

### Objective-C++

The Core ML delegate uses C++ API for Objective-C++ codes.

#### Step 1. Include `coreml_delegate.h`.

```objectivec++
#include "tensorflow/lite/experimental/delegates/coreml/coreml_delegate.h"
```

#### Step 2. Create a delegate and initialize a TensorFlow Lite Interpreter

After initializing the interpreter, call `interpreter->ModifyGraphWithDelegate`
with initialized Core ML delegate to apply the delegate.

```objectivec++
// initializer interpreter with model.
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Add following section to use the Core ML delegate.
TfLiteCoreMlDelegateOptions options = {};
delegate = TfLiteCoreMlDelegateCreate(&options);
interpreter->ModifyGraphWithDelegate(delegate);

// ...
```

#### Step 3. Dispose the delegate when it is no longer used.

Add this code to the section where you dispose of the delegate (e.g. `dealloc`
of class).

```objectivec++
TfLiteCoreMlDelegateDelete(delegate);
```

## Supported ops

Following ops are supported by the Core ML delegate.

*   Add
    *   Only certain shapes are broadcastable. In Core ML tensor layout,
        following tensor shapes are broadcastable. `[B, C, H, W]`, `[B, C, 1,
        1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
*   AveragePool2D
*   Concat
*   Conv2D
    *   Weights and bias should be constant.
*   DepthwiseConv2D
    *   Weights and bias should be constant.
*   FullyConnected (aka Dense or InnerProduct)
    *   Weights and bias (if present) should be constant.
    *   Only supports single-batch case. Input dimensions should be 1, except
        the last dimension.
*   Hardswish
*   Logistic (aka Sigmoid)
*   MaxPool2D
*   Mul
    *   Only certain shapes are broadcastable. In Core ML tensor layout,
        following tensor shapes are broadcastable. `[B, C, H, W]`, `[B, C, 1,
        1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
*   Relu
*   ReluN1To1
*   Relu6
*   Reshape
*   ResizeBilinear
*   SoftMax
*   Tanh

## Feedback

For issues, please create a
[GitHub](https://github.com/tensorflow/tensorflow/issues/new?template=50-other-issues.md)
issue with all the necessary details to reproduce.

## FAQ

* Does CoreML delegate support fallback to CPU if a graph contains unsupported
  ops?
  * Yes
* Does CoreML delegate work on iOS Simulator?
  * Yes. The library includes x86 and x86_64 targets so it can run on
    a simulator, but you will not see performance boost over CPU.
* Does TensorFlow Lite and CoreML delegate support MacOS?
  * TensorFlow Lite is only tested on iOS but not MacOS.
* Is custom TF Lite ops supported?
  * No, CoreML delegate does not support custom ops and they will fallback to
    CPU.

## APIs

### Core ML delegate Swift API

```swift
/// A delegate that uses the `Core ML` framework for performing
/// TensorFlow Lite graph operations.
///
/// - Important: This is an experimental interface that is subject to change.
public final class CoreMLDelegate: Delegate {
 /// The configuration options for the `CoreMLDelegate`.
 public let options: Options

 // Conformance to the `Delegate` protocol.
 public private(set) var cDelegate: CDelegate

 * /// Creates a new instance configured with the given `options`.
 ///
 /// - Parameters:
 ///   - options: Configurations for the delegate. The default is a new instance of
 ///       `CoreMLDelegate.Options` with the default configuration values.
 public init(options: Options = Options()) {
   self.options = options
   var delegateOptions = TfLiteCoreMlDelegateOptions()
   cDelegate = TfLiteCoreMlDelegateCreate(&delegateOptions)
 }

 deinit {
   TfLiteCoreMlDelegateDelete(cDelegate)
 }
}

extension CoreMLDelegate {
 /// Options for configuring the `CoreMLDelegate`.
 public struct Options: Equatable, Hashable {
   /// Creates a new instance with the default values.
   public init() {}
 }
}
```

### Core ML delegate C++ API

```c++
typedef struct {
 // We have dummy for now as we can't have empty struct in C.
 char dummy;
} TfLiteCoreMlDelegateOptions;

// Return a delegate that uses CoreML for ops execution.
// Must outlive the interpreter.
TfLiteDelegate* TfLiteCoreMlDelegateCreate(
   const TfLiteCoreMlDelegateOptions* options);

// Do any needed cleanup and delete 'delegate'.
void TfLiteCoreMlDelegateDelete(TfLiteDelegate* delegate);
```
