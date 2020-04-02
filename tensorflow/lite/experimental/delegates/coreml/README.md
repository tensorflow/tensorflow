# Tensorflow Lite Core ML Delegate

TensorFlow Lite Core ML Delegate enables running TensorFlow Lite models on
[Core ML framework](https://developer.apple.com/documentation/coreml),
which results in faster model inference on iOS devices.

[TOC]

## Supported iOS versions and processors

* iOS 11 and later. In the older iOS version, Core ML delegate will
  automatically fallback to CPU.
* When running on iPhone Xs and later, it will use Neural Engine for faster
  inference.

## Update code to use core ML delegate

### Swift

Initialize TensorFlow Lite interpreter with Core ML delegate.

```swift
let coreMLDelegate = CoreMLDelegate()
let interpreter = try Interpreter(modelPath: modelPath,
                                  delegates: [coreMLDelegate])
```

### Objective-C++

#### Interpreter initialization

Include `coreml_delegate.h`.

```objectivec++
#include "tensorflow/lite/experimental/delegates/coreml/coreml_delegate.h"
```

Modify code following interpreter initialization to apply delegate.

```objectivec++
// initializer interpreter with model.
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Add following section to use Core ML delegate.
TfLiteCoreMlDelegateOptions options = {};
delegate = TfLiteCoreMlDelegateCreate(&options);
interpreter->ModifyGraphWithDelegate(delegate);

// ...
```

#### Disposal

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

## Appendix

### Core ML delegate Swift API

```swift
/// A delegate that uses the `Core ML` framework for performing
/// TensorFlow Lite graph operations.
///
/// - Important: This is an experimental interface that is subject to change.
public final class CoreMlDelegate: Delegate {
 /// The configuration options for the `CoreMlDelegate`.
 public let options: Options

 // Conformance to the `Delegate` protocol.
 public private(set) var cDelegate: CDelegate

 * /// Creates a new instance configured with the given `options`.
 ///
 /// - Parameters:
 ///   - options: Configurations for the delegate. The default is a new instance of
 ///       `CoreMlDelegate.Options` with the default configuration values.
 public init(options: Options = Options()) {
   self.options = options
   var delegateOptions = TfLiteCoreMlDelegateOptions()
   cDelegate = TfLiteCoreMlDelegateCreate(&delegateOptions)
 }

 deinit {
   TfLiteCoreMlDelegateDelete(cDelegate)
 }
}

extension CoreMlDelegate {
 /// Options for configuring the `CoreMlDelegate`.
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
