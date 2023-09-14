# Tensorflow Lite Core ML Delegate

TensorFlow Lite Core ML Delegate enables running TensorFlow Lite models on
[Core ML framework](https://developer.apple.com/documentation/coreml),
which results in faster model inference on iOS devices.

[TOC]

## Supported iOS versions and processors

* iOS 12 and later. In the older iOS versions, Core ML delegate will
  automatically fallback to CPU.
* When running on iPhone Xs and later, it will use Neural Engine for faster
  inference.

## Update code to use Core ML delegate

### Swift

Initialize TensorFlow Lite interpreter with Core ML delegate.

```swift
let coreMlDelegate = CoreMLDelegate()
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

// Any calls to AllocateTensors must happen strictly AFTER all
// ModifyGraphWithDelegate calls.

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
*   FullyConnected (aka Dense or InnerProduct)
    *   Weights and bias (if present) should be constant.
    *   Only supports single-batch case. Input dimensions should be 1, except
        the last dimension.
*   Hardswish
*   Logistic (aka Sigmoid)
*   MaxPool2D
*   MirrorPad
    *   Only 4D input with `REFLECT` mode is supported. Padding should be
        constant, and is only allowed for H and W dimensions.
*   Mul
    *   Only certain shapes are broadcastable. In Core ML tensor layout,
        following tensor shapes are broadcastable. `[B, C, H, W]`, `[B, C, 1,
        1]`, `[B, 1, H, W]`, `[B, 1, 1, 1]`.
*   Pad and PadV2
    *   Only 4D input is supported. Padding should be constant, and is only
        allowed for H and W dimensions.
*   Relu
*   ReluN1To1
*   Relu6
*   Reshape
    *   Only supported when target Core ML version is 2, not supported when
        targeting Core ML 3.
*   ResizeBilinear
*   SoftMax
*   Tanh
*   TransposeConv
    *   Weights should be constant.

## FAQ

* Does Core ML delegate support fallback to CPU if a graph contains unsupported
  ops?
  * Yes.
* Does Core ML delegate work on iOS Simulator?
  * Yes. The library includes x86 and x86_64 targets so it can run on
    a simulator, but you will not see performance boost over CPU.
* Does TensorFlow Lite and Core ML delegate support macOS?
  * TensorFlow Lite is only tested on iOS but not macOS.
* Are custom TF Lite ops supported?
  * No, CoreML delegate does not support custom ops and they will fallback to
    CPU.

## Appendix

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
  // Only create delegate when Neural Engine is available on the device.
  TfLiteCoreMlDelegateEnabledDevices enabled_devices;
  // Specifies target Core ML version for model conversion.
  // Core ML 3 come with a lot more ops, but some ops (e.g. reshape) is not
  // delegated due to input rank constraint.
  // if not set to one of the valid versions, the delegate will use highest
  // version possible in the platform.
  // Valid versions: (2, 3)
  int coreml_version;
  // This sets the maximum number of Core ML delegates created.
  // Each graph corresponds to one delegated node subset in the
  // TFLite model. Set this to 0 to delegate all possible partitions.
  int max_delegated_partitions;
  // This sets the minimum number of nodes per partition delegated with
  // Core ML delegate. Defaults to 2.
  int min_nodes_per_partition;
#ifdef TFLITE_DEBUG_DELEGATE
  // This sets the index of the first node that could be delegated.
  int first_delegate_node_index;
  // This sets the index of the last node that could be delegated.
  int last_delegate_node_index;
#endif
} TfLiteCoreMlDelegateOptions;

// Return a delegate that uses CoreML for ops execution.
// Must outlive the interpreter.
TfLiteDelegate* TfLiteCoreMlDelegateCreate(
   const TfLiteCoreMlDelegateOptions* options);

// Do any needed cleanup and delete 'delegate'.
void TfLiteCoreMlDelegateDelete(TfLiteDelegate* delegate);
```
