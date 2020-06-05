# Tensorflow Lite Core ML delegate

The TensorFlow Lite Core ML delegate enables running TensorFlow Lite models on
[Core ML framework](https://developer.apple.com/documentation/coreml), which
results in faster model inference on iOS devices.

Note: This delegate is in experimental (beta) phase.

Note: Core ML delegate supports Core ML version 2 and later.

**Supported iOS versions and devices:**

*   iOS 12 and later. In the older iOS versions, Core ML delegate will
    automatically fallback to CPU.
*   By default, Core ML delegate will only be enabled on devices with A12 SoC
    and later (iPhone Xs and later) to use Neural Engine for faster inference.
    If you want to use Core ML delegate also on the older devices, please see
    [best practices](#best-practices)

**Supported models**

The Core ML delegate currently supports float32 models.

## Trying the Core ML delegate on your own model

The Core ML delegate is already included in nightly release of TensorFlow lite
CocoaPods. To use Core ML delegate, change your TensorFlow lite pod
(`TensorflowLiteC` for C API, and `TensorFlowLiteSwift` for Swift) version to
`0.0.1-nightly` in your `Podfile`, and include subspec `CoreML`

```
target 'YourProjectName'
  # pod 'TensorFlowLiteSwift'
  pod 'TensorFlowLiteSwift/CoreML', '~> 0.0.1-nightly'
```

OR

```
target 'YourProjectName'
  # pod 'TensorFlowLiteSwift'
  pod 'TensorFlowLiteSwift', '~> 0.0.1-nightly', :subspecs => ['CoreML']
```

Note: After updating `Podfile`, you should run `pod update` to reflect changes.
If you can't see the latest `CoreMLDelegate.swift` file, try running `pod cache
clean TensorFlowLiteSwift`.

### Swift

Initialize TensorFlow Lite interpreter with the Core ML delegate.

```swift
let coreMLDelegate = CoreMLDelegate()
var interpreter: Interpreter

// Core ML delegate will only be created for devices with Neural Engine
if coreMLDelegate != nil {
  interpreter = try Interpreter(modelPath: modelPath,
                                delegates: [coreMLDelegate!])
} else {
  interpreter = try Interpreter(modelPath: modelPath)
}
```

### Objective-C

The Core ML delegate uses C API for Objective-C codes.

#### Step 1. Include `coreml_delegate.h`.

```objectivec
#include "tensorflow/lite/experimental/delegates/coreml/coreml_delegate.h"
```

#### Step 2. Create a delegate and initialize a TensorFlow Lite Interpreter

After initializing the interpreter, call `interpreter->ModifyGraphWithDelegate`
with initialized Core ML delegate to apply the delegate.

```objectivec
// initializer interpreter with model.
tflite::InterpreterBuilder(*model, resolver)(&interpreter);

// Add following section to use the Core ML delegate.
TfLiteCoreMlDelegateOptions options = {};
delegate = TfLiteCoreMlDelegateCreate(&options);
if (delegate != nullptr) {
  interpreter->ModifyGraphWithDelegate(delegate);
}
// ...
```

#### Step 3. Dispose the delegate when it is no longer used.

Add this code to the section where you dispose of the delegate (e.g. `dealloc`
of class).

```objectivec
TfLiteCoreMlDelegateDelete(delegate);
```

## Best practices

### Using Core ML delegate on devices without Neural Engine

By default, Core ML delegate will only be created if the device has Neural
Engine, and will return `null` if the delegate is not created. If you want to
run Core ML delegate on other environments (for example, simulator), pass `.all`
as an option while creating delegate in Swift. On C++ (and Objective-C), you can
pass `TfLiteCoreMlDelegateAllDevices`. Following example shows how to do this:

#### Swift

```swift
var options = CoreMLDelegate.Options()
options.enabledDevices = .all
let coreMLDelegate = CoreMLDelegate(options: options)!
let interpreter: try Interpreter(modelPath: modelPath,
                                delegates: [coreMLDelegate])
```

#### Objective-C

```objectivec
TfLiteCoreMlDelegateOptions options;
options.enabled_devices = TfLiteCoreMlDelegateAllDevices;
delegate = TfLiteCoreMlDelegateCreate(&options);
interpreter->ModifyGraphWithDelegate(delegate);
```

### Using Metal(GPU) delegate as a fallback.

When the Core ML delegate is not created, alternatively you can still use
[Metal delegate](https://www.tensorflow.org/lite/performance/gpu#ios) to get
performance benefits. Following example shows how to do this:

#### Swift

```swift
var delegate = CoreMLDelegate()
if delegate == nil {
  delegate = MetalDelegate()  // Add Metal delegate options if necessary.
}

let interpreter = try Interpreter(modelPath: modelPath,
                                  delegates: [delegate!])
```

#### Objective-C

```objectivec
TfLiteCoreMlDelegateOptions options = {};
delegate = TfLiteCoreMlDelegateCreate(&options);
if (delegate == nullptr) {
  // Add Metal delegate options if necessary
  delegate = TFLGpuDelegateCreate(nullptr);
}
interpreter->ModifyGraphWithDelegate(delegate);
```

The delegate creation logic reads device's machine id (e.g. iPhone11,1)
to determine its Neural Engine availability. See the
[code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/coreml/coreml_delegate.mm)
for more detail. Alternatively, you can implement your own set of blacklist
devices using other libraries such as
[DeviceKit](https://github.com/devicekit/DeviceKit).

### Using older Core ML version

Although iOS 13 supports Core ML 3, the model might work better when it is
converted with Core ML 2 model specification. The target conversion version is
set to the latest version by default, but you can change this by setting
`coreMLVersion` (in Swift, `coreml_version` in C API) in the delegate option to
older version.

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
    *   Only supported when target Core ML version is 2, not supported when
        targeting Core ML 3.
*   ResizeBilinear
*   SoftMax
*   Tanh
*   TransposeConv
    *   Weights should be constant.

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

*   [Core ML delegate Swift API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/swift/Sources/CoreMLDelegate.swift)
*   [Core ML delegate C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/experimental/delegates/coreml/coreml_delegate.h)
    *   This can be used for Objective-C codes.
