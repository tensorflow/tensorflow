# Tensorflow Lite Core ML delegate

The TensorFlow Lite Core ML delegate enables running TensorFlow Lite models on
[Core ML framework](https://developer.apple.com/documentation/coreml), which
results in faster model inference on iOS devices.

Note: This delegate is in experimental (beta) phase. It is available from
TensorFlow Lite 2.4.0 and latest nightly releases.

Note: Core ML delegate supports Core ML version 2 and later.

**Supported iOS versions and devices:**

*   iOS 12 and later. In the older iOS versions, Core ML delegate will
    automatically fallback to CPU.
*   By default, Core ML delegate will only be enabled on devices with A12 SoC
    and later (iPhone Xs and later) to use Neural Engine for faster inference.
    If you want to use Core ML delegate also on the older devices, please see
    [best practices](#best-practices)

**Supported models**

The Core ML delegate currently supports float (FP32 and FP16) models.

## Trying the Core ML delegate on your own model

The Core ML delegate is already included in nightly release of TensorFlow lite
CocoaPods. To use Core ML delegate, change your TensorFlow lite pod to include
subspec `CoreML` in your `Podfile`.

Note: If you want to use C API instead of Objective-C API, you can include
`TensorFlowLiteC/CoreML` pod to do so.

```
target 'YourProjectName'
  pod 'TensorFlowLiteSwift/CoreML', '~> 2.4.0'  # Or TensorFlowLiteObjC/CoreML
```

OR

```
# Particularily useful when you also want to include 'Metal' subspec.
target 'YourProjectName'
  pod 'TensorFlowLiteSwift', '~> 2.4.0', :subspecs => ['CoreML']
```

Note: Core ML delegate can also use C API for Objective-C code. Prior to
TensorFlow Lite 2.4.0 release, this was the only option.

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p><pre class="prettyprint lang-swift">
    let coreMLDelegate = CoreMLDelegate()
    var interpreter: Interpreter

    // Core ML delegate will only be created for devices with Neural Engine
    if coreMLDelegate != nil {
      interpreter = try Interpreter(modelPath: modelPath,
                                    delegates: [coreMLDelegate!])
    } else {
      interpreter = try Interpreter(modelPath: modelPath)
    }
      </pre></p>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p><pre class="prettyprint lang-objc">

    // Import module when using CocoaPods with module support
    @import TFLTensorFlowLite;

    // Or import following headers manually
    # import "tensorflow/lite/objc/apis/TFLCoreMLDelegate.h"
    # import "tensorflow/lite/objc/apis/TFLTensorFlowLite.h"

    // Initialize Core ML delegate
    TFLCoreMLDelegate* coreMLDelegate = [[TFLCoreMLDelegate alloc] init];

    // Initialize interpreter with model path and Core ML delegate
    TFLInterpreterOptions* options = [[TFLInterpreterOptions alloc] init];
    NSError* error = nil;
    TFLInterpreter* interpreter = [[TFLInterpreter alloc]
                                    initWithModelPath:modelPath
                                              options:options
                                            delegates:@[ coreMLDelegate ]
                                                error:&amp;error];
    if (error != nil) { /* Error handling... */ }

    if (![interpreter allocateTensorsWithError:&amp;error]) { /* Error handling... */ }
    if (error != nil) { /* Error handling... */ }

    // Run inference ...
      </pre></p>
    </section>
    <section>
      <h3>C (Until 2.3.0)</h3>
      <p><pre class="prettyprint lang-c">
    #include "tensorflow/lite/delegates/coreml/coreml_delegate.h"

    // Initialize interpreter with model
    TfLiteModel* model = TfLiteModelCreateFromFile(model_path);

    // Initialize interpreter with Core ML delegate
    TfLiteInterpreterOptions* options = TfLiteInterpreterOptionsCreate();
    TfLiteDelegate* delegate = TfLiteCoreMlDelegateCreate(NULL);  // default config
    TfLiteInterpreterOptionsAddDelegate(options, delegate);
    TfLiteInterpreterOptionsDelete(options);

    TfLiteInterpreter* interpreter = TfLiteInterpreterCreate(model, options);

    TfLiteInterpreterAllocateTensors(interpreter);

    // Run inference ...

    /* ... */

    // Dispose resources when it is no longer used.
    // Add following code to the section where you dispose of the delegate
    // (e.g. `dealloc` of class).

    TfLiteInterpreterDelete(interpreter);
    TfLiteCoreMlDelegateDelete(delegate);
    TfLiteModelDelete(model);
      </pre></p>
    </section>
  </devsite-selector>
</div>

## Best practices

### Using Core ML delegate on devices without Neural Engine

By default, Core ML delegate will only be created if the device has Neural
Engine, and will return `null` if the delegate is not created. If you want to
run Core ML delegate on other environments (for example, simulator), pass `.all`
as an option while creating delegate in Swift. On C++ (and Objective-C), you can
pass `TfLiteCoreMlDelegateAllDevices`. Following example shows how to do this:

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p><pre class="prettyprint lang-swift">
    var options = CoreMLDelegate.Options()
    options.enabledDevices = .all
    let coreMLDelegate = CoreMLDelegate(options: options)!
    let interpreter = try Interpreter(modelPath: modelPath,
                                      delegates: [coreMLDelegate])
      </pre></p>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p><pre class="prettyprint lang-objc">
    TFLCoreMLDelegateOptions* coreMLOptions = [[TFLCoreMLDelegateOptions alloc] init];
    coreMLOptions.enabledDevices = TFLCoreMLDelegateEnabledDevicesAll;
    TFLCoreMLDelegate* coreMLDelegate = [[TFLCoreMLDelegate alloc]
                                          initWithOptions:coreMLOptions];

    // Initialize interpreter with delegate
      </pre></p>
    </section>
    <section>
      <h3>C</h3>
      <p><pre class="prettyprint lang-c">
    TfLiteCoreMlDelegateOptions options;
    options.enabled_devices = TfLiteCoreMlDelegateAllDevices;
    TfLiteDelegate* delegate = TfLiteCoreMlDelegateCreate(&amp;options);
    // Initialize interpreter with delegate
      </pre></p>
    </section>
  </devsite-selector>
</div>

### Using Metal(GPU) delegate as a fallback.

When the Core ML delegate is not created, alternatively you can still use
[Metal delegate](https://www.tensorflow.org/lite/performance/gpu#ios) to get
performance benefits. Following example shows how to do this:

<div>
  <devsite-selector>
    <section>
      <h3>Swift</h3>
      <p><pre class="prettyprint lang-swift">
    var delegate = CoreMLDelegate()
    if delegate == nil {
      delegate = MetalDelegate()  // Add Metal delegate options if necessary.
    }

    let interpreter = try Interpreter(modelPath: modelPath,
                                      delegates: [delegate!])
      </pre></p>
    </section>
    <section>
      <h3>Objective-C</h3>
      <p><pre class="prettyprint lang-objc">
    TFLDelegate* delegate = [[TFLCoreMLDelegate alloc] init];
    if (!delegate) {
      // Add Metal delegate options if necessary
      delegate = [[TFLMetalDelegate alloc] init];
    }
    // Initialize interpreter with delegate
      </pre></p>
    </section>
    <section>
      <h3>C</h3>
      <p><pre class="prettyprint lang-c">
    TfLiteCoreMlDelegateOptions options = {};
    delegate = TfLiteCoreMlDelegateCreate(&amp;options);
    if (delegate == NULL) {
      // Add Metal delegate options if necessary
      delegate = TFLGpuDelegateCreate(NULL);
    }
    // Initialize interpreter with delegate
      </pre></p>
    </section>
  </devsite-selector>
</div>

The delegate creation logic reads device's machine id (e.g. iPhone11,1) to
determine its Neural Engine availability. See the
[code](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.mm)
for more detail. Alternatively, you can implement your own set of denylist
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
    *   Concatenation should be done along the channel axis.
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

*   [Core ML delegate Swift API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/swift/Sources/CoreMLDelegate.swift)
*   [Core ML delegate C API](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/coreml/coreml_delegate.h)
    *   This can be used for Objective-C codes. ~~~
