# Hexagon Delegate

Experimental delegate which uses Hexagon SDK to delegate the processing
to QC DSP.
Note that we only support quantized models, since the DSP is efficient
with quantized versions. So all op support is for quantized versions.

Usage:

- Add dependency on hexagon_delegate rule.

- Code change example:

```
  #include "tensorflow/lite/experimental/delegates/hexagon/hexagon_delegate.h"

  // Assuming shared libraries are under "/data/local/tmp/"
  // If files are packaged with native lib in android App then it
  // will typically be equivalent to the path provided by
  // "getContext().getApplicationInfo().nativeLibraryDir"
  const char[] library_directory_path = "/data/local/tmp/";
  TfLiteHexagonInitWithPath(library_directory_path);  // Needed once at startup.
  ::tflite::TfLiteHexagonDelegateOptions params = {0};
  // 'delegate_ptr' Need to outlive the interpreter. For example,
  // If use case will need to resize input or anything that can trigger
  // re-applying delegates then 'delegate_ptr' need to outlive the interpreter.
  auto* delegate_ptr = ::tflite::TfLiteHexagonDelegateCreate(&params);
  Interpreter::TfLiteDelegatePtr delegate(delegate_ptr,
      [](TfLiteDelegate* delegate) {
        ::tflite::TfLiteHexagonDelegateDelete(delegate);
      });
  interpreter->ModifyGraphWithDelegate(delegate.get());
  TfLiteHexagonTearDown();  // Needed once at end of app/DSP usage.
```

* Shared libraries:
  - 'libhexagon_interface.so' which holds the interface that the delegate uses.
  It must be available if you linked the hexagon_delegate library to TFLite.
  You can load it either from shell by overriding
  LD_LIBRARY_PATH=$LD_LIBRARY_PATH:"path to the so",
  or add it inside your apk in a way it is available.
  - 'libhexagon_nn_skel(_v65/_v66).so' which holds the DSP code.
  Use TfLiteHexagonInitWithPath(..) and provide the path to the directory
  which holds the shared libraries for the Hexagon NN on device.
  If you're using TfLiteHexagonInit() then
  You will need to set environment variable "ADSP_LIBRARY_PATH" to
  "path_to_the_lib";/system/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp
  Note that separator here is ';' not ':'
  You can push all 3 files, and the library will pick the one needed based
  on the runtime. Or if you are sure of what you will use on the device then
  push only one of them.



## Supported Ops

Hexagon only supports ops that have inputs/outputs of <= 4 dimensions.
The following operations have been implemented, with a few constraints that
are verified in `IsNodeSupportedByHexagon`:

* Add (Support relu activations)
* ArgMax
* ArgMin
* AveragePool2D:
  * Constraints:
    - No Activation
* Concat
* Conv2D:
  * Constraints:
    - stride width/height <= 3
* DepthToSpace
* DepthwiseConv2D:
  * Constraints:
      - Filter width == 3
      - depth_multiplier == 1
      - dilation only supported when stride == 1
      - Otherwise, stride height/width <= 3
* FullyConnected (without any activation)
* Hardswish
* L2Normalization (without any activation)
* Logistic (aka Sigmoid)
* Maximum
* MaxPool2D (without any activation) (b/129276536)
* Mean
* Minimum
* MirrorPad
* Mul (Support relu activations)
* Neg
* Pack
* Pad: Only supports 0 padding (b/139277813)
* Quantize (8-bit inputs & outputs only)
* Relu
* Relu6
* Reshape
* Resize Bilinear:
  * Constraints:
    - Requested size <= 65 (b/143105433)
* Resize Nearest Neighbor
* Slice
* SoftMax
* SpaceToDepth
* Split
* Sub (Support relu activations)
* Tanh
* Transpose
* TransposeConv2D:
  * Constraints:
    - stride height/width <= 3
    - dilation height/width == 1
