# TFLite Delegate Utilities for Tooling

## TFLite Delegate Registrar

[A TFLite delegate registrar](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/delegate_provider.h)
is provided here. The registrar keeps a list of TFLite delegate providers, each
of which defines a list parameters that could be initialized from commandline
arguments and provides a TFLite delegate instance creation based on those
parameters. This delegate registrar has been used in TFLite evaluation tools and
the benchmark model tool.

A particular TFLite delegate provider can be used by linking the corresponding
library, e.g. adding it to the `deps` of a BUILD rule. Note that each delegate
provider library has been configured with `alwayslink=1` in the BUILD rule so
that it will be linked to any binary that directly or indirectly depends on it.

The following lists all implemented TFLite delegate providers and their
corresponding list of parameters that each supports to create a particular
TFLite delegate.

### Common parameters

*   `num_threads`: `int` (default=-1) \
    The number of threads to use for running the inference on CPU. By default,
    this is set to the platform default value -1.
*   `max_delegated_partitions`: `int` (default=0, i.e. no limit) \
    The maximum number of partitions that will be delegated. \
    Currently supported by the GPU, Hexagon, CoreML and NNAPI delegate.
*   `min_nodes_per_partition`: `int` (default=delegate's own choice) \
    The minimal number of TFLite graph nodes of a partition that needs to be
    reached to be delegated. A negative value or 0 means to use the default
    choice of each delegate. \
    This option is currently supported by the Hexagon and CoreML delegate.
*   `delegate_serialize_dir`: `string` (default="") \
    Directory to be used by delegates for serializing any model data. This
    allows the delegate to save data into this directory to reduce init time
    after the first run. Currently supported by GPU (OpenCL) and NNAPI delegate
    with specific backends on Android. Note that delegate_serialize_token is
    also required to enable this feature.
*   `delegate_serialize_token`: `string` (default="") \
    Model-specific token acting as a namespace for delegate serialization.
    Unique tokens ensure that the delegate doesn't read inapplicable/invalid
    data. Note that delegate_serialize_dir is also required to enable this
    feature.
*   `first_delegate_node_index`: `int` (default=0) \
    The index of the first node that could be delegated. Debug only. Add
    '--define=tflite_debug_delegate=true' in your build command line to use it.
    \
    Currently only supported by CoreML delegate.
*   `last_delegate_node_index`: `int` (default=INT_MAX) \
    The index of the last node that could be delegated. Debug only. Add
    '--define=tflite_debug_delegate=true' in your build command line to use it.
    \
    Currently only supported by CoreML delegate.

### GPU delegate provider

The GPU delegate is supported on Android and iOS devices, or platforms where the
delegate library is built with "-DCL_DELEGATE_NO_GL" macro.

#### Common options

*   `use_gpu`: `bool` (default=false) \
    Whether to use the
    [GPU accelerator delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/gpu).
*   `gpu_precision_loss_allowed`: `bool` (default=true) \
    Whether to allow the GPU delegate to carry out computation with some
    precision loss (i.e. processing in FP16) or not. If allowed, the performance
    will increase.
*   `gpu_experimental_enable_quant`: `bool` (default=true) \
    Whether to allow the GPU delegate to run a 8-bit quantized model or not.
*   `gpu_inference_for_sustained_speed`: `bool` (default=false) \
    Whether to prefer maximizing the throughput. This mode will help when the
    same delegate will be used repeatedly on multiple inputs. This is supported
    on non-iOS platforms.

#### Android options

*   `gpu_backend`: `string` (default="") \
    Force the GPU delegate to use a particular backend for execution, and fail
    if unsuccessful. Should be one of: cl, gl. By default, the GPU delegate will
    try OpenCL first and then OpenGL if the former fails.

#### iOS options

*   `gpu_wait_type`: `string` (default="") \
    Which GPU wait_type option to use. Should be one of the following: passive,
    active, do_not_wait, aggressive. When left blank, passive mode is used by
    default.

### NNAPI delegate provider

*   `use_nnapi`: `bool` (default=false) \
    Whether to use
    [Android NNAPI](https://developer.android.com/ndk/guides/neuralnetworks/).
    This API is available on recent Android devices. When on Android Q+, will
    also print the names of NNAPI accelerators accessible through the
    `nnapi_accelerator_name` flag.
*   `nnapi_accelerator_name`: `string` (default="") \
    The name of the NNAPI accelerator to use (requires Android Q+). If left
    blank, NNAPI will automatically select which of the available accelerators
    to use.
*   `nnapi_execution_preference`: `string` (default="") \
    Which
    [NNAPI execution preference](https://developer.android.com/ndk/reference/group/neural-networks.html#group___neural_networks_1gga034380829226e2d980b2a7e63c992f18af727c25f1e2d8dcc693c477aef4ea5f5)
    to use when executing using NNAPI. Should be one of the following:
    fast_single_answer, sustained_speed, low_power, undefined.
*   `nnapi_execution_priority`: `string` (default="") \
    The relative priority for executions of the model in NNAPI. Should be one of
    the following: default, low, medium and high. This option requires Android
    11+.
*   `disable_nnapi_cpu`: `bool` (default=true) \
    Excludes the
    [NNAPI CPU reference implementation](https://developer.android.com/ndk/guides/neuralnetworks#device-assignment)
    from the possible devices to be used by NNAPI to execute the model. This
    option is ignored if `nnapi_accelerator_name` is specified.
*   `nnapi_allow_fp16`: `bool` (default=false) \
    Whether to allow FP32 computation to be run in FP16.
*   `nnapi_allow_dynamic_dimensions`: `bool` (default=false) \
    Whether to allow dynamic dimension sizes without re-compilation. This
    requires Android 9+.
*   `nnapi_use_burst_mode`: `bool` (default=false) \
    use NNAPI Burst mode if supported. Burst mode allows accelerators to
    efficiently manage resources, which would significantly reduce overhead
    especially if the same delegate instance is to be used for multiple
    inferences.
*   `nnapi_support_library_path`: `string` (default=""), Path from which NNAPI
    support library will be loaded to construct the delegate. In order to use
    NNAPI delegate with support library, --nnapi_accelerator_name must be
    specified and must be equal to one of the devices provided by the support
    library.

### Hexagon delegate provider

*   `use_hexagon`: `bool` (default=false) \
    Whether to use the Hexagon delegate. Not all devices may support the Hexagon
    delegate, refer to the
    [TensorFlow Lite documentation](https://www.tensorflow.org/lite/performance/hexagon_delegate)
    for more information about which devices/chipsets are supported and about
    how to get the required libraries. To use the Hexagon delegate also build
    the hexagon_nn:libhexagon_interface.so target and copy the library to the
    device. All libraries should be copied to /data/local/tmp on the device.
*   `hexagon_profiling`: `bool` (default=false) \
    Whether to profile ops running on hexagon.

### XNNPACK delegate provider

*   `use_xnnpack`: `bool` (default=false) \
    Whether to explicitly apply the XNNPACK delegate. Note the XNNPACK delegate
    could be implicitly applied by the TF Lite runtime regardless the value of
    this parameter. To disable this implicit application, set the value to
    `false` explicitly.

### CoreML delegate provider

*   `use_coreml`: `bool` (default=false) \
    Whether to use the
    [Core ML delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/coreml).
    This option is only available in iOS.
*   `coreml_version`: `int` (default=0) \
    Target Core ML version for model conversion. The default value is 0 and it
    means using the newest version that's available on the device.

### External delegate provider

*   `external_delegate_path`: `string` (default="") \
    Path to the external delegate library to use.
*   `external_delegate_options`: `string` (default="") \
    A list of options to be passed to the external delegate library. Options
    should be in the format of `option1:value1;option2:value2;optionN:valueN`

### Stable delegate provider [Experimental API]

The stable delegate provider provides a `TfLiteOpaqueDelegate` object pointer
and its corresponding deleter by loading a dynamic library that encapsulates the
actual TFLite delegate implementation in a
[`TfLiteStableDelegate`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/experimental/acceleration/configuration/c/stable_delegate.h)
struct instance.

While the structure of the stable delegate provider is similar to the external
delegate provider, which provides the
[external delegates](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/external),
the design objectives of the stable delegates and the external delegates are
different.

-   Stable delegates are designed to work with shared object files that support
    ABI backward compatibility; that is, the delegate and the TF Lite runtime
    won't need to be built using the exact same version of TF Lite as the app.
    However, this is work in progress and the ABI stability is not yet
    guaranteed.
-   External delegates were developed mainly for delegate evaluation
    (https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/delegates/external).

The stable delegates and the external delegates use different APIs for
diagnosing errors, creating and destroying the delegates. For more details of
the concrete API differences, please check
[stable_delegate.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/core/experimental/acceleration/configuration/c/stable_delegate.h)
and
[external_delegate.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/external/external_delegate.h).

The stable delegate provider is not supported on Windows platform.

*   `stable_abi_delegate_settings_file`: `string` (default="") \
    Path to the delegate settings JSON file.
