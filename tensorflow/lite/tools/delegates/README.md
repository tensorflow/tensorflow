# TFLite Delegate Utilities for Tooling

## TFLite Delegate Registrar
[A TFLite delegate registrar](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/delegates/delegate_provider.h)
is provided here. The registrar keeps a list of TFLite delegate providers, each
of which defines a list parameters that could be initialized from commandline
argumenents and provides a TFLite delegate instance creation based on those
parameters. This delegate registrar has been used in TFLite evaluation tools and
the benchmark model tool.

A particular TFLite delegate provider can be used by
linking the corresponding library, e.g. adding it to the `deps` of a BUILD rule.
Note that each delegate provider library has been configured with
`alwayslink=1` in the BUILD rule so that it will be linked to any binary that
directly or indirectly depends on it.

The following lists all implemented TFLite delegate providers and their
corresponding list of parameters that each supports to create a particular
TFLite delegate.

### Common parameters
*   `num_threads`: `int` (default=1) \
    The number of threads to use for running the inference on CPU.
*   `max_delegated_partitions`: `int` (default=0, i.e. no limit) \
    The maximum number of partitions that will be delegated. \
    Currently supported by the GPU, Hexagon, CoreML and NNAPI delegate.
*   `min_nodes_per_partition`: `int` (default=delegate's own choice) \
    The minimal number of TFLite graph nodes of a partition that needs to be
    reached to be delegated. A negative value or 0 means to use the default
    choice of each delegate. \
    This option is currently supported by the Hexagon and CoreML delegate.

### GPU delegate provider

Only Android and iOS devices support GPU delegate.

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

#### Android options
*  `gpu_backend`: `string` (default="") \
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
    The relative priority for executions of the model in NNAPI. Should be one
    of the following: default, low, medium and high.
*   `disable_nnapi_cpu`: `bool` (default=false) \
    Excludes the
    [NNAPI CPU reference implementation](https://developer.android.com/ndk/guides/neuralnetworks#device-assignment)
    from the possible devices to be used by NNAPI to execute the model. This
    option is ignored if `nnapi_accelerator_name` is specified.
*   `nnapi_allow_fp16`: `bool` (default=false) \
    Whether to allow FP32 computation to be run in FP16.

### Hexagon delegate provider
*   `use_hexagon`: `bool` (default=false) \
    Whether to use the Hexagon delegate. Not all devices may support the Hexagon
    delegate, refer to the [TensorFlow Lite documentation](https://www.tensorflow.org/lite/performance/hexagon_delegate) for more
    information about which devices/chipsets are supported and about how to get
    the required libraries. To use the Hexagon delegate also build the
    hexagon_nn:libhexagon_interface.so target and copy the library to the
    device. All libraries should be copied to /data/local/tmp on the device.
*   `hexagon_profiling`: `bool` (default=false) \
    Whether to profile ops running on hexagon.

### XNNPACK delegate provider
*   `use_xnnpack`: `bool` (default=false) \
    Whether to use the XNNPack delegate.

### CoreML delegate provider
*   `use_coreml`: `bool` (default=false) \
    Whether to use the [Core ML delegate](https://github.com/tensorflow/tensorflow/tree/master/tensorflow/lite/experimental/delegates/coreml).
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
