# TensorFlow Lite operator versions

This document describes TensorFlow Lite's op versioning schema. Op versioning
enables developers to add new functionalities and parameters into existing ops.
In addition, it guarantees the following:

*   Backward compatibility: New TensorFlow Lite implementation should handle an
    old model file.
*   Forward compatibility: Old TensorFlow Lite implementation should handle a
    new model file produced by new version of converter, as long as no new
    features are used.
*   Forward in-compatibility detection: If an old TensorFlow Lite implementation
    reads a new model that contains a new version of an op which isn't
    supported, it should report the error.

## Example: Adding dilation into depthwise convolution

The remainder of this document explains op versioning in TFLite by showing how
to add dilation parameters to the depthwise convolution operation.

Knowledge of dilation is not required to understand this document. Note that:

*   2 new integer parameters will be added: `dilation_width_factor` and
    `dilation_height_factor`.
*   Old depthwise convolution kernels that don't support dilation are equivalent
    to setting the dilation factors to 1.

### Change FlatBuffer schema

To add new parameters into an op, change the options table in
`lite/schema/schema.fbs`.

For example, the options table of depthwise convolution looks like this:

```
table DepthwiseConv2DOptions {
  padding:Padding;
  stride_w:int;
  stride_h:int;
  depth_multiplier:int;
  fused_activation_function:ActivationFunctionType;
}
```

When adding new parameters:

*   Add comments indicating which parameters are supported by which version.
*   When the new implementation gets the default values for newly added
    parameters, it should work exactly the same as the old implementation.

The table will be like this after the new parameters are added:

```
table DepthwiseConv2DOptions {
  // Parameters for DepthwiseConv version 1 or above.
  padding:Padding;
  stride_w:int;
  stride_h:int;
  depth_multiplier:int;
  fused_activation_function:ActivationFunctionType;
  // Parameters for DepthwiseConv version 2 or above.
  dilation_w_factor:int = 1;
  dilation_h_factor:int = 1;
}
```

The file `lite/schema/schema_generated.h` should be re-generated for the new
schema.

### Change C structures and kernel implementation

In TensorFlow Lite, the kernel implementation is decoupled from FlatBuffer
definition. The kernels read the parameter from C structures defined in
`lite/c/builtin_op_data.h`.

The original depthwise convolution parameter is as follows:

```
typedef struct {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int depth_multiplier;
  TfLiteFusedActivation activation;
} TfLiteDepthwiseConvParams;
```

As with the FlatBuffer schema, add comments indicating which parameters are
supported starting from which version. The result is seen below:

```
typedef struct {
  // Parameters for DepthwiseConv version 1 or above.
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  int depth_multiplier;
  TfLiteFusedActivation activation;
  // Parameters for DepthwiseConv version 2 or above.
  int dilation_width_factor;
  int dilation_height_factor;
} TfLiteDepthwiseConvParams;
```

Please also change the kernel implementation to read the newly added parameters
from the C structures. The details are omitted here.

### Change the FlatBuffer reading code

The logic to read FlatBuffer and produce C structure is in
`lite/core/api/flatbuffer_conversions.cc`.

Update the file to handle the new parameters, as shown below:

```
TfLiteStatus ParseDepthwiseConv2D(const Operator* op,
                                  ErrorReporter* error_reporter,
                                  BuiltinDataAllocator* allocator,
                                  void** builtin_data) {
  CheckParsePointerParams(op, error_reporter, allocator, builtin_data);

  SafeBuiltinDataAllocator safe_allocator(allocator);

  std::unique_ptr<TfLiteDepthwiseConvParams,
                  SafeBuiltinDataAllocator::BuiltinDataDeleter>
      params = safe_allocator.Allocate<TfLiteDepthwiseConvParams>();
  TF_LITE_ENSURE(error_reporter, params != nullptr);

  const DepthwiseConv2DOptions* schema_params =
      op->builtin_options_as_DepthwiseConv2DOptions();

  if (schema_params != nullptr) {
    params->padding = ConvertPadding(schema_params->padding());
    params->stride_width = schema_params->stride_w();
    params->stride_height = schema_params->stride_h();
    params->depth_multiplier = schema_params->depth_multiplier();
    params->activation =
        ConvertActivation(schema_params->fused_activation_function());

    params->dilation_width_factor = schema_params->dilation_w_factor();
    params->dilation_height_factor = schema_params->dilation_h_factor();
  }

  *builtin_data = params.release();
  return kTfLiteOk;
}
```

It's not required to check the op version here. When the new implementation
reads an old model file where dilation factors are missing, it will use 1 as the
default value, and the new kernel will work consistently with the old kernel.

### Change kernel registration

The MutableOpResolver (defined in `lite/mutable_op_resolver.h`) provides a few
functions to register op kernels. The minimum and maximum version are 1 by
default:

```
void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration,
                int min_version = 1, int max_version = 1);
void AddCustom(const char* name, TfLiteRegistration* registration,
               int min_version = 1, int max_version = 1);
```

The built-in ops are registered in `lite/kernels/register.cc`. In this example,
we implemented a new op kernel which can handle `DepthwiseConv2D` version 1 and
2, so we need to change this line:

```
AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D());
```

to:

```
AddBuiltin(BuiltinOperator_DEPTHWISE_CONV_2D, Register_DEPTHWISE_CONV_2D(),
             /* min_version = */ 1,
             /* max_version = */ 2);
```

### Change TFLite op version

The next step is to make TFLite populate the minimum version that's required to
execute the op. In this example, it means:

*   Populate version=1 when dilation factors are all 1.
*   Populate version=2 otherwise.

To do this, you need to first add corresponding parameters to
`depthwise_conv_2d` inside the `OpSignature` struct:

```
struct {
      int32_t dilation_w_factor;
      int32_t dilation_h_factor;
    } depthwise_conv_2d;
```

Then populate these new parameters in `GetOpSignature` function in
`lite/tools/versioning/op_version.cc`:

```
case BuiltinOperator_DEPTHWISE_CONV_2D: {
      auto conv_option = op->builtin_options_as_DepthwiseConv2DOptions();
      if (conv_option) {
        op_sig.options.depthwise_conv_2d.dilation_w_factor =
            conv_option->dilation_w_factor();
        op_sig.options.depthwise_conv_2d.dilation_h_factor =
            conv_option->dilation_h_factor();
      }
    } break;
```

Note that if you are adding support for new types, above steps are not needed.
Input and output types are defined and populated for all ops in `OpSignature`.

Finally, modify `GetBuiltinOperatorVersion` function for the operator in
`lite/tools/versioning/op_version.cc` by adding the new version to the case of
`DepthwiseConv2D`:

```
case BuiltinOperator_DEPTHWISE_CONV_2D:
  if (op_sig.options.depthwise_conv_2d.dilation_w_factor != 1 ||
      op_sig.options.depthwise_conv_2d.dilation_h_factor != 1) {
    return 2;
  }
  return 1;
```

### Update the operator version map

The last step is to add the new version info into the operator version map. This
step is required because we need to generate the model's minimum required
runtime version based on this version map.

To do this, you need to add a new map entry in
`lite/tools/versioning/runtime_version.cc`.

In this example, you need to add the following entry into `op_version_map`:

```
{{BuiltinOperator_DEPTHWISE_CONV_2D, 2}, kPendingReleaseOpVersion}
```

(`kPendingReleaseOpVersion` will be replaced with the appropriate release
version in the next stable release.)

### Delegation implementation

TensorFlow Lite provides a delegation API which enables delegating ops to
hardware backends. In the delegate's `Prepare` function, check if the version is
supported for every node in Delegation code.

```
const int kMaxVersion = 1;
TfLiteNode* node;
TfLiteRegistration* registration = nullptr;
TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(context, node_index, &node, &registration));

if (registration->version > kMaxVersion) {
  // Reject the node if the version isn't supported.
}
```

This is required even if the delegation only supports version 1 ops, so the
delegation can detect incompatibility when getting a higher version op.
