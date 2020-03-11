# TensorFlow Lite operator versions

This document describes TensorFlow Lite's op versioning schema. Op
versioning enables developers to add new functionalities and parameters into
existing ops. In addition, it guarantees the following:

*   Backward compatibility: New TensorFlow Lite implementation should
    handle an old model file.
*   Forward compatibility: Old TensorFlow Lite implementation should
    handle a new model file produced by new version of TOCO, as long as no new
    features are used.
*   Forward in-compatibility detection: If an old TensorFlow Lite implementation
    reads a new model that contains a new version of an op which isn't
    supported, it should report the error.

## Example: Adding Dilation into Convolution

The remainder of this document explains op versioning in TFLite by showing how
to add dilation parameters to the convolution operation.

Knowledge of dilation is not required to understand this document. Note that:

*   2 new integer parameters will be added: `dilation_width_factor` and
    `dilation_height_factor`.
*   Old convolution kernels that don't support dilation are equivalent to
    setting the dilation factors to 1.

### Change FlatBuffer Schema

To add new parameters into an op, change the options table in
`lite/schema/schema.fbs`.

For example, the options table of convolution looks like this:

```
table Conv2DOptions {
  padding:Padding;
  stride_w:int;
  stride_h:int;
  fused_activation_function:ActivationFunctionType;
}
```

When adding new parameters:

*   Add comments indicating which parameters are supported by which version.
*   When the new implementation gets the default values for newly added
    parameters, it should work exactly the same as the old implementation.

The table will be like this after the new parameters are added:

```
table Conv2DOptions {
  // Parameters supported by version 1:
  padding:Padding;
  stride_w:int;
  stride_h:int;
  fused_activation_function:ActivationFunctionType;

  // Parameters supported by version 2:
  dilation_width_factor:int = 1;
  dilation_height_factor:int = 1;
}
```

### Change C Structures and Kernel Implementation

In TensorFlow Lite, the kernel implementation is decoupled from
FlatBuffer definition. The kernels read the parameter from C structures defined
in `lite/builtin_op_data.h`.

The original convolution parameter is as follows:

```
typedef struct {
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  TfLiteFusedActivation activation;
} TfLiteConvParams;
```

As with the FlatBuffer schema, add comments indicating which parameters are
supported starting from which version. The result is seen below:

```
typedef struct {
  // Parameters supported by version 1:
  TfLitePadding padding;
  int stride_width;
  int stride_height;
  TfLiteFusedActivation activation;

  // Parameters supported by version 2:
  int dilation_width_factor;
  int dilation_height_factor;
} TfLiteConvParams;
```

Please also change the kernel implementation to read the newly added parameters
from the C structures. The details are omitted here.

### Change the FlatBuffer Reading Code

The logic to read FlatBuffer and produce C structure is in `lite/model.cc`.

Update the file to handle the new parameters, as shown below:

```
case BuiltinOperator_CONV_2D: {
  TfLiteConvParams* params = MallocPOD<TfLiteConvParams>();
  if (auto* conv_params = op->builtin_options_as_Conv2DOptions()) {
    params->padding = parse_padding(conv_params->padding());
    params->stride_width = conv_params->stride_w();
    params->stride_height = conv_params->stride_h();
    params->activation =
        parse_activation(conv_params->fused_activation_function());
    params->dilation_width_factor = conv_params->dilation_width_factor();
    params->dilation_height_factor = conv_params->dilation_height_factor();
  }
  *builtin_data = reinterpret_cast<void*>(params);
  break;
}
```

It's not required to check the op version here. When the new implementation
reads an old model file where dilation factors are missing, it will use 1 as
the default value, and the new kernel will work consistently with the old
kernel.

### Change Kernel Registration

The MutableOpResolver (defined in `lite/op_resolver.h`) provides a few functions
to register op kernels. The minimum and maximum version are 1 by default:

```
void AddBuiltin(tflite::BuiltinOperator op, TfLiteRegistration* registration,
                int min_version = 1, int max_version = 1);
void AddCustom(const char* name, TfLiteRegistration* registration,
               int min_version = 1, int max_version = 1);
```

The built-in ops are registered in `lite/kernels/register.cc`. In this example,
we implemented a new op kernel which can handle `Conv2D` version 1 and 2, so we
need to change this line:

```
AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D());
```

to:

```
AddBuiltin(BuiltinOperator_CONV_2D, Register_CONV_2D(), 1, 2);
```

### Change TOCO TFLite exporter

The next step is to make TOCO populate the minimum version that's required to
execute the op. In this example, it means:

*   Populate version=1 when dilation factors are all 1.
*   Populate version=2 otherwise.

To do this, you need to override `GetVersion` function for the operator class in
`lite/tools/versioning/op_version.cc`.

For ops with only one version, the `GetVersion` function is defined as:

```
int GetVersion(const Operator& op) const override { return 1; }
```

When supporting multiple versions, check the parameters and determine the
version for the op, as shown in the following example:

```
int GetVersion(const Operator& op) const override {
  const auto& conv_op = static_cast<const ConvOperator&>(op);
  if (conv_op.dilation_width_factor != 1 ||
      conv_op.dilation_height_factor != 1) {
    return 2;
  }
  return 1;
}
```

### Update the operator version map

The last step is to add the new version info into the operator version map. This
step is required because we need generate the model's minimum required runtime
version based on this version map.

To do this, you need to add a new map entry in
`lite/tools/versioning/op_version.cc`.

In this example, it means you need to add the following into `op_version_map`:
```
{{OperatorType::kConv, 3}, "kPendingReleaseOpVersion"}
```
(`kPendingReleaseOpVersion` will be replaced with the appropriate release
version in the next stable release.)

### Delegation Implementation

TensorFlow Lite provides a delegation API which enables delegating ops to
hardware backends. In Delegate's `Prepare` function, check if the version
is supported for every node in Delegation code.

```
const int kMinVersion = 1;
TfLiteNode* node;
TfLiteRegistration* registration = nullptr;
TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(context, node_index, &node, &registration));

if (registration->version > kMinVersion) {
  // Reject the node if the version isn't supported.
}
```

This is required even if the delegation only supports version 1 ops, so the
delegation can detect incompatibility when getting a higher version op.

