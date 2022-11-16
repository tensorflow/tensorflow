# TensorFlow Lite Sample Stable Delegate

## Description

An example delegate for stable delegate testing.

**Note:** Dynamic loading of delegate shared object files is work-in-progress.
Therefore, the stable delegate library currently needs to be linked directly
with TFLite.

Here, we show a delegate implementation to illustrate our approach to create an
opaque delegate. The sample stable delegate supports addition and subtraction
operations only.

The sample stable delegate implementation uses the new stable delegate API,
which is based around TfLiteOpaqueDelegateStruct. TfLiteOpaqueDelegateStruct is
an opaque version of TfLiteDelegate; allows delegation of nodes to alternative
backends. This is an abstract type that is intended to have the same role as
TfLiteDelegate from
[common.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h),
but without exposing the implementation details of how delegates are
implemented.

## Delegate Creation

We use
[SimpleOpaqueDelegateInterface and SimpleOpaqueDelegateKernelInterface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_opaque_delegate.h).
These APIs make it easier to create a TF Lite delegate plug-in via
[TfLiteOpaqueDelegateFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_opaque_delegate.h),
which provides the static methods to deal with the delegate creation and
deletion.

## Testing & Tooling

### Benchmark Tools

TODO(b/250886376): Enable TFLite Benchmark Tool to use the delegate binary.
