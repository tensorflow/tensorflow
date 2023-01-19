# TensorFlow Lite Sample Stable Delegate

## Description

An example delegate for stable delegate testing that supports addition and
subtraction operations only.

The sample stable delegate implementation uses the stable delegate API,
which is based around `TfLiteOpaqueDelegate`. `TfLiteOpaqueDelegate` is
an opaque version of `TfLiteDelegate`; which allows delegation of nodes to
alternative backends. This is an abstract type that is intended to have the same
role as `TfLiteDelegate` from
[common.h](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h),
but without exposing the implementation details of how delegates are
implemented.

`TfLiteOpaqueDelegate`s can be loaded dynamically
(see `sample_stable_delegate_external_test.cc`) and then be supplied to the
TFLite runtime, in the same way as statically linked delegates can.

Note however that open-source TF Lite does not (yet) provide a binary stable
interface between delegates and the TF Lite runtime itself.  Therefore any
opaque delegate that is loaded dynamically into TF Lite *must* have been built
against the same version (and commit) that the TF Lite runtime itself has been
built at. Any other configuration can lead to undefined behavior.

## Delegate implementation

The sample stable delegate uses two supporting interfaces [SimpleOpaqueDelegateInterface and SimpleOpaqueDelegateKernelInterface](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_opaque_delegate.h).
These APIs make it easier to implement an opaque TF Lite delegate, though their
usage is entirely optional.

The `sample_stable_delegate_test` driver (see next section) makes use of the
[TfLiteOpaqueDelegateFactory](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/simple_opaque_delegate.h)
facility, which provides static methods that deal with delegate creation and
deletion.

## Testing

See [sample_stable_delegate_test.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate_test.cc)
for a standalone test driver that links the sample stable delegate statically
and runs inference on a TF Lite model.

See [sample_stable_delegate_external_test.cc](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/utils/experimental/sample_stable_delegate/sample_stable_delegate_external_test.cc)
for a standalone test driver that loads the sample stable delegate dynamically
and runs inference on a TF Lite model.

### Benchmark Tools

See the [Delegate Performance Benchmark app](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/tools/benchmark/experimental/delegate_performance/android/README.md)
