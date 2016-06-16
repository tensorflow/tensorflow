#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
// #include "tensorflow/core/kernels/identity_op.h"

using namespace tensorflow;

// TODO: replace with IdentityOp
class FlipGradientOp : public OpKernel {
 public:
  explicit FlipGradientOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
};

REGISTER_OP("FlipGradient")
    .Input("input: T")
    .Input("s: T")
    .Output("output: T")
    .Attr("T: type")
    .Doc(R"Doc(
Return a tensor with the same shape and contents as the input tensor or value,
but flip gradients for gradient computation.
)Doc");

REGISTER_KERNEL_BUILDER(Name("FlipGradient").Device(DEVICE_CPU), FlipGradientOp);
// REGISTER_KERNEL_BUILDER(Name("FlipGradient").Device(DEVICE_CPU), IdentityOp);
