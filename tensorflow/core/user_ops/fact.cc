// An example Op.

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"

using namespace tensorflow;

REGISTER_OP("Fact")
    .Output("fact: string")
    .Doc(R"doc(
Output a fact about factorials.
)doc");

class FactOp : public OpKernel {
 public:
  explicit FactOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Output a scalar string.
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context,
                   context->allocate_output(0, TensorShape(), &output_tensor));
    auto output = output_tensor->template scalar<string>();

    output() = "0! == 1";
  }
};

REGISTER_KERNEL_BUILDER(Name("Fact").Device(DEVICE_CPU), FactOp);
