#ifndef TENSORFLOW_KERNELS_IDENTITY_OP_H_
#define TENSORFLOW_KERNELS_IDENTITY_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class IdentityOp : public OpKernel {
 public:
  explicit IdentityOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    if (IsRefType(context->input_dtype(0))) {
      context->forward_ref_input_to_ref_output(0, 0);
    } else {
      context->set_output(0, context->input(0));
    }
  }

  bool IsExpensive() override { return false; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_IDENTITY_OP_H_
