#ifndef TENSORFLOW_KERNELS_NO_OP_H_
#define TENSORFLOW_KERNELS_NO_OP_H_

#include "tensorflow/core/framework/op_kernel.h"

namespace tensorflow {

class NoOp : public OpKernel {
 public:
  explicit NoOp(OpKernelConstruction* context) : OpKernel(context) {}
  void Compute(OpKernelContext* context) override {}
  bool IsExpensive() override { return false; }
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_NO_OP_H_
