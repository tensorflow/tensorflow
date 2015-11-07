#ifndef TENSORFLOW_KERNELS_CONSTANT_OP_H_
#define TENSORFLOW_KERNELS_CONSTANT_OP_H_

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

// ConstantOp returns a tensor specified by ConstantOpDef.
class ConstantOp : public OpKernel {
 public:
  explicit ConstantOp(OpKernelConstruction* ctx);
  void Compute(OpKernelContext* ctx) override;
  bool IsExpensive() override { return false; }
  ~ConstantOp() override;

 private:
  Tensor tensor_;
  TF_DISALLOW_COPY_AND_ASSIGN(ConstantOp);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_CONSTANT_OP_H_
