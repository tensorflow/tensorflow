#ifndef TENSORFLOW_KERNELS_TRANSPOSE_OP_H_
#define TENSORFLOW_KERNELS_TRANSPOSE_OP_H_

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"

namespace tensorflow {

template <typename Device, typename T>
class TransposeOp : public OpKernel {
 public:
  explicit TransposeOp(OpKernelConstruction* context);
  void Compute(OpKernelContext* context) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_KERNELS_TRANSPOSE_OP_H_
