#if !GOOGLE_CUDA
#error This file must only be included when building with Cuda support
#endif

#ifndef THIRD_PARTY_TENSORFLOW_CORE_KERNELS_POOLING_OPS_COMMON_GPU_H_
#define THIRD_PARTY_TENSORFLOW_CORE_KERNELS_POOLING_OPS_COMMON_GPU_H_

#include "tensorflow/stream_executor/dnn.h"
#include "tensorflow/core/framework/numeric_op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/kernels/avgpooling_op.h"
#include "tensorflow/core/kernels/maxpooling_op.h"
#include "tensorflow/core/kernels/ops_util.h"
#include "tensorflow/core/util/padding.h"
#include "tensorflow/core/public/tensor_shape.h"
#include "third_party/eigen3/unsupported/Eigen/CXX11/NeuralNetworks"
#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

namespace tensorflow {

// A helper class that launch the cudnn pooling backward operations.
// The original input and output tensors are optional for AvgPoolGrad, but
// mandatory for MaxPoolGrad.
template <typename T>
class DnnPoolingGradOp {
 public:
  typedef GPUDevice Device;
  static void Compute(OpKernelContext* context,
                      perftools::gputools::dnn::PoolingMode pooling_mode,
                      const std::vector<int32>& size,
                      const std::vector<int32>& stride, Padding padding,
                      const Tensor* tensor_in, const Tensor* tensor_out,
                      const Tensor& out_backprop,
                      const TensorShape& tensor_in_shape);
};

}  // namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_KERNELS_POOLING_OPS_COMMON_GPU_H_
