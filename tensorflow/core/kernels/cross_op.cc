/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// See docs in ../ops/math_ops.cc.
#define EIGEN_USE_THREADS

#include <algorithm>
#include <cmath>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/cross_op.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename Type>
class CrossOp : public OpKernel {
 public:
  explicit CrossOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor& in0 = context->input(0);
    const Tensor& in1 = context->input(1);
    OP_REQUIRES(context, in0.shape() == in1.shape(),
                errors::InvalidArgument("Both inputs must be of same shape: ",
                                        in0.shape().DebugString(), " vs. ",
                                        in1.shape().DebugString()));
    OP_REQUIRES(context, in0.dims() >= 1,
                errors::InvalidArgument("Input must be at least 1D",
                                        in0.shape().DebugString()));

    // Cross-products only really make sense for three and
    // seven dimensions, and the latter is very obscure. If there is
    // demand, we could perhaps allow 2D vectors where the last
    // element is taken to be zero, but for now, we simply require
    // that all are 3D.
    auto inner_dim = in0.dim_size(in0.dims() - 1);
    OP_REQUIRES(context, inner_dim == 3,
                errors::FailedPrecondition(
                    "Cross-products are only defined for 3-element vectors."));

    // Create the output Tensor with the same dimensions as the input Tensors.
    Tensor* output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, in0.shape(), &output));

    // Make a canonical tensor, maintaining the last (3-vector) dimension,
    // while flattening all others do give the functor easy to work with data.
    typename TTypes<Type, 2>::ConstTensor in0_data =
        in0.flat_inner_dims<Type>();
    typename TTypes<Type, 2>::ConstTensor in1_data =
        in1.flat_inner_dims<Type>();
    typename TTypes<Type, 2>::Tensor output_data =
        output->flat_inner_dims<Type>();

    functor::Cross<Device, Type>()(context->eigen_device<Device>(), in0_data,
                                   in1_data, output_data);
  }
};

#define REGISTER_CPU_KERNEL(type)                                 \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Cross").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      CrossOp<CPUDevice, type>);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNEL);
#undef REGISTER_CPU_KERNEL

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
// Forward declarations of the function specializations for GPU (to prevent
// building the GPU versions here, they will be built compiling _gpu.cu.cc).
namespace functor {
#define DECLARE_GPU_KERNEL(type)                                 \
  template <>                                                    \
  void Cross<GPUDevice, type>::operator()(                       \
      const GPUDevice& d, TTypes<type, 2>::ConstTensor in0_data, \
      TTypes<type, 2>::ConstTensor in1_data,                     \
      TTypes<type, 2>::Tensor output_data);                      \
  extern template struct Cross<GPUDevice, type>;
TF_CALL_REAL_NUMBER_TYPES(DECLARE_GPU_KERNEL);
#undef DECLARE_GPU_KERNEL
}  // namespace functor
#define REGISTER_GPU_KERNEL(type)                                 \
  REGISTER_KERNEL_BUILDER(                                        \
      Name("Cross").Device(DEVICE_GPU).TypeConstraint<type>("T"), \
      CrossOp<GPUDevice, type>);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_KERNEL);
#undef REGISTER_GPU_KERNEL
#endif

}  // namespace tensorflow
