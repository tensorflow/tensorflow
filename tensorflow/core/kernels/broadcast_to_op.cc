/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#define EIGEN_USE_THREADS

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/broadcast_to_op.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

template <typename Device, typename T>
class BroadcastToOp : public OpKernel {
 public:
  explicit BroadcastToOp(OpKernelConstruction* ctx) : OpKernel(ctx) {}

  void Compute(OpKernelContext* ctx) override {
    const Tensor& input_tensor = ctx->input(0);
    const TensorShape& input_shape = input_tensor.shape();

    const Tensor& shape_tensor = ctx->input(1);

    TensorShape output_shape;
    OP_REQUIRES_OK(ctx,
                   ctx->op_kernel().MakeShape(shape_tensor, &output_shape));

    Tensor* output_tensor = nullptr;
    OP_REQUIRES_OK(ctx, ctx->allocate_output(0, output_shape, &output_tensor));

    const Device& d = ctx->eigen_device<Device>();
    functor::BroadcastTo<Device, T>()(d, ctx, *output_tensor, output_shape,
                                      input_tensor, input_shape);
  }
};

// As MakeShape is able to handle both DT_INT32 and DT_INT64,
// no need to have TypeConstraint for `Tidx`
#define REGISTER_KERNEL(type)                                           \
  REGISTER_KERNEL_BUILDER(                                              \
      Name("BroadcastTo").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      BroadcastToOp<CPUDevice, type>);

TF_CALL_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL

#if GOOGLE_CUDA

namespace functor {
#define DECLARE_GPU_TEMPLATE(Type)                              \
  template <>                                                   \
  void BroadcastTo<GPUDevice, Type>::operator()(                \
      const GPUDevice& d, OpKernelContext* ctx, Tensor& output, \
      const TensorShape& output_shape, const Tensor& input,     \
      const TensorShape& input_shape);                          \
  extern template struct BroadcastTo<GPUDevice, Type>;

TF_CALL_GPU_ALL_TYPES(DECLARE_GPU_TEMPLATE);
#undef DECLARE_GPU_KERNEL
}  // namespace functor

#define REGISTER_KERNEL(type)                            \
  REGISTER_KERNEL_BUILDER(Name("BroadcastTo")            \
                              .Device(DEVICE_GPU)        \
                              .TypeConstraint<type>("T") \
                              .HostMemory("shape"),      \
                          BroadcastToOp<GPUDevice, type>);

TF_CALL_GPU_ALL_TYPES(REGISTER_KERNEL);
#undef REGISTER_KERNEL
#endif

}  // namespace tensorflow
