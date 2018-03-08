/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA
#define EIGEN_USE_GPU
#endif  // GOOGLE_CUDA

#include "tensorflow/contrib/framework/kernels/zero_initializer_op.h"

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

using CPUDevice = Eigen::ThreadPoolDevice;
using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename T>
class ZeroInitializerOp : public OpKernel {
 public:
  explicit ZeroInitializerOp(OpKernelConstruction* ctx) : OpKernel(ctx) {
    OP_REQUIRES(ctx, IsRefType(ctx->input_type(0)),
                errors::InvalidArgument("input needs to be a ref type"));
  }

  void Compute(OpKernelContext* ctx) override {
    mutex_lock l(*ctx->input_ref_mutex(0));
    Tensor input = ctx->mutable_input(0, true);
    OP_REQUIRES(ctx, !input.IsInitialized(),
                errors::InvalidArgument("input is already initialized"));
    AllocatorAttributes attr;
    attr.set_gpu_compatible(true);
    attr.set_nic_compatible(true);
    PersistentTensor out_persistent;
    Tensor* out_tensor = nullptr;
    OP_REQUIRES_OK(
        ctx, ctx->allocate_persistent(input.dtype(), input.shape(),
                                      &out_persistent, &out_tensor, attr));
    functor::TensorSetZero<Device, T>()(ctx->eigen_device<Device>(),
                                        out_tensor->flat<T>());
    ctx->replace_ref_input(0, *out_tensor, true);
    // we always return the input ref.
    ctx->forward_ref_input_to_ref_output(0, 0);
  }
};

#define REGISTER_KERNELS(D, T)                                           \
  REGISTER_KERNEL_BUILDER(                                               \
      Name("ZeroInitializer").Device(DEVICE_##D).TypeConstraint<T>("T"), \
      ZeroInitializerOp<D##Device, T>);
#define REGISTER_CPU_KERNELS(T) REGISTER_KERNELS(CPU, T);
TF_CALL_REAL_NUMBER_TYPES(REGISTER_CPU_KERNELS);
#undef REGISTER_CPU_KERNELS

#if GOOGLE_CUDA
namespace functor {
#define DECLARE_GPU_SPEC(T)                                                 \
  template <>                                                               \
  void TensorSetZero<GPUDevice, T>::operator()(const GPUDevice& d,          \
                                               typename TTypes<T>::Flat t); \
  extern template struct TensorSetZero<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);
#undef DECLARE_GPU_SPEC
}  // namespace functor

#define REGISTER_GPU_KERNELS(T) REGISTER_KERNELS(GPU, T);
TF_CALL_GPU_NUMBER_TYPES(REGISTER_GPU_KERNELS);
#undef REGISTER_GPU_KERNELS
#endif  // GOOGLE_CUDA

#undef REGISTER_KERNELS

}  // namespace tensorflow
