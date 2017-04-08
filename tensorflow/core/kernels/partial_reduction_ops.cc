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

#include "tensorflow/core/kernels/partial_reduction_ops.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

template <typename Device, typename T, typename Index, T beginning(), T reduce(T,T)>
class PartialReduce : public OpKernel {
public:
  explicit PartialReduce(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    const Tensor &indices = context->input(1);
    const Tensor &data = context->input(0);
    TensorShape output_shape = data.shape();
    output_shape.set_dim(0,indices.shape().dim_size(0));
    Tensor *output = nullptr;
    OP_REQUIRES_OK(context, context->allocate_output(0, output_shape, &output));
    auto functor = functor::PartialReductionFunctor<Device, T, Index, beginning, reduce>();
    functor(context, context->eigen_device<Device>(), indices.matrix<Index>(), data.matrix<T>(), output->matrix<T>());
  }
};

#if GOOGLE_CUDA

#define REGISTER_GPU_PARTIAL_REDUCE_KERNELS(type, index_type)          \
  REGISTER_KERNEL_BUILDER(Name("PartialSum")                           \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          PartialReduce<GPUDevice, type, index_type,   \
                          functor::reduce_functions::zero<type>,               \
                          functor::reduce_functions::sum<type>>);          \
  REGISTER_KERNEL_BUILDER(Name("PartialProd")                          \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          PartialReduce<GPUDevice, type, index_type,   \
                          functor::reduce_functions::one<type>,                \
                          functor::reduce_functions::prod<type>>);         \
  REGISTER_KERNEL_BUILDER(Name("PartialMax")                           \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          PartialReduce<GPUDevice, type, index_type,   \
                          functor::reduce_functions::negative_infinity<type>,  \
                          functor::reduce_functions::max<type>>);   \
  REGISTER_KERNEL_BUILDER(Name("PartialMin")                           \
                              .Device(DEVICE_GPU)                      \
                              .TypeConstraint<type>("T")               \
                              .TypeConstraint<index_type>("Tindices"), \
                          PartialReduce<GPUDevice, type, index_type,   \
                          functor::reduce_functions::infinity<type>,   \
                          functor::reduce_functions::min<type>>);

#define REGISTER_GPU_PARTIAL_REDUCE_KERNELS_ALL(type) \
  REGISTER_GPU_PARTIAL_REDUCE_KERNELS(type, int32);   \
  REGISTER_GPU_PARTIAL_REDUCE_KERNELS(type, int64);

TF_CALL_REAL_NUMBER_TYPES(REGISTER_GPU_PARTIAL_REDUCE_KERNELS_ALL);

#undef REGISTER_GPU_PARTIAL_REDUCE_KERNELS
#undef REGISTER_GPU_PARTIAL_REDUCE_KERNELS_ALL

#endif  // GOOGLE_CUDA

}  // namespace tensorflow
