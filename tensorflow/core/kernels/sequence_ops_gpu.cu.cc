/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/sequence_ops.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace {

template <typename T>
__global__ void RangeKernel(int64_t size, T start, T delta,
                            T* __restrict__ output) {
  for (int64_t i : GpuGridRangeX(size)) {
    output[i] = start + static_cast<T>(i) * delta;
  }
}

}  // namespace

namespace functor {

template <typename T>
struct RangeFunctor<GPUDevice, T> {
  void operator()(OpKernelContext* context, int64_t size, T start, T delta,
                  typename TTypes<T>::Flat output) const {
    const GPUDevice& device = context->eigen_gpu_device();
    GpuLaunchConfig config = GetGpuLaunchConfig(
        size, device, &RangeKernel<T>,
        /*dynamic_shared_memory_size=*/0, /*block_size_limit=*/0);
    OP_REQUIRES_OK(context,
                   GpuLaunchKernel(RangeKernel<T>, config.block_count,
                                   config.thread_per_block, 0, device.stream(),
                                   size, start, delta, output.data()));
  }
};

}  // namespace functor

#define DEFINE_FUNCTOR(T) template struct functor::RangeFunctor<GPUDevice, T>;
TF_CALL_float(DEFINE_FUNCTOR);
TF_CALL_double(DEFINE_FUNCTOR);
TF_CALL_int32(DEFINE_FUNCTOR);
TF_CALL_int64(DEFINE_FUNCTOR);
#undef DEFINE_FUNCTOR

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
