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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/roll_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
__global__ void RollKernel(const int32 nthreads, const int32 num_dims,
                           const T* __restrict__ input, T* __restrict__ output,
                           const int32* __restrict__ dim_size,
                           const int32* __restrict__ threshold,
                           const int64* __restrict__ dim_range) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    int64 offset = 0;
    for (int i = 0; i < num_dims; i++) {
      const int64 stride = dim_range[i] / dim_size[i];
      const int shift = dim_size[i] - threshold[i];
      const int indx = (out_idx / stride) % dim_size[i];
      const int shifted_indx = (indx + shift) % dim_size[i];
      offset += (shifted_indx - indx) * stride;
    }
    output[out_idx + offset] = input[out_idx];
  }
}
}  // namespace

namespace functor {

template <typename T>
struct Roll<GPUDevice, T> {
  void operator()(const OpKernelContext* context, const int64 num_elements,
                  const int num_dims, const gtl::ArraySlice<int32> dim_size,
                  const T* input, T* output,
                  const gtl::ArraySlice<int32> threshold,
                  const gtl::ArraySlice<int64> dim_range, const int64 isd) {
    if (!num_elements) return;
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    auto dim_bytes = sizeof(int32) * dim_size.size();
    auto dim_buf = d.allocate(dim_bytes);

    auto thres_bytes = sizeof(int32) * threshold.size();
    auto thres_buf = d.allocate(thres_bytes);

    auto range_bytes = sizeof(int64) * dim_range.size();
    auto range_buf = d.allocate(range_bytes);

    d.memcpyHostToDevice(dim_buf, dim_size.data(), dim_bytes);
    d.memcpyHostToDevice(thres_buf, threshold.data(), thres_bytes);
    d.memcpyHostToDevice(range_buf, dim_range.data(), range_bytes);

    GpuLaunchConfig cfg = GetGpuLaunchConfig(num_elements, d);

    TF_CHECK_OK(GpuLaunchKernel(RollKernel<T>, cfg.block_count,
                                cfg.thread_per_block, 0, d.stream(),
                                cfg.virtual_thread_count, num_dims, input,
                                output, reinterpret_cast<const int32*>(dim_buf),
                                reinterpret_cast<const int32*>(thres_buf),
                                reinterpret_cast<const int64*>(range_buf)));

    d.deallocate(dim_buf);
    d.deallocate(thres_buf);
    d.deallocate(range_buf);
  }
};

#define DEFINE_GPU_SPECS(T) template struct Roll<GPUDevice, T>;

TF_CALL_int32(DEFINE_GPU_SPECS);
TF_CALL_int64(DEFINE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);
TF_CALL_complex64(DEFINE_GPU_SPECS);
TF_CALL_complex128(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
