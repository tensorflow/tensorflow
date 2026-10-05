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

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/kernels/roll_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
__global__ void RollKernel(const int64_t nthreads, const int32_t num_dims,
                           const T* __restrict__ input, T* __restrict__ output,
                           const int32_t* __restrict__ dim_size,
                           const int32_t* __restrict__ shifts,
                           const int64_t* __restrict__ strides) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads, int64_t) {
    int64_t offset = 0;
    for (int i = 0; i < num_dims; i++) {
      const int64_t stride = strides[i];
      const int64_t shift = shifts[i];
      const int64_t indx = (out_idx / stride) % dim_size[i];
      const int64_t shifted_indx = (indx + shift) % dim_size[i];
      offset += (shifted_indx - indx) * stride;
    }
    output[out_idx + offset] = input[out_idx];
  }
}
}  // namespace

namespace functor {

template <typename T>
struct Roll<GPUDevice, T> {
  void operator()(OpKernelContext* context, const int64_t num_elements,
                  const int num_dims, const absl::Span<const int32_t> dim_size,
                  const T* input, T* output,
                  const absl::Span<const int32_t> threshold,
                  const absl::Span<const int64_t> dim_range,
                  const int64_t isd) {
    if (!num_elements) return;
    const GPUDevice& d = context->eigen_device<GPUDevice>();

    auto config_or = GetGpuLaunchConfig64(num_elements, d);
    OP_REQUIRES_OK(context, config_or.status());
    const GpuLaunchConfig64& cfg = *config_or;

    std::vector<int32_t> shifts(num_dims);
    std::vector<int64_t> strides(num_dims);
    for (int i = 0; i < num_dims; ++i) {
      shifts[i] = dim_size[i] - threshold[i];
      strides[i] = dim_range[i] / dim_size[i];
    }

    Tensor dim_tensor;
    Tensor shift_tensor;
    Tensor stride_tensor;
    OP_REQUIRES_OK(
        context,
        context->allocate_temp(DT_INT32, TensorShape({num_dims}), &dim_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT32, TensorShape({num_dims}),
                                          &shift_tensor));
    OP_REQUIRES_OK(context,
                   context->allocate_temp(DT_INT64, TensorShape({num_dims}),
                                          &stride_tensor));
    auto* dim_buf = dim_tensor.flat<int32_t>().data();
    auto* shift_buf = shift_tensor.flat<int32_t>().data();
    auto* stride_buf = stride_tensor.flat<int64_t>().data();
    d.memcpyHostToDevice(dim_buf, dim_size.data(), dim_tensor.TotalBytes());
    d.memcpyHostToDevice(shift_buf, shifts.data(), shift_tensor.TotalBytes());
    d.memcpyHostToDevice(stride_buf, strides.data(),
                         stride_tensor.TotalBytes());

    OP_REQUIRES_OK(
        context,
        GpuLaunchKernel(RollKernel<T>, cfg.block_count, cfg.thread_per_block, 0,
                        d.stream(), cfg.virtual_thread_count, num_dims, input,
                        output, dim_buf, shift_buf, stride_buf));
  }
};

#define DEFINE_GPU_SPECS(T) template struct Roll<GPUDevice, T>;

TF_CALL_int32(DEFINE_GPU_SPECS);
TF_CALL_int64(DEFINE_GPU_SPECS);
TF_CALL_uint32(DEFINE_GPU_SPECS);
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);
TF_CALL_COMPLEX_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS
}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
