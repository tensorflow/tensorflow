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

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM

#define EIGEN_USE_GPU

#include <stdio.h>

#include "tensorflow/core/kernels/resize_nearest_neighbor_op.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T, bool align_corners>
__global__ void ResizeNearestNeighborNHWC(
    const int nthreads, const T* bottom_data, const int in_height,
    const int in_width, const int channels, const int out_height,
    const int out_width, const float height_scale, const float width_scale,
    T* top_data) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int out_x = n % out_width;
    n /= out_width;
    int out_y = n % out_height;
    n /= out_height;

    const T* bottom_data_n = bottom_data + n * channels * in_height * in_width;
    const int in_y =
        min((align_corners) ? static_cast<int>(roundf(out_y * height_scale))
                            : static_cast<int>(floorf(out_y * height_scale)),
            in_height - 1);
    const int in_x =
        min((align_corners) ? static_cast<int>(roundf(out_x * width_scale))
                            : static_cast<int>(floorf(out_x * width_scale)),
            in_width - 1);
    const int idx = (in_y * in_width + in_x) * channels + c;
    top_data[index] = ldg(bottom_data_n + idx);
  }
}

template <typename T, bool align_corners>
__global__ void ResizeNearestNeighborBackwardNHWC(
    const int nthreads, const T* top_diff, const int in_height,
    const int in_width, const int channels, const int out_height,
    const int out_width, const float height_scale, const float width_scale,
    T* bottom_diff) {
  GPU_1D_KERNEL_LOOP(index, nthreads) {
    int n = index;
    int c = n % channels;
    n /= channels;
    int in_x = n % in_width;
    n /= in_width;
    int in_y = n % in_height;
    n /= in_height;

    T* bottom_diff_n = bottom_diff + n * channels * out_height * out_width;
    const int out_y =
        min((align_corners) ? static_cast<int>(roundf(in_y * height_scale))
                            : static_cast<int>(floorf(in_y * height_scale)),
            out_height - 1);
    const int out_x =
        min((align_corners) ? static_cast<int>(roundf(in_x * width_scale))
                            : static_cast<int>(floorf(in_x * width_scale)),
            out_width - 1);
    const int idx = (out_y * out_width + out_x) * channels + c;
    GpuAtomicAdd(bottom_diff_n + idx, ldg(top_diff + index));
  }
}

}  // namespace

namespace functor {

// Partial specialization of ResizeNearestNeighbor functor for a GPUDevice.
template <typename T, bool align_corners>
struct ResizeNearestNeighbor<GPUDevice, T, align_corners> {
  bool operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int64 in_height = input.dimension(1);
    const int64 in_width = input.dimension(2);
    const int channels = input.dimension(3);

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    const int output_size = batch_size * out_height * out_width * channels;
    if (output_size == 0) return true;

    GpuLaunchConfig config = GetGpuLaunchConfig(output_size, d);
    GPU_LAUNCH_KERNEL((ResizeNearestNeighborNHWC<T, align_corners>),
        dim3(config.block_count), dim3(config.thread_per_block), 0, d.stream(),
        output_size, input.data(),
        static_cast<int>(in_height),
        static_cast<int>(in_width),
        channels,
        static_cast<int>(out_height),
        static_cast<int>(out_width),
        height_scale, width_scale, output.data());
    return d.ok();
  }
};

#define DECLARE_GPU_SPEC(T)                                   \
  template struct ResizeNearestNeighbor<GPUDevice, T, false>; \
  template struct ResizeNearestNeighbor<GPUDevice, T, true>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

// Partial specialization of ResizeNearestNeighborGrad functor for a GPUDevice.
template <typename T, bool align_corners>
struct ResizeNearestNeighborGrad<GPUDevice, T, align_corners> {
  bool operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  const float height_scale, const float width_scale,
                  typename TTypes<T, 4>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int64 in_height = input.dimension(1);
    const int64 in_width = input.dimension(2);
    const int channels = input.dimension(3);

    const int64 out_height = output.dimension(1);
    const int64 out_width = output.dimension(2);

    const int output_size = batch_size * channels * out_height * out_width;

    GpuLaunchConfig output_config = GetGpuLaunchConfig(output_size, d);
    GPU_LAUNCH_KERNEL(SetZero,
        dim3(output_config.block_count), dim3(output_config.thread_per_block),
        0, d.stream(),
        output_size, output.data());
    if (!d.ok()) return false;

    const int input_size = batch_size * channels * in_height * in_width;
    if (input_size == 0) return true;

    GpuLaunchConfig input_config = GetGpuLaunchConfig(input_size, d);
    GPU_LAUNCH_KERNEL((ResizeNearestNeighborBackwardNHWC<T, align_corners>),
        dim3(input_config.block_count), dim3(input_config.thread_per_block), 0,
        d.stream(),
        input_config.virtual_thread_count, input.data(),
        static_cast<int>(in_height),
        static_cast<int>(in_width),
        channels,
        static_cast<int>(out_height),
        static_cast<int>(out_width),
        height_scale, width_scale,
        output.data());
    return d.ok();
  }
};

#define DECLARE_GPU_SPEC(T)                                       \
  template struct ResizeNearestNeighborGrad<GPUDevice, T, false>; \
  template struct ResizeNearestNeighborGrad<GPUDevice, T, true>;

TF_CALL_GPU_NUMBER_TYPES(DECLARE_GPU_SPEC);

#undef DECLARE_GPU_SPEC

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
