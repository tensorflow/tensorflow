/* Copyright 2015 Google Inc. All Rights Reserved.

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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/spacetobatch_op.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void S2B(
    const int32 nthreads, const T* input_ptr,
    const int block_size, const int pad_top, const int pad_left,
    const int output_batch, const int output_height, const int output_width,
    const int depth, const int input_batch, const int input_height,
    const int input_width, T* output_ptr) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = d + depth * (w + output_width * (h + output_height * b))
    const int d = out_idx % depth;
    const int out_idx2 = out_idx / depth;
    const int w = out_idx2 % output_width;
    const int out_idx3 = out_idx2 / output_width;
    const int h = out_idx3 % output_height;
    const int b = out_idx3 / output_height;

    const int in_b = b % input_batch;
    const int offset_w = (b / input_batch) % block_size;
    const int offset_h = (b / input_batch) / block_size;
    const int in_h = h * block_size + offset_h - pad_top;
    const int in_w = w * block_size + offset_w - pad_left;

    if (in_h >= 0 && in_w >= 0 && in_h < input_height && in_w < input_width) {
      const int inp_idx =
          d + depth * (in_w + input_width * (in_h + input_height * in_b));
      output_ptr[out_idx] = ldg(input_ptr + inp_idx);
    } else {
      output_ptr[out_idx] = static_cast<T>(0);
    }
  }
}

// Specialization of SpaceToBatchOpFunctor for a GPUDevice.
namespace functor {
template <typename T>
struct SpaceToBatchOpFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<int32>::ConstMatrix paddings,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int output_batch = output.dimension(0);
    const int output_height = output.dimension(1);
    const int output_width = output.dimension(2);
    const int depth = output.dimension(3);

    const int input_batch = input.dimension(0);
    const int input_height = input.dimension(1);
    const int input_width = input.dimension(2);

    const int pad_top = paddings(0, 0);
    const int pad_left = paddings(1, 0);

    const int total_count =
        output_batch * output_height * output_width * depth;
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
    S2B<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, input.data(), block_size, pad_top,
        pad_left, output_batch, output_height, output_width, depth,
        input_batch, input_height, input_width, output.data());
  }
};
}  // end namespace functor

// Instantiate the GPU implementation.
template struct functor::SpaceToBatchOpFunctor<GPUDevice, float>;
template struct functor::SpaceToBatchOpFunctor<GPUDevice, double>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
