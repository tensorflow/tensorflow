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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/batchtospace_op.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename T>
__global__ void B2S(
    const int32 nthreads, const T* input_ptr,
    const int block_size, const int crop_top, const int crop_left,
    const int input_batch, const int input_height, const int input_width,
    const int depth, const int output_batch, const int output_height,
    const int output_width, T* output_ptr) {
  CUDA_1D_KERNEL_LOOP(inp_idx, nthreads) {
    // inp_idx = d + input_depth * (w + input_width * (h + input_height * b))
    const int d = inp_idx % depth;
    const int inp_idx2 = inp_idx / depth;
    const int w = inp_idx2 % input_width;
    const int inp_idx3 = inp_idx2 / input_width;
    const int h = inp_idx3 % input_height;
    const int b = inp_idx3 / input_height;

    const int out_b = b % output_batch;
    const int offset_w = (b / output_batch) % block_size;
    const int offset_h = (b / output_batch) / block_size;
    const int out_h = h * block_size + offset_h - crop_top;
    const int out_w = w * block_size + offset_w - crop_left;

    if (out_h >= 0 && out_w >= 0 &&
        out_h < output_height && out_w < output_width) {
      const int out_idx =
          d + depth * (out_w + output_width * (out_h + output_height * out_b));
      output_ptr[out_idx] = ldg(input_ptr + inp_idx);
    }
  }
}

// Specialization of BatchToSpaceOpFunctor for a GPUDevice.
namespace functor {
template <typename T>
struct BatchToSpaceOpFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<int32>::ConstMatrix crops,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int input_batch = input.dimension(0);
    const int input_height = input.dimension(1);
    const int input_width = input.dimension(2);
    const int depth = input.dimension(3);

    const int output_batch = output.dimension(0);
    const int output_height = output.dimension(1);
    const int output_width = output.dimension(2);

    const int crop_top = crops(0, 0);
    const int crop_left = crops(1, 0);

    const int total_count =
        input_batch * input_height * input_width * depth;
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
    B2S<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, input.data(), block_size, crop_top,
        crop_left, input_batch, input_height, input_width, depth,
        output_batch, output_height, output_width, output.data());
  }
};
}  // end namespace functor

// Instantiate the GPU implementation.
template struct functor::BatchToSpaceOpFunctor<GPUDevice, float>;
template struct functor::BatchToSpaceOpFunctor<GPUDevice, double>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
