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

#include "tensorflow/core/kernels/spacetodepth_op.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

template <typename dtype>
__global__ void S2D(const int32 nthreads, const dtype* input_ptr,
                    const int block_size, const int batch_size,
                    const int input_height, const int input_width,
                    const int input_depth, const int output_height,
                    const int output_width, const int output_depth,
                    dtype* output_ptr) {
  CUDA_1D_KERNEL_LOOP(inp_idx, nthreads) {
    // inp_idx = d + input_depth * (w + input_width * (h + input_height * b))
    const int d = inp_idx % input_depth;
    const int inp_idx2 = inp_idx / input_depth;
    const int w = inp_idx2 % input_width;
    const int inp_idx3 = inp_idx2 / input_width;
    const int h = inp_idx3 % input_height;
    const int b = inp_idx3 / input_height;

    const int out_h = h / block_size;
    const int offset_h = h % block_size;
    const int out_w = w / block_size;
    const int offset_w = w % block_size;
    const int offset_d = (offset_h * block_size + offset_w) * input_depth;
    const int out_d = d + offset_d;
    const int out_idx =
        out_d +
        output_depth * (out_w + output_width * (out_h + output_height * b));
    *(output_ptr + out_idx) = ldg(input_ptr + inp_idx);
  }
}

// Specialization of SpaceToDepthOpFunctor for a CPUDevice.
namespace functor {
template <typename T>
struct SpaceToDepthOpFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int input_height = input.dimension(1);
    const int input_width = input.dimension(2);
    const int input_depth = input.dimension(3);
    const int output_height = output.dimension(1);
    const int output_width = output.dimension(2);
    const int output_depth = output.dimension(3);

    const int total_count =
        batch_size * input_height * input_width * input_depth;
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
    S2D<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, input.data(), block_size, batch_size,
        input_height, input_width, input_depth, output_height, output_width,
        output_depth, output.data());
  }
};
}  // end namespace functor

// Instantiate the GPU implementation for float.
template struct functor::SpaceToDepthOpFunctor<GPUDevice, float>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
