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

// Space2Depth kernel for FORMAT_NHWC.
// See 'spacetodepth_op.h' for a more detailed description.
template <typename dtype>
__global__ void S2D_NHWC(const int32 nthreads, const dtype* input_ptr,
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

// Space2Depth kernel for FORMAT_NCHW.
// See 'spacetodepth_op.h' for a more detailed description.
template <typename dtype>
__global__ void S2D_NCHW(const int32 nthreads,
                         const dtype* __restrict__ input_ptr,
                         const int block_size, const int output_width,
                         const int input_depth_by_output_height,
                         dtype* __restrict__ output_ptr) {
  // TODO(pauldonnelly): This kernel gets input coalescing, but not output
  // coalescing. We could use shared memory to get both. It may also help
  // to amortize the address calculations via an inner loop over block_size.
  // A template parameter for the block_size is another potential optimization.
  CUDA_1D_KERNEL_LOOP(input_idx, nthreads) {
    // We assume both the input and output are packed NCHW tensors.
    // input_idx represents an index within the flattened input tensor.
    // We can consider the block width and height as extra tensor dimensions,
    // then isolate the relevant components of input_idx and recombine them to
    // form output_idx. The layout transform performed is:
    // n, iC, oY, bY, oX, bX    (== input_idx)   to
    // n, bY, bX, iC, oY, oX    (== output_idx).

    const int n_iC_oY_bY_oX = input_idx / block_size;
    const int bX = input_idx - n_iC_oY_bY_oX * block_size;

    const int n_iC_oY_bY = n_iC_oY_bY_oX / output_width;
    const int oX = n_iC_oY_bY_oX - n_iC_oY_bY * output_width;

    const int n_iC_oY = n_iC_oY_bY / block_size;
    const int bY = n_iC_oY_bY - n_iC_oY * block_size;

    const int n = n_iC_oY / input_depth_by_output_height;
    const int iC_oY = n_iC_oY - n * input_depth_by_output_height;

    const int output_idx = oX + (((n * block_size + bY) * block_size + bX) *
                                     input_depth_by_output_height +
                                 iC_oY) *
                                    output_width;

    *(output_ptr + output_idx) = ldg(input_ptr + input_idx);
  }
}

// Specialization of SpaceToDepthOpFunctor for a CPUDevice.
namespace functor {
template <typename T>
struct SpaceToDepthOpFunctor<GPUDevice, T, FORMAT_NHWC> {
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
    S2D_NHWC<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, input.data(), block_size, batch_size,
        input_height, input_width, input_depth, output_height, output_width,
        output_depth, output.data());
  }
};

template <typename T>
struct SpaceToDepthOpFunctor<GPUDevice, T, FORMAT_NCHW> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int batch_size = output.dimension(0);
    const int input_depth = input.dimension(1);
    const int output_depth = output.dimension(1);
    const int output_height = output.dimension(2);
    const int output_width = output.dimension(3);

    const int total_count =
        batch_size * output_height * output_width * output_depth;
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
    S2D_NCHW<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, input.data(), block_size, output_width,
        input_depth * output_height, output.data());
  }
};
}  // end namespace functor

// Instantiate the GPU implementations for float.
template struct functor::SpaceToDepthOpFunctor<GPUDevice, float, FORMAT_NCHW>;
template struct functor::SpaceToDepthOpFunctor<GPUDevice, float, FORMAT_NHWC>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
