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

#include "tensorflow/core/kernels/depthtospace_op.h"

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {
namespace {

using GPUDevice = Eigen::GpuDevice;

// Depth2Space kernel for FORMAT_NHWC.
// See 'depthtospace_op.h' for a more detailed description.
template <typename dtype>
__global__ void D2S_NHWC(const int32 nthreads,
                         const dtype* __restrict__ input_ptr,
                         const int block_size, const int batch_size,
                         const int input_height, const int input_width,
                         const int input_depth, const int output_height,
                         const int output_width, const int output_depth,
                         dtype* __restrict__ output_ptr) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = d + output_depth * (w + output_width * (h + output_height * b))
    const int d = out_idx % output_depth;
    const int out_idx2 = out_idx / output_depth;
    const int w = out_idx2 % output_width;
    const int out_idx3 = out_idx2 / output_width;
    const int h = out_idx3 % output_height;
    const int b = out_idx3 / output_height;

    const int in_h = h / block_size;
    const int offset_h = h % block_size;
    const int in_w = w / block_size;
    const int offset_w = w % block_size;
    const int offset_d = (offset_h * block_size + offset_w) * output_depth;
    const int in_d = d + offset_d;
    const int inp_idx =
        in_d + input_depth * (in_w + input_width * (in_h + input_height * b));
    *(output_ptr + out_idx) = ldg(input_ptr + inp_idx);
  }
}

// Depth2Space kernel for FORMAT_NCHW.
// See 'spacetodepth_op.h' for a more detailed description.
template <typename dtype>
__global__ void D2S_NCHW(const int32 nthreads,
                         const dtype* __restrict__ input_ptr,
                         const int block_size, const int input_width,
                         const int output_depth_by_input_height,
                         dtype* __restrict__ output_ptr) {
  // TODO(pauldonnelly): Implement more optimized kernels.
  CUDA_1D_KERNEL_LOOP(input_idx, nthreads) {
    // We will be converting the image from ordering:
    // n, bY, bX, oC, iY, iX    (== input_idx)   to
    // n, oC, iY, bY, iX, bX

    // Start reading the input data straight away since we know the address.
    // We calculate the output address in parallel while this is being fetched.

    const int n_bY_bX_oC_iY = input_idx / input_width;
    const int iX = input_idx - n_bY_bX_oC_iY * input_width;

    const int n_bY_bX = n_bY_bX_oC_iY / output_depth_by_input_height;
    const int oC_iY = n_bY_bX_oC_iY - n_bY_bX * output_depth_by_input_height;

    const int n_bY = n_bY_bX / block_size;
    const int bX = n_bY_bX - n_bY * block_size;

    const int n = n_bY / block_size;
    const int bY = n_bY - n * block_size;

    const int output_idx =
        bX +
        block_size *
            (iX + input_width *
                      (bY + block_size *
                                (oC_iY + n * output_depth_by_input_height)));

    *(output_ptr + output_idx) = ldg(input_ptr + input_idx);
  }
}

}  // namespace

// Specialization of DepthToSpaceOpFunctor for a GPUDevice.
namespace functor {

template <typename T>
struct DepthToSpaceOpFunctor<GPUDevice, T, FORMAT_NHWC> {
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
        batch_size * output_height * output_width * output_depth;
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);
    D2S_NHWC<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, input.data(), block_size, batch_size,
        input_height, input_width, input_depth, output_height, output_width,
        output_depth, output.data());
  }
};

template <typename T>
struct DepthToSpaceOpFunctor<GPUDevice, T, FORMAT_NCHW> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  int block_size, typename TTypes<T, 4>::Tensor output) {
    const int batch_size = input.dimension(0);
    const int input_depth = input.dimension(1);
    const int input_height = input.dimension(2);
    const int input_width = input.dimension(3);
    const int output_depth = output.dimension(1);
    const int total_count =
        batch_size * input_height * input_width * input_depth;
    auto config = GetCudaLaunchConfig(total_count, d);

    D2S_NCHW<<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
        config.virtual_thread_count, input.data(), block_size, input_width,
        output_depth * input_height, output.data());
  }
};
}  // end namespace functor

// Instantiate the GPU implementations for float.
template struct functor::DepthToSpaceOpFunctor<GPUDevice, float, FORMAT_NCHW>;
template struct functor::DepthToSpaceOpFunctor<GPUDevice, float, FORMAT_NHWC>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
