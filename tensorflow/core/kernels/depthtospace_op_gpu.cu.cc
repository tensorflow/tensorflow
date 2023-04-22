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

#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/kernels/depthtospace_op.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

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
  GPU_1D_KERNEL_LOOP(out_idx, nthreads) {
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
  GPU_1D_KERNEL_LOOP(input_idx, nthreads) {
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

template <typename dtype, int block_size>
__global__ void D2S_NCHW_LOOP(const int32 nthreads,
                              const dtype* __restrict__ input,
                              const int input_width, const int output_width,
                              const int output_depth_by_input_area,
                              const int input_depth_by_input_area,
                              dtype* __restrict__ output) {
  GPU_1D_KERNEL_LOOP(thread_idx, nthreads) {
    // We will be converting the image from ordering:
    // n, bY, bX, oC, iY, iX   to
    // n, oC, iY, bY, iX, bX

    // We assume thread_idx encodes n_oC_iY_iX, and use an unrolled loop over
    // bY and bX coordinates within the block. This kernel is significantly
    // more performant than the D2S_NCHW kernel.
    //   A likely explanation of the improvement is that although both kernels
    // get input coalescing, this one would write the output data more densely
    // per warp, so would benefit assuming delayed cache writeback is used.

    const int n_oC_iY = thread_idx / input_width;
    const int iX = thread_idx - n_oC_iY * input_width;

    const int n = thread_idx / output_depth_by_input_area;
    const int oC_iY_iX = thread_idx - n * output_depth_by_input_area;

    // Recombine the components and apply to the input and output pointers.
    auto input_ptr = input + n * input_depth_by_input_area + oC_iY_iX;
    auto output_ptr = output + (n_oC_iY * output_width + iX) * block_size;

#pragma unroll
    // Copy a patch of data to the output batch image.
    for (int bY = 0; bY < block_size; ++bY) {
#pragma unroll
      for (int bX = 0; bX < block_size; ++bX) {
        output_ptr[bY * output_width + bX] = ldg(
            input_ptr + (bY * block_size + bX) * output_depth_by_input_area);
      }
    }
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
    if (total_count == 0) {
      return;
    }
    GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);
    TF_CHECK_OK(GpuLaunchKernel(
        D2S_NHWC<T>, config.block_count, config.thread_per_block, 0, d.stream(),
        config.virtual_thread_count, input.data(), block_size, batch_size,
        input_height, input_width, input_depth, output_height, output_width,
        output_depth, output.data()));
  }
  void operator()(const GPUDevice& d, typename TTypes<T, 5>::ConstTensor input,
                  int block_size, typename TTypes<T, 5>::Tensor output) {
    LOG(FATAL) << "5-D tensors should not be used with NHWC format";
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
    const int input_area = input_width * input_height;
    const int input_depth_by_input_area = input_depth * input_area;

    // We improve performance by generating instantiations of the loop kernel
    // for the most common block sizes.
    if (block_size <= 4) {
      const int output_width = output.dimension(3);
      const int output_depth_by_input_area = output_depth * input_area;
      const int total_count = batch_size * output_depth_by_input_area;
      if (total_count == 0) {
        return;
      }
      GpuLaunchConfig config = GetGpuLaunchConfig(total_count, d);
      switch (block_size) {
        case 2:
          TF_CHECK_OK(GpuLaunchKernel(
              D2S_NCHW_LOOP<T, 2>, config.block_count, config.thread_per_block,
              0, d.stream(), total_count, input.data(), input_width,
              output_width, output_depth_by_input_area,
              input_depth_by_input_area, output.data()));
          return;
        case 3:
          TF_CHECK_OK(GpuLaunchKernel(
              D2S_NCHW_LOOP<T, 3>, config.block_count, config.thread_per_block,
              0, d.stream(), total_count, input.data(), input_width,
              output_width, output_depth_by_input_area,
              input_depth_by_input_area, output.data()));
          return;
        case 4:
          TF_CHECK_OK(GpuLaunchKernel(
              D2S_NCHW_LOOP<T, 4>, config.block_count, config.thread_per_block,
              0, d.stream(), total_count, input.data(), input_width,
              output_width, output_depth_by_input_area,
              input_depth_by_input_area, output.data()));
          return;
      }
    }

    // Other block sizes are processed by the generic kernel.
    const int total_count = batch_size * input_depth_by_input_area;
    if (total_count == 0) {
      return;
    }
    auto config = GetGpuLaunchConfig(total_count, d);
    TF_CHECK_OK(GpuLaunchKernel(
        D2S_NCHW<T>, config.block_count, config.thread_per_block, 0, d.stream(),
        config.virtual_thread_count, input.data(), block_size, input_width,
        output_depth * input_height, output.data()));
  }
  void operator()(const GPUDevice& d, typename TTypes<T, 5>::ConstTensor input,
                  int block_size, typename TTypes<T, 5>::Tensor output) {
    LOG(FATAL) << "5-D tensors should not be used with NCHW format";
  }
};
}  // end namespace functor

// Instantiate the GPU implementations for float.
template struct functor::DepthToSpaceOpFunctor<GPUDevice, float, FORMAT_NCHW>;
template struct functor::DepthToSpaceOpFunctor<GPUDevice, float, FORMAT_NHWC>;

// Instantiate the GPU implementations for Eigen::half.
template struct functor::DepthToSpaceOpFunctor<GPUDevice, Eigen::half,
                                               FORMAT_NCHW>;
template struct functor::DepthToSpaceOpFunctor<GPUDevice, Eigen::half,
                                               FORMAT_NHWC>;

// NCHW_VECT_C with 4 x qint8 can be treated as NCHW int32.
template struct functor::DepthToSpaceOpFunctor<GPUDevice, int32, FORMAT_NCHW>;

}  // end namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
