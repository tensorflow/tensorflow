// Copyright 2016 The Sonnet Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or  implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// =============================================================================

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/contrib/resampler/kernels/resampler_ops.h"

#include <stdio.h>
#include <cmath>

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace {

#define GET_DATA_POINT(x, y)                   \
  data[batch_id * data_batch_stride +          \
       data_channels * (y * data_width + x) +  \
       chan]

template <typename T>
__global__ void Resampler2DKernel(const T* __restrict__ data,
                                  const T* __restrict__ warp,
                                  T* __restrict__ output,
                                  const int batch_size,
                                  const int data_height,
                                  const int data_width,
                                  const int data_channels,
                                  const int num_sampling_points) {
  const int output_data_size = batch_size * num_sampling_points * data_channels;
  CUDA_1D_KERNEL_LOOP(index, output_data_size) {
    const int out_index = index;

    // Get (idxSample, channel, point) from the index.
    // Use this formula
    //   index = batch_id * num_sampling_points * num_chans +
    //           sample_id * num_chans + chan_id,
    // with sample_id = [0, ... ,num_sampling_points)
    const int data_batch_stride = data_height * data_width * data_channels;
    const int warp_batch_stride = num_sampling_points * 2;
    const int output_batch_stride = num_sampling_points * data_channels;

    const int batch_id = index / output_batch_stride;
    const int index_in_batch = index % output_batch_stride;
    const int chan = index_in_batch % data_channels;
    const int sample_id = index_in_batch / data_channels;

    // Get coords of 2D point where data will be resampled
    const T x = warp[batch_id * warp_batch_stride + sample_id * 2];
    const T y = warp[batch_id * warp_batch_stride + sample_id * 2 + 1];
    const T zero = static_cast<T>(0.0);
    const T one = static_cast<T>(1.0);
    // The interpolation function:
    // a) implicitly pads the input data with 0s (hence the unusual checks
    // with {x,y} > -1)
    // b) returns 0 when sampling outside the (padded) image.
    // The effect is that the sampled signal smoothly goes to 0 outside
    // the original input domain, rather than presenting a jump
    // discontinuity at the image boundaries.
    if (x > static_cast<T>(-1.0) &&
        y > static_cast<T>(-1.0) &&
        x < static_cast<T>(data_width) &&
        y < static_cast<T>(data_height)) {
      // Precompute floor (f) and ceil (c) values for x and y.
      const int fx = std::floor(static_cast<float>(x));
      const int fy = std::floor(static_cast<float>(y));
      const int cx = fx + 1;
      const int cy = fy + 1;
      const T dx = static_cast<T>(cx) - x;
      const T dy = static_cast<T>(cy) - y;

      const T img_fxfy = (fx >= 0 && fy >= 0)
                         ? dx * dy * GET_DATA_POINT(fx, fy)
                         : zero;

      const T img_cxcy = (cx <= data_width - 1 && cy <= data_height - 1)
                         ? (one - dx) * (one - dy) * GET_DATA_POINT(cx, cy)
                         : zero;

      const T img_fxcy = (fx >= 0 && cy <= data_height - 1)
                         ? dx * (one - dy) * GET_DATA_POINT(fx, cy)
                         : zero;

      const T img_cxfy = (cx <= data_width - 1 && fy >= 0)
                         ? (one - dx) * dy * GET_DATA_POINT(cx, fy)
                         : zero;

      output[out_index] = img_fxfy + img_cxcy + img_fxcy + img_cxfy;
    } else {
      output[out_index] = zero;
    }
  }
}

}  // namespace

namespace functor {

template <typename T>
struct Resampler2DFunctor<GPUDevice, T>{
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const GPUDevice& d,
                   const T* __restrict__ data,
                   const T* __restrict__ warp,
                   T* __restrict__ output,
                   const int batch_size,
                   const int data_height,
                   const int data_width,
                   const int data_channels,
                   const int num_sampling_points) {
  const int output_data_size = batch_size * num_sampling_points * data_channels;
  ::tensorflow::CudaLaunchConfig config =
      ::tensorflow::GetCudaLaunchConfig(output_data_size, d);
  Resampler2DKernel<T>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          data, warp, output, batch_size, data_height, data_width,
          data_channels, num_sampling_points);
  }
};

// TODO(fviola): gcudacc fails at compile time with Eigen::half.
// template struct Resampler2DFunctor<GPUDevice, Eigen::half>;
template struct Resampler2DFunctor<GPUDevice, float>;
template struct Resampler2DFunctor<GPUDevice, double>;

}  // namespace functor

namespace {

#define UPDATE_GRAD_DATA_POINT(x, y, v)                  \
  atomicAdd(grad_data + (batch_id * data_batch_stride +  \
            data_channels * (y * data_width + x) +       \
            chan),                                       \
            v)


template <typename T>
__global__ void ResamplerGrad2DKernel(const T* __restrict__ data,
                                      const T* __restrict__ warp,
                                      const T* __restrict__ grad_output,
                                      T* __restrict__ grad_data,
                                      T* __restrict__ grad_warp,
                                      const int batch_size,
                                      const int data_height,
                                      const int data_width,
                                      const int data_channels,
                                      const int num_sampling_points) {
  const int resampler_output_size = batch_size * num_sampling_points *
      data_channels;
  CUDA_1D_KERNEL_LOOP(index, resampler_output_size) {
    const int out_index = index;

    // Get (idxSample, channel, point) from the index.
    // Use this formula
    //   index = batch_id * num_sampling_points * num_chans +
    //           sample_id * num_chans + chan_id,
    // with sample_id = [0, ... ,num_sampling_points)
    const int data_batch_stride = data_height * data_width * data_channels;
    const int warp_batch_stride = num_sampling_points * 2;
    const int output_batch_stride = num_sampling_points * data_channels;

    const int batch_id = index / output_batch_stride;
    const int index_in_batch = index % output_batch_stride;
    const int chan = index_in_batch % data_channels;
    const int sample_id = index_in_batch / data_channels;

    // Get coords of 2D point where data will be resampled
    const int warp_id_x = batch_id * warp_batch_stride + sample_id * 2;
    const int warp_id_y = warp_id_x + 1;
    const T x = warp[warp_id_x];
    const T y = warp[warp_id_y];
    const T zero = static_cast<T>(0.0);
    const T one = static_cast<T>(1.0);

    // Get grad output
    const T grad_output_value = grad_output[out_index];
    // The interpolation function whose gradient this kernel implements:
    // a) implicitly pads the input data with 0s (hence the unusual checks
    // with {x,y} > -1)
    // b) returns 0 when sampling outside the (padded) image.
    // The effect is that the sampled signal smoothly goes to 0 outside
    // the original input domain, rather than presenting a jump
    // discontinuity at the image boundaries.
    if (x > static_cast<T>(-1.0) &&
        y > static_cast<T>(-1.0) &&
        x < static_cast<T>(data_width) &&
        y < static_cast<T>(data_height)) {
      // Precompute floor (f) and ceil (c) values for x and y.
      const int fx = std::floor(static_cast<float>(x));
      const int fy = std::floor(static_cast<float>(y));
      const int cx = fx + 1;
      const int cy = fy + 1;
      const T dx = static_cast<T>(cx) - x;
      const T dy = static_cast<T>(cy) - y;

      const T img_fxfy = (fx >= 0 && fy >= 0)
                         ? GET_DATA_POINT(fx, fy)
                         : zero;

      const T img_cxcy = (cx <= data_width - 1 && cy <= data_height - 1)
                         ? GET_DATA_POINT(cx, cy)
                         : zero;

      const T img_fxcy = (fx >= 0 && cy <= data_height - 1)
                         ? GET_DATA_POINT(fx, cy)
                         : zero;

      const T img_cxfy = (cx <= data_width - 1 && fy >= 0)
                         ? GET_DATA_POINT(cx, fy)
                         : zero;

      // Update partial gradients wrt relevant warp field entries
      atomicAdd(grad_warp + warp_id_x,
                grad_output_value * ((one - dy) * (img_cxcy - img_fxcy) +
                                     dy * (img_cxfy - img_fxfy)));
      atomicAdd(grad_warp + warp_id_y,
                grad_output_value * ((one - dx) * (img_cxcy - img_cxfy) +
                                     dx * (img_fxcy - img_fxfy)));

      // Update partial gradients wrt sampled data
      if (fx >= 0 && fy >= 0) {
        UPDATE_GRAD_DATA_POINT(fx, fy, grad_output_value * dx * dy);
      }
      if (cx <= data_width - 1 && cy <= data_height - 1) {
        UPDATE_GRAD_DATA_POINT(cx, cy,
                               grad_output_value  * (one - dx) * (one - dy));
      }
      if (fx >= 0 && cy <= data_height - 1) {
        UPDATE_GRAD_DATA_POINT(fx, cy, grad_output_value * dx * (one - dy));
      }
      if (cx <= data_width - 1 && fy >= 0) {
        UPDATE_GRAD_DATA_POINT(cx, fy, grad_output_value * (one - dx) * dy);
      }
    }
  }
}

#undef GET_DATA_POINT
#undef UPDATE_GRAD_DATA_POINT

}  // namespace

namespace functor {

template <typename T>
struct ResamplerGrad2DFunctor<GPUDevice, T>{
  void operator ()(::tensorflow::OpKernelContext* ctx,
                   const GPUDevice& d,
                   const T* __restrict__ data,
                   const T* __restrict__ warp,
                   const T* __restrict__ grad_output,
                   T* __restrict__ grad_data,
                   T* __restrict__ grad_warp,
                   const int batch_size,
                   const int data_height,
                   const int data_width,
                   const int data_channels,
                   const int num_sampling_points) {
  // Set gradients to 0, because the kernel incrementally updates the
  // tensor entries by adding partial contributions.
  const int grad_warp_size = batch_size * num_sampling_points * 2;
  const int grad_data_size = batch_size * data_height * data_width *
      data_channels;

  ::tensorflow::CudaLaunchConfig config =
     ::tensorflow::GetCudaLaunchConfig(grad_warp_size, d);
  ::tensorflow::SetZero
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          grad_warp_size, grad_warp);

  config = ::tensorflow::GetCudaLaunchConfig(grad_data_size, d);
  ::tensorflow::SetZero
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          grad_data_size, grad_data);

  const int resampler_output_size = batch_size * num_sampling_points *
      data_channels;
  config = ::tensorflow::GetCudaLaunchConfig(resampler_output_size, d);
  ResamplerGrad2DKernel<T>
      <<<config.block_count, config.thread_per_block, 0, d.stream()>>>(
          data, warp, grad_output, grad_data, grad_warp, batch_size,
          data_height, data_width, data_channels, num_sampling_points);
  }
};

template struct ResamplerGrad2DFunctor<GPUDevice, float>;

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
