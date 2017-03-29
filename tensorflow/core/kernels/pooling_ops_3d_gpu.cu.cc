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

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "tensorflow/core/kernels/pooling_ops_3d_gpu.h"
#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"
#include "tensorflow/core/util/tensor_format.h"

namespace tensorflow {

namespace {

template <typename dtype>
__global__ void MaxPoolGradBackwardNoMaskNCDHW(
    const int nthreads, const dtype* bottom_data, const dtype* output_data,
    const int pooled_plane, const int pooled_height, const int pooled_width,
    const int channels, const int plane, const int height, const int width,
    const int kernel_p, const int kernel_h, const int kernel_w,
    const int stride_p, const int stride_h, const int stride_w, const int pad_p,
    const int pad_t, const int pad_l, const dtype* top_diff,
    dtype* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // First find out the index to the maximum, since we have no mask.
    int pw = index % pooled_width;
    int ph = (index / pooled_width) % pooled_height;
    int pp = (index / pooled_width / pooled_height) % pooled_plane;
    int c = (index / pooled_width / pooled_height / pooled_plane) % channels;
    int n = (index / pooled_width / pooled_height / pooled_plane / channels);
    int pstart = pp * stride_p - pad_p;
    int hstart = ph * stride_h - pad_t;
    int wstart = pw * stride_w - pad_l;
    const int pend = min(pstart + kernel_p, plane);
    const int hend = min(hstart + kernel_h, height);
    const int wend = min(wstart + kernel_w, width);
    pstart = max(pstart, 0);
    hstart = max(hstart, 0);
    wstart = max(wstart, 0);
    bool should_stop = false;
    int maxidx = -1;
    const dtype* bottom_data_n =
        bottom_data + n * channels * plane * height * width;
    // Propagate only first value from top_diff corresponding to the maximum.
    for (int p = pstart; p < pend && !should_stop; ++p) {
      for (int h = hstart; h < hend && !should_stop; ++h) {
        for (int w = wstart; w < wend && !should_stop; ++w) {
          int idx = c * plane * height * width + (p * height + h) * width + w;
          if (output_data[index] == bottom_data_n[idx]) {
            maxidx = idx;
            should_stop = true;
          }
        }
      }
    }
    // Set the bottom diff (atomic is not necessary). The index could still be
    // uninitialized, if all the bottom_data are NaN.
    if (maxidx != -1) {
      bottom_diff[index] =
          top_diff[n * channels * plane * height * width + maxidx];
    }
  }
}

template <typename dtype>
__global__ void MaxPoolGradBackwardNoMaskNDHWC(
    const int nthreads, const dtype* bottom_data, const dtype* output_data,
    const int pooled_plane, const int pooled_height, const int pooled_width,
    const int channels, const int plane, const int height, const int width,
    const int kernel_p, const int kernel_h, const int kernel_w,
    const int stride_p, const int stride_h, const int stride_w, const int pad_p,
    const int pad_t, const int pad_l, const dtype* top_diff,
    dtype* bottom_diff) {
  CUDA_1D_KERNEL_LOOP(index, nthreads) {
    // First find out the index to the maximum, since we have no mask.
    int n = index;
    int c = n % channels;
    n /= channels;
    int wstart = (n % pooled_width) * stride_w - pad_l;
    int wend = min(wstart + kernel_w, width);
    wstart = max(wstart, 0);
    n /= pooled_width;
    int hstart = (n % pooled_height) * stride_h - pad_t;
    int hend = min(hstart + kernel_h, height);
    hstart = max(hstart, 0);
    n /= pooled_height;
    int pstart = (n % pooled_plane) * stride_p - pad_p;
    int pend = min(pstart + kernel_p, plane);
    pstart = max(pstart, 0);
    n /= pooled_plane;
    bool should_stop = false;
    int maxidx = -1;
    const dtype* bottom_data_n =
        bottom_data + n * plane * height * width * channels;
    // Propagate only first value from top_diff corresponding to the maximum.
    for (int p = pstart; p < pend && !should_stop; ++p) {
      for (int h = hstart; h < hend && !should_stop; ++h) {
        for (int w = wstart; w < wend && !should_stop; ++w) {
          int idx = ((p * height + h) * width + w) * channels + c;
          if (output_data[index] == bottom_data_n[idx]) {
            maxidx = idx;
            should_stop = true;
          }
        }
      }
    }
    // Set the bottom diff (atomic is not necessary). The index could still be
    // uninitialized, if all the bottom_data are NaN.
    if (maxidx != -1) {
      bottom_diff[index] =
          top_diff[n * plane * height * width * channels + maxidx];
    }
  }
}

}  // namespace

namespace functor {

template <typename T>
bool MaxPool3dGradBackward<T>::operator()(
    TensorFormat data_format, const T* bottom_data, const T* output_data,
    const int batch, const int pooled_plane, const int pooled_height,
    const int pooled_width, const int channels, const int plane,
    const int height, const int width, const int kernel_p, const int kernel_h,
    const int kernel_w, const int stride_p, const int stride_h,
    const int stride_w, const int pad_p, const int pad_t, const int pad_l,
    const T* top_diff, T* bottom_diff, const Eigen::GpuDevice& d) {
  int num_kernels =
      batch * channels * pooled_plane * pooled_height * pooled_width;
  CudaLaunchConfig config = GetCudaLaunchConfig(num_kernels, d);
  if (data_format == FORMAT_NHWC) {
    MaxPoolGradBackwardNoMaskNDHWC<<<config.block_count,
                                     config.thread_per_block, 0, d.stream()>>>(
        num_kernels, bottom_data, output_data, pooled_plane, pooled_height,
        pooled_width, channels, plane, height, width, kernel_p, kernel_h,
        kernel_w, stride_p, stride_h, stride_w, pad_p, pad_t, pad_l, top_diff,
        bottom_diff);
  } else {
    MaxPoolGradBackwardNoMaskNCDHW<<<config.block_count,
                                     config.thread_per_block, 0, d.stream()>>>(
        num_kernels, bottom_data, output_data, pooled_plane, pooled_height,
        pooled_width, channels, plane, height, width, kernel_p, kernel_h,
        kernel_w, stride_p, stride_h, stride_w, pad_p, pad_t, pad_l, top_diff,
        bottom_diff);
  }
  return d.ok();
};

}  // namespace functor

#define DEFINE_GPU_SPECS(T) \
  template struct functor::MaxPool3dGradBackward<T>;
TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);
#undef DEFINE_GPU_SPECS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
