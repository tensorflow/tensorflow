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

// See docs in ../ops/nn_ops.cc.

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include <cfloat>
#include <vector>

#include "tensorflow/core/kernels/dilation_ops.h"

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/util/cuda_kernel_helper.h"

namespace tensorflow {

typedef Eigen::GpuDevice GPUDevice;

namespace {

template <typename T>
__global__ void DilationKernel(const int32 nthreads, const T* input_ptr,
                               const T* filter_ptr, int batch, int input_rows,
                               int input_cols, int depth, int filter_rows,
                               int filter_cols, int output_rows,
                               int output_cols, int stride_rows,
                               int stride_cols, int rate_rows, int rate_cols,
                               int pad_top, int pad_left, T* output_ptr) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = d + depth * (w_out + output_cols * (h_out + output_rows * b))
    const int d = out_idx % depth;
    const int out_idx2 = out_idx / depth;
    const int w_out = out_idx2 % output_cols;
    const int out_idx3 = out_idx2 / output_cols;
    const int h_out = out_idx3 % output_rows;
    const int b = out_idx3 / output_rows;
    int h_beg = h_out * stride_rows - pad_top;
    int w_beg = w_out * stride_cols - pad_left;
    T cur_val = Eigen::NumTraits<T>::lowest();
    for (int h = 0; h < filter_rows; ++h) {
      const int h_in = h_beg + h * rate_rows;
      if (h_in >= 0 && h_in < input_rows) {
        for (int w = 0; w < filter_cols; ++w) {
          const int w_in = w_beg + w * rate_cols;
          if (w_in >= 0 && w_in < input_cols) {
            const T val =
                input_ptr[d + depth * (w_in +
                                       input_cols * (h_in + input_rows * b))] +
                filter_ptr[d + depth * (w + filter_cols * h)];
            if (val > cur_val) {
              cur_val = val;
            }
          }
        }
      }
    }
    output_ptr[out_idx] = cur_val;
  }
}

template <typename T>
__global__ void DilationBackpropInputKernel(
    const int32 nthreads, const T* input_ptr, const T* filter_ptr,
    const T* out_backprop_ptr, int batch, int input_rows, int input_cols,
    int depth, int filter_rows, int filter_cols, int output_rows,
    int output_cols, int stride_rows, int stride_cols, int rate_rows,
    int rate_cols, int pad_top, int pad_left, T* in_backprop_ptr) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = d + depth * (w_out + output_cols * (h_out + output_rows * b))
    const int d = out_idx % depth;
    const int out_idx2 = out_idx / depth;
    const int w_out = out_idx2 % output_cols;
    const int out_idx3 = out_idx2 / output_cols;
    const int h_out = out_idx3 % output_rows;
    const int b = out_idx3 / output_rows;
    int h_beg = h_out * stride_rows - pad_top;
    int w_beg = w_out * stride_cols - pad_left;
    T cur_val = Eigen::NumTraits<T>::lowest();
    int h_in_max = (h_beg < 0) ? 0 : h_beg;
    int w_in_max = (w_beg < 0) ? 0 : w_beg;
    // In the case of multiple argmax branches, we only back-propagate along the
    // last branch, i.e., the one with largest value of `h * filter_cols + w`,
    // similarly to the max-pooling backward routines.
    for (int h = 0; h < filter_rows; ++h) {
      const int h_in = h_beg + h * rate_rows;
      if (h_in >= 0 && h_in < input_rows) {
        for (int w = 0; w < filter_cols; ++w) {
          const int w_in = w_beg + w * rate_cols;
          if (w_in >= 0 && w_in < input_cols) {
            const T val =
                input_ptr[d + depth * (w_in +
                                       input_cols * (h_in + input_rows * b))] +
                filter_ptr[d + depth * (w + filter_cols * h)];
            if (val > cur_val) {
              cur_val = val;
              h_in_max = h_in;
              w_in_max = w_in;
            }
          }
        }
      }
    }
    CudaAtomicAdd(
        in_backprop_ptr + d +
            depth * (w_in_max + input_cols * (h_in_max + input_rows * b)),
        out_backprop_ptr[out_idx]);
  }
}

template <typename T>
__global__ void DilationBackpropFilterKernel(
    const int32 nthreads, const T* input_ptr, const T* filter_ptr,
    const T* out_backprop_ptr, int batch, int input_rows, int input_cols,
    int depth, int filter_rows, int filter_cols, int output_rows,
    int output_cols, int stride_rows, int stride_cols, int rate_rows,
    int rate_cols, int pad_top, int pad_left, T* filter_backprop_ptr) {
  CUDA_1D_KERNEL_LOOP(out_idx, nthreads) {
    // out_idx = d + depth * (w_out + output_cols * (h_out + output_rows * b))
    const int d = out_idx % depth;
    const int out_idx2 = out_idx / depth;
    const int w_out = out_idx2 % output_cols;
    const int out_idx3 = out_idx2 / output_cols;
    const int h_out = out_idx3 % output_rows;
    const int b = out_idx3 / output_rows;
    int h_beg = h_out * stride_rows - pad_top;
    int w_beg = w_out * stride_cols - pad_left;
    T cur_val = Eigen::NumTraits<T>::lowest();
    int h_max = 0;
    int w_max = 0;
    // In the case of multiple argmax branches, we only back-propagate along the
    // last branch, i.e., the one with largest value of `h * filter_cols + w`,
    // similarly to the max-pooling backward routines.
    for (int h = 0; h < filter_rows; ++h) {
      const int h_in = h_beg + h * rate_rows;
      if (h_in >= 0 && h_in < input_rows) {
        for (int w = 0; w < filter_cols; ++w) {
          const int w_in = w_beg + w * rate_cols;
          if (w_in >= 0 && w_in < input_cols) {
            const T val =
                input_ptr[d + depth * (w_in +
                                       input_cols * (h_in + input_rows * b))] +
                filter_ptr[d + depth * (w + filter_cols * h)];
            if (val > cur_val) {
              cur_val = val;
              h_max = h;
              w_max = w;
            }
          }
        }
      }
    }
    CudaAtomicAdd(
        filter_backprop_ptr + d + depth * (w_max + filter_cols * h_max),
        out_backprop_ptr[out_idx]);
  }
}

}  // namespace

namespace functor {

template <typename T>
struct Dilation<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter, int stride_rows,
                  int stride_cols, int rate_rows, int rate_cols, int pad_top,
                  int pad_left, typename TTypes<T, 4>::Tensor output) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = output.dimension(1);
    const int output_cols = output.dimension(2);

    const int total_count = batch * output_rows * output_cols * depth;
    CudaLaunchConfig config = GetCudaLaunchConfig(total_count, d);

    TF_CHECK_OK(CudaLaunchKernel(
        DilationKernel<T>, config.block_count, config.thread_per_block, 0,
        d.stream(), config.virtual_thread_count, input.data(), filter.data(),
        batch, input_rows, input_cols, depth, filter_rows, filter_cols,
        output_rows, output_cols, stride_rows, stride_cols, rate_rows,
        rate_cols, pad_top, pad_left, output.data()));
  }
};

template <typename T>
struct DilationBackpropInput<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor out_backprop,
                  int stride_rows, int stride_cols, int rate_rows,
                  int rate_cols, int pad_top, int pad_left,
                  typename TTypes<T, 4>::Tensor in_backprop) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = out_backprop.dimension(1);
    const int output_cols = out_backprop.dimension(2);

    int total_count;
    CudaLaunchConfig config;

    // Initialize in_backprop with all zeros.
    total_count = batch * input_rows * input_cols * depth;
    config = GetCudaLaunchConfig(total_count, d);
    TF_CHECK_OK(CudaLaunchKernel(SetZero<T>, config.block_count,
                                 config.thread_per_block, 0, d.stream(),
                                 total_count, in_backprop.data()));

    // Accumulate.
    total_count = batch * output_rows * output_cols * depth;
    config = GetCudaLaunchConfig(total_count, d);
    TF_CHECK_OK(CudaLaunchKernel(
        DilationBackpropInputKernel<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
        input.data(), filter.data(), out_backprop.data(), batch, input_rows,
        input_cols, depth, filter_rows, filter_cols, output_rows, output_cols,
        stride_rows, stride_cols, rate_rows, rate_cols, pad_top, pad_left,
        in_backprop.data()));
  }
};

template <typename T>
struct DilationBackpropFilter<GPUDevice, T> {
  void operator()(const GPUDevice& d, typename TTypes<T, 4>::ConstTensor input,
                  typename TTypes<T, 3>::ConstTensor filter,
                  typename TTypes<T, 4>::ConstTensor out_backprop,
                  int stride_rows, int stride_cols, int rate_rows,
                  int rate_cols, int pad_top, int pad_left,
                  typename TTypes<T, 3>::Tensor filter_backprop) {
    const int batch = input.dimension(0);
    const int input_rows = input.dimension(1);
    const int input_cols = input.dimension(2);
    const int depth = input.dimension(3);

    const int filter_rows = filter.dimension(0);
    const int filter_cols = filter.dimension(1);

    const int output_rows = out_backprop.dimension(1);
    const int output_cols = out_backprop.dimension(2);

    int total_count;
    CudaLaunchConfig config;

    // Initialize filter_backprop with all zeros.
    total_count = filter_rows * filter_cols * depth;
    config = GetCudaLaunchConfig(total_count, d);
    TF_CHECK_OK(CudaLaunchKernel(SetZero<T>, config.block_count,
                                 config.thread_per_block, 0, d.stream(),
                                 total_count, filter_backprop.data()));

    // Accumulate.
    total_count = batch * output_rows * output_cols * depth;
    config = GetCudaLaunchConfig(total_count, d);
    TF_CHECK_OK(CudaLaunchKernel(
        DilationBackpropFilterKernel<T>, config.block_count,
        config.thread_per_block, 0, d.stream(), config.virtual_thread_count,
        input.data(), filter.data(), out_backprop.data(), batch, input_rows,
        input_cols, depth, filter_rows, filter_cols, output_rows, output_cols,
        stride_rows, stride_cols, rate_rows, rate_cols, pad_top, pad_left,
        filter_backprop.data()));
  }
};

}  // namespace functor

#define DEFINE_GPU_SPECS(T)                                     \
  template struct functor::Dilation<GPUDevice, T>;              \
  template struct functor::DilationBackpropInput<GPUDevice, T>; \
  template struct functor::DilationBackpropFilter<GPUDevice, T>;

TF_CALL_GPU_NUMBER_TYPES(DEFINE_GPU_SPECS);

#undef DEFINE_GPU_SPECS

}  // namespace tensorflow

#endif  // GOOGLE_CUDA
