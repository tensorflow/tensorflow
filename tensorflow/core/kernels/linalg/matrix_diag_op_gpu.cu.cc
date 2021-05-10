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

#include "tensorflow/core/framework/register_types.h"
#include "tensorflow/core/kernels/linalg/matrix_diag_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

__device__ inline int ComputeContentOffset(const int diag_index,
                                           const int max_diag_len,
                                           const int num_rows,
                                           const int num_cols,
                                           const bool left_align_superdiagonal,
                                           const bool left_align_subdiagonal) {
  const bool left_align = (diag_index >= 0 && left_align_superdiagonal) ||
                          (diag_index <= 0 && left_align_subdiagonal);
  if (left_align) return 0;
  const int y_offset = min(0, diag_index);
  const int x_offset = max(0, diag_index);
  const int diag_len = min(num_rows + y_offset, num_cols - x_offset);
  return max_diag_len - diag_len;
}

template <typename T>
__global__ void MatrixDiagKernel(
    const int num_threads, const int num_rows, const int num_cols,
    const int num_diags, const int max_diag_len, const int lower_diag_index,
    const int upper_diag_index, const T padding_value,
    const bool left_align_superdiagonal, const bool left_align_subdiagonal,
    const T* diag_ptr, T* output_ptr) {
  GPU_1D_KERNEL_LOOP(index, num_threads) {
    const int batch_and_row_index = index / num_cols;
    const int col = index - batch_and_row_index * num_cols;
    const int batch = batch_and_row_index / num_rows;
    const int row = batch_and_row_index - batch * num_rows;
    const int diag_index = col - row;
    const int diag_index_in_input = upper_diag_index - diag_index;
    const int content_offset =
        ComputeContentOffset(diag_index, max_diag_len, num_rows, num_cols,
                             left_align_superdiagonal, left_align_subdiagonal);
    const int index_in_the_diagonal = col - max(diag_index, 0) + content_offset;
    if (lower_diag_index <= diag_index && diag_index <= upper_diag_index) {
      output_ptr[index] =
          diag_ptr[batch * num_diags * max_diag_len +
                   diag_index_in_input * max_diag_len + index_in_the_diagonal];
    } else {
      output_ptr[index] = padding_value;
    }
  }
}

template <typename T>
struct MatrixDiag<GPUDevice, T> {
  static void Compute(OpKernelContext* context, const GPUDevice& device,
                      typename TTypes<T>::ConstTensor& diag,
                      typename TTypes<T, 3>::Tensor& output,
                      const Eigen::Index lower_diag_index,
                      const Eigen::Index upper_diag_index,
                      const Eigen::Index max_diag_len, const T padding_value,
                      const bool left_align_superdiagonal,
                      const bool left_align_subdiagonal) {
    const int batch_size = output.dimension(0);
    const int num_rows = output.dimension(1);
    const int num_cols = output.dimension(2);
    const int num_diags = upper_diag_index - lower_diag_index + 1;
    if (batch_size == 0 || max_diag_len == 0 || num_rows == 0 ||
        num_cols == 0) {
      return;
    }
    GpuLaunchConfig config =
        GetGpuLaunchConfig(batch_size * num_rows * num_cols, device);
    TF_CHECK_OK(GpuLaunchKernel(
        MatrixDiagKernel<T>, config.block_count, config.thread_per_block, 0,
        device.stream(), config.virtual_thread_count, num_rows, num_cols,
        num_diags, max_diag_len, lower_diag_index, upper_diag_index,
        padding_value, left_align_superdiagonal, left_align_subdiagonal,
        diag.data(), output.data()));
  }
};

template <typename T>
__global__ void MatrixDiagPartKernel(
    const int num_threads, const int num_rows, const int num_cols,
    const int num_diags, const int max_diag_len, const int lower_diag_index,
    const int upper_diag_index, const T padding_value,
    const bool left_align_superdiagonal, const bool left_align_subdiagonal,
    const T* input_ptr, T* output_ptr) {
  GPU_1D_KERNEL_LOOP(index, num_threads) {
    const int batch_and_mapped_diag_index = index / max_diag_len;
    const int index_in_the_diagonal =
        index - batch_and_mapped_diag_index * max_diag_len;
    const int batch = batch_and_mapped_diag_index / num_diags;
    const int mapped_diag_index =
        batch_and_mapped_diag_index - batch * num_diags;
    const int diag_index = upper_diag_index - mapped_diag_index;
    const int content_offset =
        ComputeContentOffset(diag_index, max_diag_len, num_rows, num_cols,
                             left_align_superdiagonal, left_align_subdiagonal);
    const int y_offset = max(0, -diag_index);
    const int x_offset = max(0, diag_index);
    const int y_index = index_in_the_diagonal + y_offset - content_offset;
    const int x_index = index_in_the_diagonal + x_offset - content_offset;
    if (0 <= y_index && y_index < num_rows && 0 <= x_index &&
        x_index < num_cols) {
      output_ptr[index] =
          input_ptr[batch * num_rows * num_cols + y_index * num_cols + x_index];
    } else {
      output_ptr[index] = padding_value;
    }
  }
}

template <typename T>
struct MatrixDiagPart<GPUDevice, T> {
  static void Compute(OpKernelContext* context, const GPUDevice& device,
                      typename TTypes<T, 3>::ConstTensor& input,
                      typename TTypes<T>::Tensor& output,
                      const Eigen::Index lower_diag_index,
                      const Eigen::Index upper_diag_index,
                      const Eigen::Index max_diag_len, const T padding_value,
                      const bool left_align_superdiagonal,
                      const bool left_align_subdiagonal) {
    const int batch_size = input.dimension(0);
    const int num_rows = input.dimension(1);
    const int num_cols = input.dimension(2);
    const int num_diags = upper_diag_index - lower_diag_index + 1;
    if (batch_size == 0 || max_diag_len == 0 || num_rows == 0 ||
        num_cols == 0) {
      return;
    }
    GpuLaunchConfig config =
        GetGpuLaunchConfig(batch_size * num_diags * max_diag_len, device);
    TF_CHECK_OK(GpuLaunchKernel(
        MatrixDiagPartKernel<T>, config.block_count, config.thread_per_block, 0,
        device.stream(), config.virtual_thread_count, num_rows, num_cols,
        num_diags, max_diag_len, lower_diag_index, upper_diag_index,
        padding_value, left_align_superdiagonal, left_align_subdiagonal,
        input.data(), output.data()));
  }
};

#define DEFINE_GPU_SPEC(T)                  \
  template struct MatrixDiag<GPUDevice, T>; \
  template struct MatrixDiagPart<GPUDevice, T>;

TF_CALL_GPU_ALL_TYPES(DEFINE_GPU_SPEC);

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
