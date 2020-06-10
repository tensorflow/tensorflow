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
#include "tensorflow/core/kernels/matrix_set_diag_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// TODO(penporn): Merge this file with matrix_diag_op_gpu.cu.cc.
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

template <typename Scalar>
__global__ void MatrixSetDiagKernel(
    const int num_threads, const int m, const int n, const int num_diags,
    const int max_diag_len, const int upper_diag_index,
    const bool left_align_superdiagonal, const bool left_align_subdiagonal,
    const Scalar* __restrict__ diag_ptr, Scalar* __restrict__ output_ptr) {
  GPU_1D_KERNEL_LOOP(index, num_threads) {
    const int batch_and_diag_index = index / max_diag_len;
    int index_in_the_diagonal = index - batch_and_diag_index * max_diag_len;
    const int batch = batch_and_diag_index / num_diags;
    const int diag_index_in_input = batch_and_diag_index - batch * num_diags;
    const int diag_index = upper_diag_index - diag_index_in_input;
    index_in_the_diagonal -=
        ComputeContentOffset(diag_index, max_diag_len, m, n,
                             left_align_superdiagonal, left_align_subdiagonal);
    const int y_index = index_in_the_diagonal - min(0, diag_index);
    const int x_index = index_in_the_diagonal + max(0, diag_index);

    // Upper-bound checks for diagonals shorter than max_diag_len.
    if (index_in_the_diagonal >= 0 && y_index < m && x_index < n) {
      const int out_index = batch * m * n + y_index * n + x_index;
      output_ptr[out_index] = diag_ptr[index];
    }
  }
}

template <typename Scalar>
__global__ void MatrixCopyInputAndSetDiagKernel(
    const int num_threads, const int m, const int n, const int num_diags,
    const int max_diag_len, const int lower_diag_index,
    const int upper_diag_index, const bool left_align_superdiagonal,
    const bool left_align_subdiagonal, const Scalar* __restrict__ input_ptr,
    const Scalar* __restrict__ diag_ptr, Scalar* __restrict__ output_ptr) {
  GPU_1D_KERNEL_LOOP(index, num_threads) {
    const int batch_and_row_index = index / n;
    const int col = index - batch_and_row_index * n;
    const int batch = batch_and_row_index / m;
    const int row = batch_and_row_index - batch * m;
    const int diag_index = col - row;
    const int diag_index_in_input = upper_diag_index - diag_index;
    const int index_in_the_diagonal =
        col - max(0, diag_index) +
        ComputeContentOffset(diag_index, max_diag_len, m, n,
                             left_align_superdiagonal, left_align_subdiagonal);
    if (lower_diag_index <= diag_index && diag_index <= upper_diag_index) {
      output_ptr[index] =
          diag_ptr[batch * num_diags * max_diag_len +
                   diag_index_in_input * max_diag_len + index_in_the_diagonal];
    } else {
      output_ptr[index] = input_ptr[index];
    }
  }
}

template <typename Scalar>
struct MatrixSetDiag<GPUDevice, Scalar> {
  static void Compute(OpKernelContext* context, const GPUDevice& device,
                      typename TTypes<Scalar, 3>::ConstTensor& input,
                      typename TTypes<Scalar>::ConstTensor& diag,
                      typename TTypes<Scalar, 3>::Tensor& output,
                      const Eigen::Index lower_diag_index,
                      const Eigen::Index upper_diag_index,
                      const Eigen::Index max_diag_len,
                      const bool left_align_superdiagonal,
                      const bool left_align_subdiagonal) {
    const int batch_size = input.dimension(0);
    const int m = input.dimension(1);
    const int n = input.dimension(2);
    const int num_diags = upper_diag_index - lower_diag_index + 1;

    if (batch_size == 0 || max_diag_len == 0 || m == 0 || n == 0) return;
    if (input.data() == output.data()) {
      GpuLaunchConfig config =
          GetGpuLaunchConfig(batch_size * num_diags * max_diag_len, device);
      TF_CHECK_OK(GpuLaunchKernel(
          MatrixSetDiagKernel<Scalar>, config.block_count,
          config.thread_per_block, 0, device.stream(),
          config.virtual_thread_count, m, n, num_diags, max_diag_len,
          upper_diag_index, left_align_superdiagonal, left_align_subdiagonal,
          diag.data(), output.data()));
    } else {
      GpuLaunchConfig config = GetGpuLaunchConfig(batch_size * m * n, device);
      TF_CHECK_OK(GpuLaunchKernel(
          MatrixCopyInputAndSetDiagKernel<Scalar>, config.block_count,
          config.thread_per_block, 0, device.stream(),
          config.virtual_thread_count, m, n, num_diags, max_diag_len,
          lower_diag_index, upper_diag_index, left_align_superdiagonal,
          left_align_subdiagonal, input.data(), diag.data(), output.data()));
    }
  }
};

#define DEFINE_GPU_SPEC(T) template struct MatrixSetDiag<GPUDevice, T>;

TF_CALL_GPU_ALL_TYPES(DEFINE_GPU_SPEC);

}  // namespace functor
}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
