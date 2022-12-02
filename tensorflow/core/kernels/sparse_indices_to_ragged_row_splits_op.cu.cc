/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/sparse_indices_to_ragged_row_splits_op.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"
#include "tensorflow/core/util/gpu_solvers.h"

namespace tensorflow {

using GPUDevice = Eigen::GpuDevice;

namespace functor {

namespace {

template <typename IndexType>
__global__ void SparseIndicesToRaggedRowSplitsKernel(
    const int num_nonzero, const bool validate_ragged_right,
    const IndexType* indices_flat_2d, const IndexType* dense_shape,
    IndexType* row_splits, int32_t* invalid_flag) {
  auto num_rows = dense_shape[0];
  auto num_cols = dense_shape[1];

  int kernel_index = blockIdx.x * blockDim.x + threadIdx.x;
  int kernel_count = blockDim.x * gridDim.x;
  int kernel_dim = (kernel_count == 1)
                       ? num_nonzero
                       : ((num_nonzero / (kernel_count - 1)) + 1);
  if (kernel_dim == 0) {
    kernel_dim = 1;
  }
  int start_index = kernel_dim * kernel_index;
  // In case of very unusual dimensions, return early if there is no work to do.
  if (start_index >= num_nonzero) {
    return;
  }
  int end_index = kernel_dim * (kernel_index + 1);
  if (end_index > num_nonzero) {
    end_index = num_nonzero;
  }
  IndexType prev_row =
      (kernel_index == 0) ? -1 : indices_flat_2d[(start_index - 1) * 2];
  IndexType prev_col = -1;

  IndexType n = start_index;
  if (validate_ragged_right && (n > 0)) {
    // If starting in the middle of the row, set the previous column idx for
    // comparison.
    prev_col = indices_flat_2d[2 * n - 1];
  }
  for (; n < end_index; ++n) {
    IndexType curr_row = indices_flat_2d[2 * n];
    if (validate_ragged_right) {
      // At the end of a row, check that the row index increased and is not
      // outside of the tensor's recorded dimensions.
      if (curr_row != prev_row) {
        if ((curr_row < prev_row) || (curr_row >= num_rows)) {
          GpuAtomicMax(invalid_flag, 1);
          return;
        }
        prev_col = -1;
      }
      // Within each row, check that each column index is one greater than the
      // last and that the first column index in the row is zero (checked
      // against prev_col + 1 where prev_col = -1). This ensures the tensor is
      // ragged-right and its indices are in order.
      IndexType curr_col = indices_flat_2d[2 * n + 1];
      if ((curr_col != prev_col + 1) || (curr_col >= num_cols)) {
        GpuAtomicMax(invalid_flag, 1);
        return;
      } else {
        prev_col = curr_col;
      }
    }
    // Fill in the row_splits with the current index; the loop will fill in any
    // needed entries for empty rows between the previous and current index.
    for (IndexType r = prev_row; r < curr_row; ++r) {
      row_splits[r + 1] = n;
    }
    prev_row = curr_row;
  }
  // At the end of the tensor, fill in the final row split and those for any
  // trailing empty rows.
  if ((start_index < num_nonzero) && (end_index == num_nonzero)) {
    for (IndexType r = prev_row; r < num_rows; ++r) {
      row_splits[r + 1] = n;
    }
  }
}

}  // namespace

template <typename IndexType>
struct SparseIndicesToRaggedRowSplitsFunctor<GPUDevice, IndexType> {
  Status operator()(OpKernelContext* context, int num_nonzero,
                    bool validate_ragged_right,
                    const IndexType* indices_flat_2d,
                    const IndexType* dense_shape, int32_t* invalid_flag) {
    auto num_rows = dense_shape[0];
    Tensor* output;
    TF_RETURN_IF_ERROR(context,
                       context->allocate_output(
                           "row_splits", TensorShape({num_rows + 1}), &output));
    IndexType* row_splits = output->flat<IndexType>().data();

    se::Stream* stream = context->op_device_context()->stream();
    if (!stream) return errors::Internal("No GPU stream available.");
    se::DeviceMemoryBase invalid_flag_gpu_memory(invalid_flag, sizeof(int32_t));
    if (!stream->ThenMemZero(&invalid_flag_gpu_memory, sizeof(int32_t)).ok()) {
      return errors::Internal("Failed to zero invalid_flag.");
    }

    const GPUDevice& d = context->eigen_gpu_device();
    GpuLaunchConfig config = GetGpuLaunchConfig(num_rows, d);
    int block_count = config.block_count;
    int thread_per_block = config.thread_per_block;

    return GpuLaunchKernel(SparseIndicesToRaggedRowSplitsKernel<IndexType>,
                           block_count, thread_per_block, 0, d.stream(),
                           num_nonzero, validate_ragged_right, indices_flat_2d,
                           dense_shape, row_splits, invalid_flag);
  }
};

template struct SparseIndicesToRaggedRowSplitsFunctor<GPUDevice, int32>;
template struct SparseIndicesToRaggedRowSplitsFunctor<GPUDevice, int64>;

}  // namespace functor

}  // namespace tensorflow

#endif  // GOOGLE_CUDA || TENSORFLOW_USE_ROCM
