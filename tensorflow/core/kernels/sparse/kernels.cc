/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/sparse/kernels.h"

#include <cstdint>
#include <numeric>

#include "absl/status/status.h"
#include "unsupported/Eigen/CXX11/Tensor"  // from @eigen_archive
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace functor {

absl::Status SparseTensorToCSRSparseMatrixCPUFunctor::operator()(
    int64_t batch_size, int num_rows, int num_cols,
    TTypes<int64_t>::ConstMatrix indices, TTypes<int32>::Vec batch_ptr,
    TTypes<int32>::Vec csr_row_ptr, TTypes<int32>::Vec csr_col_ind) {
  // Validate inputs.
  if (batch_ptr.size() != batch_size + 1) {
    return errors::InvalidArgument(
        "Expected batch_ptr.size() == batch_size + 1. Got: ", batch_ptr.size(),
        " vs. ", batch_size + 1);
  }
  if (csr_row_ptr.size() != batch_size * (num_rows + 1)) {
    return errors::InvalidArgument(
        "Expected csr_row_ptr.size() == batch_size * (num_rows + 1). Got: ",
        csr_row_ptr.size(), " vs. ", batch_size * (num_rows + 1));
  }

  const int64_t total_nnz = indices.dimension(0);
  const int rank = indices.dimension(1);
  if (rank == 2 && batch_size != 1) {
    return errors::InvalidArgument(
        "Expected batch_size == 1 when rank is 2. Got batch_size: ",
        batch_size);
  }
  if (rank < 2 || rank > 3) {
    return errors::InvalidArgument(
        "Indices must have either 2 or 3 columns.  Got size ",
        indices.dimensions());
  }
  if (csr_col_ind.size() != total_nnz) {
    return errors::InvalidArgument(
        "Expected csr_col_ind.size() == total_nnz. Got: ", csr_col_ind.size(),
        " vs. ", total_nnz);
  }

  int prev_batch = -1;
  if (rank == 2) {
    // For a single batch, the batch_ptrs are {0, total_nnz}.
    batch_ptr(0) = 0;
    ++prev_batch;

    for (int64_t i = 0; i < total_nnz; ++i) {
      int64_t row = indices(i, 0);
      if (row < 0 || row >= num_rows) {
        return errors::InvalidArgument("Row index ", row,
                                       " is outside of valid range [0, ",
                                       num_rows, ")");
      }
      int64_t col = indices(i, 1);
      if (col < 0 || col >= num_cols) {
        return errors::InvalidArgument("Column index ", col,
                                       " is outside of valid range [0, ",
                                       num_cols, ")");
      }
      // For now, the rows pointers store the corresponding row counts.
      int64_t ix = row + 1;
      if (ix >= csr_row_ptr.size()) {
        return errors::InvalidArgument("Got an index ", ix,
                                       " that is outside of csr_row_ptr");
      }

      csr_row_ptr(ix) += 1;
      csr_col_ind(i) = col;
    }
  } else {  // rank == 3
    for (int64_t i = 0; i < total_nnz; ++i) {
      const int cur_batch = indices(i, 0);
      if (cur_batch < 0 || cur_batch >= batch_size) {
        return errors::InvalidArgument("Batch index ", cur_batch,
                                       " is outside of valid range [0, ",
                                       batch_size, ")");
      }
      int64_t row = indices(i, 1);
      if (row < 0 || row >= num_rows) {
        return errors::InvalidArgument("Row index ", row,
                                       " is outside of valid range [0, ",
                                       num_rows, ")");
      }
      int64_t col = indices(i, 2);
      if (col < 0 || col >= num_cols) {
        return errors::InvalidArgument("Column index ", col,
                                       " is outside of valid range [0, ",
                                       num_cols, ")");
      }

      // For now, the rows pointers store the corresponding row counts.
      int64_t ix = cur_batch * (num_rows + 1) + row + 1;
      if (ix >= csr_row_ptr.size()) {
        return errors::InvalidArgument("Got an index ", ix,
                                       " that is outside of csr_row_ptr");
      }
      csr_row_ptr(ix) += 1;
      csr_col_ind(i) = col;

      // We're at a new batch and might have skipped over empty batches.
      while (prev_batch < cur_batch) {
        // The previous batch ends at position i.
        batch_ptr(prev_batch + 1) = i;
        ++prev_batch;
      }
    }
  }
  // Set the last element of batch_ptr and account for trailing empty batches.
  while (prev_batch < batch_size) {
    batch_ptr(prev_batch + 1) = total_nnz;
    ++prev_batch;
  }

  // Compute the cumulative row counts for each batch.
  for (int batch_idx = 0; batch_idx < batch_size; ++batch_idx) {
    auto* row_ptr_batch = csr_row_ptr.data() + batch_idx * (num_rows + 1);
    std::partial_sum(row_ptr_batch, row_ptr_batch + num_rows + 1,
                     row_ptr_batch);
  }
  return absl::OkStatus();
}

}  // namespace functor
}  // namespace tensorflow
