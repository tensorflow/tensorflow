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

#include <numeric>

#include "third_party/eigen3/unsupported/Eigen/CXX11/Tensor"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_types.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/errors.h"

namespace tensorflow {
namespace functor {

Status SparseTensorToCSRSparseMatrixCPUFunctor::operator()(
    const int64 batch_size, const int num_rows,
    TTypes<int64>::ConstMatrix indices, TTypes<int32>::Vec batch_ptr,
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

  const int64 total_nnz = indices.dimension(0);
  const int rank = indices.dimension(1);
  if (rank == 2 && batch_size != 1) {
    return errors::InvalidArgument(
        "Expected batch_size == 1 when rank is 2. Got batch_size: ",
        batch_size);
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
      // For now, the rows pointers store the corresponding row counts.
      int64_t ix = indices(i, 0) + 1;
      if (ix >= csr_row_ptr.size()) {
        return errors::InvalidArgument("Got an index ", ix,
                                       " that is outside of csr_row_ptr");
      }
      csr_row_ptr(indices(i, 0) + 1) += 1;
      csr_col_ind(i) = indices(i, 1);
    }
  } else {  // rank == 3
    for (int64_t i = 0; i < total_nnz; ++i) {
      const int cur_batch = indices(i, 0);
      // For now, the rows pointers store the corresponding row counts.
      csr_row_ptr(cur_batch * (num_rows + 1) + indices(i, 1) + 1) += 1;
      csr_col_ind(i) = indices(i, 2);

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
  return Status::OK();
}

}  // namespace functor
}  // namespace tensorflow
