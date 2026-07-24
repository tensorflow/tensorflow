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

#define EIGEN_USE_THREADS

#if GOOGLE_CUDA || TENSORFLOW_USE_ROCM
#define EIGEN_USE_GPU
#endif

#include <cstdint>

#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/kernels/sparse/sparse_matrix.h"

namespace tensorflow {

constexpr const char CSRSparseMatrix::kTypeName[];

absl::Status CSRSparseMatrix::ValidateComponentValues(
    const Tensor& dense_shape, const Tensor& batch_pointers,
    const Tensor& row_pointers, const Tensor& col_indices) {
  // dense_shape has already been checked to be an int64 vector of size 2 or 3
  // by ValidateTypesAndShapes, and the index arrays to be int32 vectors of the
  // matching sizes: batch_pointers -> batch_size + 1,
  // row_pointers -> batch_size * (num_rows + 1), col_indices -> total nnz.
  const auto dense_shape_vec = dense_shape.vec<int64_t>();
  const int rank = dense_shape.dim_size(0);
  const int64_t batch_size = (rank == 2) ? 1 : dense_shape_vec(0);
  const int64_t num_rows = (rank == 2) ? dense_shape_vec(0) : dense_shape_vec(1);
  const int64_t num_cols = (rank == 2) ? dense_shape_vec(1) : dense_shape_vec(2);
  if (batch_size < 0 || num_rows < 0 || num_cols < 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CSRSparseMatrix::Validate: dense_shape has a negative dimension: ",
        dense_shape.SummarizeValue(5)));
  }

  const int64_t total_nnz = col_indices.NumElements();
  const auto batch_ptr = batch_pointers.vec<int32_t>();
  const auto row_ptr = row_pointers.vec<int32_t>();
  const auto col_ind = col_indices.vec<int32_t>();

  // batch_pointers: 0, ..., total_nnz (non-decreasing offsets into the values).
  if (batch_ptr(0) != 0) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CSRSparseMatrix::Validate: batch_pointers[0] = ", batch_ptr(0),
        " but should be 0"));
  }
  for (int64_t b = 0; b < batch_size; ++b) {
    if (batch_ptr(b) < 0 || batch_ptr(b + 1) < batch_ptr(b)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CSRSparseMatrix::Validate: batch_pointers must be non-negative and "
          "non-decreasing, saw ",
          batch_ptr(b), " -> ", batch_ptr(b + 1), " at batch ", b));
    }
  }
  if (batch_ptr(batch_size) != total_nnz) {
    return absl::InvalidArgumentError(absl::StrCat(
        "CSRSparseMatrix::Validate: batch_pointers[batch_size] = ",
        batch_ptr(batch_size), " but should equal nnz = ", total_nnz));
  }

  // row_pointers: within each batch, 0, ..., nnz(batch) (non-decreasing).
  for (int64_t b = 0; b < batch_size; ++b) {
    const int64_t base = b * (num_rows + 1);
    const int32_t batch_nnz = batch_ptr(b + 1) - batch_ptr(b);
    if (row_ptr(base) != 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CSRSparseMatrix::Validate: row_pointers for batch ", b,
          " should start at 0, saw ", row_ptr(base)));
    }
    for (int64_t r = 0; r < num_rows; ++r) {
      if (row_ptr(base + r) < 0 || row_ptr(base + r + 1) < row_ptr(base + r)) {
        return absl::InvalidArgumentError(absl::StrCat(
            "CSRSparseMatrix::Validate: row_pointers must be non-negative and "
            "non-decreasing, saw ",
            row_ptr(base + r), " -> ", row_ptr(base + r + 1), " in batch ", b));
      }
    }
    if (row_ptr(base + num_rows) != batch_nnz) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CSRSparseMatrix::Validate: last row_pointer for batch ", b, " = ",
          row_ptr(base + num_rows), " but should equal the batch nnz = ",
          batch_nnz));
    }
  }

  // col_indices: every column index is in [0, num_cols).
  for (int64_t i = 0; i < total_nnz; ++i) {
    if (col_ind(i) < 0 || col_ind(i) >= num_cols) {
      return absl::InvalidArgumentError(absl::StrCat(
          "CSRSparseMatrix::Validate: column index ", col_ind(i),
          " is outside of the valid range [0, ", num_cols, ")"));
    }
  }

  return absl::OkStatus();
}

// Register variant decoding function for TF's RPC.
REGISTER_UNARY_VARIANT_DECODE_FUNCTION(CSRSparseMatrix,
                                       CSRSparseMatrix::kTypeName);

#define REGISTER_CSR_COPY(DIRECTION)                    \
  INTERNAL_REGISTER_UNARY_VARIANT_DEVICE_COPY_FUNCTION( \
      CSRSparseMatrix, DIRECTION, CSRSparseMatrix::DeviceCopy)

REGISTER_CSR_COPY(VariantDeviceCopyDirection::HOST_TO_DEVICE);
REGISTER_CSR_COPY(VariantDeviceCopyDirection::DEVICE_TO_HOST);
REGISTER_CSR_COPY(VariantDeviceCopyDirection::DEVICE_TO_DEVICE);

#undef REGISTER_CSR_COPY

}  // namespace tensorflow
