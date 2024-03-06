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

#include "tensorflow/core/util/sparse/sparse_tensor.h"

#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace sparse {

namespace {

int UnsafeGetDimsFromIx(const Tensor& ix) {
  DCHECK(TensorShapeUtils::IsMatrix(ix.shape()));
  return ix.dim_size(1);
}

Status GetDimsFromIx(const Tensor& ix, int* result) {
  if (!TensorShapeUtils::IsMatrix(ix.shape())) {
    return errors::InvalidArgument("indices must be a matrix, but got: ",
                                   ix.shape().DebugString());
  }
  *result = UnsafeGetDimsFromIx(ix);
  return Status();
}

}  // namespace

/* static */ Status SparseTensor::Create(Tensor ix, Tensor vals,
                                         const VarDimArray shape,
                                         const VarDimArray order,
                                         SparseTensor* result) {
  if (ix.dtype() != DT_INT64) {
    return errors::InvalidArgument("indices must be type int64 but got: ",
                                   ix.dtype());
  }
  if (!TensorShapeUtils::IsVector(vals.shape())) {
    return errors::InvalidArgument("vals must be a vec, but got: ",
                                   vals.shape().DebugString());
  }
  if (ix.shape().dim_size(0) != vals.shape().dim_size(0)) {
    return errors::InvalidArgument(
        "indices and values rows (indexing "
        "dimension) must match. (indices = ",
        ix.shape().dim_size(0), ", values = ", vals.shape().dim_size(0), ")");
  }
  int dims = 0;
  TF_RETURN_IF_ERROR(GetDimsFromIx(ix, &dims));
  if (order.size() != dims) {
    return errors::InvalidArgument("Order length must be SparseTensor rank.");
  }
  if (shape.size() != dims) {
    return errors::InvalidArgument("Shape rank must be SparseTensor rank.");
  }

  result->ix_ = std::move(ix);
  result->vals_ = std::move(vals);
  result->shape_.assign(shape.begin(), shape.end());
  result->order_.assign(order.begin(), order.end());
  result->dims_ = dims;
  return absl::OkStatus();
}

/* static */ Status SparseTensor::Create(Tensor ix, Tensor vals,
                                         const TensorShape& shape,
                                         SparseTensor* result) {
  return Create(std::move(ix), std::move(vals), TensorShapeToVector(shape),
                UndefinedOrder(TensorShapeToVector(shape)), result);
}

/* static */ Status SparseTensor::Create(Tensor ix, Tensor vals,
                                         const VarDimArray shape,
                                         SparseTensor* result) {
  return Create(std::move(ix), std::move(vals), shape, UndefinedOrder(shape),
                result);
}

/* static */ Status SparseTensor::Create(Tensor ix, Tensor vals,
                                         const TensorShape& shape,
                                         const VarDimArray order,
                                         SparseTensor* result) {
  return Create(std::move(ix), std::move(vals), TensorShapeToVector(shape),
                order, result);
}

SparseTensor::SparseTensor(Tensor ix, Tensor vals, const VarDimArray shape,
                           const VarDimArray order)
    : ix_(std::move(ix)),
      vals_(std::move(vals)),
      shape_(shape.begin(), shape.end()),
      order_(order.begin(), order.end()),
      dims_(UnsafeGetDimsFromIx(ix_)) {
  DCHECK_EQ(ix_.dtype(), DT_INT64)
      << "indices must be type int64 but got: " << ix_.dtype();
  DCHECK(TensorShapeUtils::IsVector(vals_.shape()))
      << "vals must be a vec, but got: " << vals_.shape().DebugString();
  DCHECK_EQ(ix_.shape().dim_size(0), vals_.shape().dim_size(0))
      << "indices and values rows (indexing dimension) must match.";
  DCHECK_EQ(order.size(), dims_) << "Order length must be SparseTensor rank.";
  DCHECK_EQ(shape.size(), dims_) << "Shape rank must be SparseTensor rank.";
}

// Optimized version of `IndicesValid()` with the following requirements:
// * The sparse tensor is one-dimensional.
//
// Returns true if the indices are valid, otherwise false.
// NOTE(mrry): If this method returns false, call IndicesValidHelper<true>()
// to obtain a meaningful error message.
bool SparseTensor::IndicesValidVectorFastPath() const {
  DCHECK_EQ(shape_.size(), 1);
  DCHECK_EQ(order_[0], 0);

  const int64_t max_index = shape_[0];

  // We maintain separate bools for each validation predicate to enable
  // vectorization across loop iterations.
  bool index_in_range_valid = true;
  bool order_valid = true;

  int64_t prev_index = -1;
  const auto ix_t = ix_.matrix<int64_t>();
  const int64_t* const index_base_ptr = ix_t.data();

  for (std::size_t n = 0; n < ix_t.dimension(0); ++n) {
    const int64_t index = index_base_ptr[n];
    index_in_range_valid = index_in_range_valid & (index < max_index);
    order_valid = order_valid & (index > prev_index);
    prev_index = index;
  }

  return index_in_range_valid & order_valid;
}

// Optimized version of `IndicesValid()` with the following requirements:
// * The sparse tensor is two-dimensional.
// * The tensor's indices are in the "standard" (lexicographic) order.
// * All of the tensor's indices fit within the range of a signed int32.
//
// Returns true if the indices are valid, otherwise false.
// NOTE(mrry): If this method returns false, call IndicesValidHelper<true>()
// to obtain a meaningful error message.
bool SparseTensor::IndicesValidMatrix32BitFastPath() const {
  const auto ix_t = ix_.matrix<int64_t>();
  const int64_t* const shape_ptr = shape_.data();

  DCHECK_EQ(shape_.size(), 2);
  DCHECK_EQ(order_[0], 0);
  DCHECK_EQ(order_[1], 1);
  DCHECK_LE(shape_ptr[0], std::numeric_limits<int32>::max());
  DCHECK_LE(shape_ptr[1], std::numeric_limits<int32>::max());

  const int32_t max_rows = static_cast<int32>(shape_ptr[0]);
  const int32_t max_cols = static_cast<int32>(shape_ptr[1]);

  // We maintain separate bools for each validation predicate to enable
  // vectorization across loop iterations.
  bool row_zeros_valid = true;
  bool row_in_range_valid = true;
  bool col_zeros_valid = true;
  bool col_in_range_valid = true;
  bool order_valid = true;

  int64_t prev_index = -1;

  // Points to the beginning of the current row of the indices matrix.
  // Each row has two int64 elements, but we use an int32 pointer to access
  // the low and high 32 bits of each element separately. This means that our
  // stride per row is 4 elements.
  const int32* const index_base_ptr =
      reinterpret_cast<const int32*>(ix_t.data());
  const size_t kInt32ElementsPerRow = 4;

  for (std::size_t n = 0; n < ix_t.dimension(0); ++n) {
    const int32* const index_ptr = index_base_ptr + n * kInt32ElementsPerRow;

    // Unpack the values on the current row of the indices matrix.
#if __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
    const int32 row_zeros = index_ptr[0];
    const int32 row_32 = index_ptr[1];
    const int32 col_zeros = index_ptr[2];
    const int32 col_32 = index_ptr[3];
#else
    const int32_t row_32 = index_ptr[0];
    const int32_t row_zeros = index_ptr[1];
    const int32_t col_32 = index_ptr[2];
    const int32_t col_zeros = index_ptr[3];
#endif

    // Validate that the high 32 bits of the row and column indices are zero.
    row_zeros_valid = row_zeros_valid & (row_zeros == 0);
    col_zeros_valid = col_zeros_valid & (col_zeros == 0);

    // Validate that the low 32 bits of the row and column indices are within
    // range of the shape.
    row_in_range_valid =
        row_in_range_valid & (row_32 >= 0) & (row_32 < max_rows);
    col_in_range_valid =
        col_in_range_valid & (col_32 >= 0) & (col_32 < max_cols);

    // Interpret the row and column as a concatenated 64-bit integer, and
    // validate that the concatenated indices are in strictly increasing order.
    const int64_t concatenated_index =
        (static_cast<int64_t>(row_32) << 32) + col_32;
    order_valid = order_valid & (concatenated_index > prev_index);
    prev_index = concatenated_index;
  }

  return row_zeros_valid & row_in_range_valid & col_zeros_valid &
         col_in_range_valid & order_valid;
}

template <bool standard_order>
Status SparseTensor::IndicesValidHelper() const {
  const auto ix_t = ix_.matrix<int64_t>();
  const int64_t* const shape_ptr = shape_.data();

  for (std::size_t n = 0; n < num_entries(); ++n) {
    bool valid = true;
    bool different = false;
    bool increasing = true;
    if (n == 0) {
      for (int di = 0; di < dims_; ++di) {
        if (ix_t(n, di) < 0 || ix_t(n, di) >= shape_ptr[di]) valid = false;
      }
      different = true;
    } else {
      for (int di = 0; di < dims_; ++di) {
        if (ix_t(n, di) < 0 || ix_t(n, di) >= shape_ptr[di]) valid = false;
        int ordered_dim;
        if (standard_order) {
          ordered_dim = di;
        } else {
          ordered_dim = order_[di];
        }
        int64_t diff = ix_t(n, ordered_dim) - ix_t(n - 1, ordered_dim);
        if (diff > 0) different = true;
        if (!different && diff < 0) increasing = false;
      }
    }
    if (TF_PREDICT_FALSE(!valid || !increasing || !different)) {
      string index = strings::StrCat("indices[", n, "] = [");
      for (int di = 0; di < dims_; ++di) {
        strings::StrAppend(&index, ix_t(n, di), di < dims_ - 1 ? "," : "]");
      }
      if (!valid) {
        return errors::InvalidArgument(index,
                                       " is out of bounds: need 0 <= index < [",
                                       str_util::Join(shape_, ","), "]");
      }
      if (!increasing) {
        return errors::InvalidArgument(
            index,
            " is out of order. Many sparse ops require sorted indices.\n"
            "    Use `tf.sparse.reorder` to create a correctly ordered copy."
            "\n\n");
      }
      if (!different) {
        return errors::InvalidArgument(index, " is repeated");
      }
    }
  }

  return absl::OkStatus();
}

Status SparseTensor::IndicesValid() const {
  if (shape_.size() == 1 && IndicesValidVectorFastPath()) {
    return absl::OkStatus();
  }

  bool standard_order = true;
  for (size_t i = 0; i < order_.size(); ++i) {
    if (order_[i] < 0) {
      return errors::FailedPrecondition(
          "Order was not provided.  Provide an order at "
          "construction time or run ReorderInPlace");
    }
    standard_order = standard_order && order_[i] == i;
  }

  if (standard_order) {
    if (shape_.size() == 1) {
      if (IndicesValidVectorFastPath()) {
        return absl::OkStatus();
      }
    } else if (shape_.size() == 2 &&
               shape_[0] <= std::numeric_limits<int32>::max() &&
               shape_[1] <= std::numeric_limits<int32>::max()) {
      if (IndicesValidMatrix32BitFastPath()) {
        return absl::OkStatus();
      }
    }
    return IndicesValidHelper<true>();
  } else {
    return IndicesValidHelper<false>();
  }
}

}  // namespace sparse
}  // namespace tensorflow
