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

  *result = SparseTensor(std::move(ix), std::move(vals), shape, order);
  return Status::OK();
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

template <bool standard_order>
Status SparseTensor::IndicesValidHelper() const {
  const auto ix_t = ix_.matrix<int64>();
  const int64* const shape_ptr = shape_.data();

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
        int64 diff = ix_t(n, ordered_dim) - ix_t(n - 1, ordered_dim);
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

  return Status::OK();
}

Status SparseTensor::IndicesValid() const {
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
    return IndicesValidHelper<true>();
  } else {
    return IndicesValidHelper<false>();
  }
}

}  // namespace sparse
}  // namespace tensorflow
