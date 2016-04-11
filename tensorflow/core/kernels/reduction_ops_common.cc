/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/kernels/reduction_ops_common.h"

namespace tensorflow {

TensorShape ReductionHelper::out_reshape() const {
  TensorShape shape;
  for (auto size : out_reshape_) shape.AddDim(size);
  return shape;
}

// The final output shape must be allocated with this shape.
TensorShape ReductionHelper::out_shape() const {
  TensorShape shape;
  for (auto size : out_shape_) shape.AddDim(size);
  return shape;
}

TensorShape ReductionHelper::shuffled_shape() {
  const int dims = data_reshape_.size();
  TensorShape shape;
  for (int i = reduce_first_axis_; i < dims; i += 2) {
    shape.AddDim(data_reshape_[i]);
  }
  for (int i = !reduce_first_axis_; i < dims; i += 2) {
    shape.AddDim(data_reshape_[i]);
  }
  return shape;
}

gtl::InlinedVector<int32, 8> ReductionHelper::permutation() {
  const int dims = data_reshape_.size();
  const int unreduced_dims = (dims + !reduce_first_axis_) / 2;
  gtl::InlinedVector<int32, 8> perm(dims);
  for (int i = 0; i < unreduced_dims; i++) {
    perm[i] = 2 * i + reduce_first_axis_;
  }
  for (int i = unreduced_dims; i < dims; i++) {
    perm[i] = 2 * (i - unreduced_dims) + !reduce_first_axis_;
  }
  return perm;
}

Status ReductionHelper::Simplify(const Tensor& data, const Tensor& axis,
                                 const bool keep_dims) {
  // bitmap[i] indicates whether to reduce data along i-th axis.
  gtl::InlinedVector<bool, 4> bitmap(data.dims(), false);
  auto axis_vec = axis.flat<int32>();
  for (int64 i = 0; i < axis.NumElements(); ++i) {
    const int32 index = axis_vec(i);
    if (index < 0 || index >= data.dims()) {
      return errors::InvalidArgument("Invalid reduction dimension (", index,
                                     " for input with ", data.dims(),
                                     " dimension(s)");
    }
    bitmap[index] = true;
  }

  // Output tensor's dim sizes.
  out_shape_.clear();
  for (int i = 0; i < data.dims(); ++i) {
    if (!bitmap[i]) {
      // If we are not reducing along dimension i.
      out_shape_.push_back(data.dim_size(i));
    } else if (keep_dims) {
      // We are reducing along dimension i, but we want to keep the
      // same number of dimensions, so we set the dimension of i to
      // '1'.
      out_shape_.push_back(1);
    }
  }

  // Depending on bitmap[i] and bitmap[i-1], we can collapse axis of
  // the input data before doing the reduction on the resulting
  // tensor.  The shape of the reduction is a reshape of the final
  // output.

  // We'll skip the leading 1s.
  int dim_index = 0;
  for (; dim_index < data.dims(); ++dim_index) {
    if (data.dim_size(dim_index) != 1) break;
  }
  if (dim_index >= data.dims()) {
    // Special case. The input is essentially a scalar.
    reduce_first_axis_ = true;
  } else {
    // Starting from the (dim_index)-th dimension, dimensions
    // alternates between runs that need to be reduced and runs that
    // don't.
    //
    // NOTE: If a dimension has size 1, we group it as the current
    // run so that we can minimize the number of runs.
    //
    // E.g., when we want to reduce a tensor of shape [2, 1, 3, 1,
    // 5] by axes = [1, 4], we should treat the tensor as a [6, 5]
    // and reduce by axes = [1] (i.e., the output is shape [6]).
    reduce_first_axis_ = bitmap[dim_index];
    data_reshape_.push_back(data.dim_size(dim_index));
    ++dim_index;
    for (; dim_index < data.dims(); ++dim_index) {
      const auto size = data.dim_size(dim_index);
      if (size == 1) {
        bitmap[dim_index] = bitmap[dim_index - 1];
      }
      if (bitmap[dim_index - 1] != bitmap[dim_index]) {
        // Starts a new run of reduce or !reduce.
        data_reshape_.push_back(size);
      } else {
        // Continue a run of reduce or !reduce.
        data_reshape_.back() *= size;
      }
    }
    // If reduce_first_axis_ is true (input's dimension 0, 2, 4, etc
    // are reduced), data_reshape_[1, 3, 5, ...]  is out_reshape_,
    // otherwise, data_reshape_[0, 2, 4, ...] is.
    for (size_t i = reduce_first_axis_ ? 1 : 0; i < data_reshape_.size();
         i += 2) {
      out_reshape_.push_back(data_reshape_[i]);
    }
  }

  VLOG(1) << "data reshape: " << str_util::Join(data_reshape_, ",");
  VLOG(1) << "out  reshape: " << str_util::Join(out_reshape_, ",");
  VLOG(1) << "out    shape: " << str_util::Join(out_shape_, ",");
  return Status::OK();
}

}  // namespace tensorflow
