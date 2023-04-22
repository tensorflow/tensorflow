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

#ifndef TENSORFLOW_CORE_UTIL_MATMUL_BCAST_H_
#define TENSORFLOW_CORE_UTIL_MATMUL_BCAST_H_

#include <vector>

#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/lib/gtl/inlined_vector.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

// Simple wrapper over BCast specialized for MatMul.
// Provides utilities for broadcasting across batch dimensions for binary
// MatMul-like operations. If neither argument has batch dimensions (rank <= 2)
// then no broadcasting is needed and the operation MatMul operation is
// considered valid.
class MatMulBCast {
 public:
  using Vec = BCast::Vec;

  MatMulBCast(const Vec& x, const Vec& y) {
    if (std::max(x.size(), y.size()) == 2) return;
    const Vec x_resized(x.begin(), x.end() - 2);
    const Vec y_resized(y.begin(), y.end() - 2);

    batch_bcast_ =
        absl::make_unique<BCast>(std::move(x_resized), std::move(y_resized));
    if (!batch_bcast_->IsValid()) {
      // Set broadcasting_required_ to true to make IsValid() return false;
      broadcasting_required_ = true;
      return;
    }

    x_batch_size_ = TensorShape(batch_bcast_->x_reshape()).num_elements();
    y_batch_size_ = TensorShape(batch_bcast_->y_reshape()).num_elements();
    output_batch_shape_ = TensorShape(batch_bcast_->output_shape());
    output_batch_size_ = output_batch_shape_.num_elements();
    broadcasting_required_ =
        std::min(x_batch_size_, y_batch_size_) != output_batch_size_;

    if (broadcasting_required_) {
      ComputeBatchIndices(output_batch_size_, batch_bcast_->x_reshape(),
                          batch_bcast_->x_bcast(), &x_batch_indices_);
      ComputeBatchIndices(output_batch_size_, batch_bcast_->y_reshape(),
                          batch_bcast_->y_bcast(), &y_batch_indices_);
    }
  }

  bool IsValid() const {
    return !broadcasting_required_ || (batch_bcast_ && batch_bcast_->IsValid());
  }
  bool IsBroadcastingRequired() const { return broadcasting_required_; }

  const int64 output_batch_size() const { return output_batch_size_; }
  const int64 x_batch_size() const { return x_batch_size_; }
  const int64 y_batch_size() const { return y_batch_size_; }
  const TensorShape& output_batch_shape() const { return output_batch_shape_; }

  // Returns the mapping from the flattened output batch indices to x's
  // flattened batch indices. The result is a vector of length
  // output_batch_size(). To compute the i'th batch output, a binary matmul-like
  // operation should use the `x_batch_indices()[i]`th batch index of `x`.
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64>& x_batch_indices() const { return x_batch_indices_; }
  // Returns the mapping from the flattened output batch indices to y's
  // flattened batch indices. Similar to x_batch_indices().
  // Note: Returns an empty vector if broadcasting is not required. Callers
  // should only use this when IsBroadcastingRequired() returns true.
  const std::vector<int64>& y_batch_indices() const { return y_batch_indices_; }

 private:
  std::unique_ptr<BCast> batch_bcast_;
  bool broadcasting_required_ = false;
  int64 x_batch_size_ = 1;
  int64 y_batch_size_ = 1;
  TensorShape output_batch_shape_;
  int64 output_batch_size_ = 1;
  std::vector<int64> x_batch_indices_;
  std::vector<int64> y_batch_indices_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_UTIL_MATMUL_BCAST_H_
