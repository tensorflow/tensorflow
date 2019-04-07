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

#include "tensorflow/core/kernels/batch_matmul_op_common.h"

namespace tensorflow {
namespace {

// Returns the mapping from the output batch indices to the corresponding
// input's batch indices, given the input's "reshape" and "bcast" shapes as
// returned by the BCast helper class. The i'th element denotes the (flattened)
// batch index of the input that must be used to compute the i'th batch output.
void ComputeBatchIndices(const int64 output_batch_size,
                         const MatMulBCast::Vec& reshape,
                         const MatMulBCast::Vec& bcast,
                         std::vector<int64>* out_indices) {
  // Populates the mapping in out_indices. This algorithm is identical to
  // the following steps:
  //  - Reshape {0, 1, ..., input_batch_size - 1} to the input shape.
  //  - Broadcast to the output shape.
  //  - Reshape back to a flat 1D vector.
  out_indices->resize(output_batch_size);
  int64 num_output_elements = 1;
  int64 num_input_elements = 1;
  for (int64 i = reshape.size() - 1; i >= 0; --i) {
    // Replicate the already populated mapping an additional (dim - 1) times.
    // If we are broadcasting, just copy the existing mapping.
    // Otherwise, add another dimension from the input shape.
    const int64 dim = std::max(reshape[i], bcast[i]);
    const int64 incr = bcast[i] > 1 ? 0 : num_input_elements;
    for (int64 k = 0; k < (dim - 1) * num_output_elements; ++k) {
      (*out_indices)[num_output_elements + k] = (*out_indices)[k] + incr;
    }
    num_output_elements *= dim;
    num_input_elements *= reshape[i];
  }
}

}  // namespace

MatMulBCast::MatMulBCast(Vec x, Vec y) {
  if (x.size() < 2 || y.size() < 2) return;
  x.resize(x.size() - 2);
  y.resize(y.size() - 2);

  batch_bcast_ = absl::make_unique<BCast>(std::move(x), std::move(y));
  if (!batch_bcast_->IsValid()) return;

  x_batch_size_ = TensorShape(batch_bcast_->x_reshape()).num_elements();
  y_batch_size_ = TensorShape(batch_bcast_->y_reshape()).num_elements();
  output_shape_ = TensorShape(batch_bcast_->output_shape());
  output_batch_size_ = output_shape_.num_elements();
  broadcasting_required_ =
      std::min(x_batch_size_, y_batch_size_) != output_batch_size_;

  if (broadcasting_required_) {
    ComputeBatchIndices(output_batch_size_, batch_bcast_->x_reshape(),
                        batch_bcast_->x_bcast(), &x_batch_indices_);
    ComputeBatchIndices(output_batch_size_, batch_bcast_->y_reshape(),
                        batch_bcast_->y_bcast(), &y_batch_indices_);
  }
}

}  // namespace tensorflow
