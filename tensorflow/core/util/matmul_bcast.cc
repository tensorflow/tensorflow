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

#include "tensorflow/core/util/matmul_bcast.h"

namespace tensorflow {
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
