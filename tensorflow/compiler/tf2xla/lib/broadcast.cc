/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/tf2xla/lib/broadcast.h"

#include <vector>

#include "absl/algorithm/container.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/tf2xla/shape_util.h"
#include "tensorflow/compiler/xla/client/lib/broadcast.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/util/bcast.h"

namespace tensorflow {

StatusOr<xla::XlaOp> BroadcastTo(xla::XlaOp input,
                                 absl::Span<int64_t const> output_dims) {
  return xla::BroadcastTo(input, output_dims);
}

Status BroadcastOpsToSame(xla::XlaOp* lhs, xla::XlaOp* rhs) {
  return xla::BroadcastOpsToSame(lhs, rhs);
}

}  // namespace tensorflow
