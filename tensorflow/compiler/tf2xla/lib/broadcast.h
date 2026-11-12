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

#ifndef TENSORFLOW_COMPILER_TF2XLA_LIB_BROADCAST_H_
#define TENSORFLOW_COMPILER_TF2XLA_LIB_BROADCAST_H_

#include <cstdint>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/hlo/builder/xla_builder.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {

// Forwards to xla::BroadcastTo.
// TODO(cheshire): Call the underlying function directly.
absl::StatusOr<xla::XlaOp> BroadcastTo(xla::XlaOp input,
                                       absl::Span<int64_t const> output_dims);

// Applies BroadcastDynamicDimension to each dynamic dimension of `input` whose
// bound and matching dimension in `output_dims` are larger than 1, so that a
// following BroadcastTo broadcasts a run-time size of 1 as TensorFlow does.
// BroadcastTo alone only sees the bound of a dynamic dimension.
xla::XlaOp BroadcastDynamicDimensions(xla::XlaOp input,
                                      absl::Span<int64_t const> output_dims);

// Returns `input` with its dynamic dimension `dimension` sliced or zero-padded
// to a bound of `size`. A run-time size of 1 is broadcast to `size`, as
// TensorFlow broadcasts it against a dimension of that size, and other
// run-time sizes are kept, capped at `size`. Requires that the dimension's
// bound and `size` are both at least 1.
xla::XlaOp BroadcastDynamicDimension(xla::XlaOp input, int64_t dimension,
                                     int64_t size);

// Forwards to xla::BroadcastOpsToSame.
absl::Status BroadcastOpsToSame(xla::XlaOp* lhs, xla::XlaOp* rhs);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_BROADCAST_H_
