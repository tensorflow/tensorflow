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

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/client/xla_builder.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {

// Forwards to xla::BroadcastTo.
// TODO(cheshire): Call the underlying function directly.
absl::StatusOr<xla::XlaOp> BroadcastTo(xla::XlaOp input,
                                       absl::Span<int64_t const> output_dims);

// Forwards to xla::BroadcastOpsToSame.
Status BroadcastOpsToSame(xla::XlaOp* lhs, xla::XlaOp* rhs);
}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_LIB_BROADCAST_H_
