/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_CLIENT_LIB_BROADCAST_H_
#define XLA_CLIENT_LIB_BROADCAST_H_

#include "xla/client/xla_builder.h"
#include "xla/primitive_util.h"
#include "xla/types.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Broadcasts 'input' up to shape 'output_dims', using TensorFlow broadcasting
// rules. Supports broadcasting a dimension of size x to size x*y, i.e., tiling.
StatusOr<XlaOp> BroadcastTo(XlaOp input, absl::Span<int64_t const> output_dims);

}  // namespace xla

#endif  // XLA_CLIENT_LIB_BROADCAST_H_
