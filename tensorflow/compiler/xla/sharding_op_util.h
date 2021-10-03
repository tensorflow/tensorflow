/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SHARDING_OP_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SHARDING_OP_UTIL_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/util.h"

namespace xla {
namespace sharding_op_util {

// Encodes the attributes string for Sharding and auto/manual covnersion custom
// ops. This will be used in the opaque field.
std::string EncodeAttributes(absl::Span<const int64_t> unspecified_dims);

// Parses the opaque string of Sharding and auto/manual covnersion custom ops.
Status ParseAttributes(absl::string_view opaque,
                       std::vector<int64_t>* unspecified_dims);

}  // namespace sharding_op_util
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SHARDING_OP_UTIL_H_
