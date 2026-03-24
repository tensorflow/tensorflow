/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_UTIL_STRIDES_H_
#define XLA_UTIL_STRIDES_H_

#include <cstdint>
#include <vector>

#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Returns the strides for `shape`.
std::vector<int64_t> ByteStridesForShape(const Shape& shape);
std::vector<int64_t> ByteStridesForShape(PrimitiveType element_type,
                                         absl::Span<const int64_t> dimensions,
                                         const xla::Layout& layout);
std::vector<int64_t> StridesForShape(PrimitiveType element_type,
                                     absl::Span<const int64_t> dimensions,
                                     const xla::Layout& layout);

}  // namespace xla

#endif  // XLA_UTIL_STRIDES_H_
