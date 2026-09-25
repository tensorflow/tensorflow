/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_SHUFFLE_H_
#define XLA_SHUFFLE_H_

#include <cstdint>

#include "absl/types/span.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace shuffle {

// The mode of a shuffle is the pattern in which it moves the elements of its
// shuffled dimensions, and the attributes that parameterize the pattern.

// Returns the mode of a shuffle that rotates each shuffled dimension to the
// left by the corresponding entry of `shifts`.
ShuffleMode Rotate(absl::Span<const int64_t> shifts);

// Returns the normalized shift in `[0, dim_size)`. Returns 0 if `dim_size` is
// 0, because every rotation of an empty dimension is a no-op.
int64_t NormalizeShift(int64_t shift, int64_t dim_size);

}  // namespace shuffle
}  // namespace xla

#endif  // XLA_SHUFFLE_H_
