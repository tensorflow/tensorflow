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

#include "xla/shuffle.h"

#include <cstdint>

#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/literal.h"
#include "xla/shape.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace shuffle {

ShuffleMode MakePermuteMode(const LiteralSlice& indices) {
  ShuffleMode mode;
  *mode.mutable_permute()->mutable_indices() = indices.ToProto();
  return mode;
}

ShuffleMode MakeRotateMode(absl::Span<const int64_t> shifts) {
  ShuffleMode mode;
  mode.mutable_rotate()->mutable_shifts()->Assign(shifts.begin(), shifts.end());
  return mode;
}

absl::StatusOr<Literal> GetPermuteIndices(const ShuffleMode& mode) {
  CHECK(mode.has_permute()) << "the mode of the shuffle is not permute";
  return Literal::CreateFromProto(mode.permute().indices());
}

absl::StatusOr<Shape> GetPermuteIndicesShape(const ShuffleMode& mode) {
  CHECK(mode.has_permute()) << "the mode of the shuffle is not permute";
  return Shape::FromProto(mode.permute().indices().shape());
}

int64_t NormalizeShift(int64_t shift, int64_t dim_size) {
  CHECK_GE(dim_size, 0) << "a dimension cannot have a negative size";
  return dim_size == 0 ? 0 : ((shift % dim_size) + dim_size) % dim_size;
}

}  // namespace shuffle
}  // namespace xla
