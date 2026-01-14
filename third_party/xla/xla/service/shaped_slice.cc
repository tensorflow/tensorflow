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

#include "xla/service/shaped_slice.h"

#include <optional>
#include <utility>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.pb.h"
#include "xla/shape.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {

absl::StatusOr<ShapedSlice> ShapedSlice::FromProto(
    const ShapedSliceProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  ShapedSlice shaped_slice;
  TF_ASSIGN_OR_RETURN(
      shaped_slice.slice,
      BufferAllocation::Slice::FromProto(proto.slice(), buffer_allocations));
  TF_ASSIGN_OR_RETURN(shaped_slice.shape, Shape::FromProto(proto.shape()));
  return shaped_slice;
}

absl::StatusOr<ShapedSliceProto> ShapedSlice::ToProto() const {
  ShapedSliceProto proto;
  TF_ASSIGN_OR_RETURN(*proto.mutable_slice(), slice.ToProto());
  *proto.mutable_shape() = shape.ToProto();
  return proto;
}

absl::StatusOr<NullableShapedSlice> NullableShapedSlice::FromProto(
    const NullableShapedSliceProto& proto,
    absl::Span<const BufferAllocation> buffer_allocations) {
  if (proto.has_shaped_slice()) {
    TF_ASSIGN_OR_RETURN(
        ShapedSlice shaped_slice,
        ShapedSlice::FromProto(proto.shaped_slice(), buffer_allocations));
    return NullableShapedSlice(std::move(shaped_slice));
  }
  return NullableShapedSlice(std::nullopt);
}

absl::StatusOr<NullableShapedSliceProto> NullableShapedSlice::ToProto() const {
  NullableShapedSliceProto proto;
  if (has_value()) {
    TF_ASSIGN_OR_RETURN(*proto.mutable_shaped_slice(), value().ToProto());
  }
  return proto;
}

}  // namespace xla
