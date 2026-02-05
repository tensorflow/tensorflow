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

#ifndef XLA_SERVICE_SHAPED_SLICE_H_
#define XLA_SERVICE_SHAPED_SLICE_H_

#include <optional>

#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/shaped_slice.pb.h"
#include "xla/shape.h"

namespace xla {

// A struct that defines a shaped slice, i.e., a BufferAllocation::Slice and its
// shape.
struct ShapedSlice {
  BufferAllocation::Slice slice;
  Shape shape;

  static absl::StatusOr<ShapedSlice> FromProto(
      const ShapedSliceProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations);
  absl::StatusOr<ShapedSliceProto> ToProto() const;

  friend bool operator==(const ShapedSlice& lhs, const ShapedSlice& rhs) {
    return lhs.slice == rhs.slice && lhs.shape == rhs.shape;
  }

  friend bool operator!=(const ShapedSlice& lhs, const ShapedSlice& rhs) {
    return !(lhs == rhs);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const ShapedSlice& shaped_slice) {
    absl::Format(&sink, "ShapedSlice{slice: %v, shape: %v}", shaped_slice.slice,
                 shaped_slice.shape.ToString(/*print_layout=*/true));
  }
};

// A nullable shaped slice is either a ShapedSlice or a nullopt. This is used
// to represent the operands and results of a thunk, where a nullopt represents
// a null pointer argument to the thunk.
class NullableShapedSlice : public std::optional<ShapedSlice> {
 public:
  using std::optional<ShapedSlice>::optional;

  static absl::StatusOr<NullableShapedSlice> FromProto(
      const NullableShapedSliceProto& proto,
      absl::Span<const BufferAllocation> buffer_allocations);
  absl::StatusOr<NullableShapedSliceProto> ToProto() const;

  friend bool operator==(const NullableShapedSlice& lhs,
                         const NullableShapedSlice& rhs) {
    return static_cast<const std::optional<ShapedSlice>&>(lhs) ==
           static_cast<const std::optional<ShapedSlice>&>(rhs);
  }

  friend bool operator!=(const NullableShapedSlice& lhs,
                         const NullableShapedSlice& rhs) {
    return !(lhs == rhs);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            const NullableShapedSlice& shaped_slice) {
    if (shaped_slice.has_value()) {
      absl::Format(&sink, "%v", *shaped_slice);
    } else {
      absl::Format(&sink, "null");
    }
  }
};

}  // namespace xla

#endif  // XLA_SERVICE_SHAPED_SLICE_H_
