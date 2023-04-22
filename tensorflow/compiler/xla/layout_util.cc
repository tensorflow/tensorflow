/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/layout_util.h"

#include <stddef.h>

#include <algorithm>
#include <functional>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/numbers.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace xla {
namespace {

// Internal helper for GetDefaultLayoutForShape and SetToDefaultLayout. Sets
// minor_to_major to the value that represents the default layout.
template <typename T>
void SetDefaultLayoutToContainer(T* minor_to_major) {
  // The default XLA layout is major-to-minor (dim 0 is major).
  // For more information on XLA layouts, see:
  // https://www.tensorflow.org/performance/xla/shapes
  const int64 size = minor_to_major->size();
  for (int64 i = 0; i < size; ++i) {
    (*minor_to_major)[i] = size - 1 - i;
  }
}

}  // namespace

/* static */ Layout LayoutUtil::MakeLayout(
    absl::Span<const int64> minor_to_major, absl::Span<const Tile> tiles,
    int64 element_size_in_bits, int64 memory_space) {
  Layout layout;
  layout.set_format(DENSE);
  for (int64 dimension_number : minor_to_major) {
    layout.add_minor_to_major(dimension_number);
  }
  for (const Tile& tile : tiles) {
    for (int64 dim : tile.dimensions()) {
      if (dim < 0 && dim != Tile::kCombineDimension) {
        LOG(FATAL) << "Tile dimension size needs to be minimum int64 value if "
                      "it's negative. Value is "
                   << dim;
      }
    }
    *layout.add_tiles() = tile;
  }
  layout.set_element_size_in_bits(element_size_in_bits);
  layout.set_memory_space(memory_space);
  return layout;
}

/* static */ Layout LayoutUtil::MakeDescendingLayout(int64 rank) {
  std::vector<int64> layout(rank);
  std::iota(layout.rbegin(), layout.rend(), static_cast<int64>(0));
  return MakeLayout(layout);
}

/* static */ Layout LayoutUtil::MakeAscendingLayout(int64 rank) {
  std::vector<int64> layout(rank);
  std::iota(layout.begin(), layout.end(), static_cast<int64>(0));
  return MakeLayout(layout);
}

/* static */ Layout LayoutUtil::MakeLayoutFromMajorToMinor(
    absl::Span<const int64> major_to_minor) {
  Layout layout;
  layout.set_format(DENSE);
  for (int i = major_to_minor.size() - 1; i >= 0; i--) {
    layout.add_minor_to_major(major_to_minor[i]);
  }
  return layout;
}

namespace {

// Internal helper that creates a default layout for an array of the given rank.
Layout CreateDefaultLayoutForRank(int64 rank) {
  Layout layout;
  layout.set_format(DENSE);
  auto* minor_to_major = layout.mutable_minor_to_major();
  minor_to_major->resize(rank, 0);
  SetDefaultLayoutToContainer(minor_to_major);
  return layout;
}

}  // namespace

/* static */ Layout LayoutUtil::GetDefaultLayoutForShape(const Shape& shape) {
  if (shape.IsOpaque() || shape.IsToken()) {
    // Opaque and token types have empty layouts.
    return Layout();
  }

  // A Layout proto corresponds to a single array, not a tuple.
  CHECK(shape.IsArray());
  return CreateDefaultLayoutForRank(shape.dimensions_size());
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForRank(int64 rank) {
  return CreateDefaultLayoutForRank(rank);
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForR2() {
  return CreateDefaultLayoutForRank(2);
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForR3() {
  return CreateDefaultLayoutForRank(3);
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForR4() {
  return CreateDefaultLayoutForRank(4);
}

/* static */ void LayoutUtil::SetToDefaultLayout(Shape* shape) {
  if (shape->IsTuple()) {
    // Tuple shape.
    for (auto& element_shape : *shape->mutable_tuple_shapes()) {
      SetToDefaultLayout(&element_shape);
    }
    shape->clear_layout();
  } else if (shape->IsArray()) {
    shape->mutable_layout()->set_format(DENSE);
    auto* minor_to_major = shape->mutable_layout()->mutable_minor_to_major();
    minor_to_major->resize(shape->dimensions_size(), 0);
    SetDefaultLayoutToContainer(minor_to_major);
  } else {
    // Opaque, token types etc. have no layout.
    shape->clear_layout();
  }
}

/* static */ Shape LayoutUtil::GetWithDefaultLayout(const Shape& shape) {
  Shape copy(shape);
  LayoutUtil::SetToDefaultLayout(&copy);
  return copy;
}

/* static */ void LayoutUtil::SetToDefaultLayout(ProgramShape* program_shape) {
  for (auto& parameter_shape : *program_shape->mutable_parameters()) {
    LayoutUtil::SetToDefaultLayout(&parameter_shape);
  }
  LayoutUtil::SetToDefaultLayout(program_shape->mutable_result());
}

/* static */ Status LayoutUtil::ValidateLayoutInShape(
    const Shape& shape, bool allow_missing_layouts) {
  if (shape.IsTuple()) {
    // Tuple shape.
    if (shape.has_layout()) {
      return InvalidArgument("tuple should not have a layout field");
    }
    for (auto& element_shape : shape.tuple_shapes()) {
      TF_RETURN_IF_ERROR(
          ValidateLayoutInShape(element_shape, allow_missing_layouts));
    }
    return Status::OK();
  } else if (shape.IsArray()) {
    if (!shape.has_layout()) {
      if (allow_missing_layouts) {
        return Status::OK();
      }
      return InvalidArgument("shape %s does not have a layout",
                             ShapeUtil::HumanString(shape));
    }
    return ValidateLayoutForShape(shape.layout(), shape);
  } else {
    // Token, opaque, etc. shape.
    if (shape.has_layout()) {
      return InvalidArgument(
          "shape of primitive type %s should not have a layout",
          PrimitiveType_Name(shape.element_type()));
    }
    return Status::OK();
  }
}

/* static */ Status LayoutUtil::ValidateLayoutForShape(const Layout& layout,
                                                       const Shape& shape) {
  if (shape.IsTuple()) {
    return InvalidArgument("a single Layout is not valid for tuple shapes");
  }

  if (!shape.IsArray()) {
    if (layout.minor_to_major_size() != 0) {
      return InvalidArgument(
          "shape of primitive type %s should not have a non-trivial layout",
          PrimitiveType_Name(shape.element_type()));
    }
    return Status::OK();
  }

  if (layout.format() == INVALID_FORMAT || !Format_IsValid(layout.format())) {
    return InvalidArgument("Layout has an invalid format (%d)",
                           layout.format());
  }

  if (layout.format() == DENSE) {
    if (layout.minor_to_major_size() != shape.rank()) {
      return InvalidArgument(
          "layout minor_to_major field contains %d elements, "
          "but shape is rank %d: {%s}; shape: %s",
          layout.minor_to_major_size(), shape.rank(),
          absl::StrJoin(layout.minor_to_major(), ", "),
          shape.ShortDebugString());
    }

    std::vector<bool> dimensions_in_layout(shape.rank(), false);
    for (int64 i = 0; i < shape.rank(); ++i) {
      int64 dim = layout.minor_to_major(i);
      if (dim < 0 || dim >= shape.rank()) {
        return InvalidArgument(
            "layout minor_to_major field has out-of-bounds value: %s",
            HumanString(layout));
      }
      if (dimensions_in_layout[dim]) {
        return InvalidArgument(
            "layout minor_to_major field has duplicate values: {%s}",
            HumanString(layout));
      }
      dimensions_in_layout[dim] = true;
    }
  } else {
    if (layout.tiles_size() != 0) {
      return InvalidArgument("Only dense layouts can be tiled.");
    }
  }

  return Status::OK();
}

/* static */ void LayoutUtil::ClearLayout(Shape* shape) {
  shape->clear_layout();
  for (auto& element_shape : *shape->mutable_tuple_shapes()) {
    ClearLayout(&element_shape);
  }
}

/* static */ void LayoutUtil::ClearLayout(ProgramShape* program_shape) {
  for (auto& parameter_shape : *program_shape->mutable_parameters()) {
    LayoutUtil::ClearLayout(&parameter_shape);
  }
  LayoutUtil::ClearLayout(program_shape->mutable_result());
}

/* static */ bool LayoutUtil::IsDenseArray(const Shape& shape) {
  return shape.IsArray() && shape.has_layout() && IsDense(shape.layout());
}

/* static */ bool LayoutUtil::IsDense(const Layout& layout) {
  return layout.format() == DENSE;
}

/* static */ bool LayoutUtil::IsMonotonicWithDim0Minor(const Layout& layout) {
  CHECK(layout.format() == DENSE);
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end());
}

/* static */ bool LayoutUtil::IsMonotonicWithDim0Major(const Layout& layout) {
  CHECK(layout.format() == DENSE);
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end(), std::greater<int64>());
}

/* static */ bool LayoutUtil::HasLayout(const Shape& shape) {
  if (shape.IsTuple()) {
    // Tuple shape: all subshapes must have a layout.
    return absl::c_all_of(shape.tuple_shapes(),
                          [](const Shape& s) { return HasLayout(s); });
  } else if (!shape.IsArray()) {
    // Opaque, token types etc. ignore layout.
    return true;
  }
  return shape.has_layout() && shape.layout().format() != INVALID_FORMAT;
}

/* static */ bool LayoutUtil::HasLayout(const ProgramShape& program_shape) {
  for (auto& parameter_shape : program_shape.parameters()) {
    if (!LayoutUtil::HasLayout(parameter_shape)) {
      return false;
    }
  }
  return LayoutUtil::HasLayout(program_shape.result());
}

/* static */ bool LayoutUtil::Equal(const Layout& lhs, const Layout& rhs) {
  return lhs == rhs;
}

/* static */ absl::Span<const int64> LayoutUtil::MinorToMajor(
    const Shape& shape) {
  CHECK(IsDenseArray(shape));
  return AsInt64Slice(shape.layout().minor_to_major());
}

/* static */ absl::Span<const int64> LayoutUtil::MinorToMajor(
    const Layout& layout) {
  CHECK(layout.format() == DENSE);
  return AsInt64Slice(layout.minor_to_major());
}

/* static */ int64 LayoutUtil::Major(const Layout& layout,
                                     int64 physical_dimension_number) {
  CHECK_LE(0, physical_dimension_number);
  CHECK_LT(physical_dimension_number, layout.minor_to_major_size());
  return Minor(layout,
               layout.minor_to_major_size() - 1 - physical_dimension_number);
}

/* static */ int64 LayoutUtil::Minor(const Layout& layout,
                                     int64 physical_dimension_number) {
  CHECK_EQ(layout.format(), DENSE);
  CHECK_LE(0, physical_dimension_number);
  CHECK_LT(physical_dimension_number, layout.minor_to_major_size());
  return layout.minor_to_major(physical_dimension_number);
}

/* static */ std::vector<int64> LayoutUtil::MakeLogicalToPhysical(
    const Layout& layout) {
  std::vector<int64> logical_to_physical(layout.minor_to_major_size());
  for (int64 physical = 0, end = logical_to_physical.size(); physical < end;
       ++physical) {
    const int64 logical = Major(layout, physical);
    logical_to_physical[logical] = physical;
  }
  return logical_to_physical;
}

/* static */ string LayoutUtil::HumanString(const Layout& layout) {
  return layout.ToString();
}

namespace {

// Internal helper for recursively copying layouts.
Status CopyLayoutInternal(const Shape& src, Shape* dst) {
  if (src.IsTuple() != dst->IsTuple()) {
    return InvalidArgument(
        "cannot copy layout from shape: shape structure differs");
  }
  if (src.IsTuple()) {
    if (ShapeUtil::TupleElementCount(src) !=
        ShapeUtil::TupleElementCount(*dst)) {
      return InvalidArgument(
          "cannot copy layout from shape: tuple element count differs");
    }
    for (int64 i = 0; i < ShapeUtil::TupleElementCount(src); ++i) {
      TF_RETURN_IF_ERROR(CopyLayoutInternal(src.tuple_shapes(i),
                                            dst->mutable_tuple_shapes(i)));
    }
  } else {
    if (src.has_layout()) {
      if (src.rank() != dst->rank()) {
        return InvalidArgument("cannot copy layout from shape: ranks differs");
      }
      TF_RETURN_IF_ERROR(
          LayoutUtil::ValidateLayoutForShape(src.layout(), *dst));
      *dst->mutable_layout() = src.layout();
    } else {
      dst->clear_layout();
    }
  }
  return Status::OK();
}

}  // namespace

/* static */
Status LayoutUtil::CopyLayoutBetweenShapes(const Shape& src, Shape* dst) {
  return CopyLayoutInternal(src, dst);
}

/* static */ bool LayoutUtil::LayoutsInShapesEqual(const Shape& lhs,
                                                   const Shape& rhs) {
  if (lhs.IsTuple()) {
    if (!rhs.IsTuple() || ShapeUtil::TupleElementCount(lhs) !=
                              ShapeUtil::TupleElementCount(rhs)) {
      return false;
    }
    for (int i = 0; i < ShapeUtil::TupleElementCount(lhs); ++i) {
      if (!LayoutsInShapesEqual(lhs.tuple_shapes(i), rhs.tuple_shapes(i))) {
        return false;
      }
    }
    return true;
  } else if (lhs.IsArray()) {
    return lhs.rank() == rhs.rank() &&
           LayoutUtil::Equal(lhs.layout(), rhs.layout());
  } else {
    // Layouts of non-array and non-tuple shapes is ignored.
    return true;
  }
}

/* static */ bool LayoutUtil::AreDimensionsConsecutive(
    const Layout& layout, absl::Span<const int64> dims) {
  CHECK(IsDense(layout));
  std::vector<int64> positions_in_layout;
  for (int64 dim : dims) {
    positions_in_layout.push_back(
        PositionInContainer(layout.minor_to_major(), dim));
  }
  absl::c_sort(positions_in_layout);
  for (size_t i = 1; i < positions_in_layout.size(); ++i) {
    if (1 != positions_in_layout[i] - positions_in_layout[i - 1]) {
      return false;
    }
  }
  return true;
}

/*static*/ Layout LayoutUtil::MoveDimToMajor(const Layout& layout, int64 dim) {
  if (dim == MinorToMajor(layout).back()) return layout;
  Layout ret = layout;
  ret.clear_minor_to_major();
  for (auto d : MinorToMajor(layout)) {
    if (d != dim) {
      ret.add_minor_to_major(d);
    }
  }
  ret.add_minor_to_major(dim);
  return ret;
}

/*static*/ size_t LayoutUtil::Hash(const Layout& layout) {
  using tensorflow::hash;
  using tensorflow::Hash64Combine;

  size_t hash_value = hash<Format>()(layout.format());

  for (int64 minor_to_major : layout.minor_to_major()) {
    hash_value = Hash64Combine(hash_value, hash<int64>()(minor_to_major));
  }
  for (const Tile& tile : layout.tiles()) {
    for (int64 tile_dim : tile.dimensions()) {
      hash_value = Hash64Combine(hash_value, hash<int64>()(tile_dim));
    }
  }
  hash_value = Hash64Combine(hash_value, layout.element_size_in_bits());
  hash_value = Hash64Combine(hash_value, layout.memory_space());

  return hash_value;
}

}  // namespace xla
