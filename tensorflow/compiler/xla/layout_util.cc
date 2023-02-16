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
#include <numeric>
#include <optional>
#include <random>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/printer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/platform/logging.h"

namespace xla {
namespace {

// Internal helper for GetDefaultLayoutForShape and SetToDefaultLayout. Sets
// minor_to_major to the value that represents the default layout.
template <typename T>
void SetDefaultLayoutToContainer(T* minor_to_major) {
  // The default XLA layout is major-to-minor (dim 0 is major).
  // For more information on XLA layouts, see:
  // https://www.tensorflow.org/performance/xla/shapes
  const int64_t size = minor_to_major->size();
  for (int64_t i = 0; i < size; ++i) {
    (*minor_to_major)[i] = size - 1 - i;
  }
}

absl::string_view BoolToString(bool b) { return b ? "true" : "false"; }

}  // namespace

/* static */ Layout LayoutUtil::MakeLayout(
    absl::Span<const int64_t> minor_to_major,
    absl::Span<const DimLevelType> dim_level_types,
    absl::Span<const bool> dim_unique, absl::Span<const bool> dim_ordered,
    absl::Span<const Tile> tiles, PrimitiveType index_primitive_type,
    PrimitiveType pointer_primitive_type, int64_t memory_space,
    std::optional<Shape> physical_shape,
    int64_t dynamic_shape_metadata_prefix_bytes) {
  Layout layout;
  for (int64_t dimension_number : minor_to_major) {
    layout.add_minor_to_major(dimension_number);
  }
  for (DimLevelType dim_level_type : dim_level_types) {
    layout.add_dim_level_type(dim_level_type);
  }
  for (bool unique : dim_unique) {
    layout.add_dim_unique(unique);
  }
  for (bool ordered : dim_ordered) {
    layout.add_dim_ordered(ordered);
  }
  for (const Tile& tile : tiles) {
    for (int64_t dim : tile.dimensions()) {
      if (dim < 0 && dim != Tile::kCombineDimension) {
        LOG(FATAL)
            << "Tile dimension size needs to be minimum int64_t value if "
               "it's negative. Value is "
            << dim;
      }
    }
    *layout.add_tiles() = tile;
  }
  layout.set_index_primitive_type(index_primitive_type);
  layout.set_pointer_primitive_type(pointer_primitive_type);
  layout.set_memory_space(memory_space);
  if (physical_shape != std::nullopt) {
    *layout.mutable_physical_shape() = *std::move(physical_shape);
  }
  layout.set_dynamic_shape_metadata_prefix_bytes(
      dynamic_shape_metadata_prefix_bytes);
  return layout;
}

/* static */ Layout LayoutUtil::MakeDescendingLayout(int64_t rank) {
  std::vector<int64_t> layout(rank);
  std::iota(layout.rbegin(), layout.rend(), static_cast<int64_t>(0));
  return MakeLayout(layout);
}

/* static */ Layout LayoutUtil::MakeAscendingLayout(int64_t rank) {
  std::vector<int64_t> layout(rank);
  std::iota(layout.begin(), layout.end(), static_cast<int64_t>(0));
  return MakeLayout(layout);
}

/* static */ Layout LayoutUtil::MakeLayoutFromMajorToMinor(
    absl::Span<const int64_t> major_to_minor) {
  Layout layout;
  for (int i = major_to_minor.size() - 1; i >= 0; i--) {
    layout.add_minor_to_major(major_to_minor[i]);
  }
  return layout;
}

namespace {

// Internal helper that creates a default layout for an array of the given rank.
Layout CreateDefaultLayoutForRank(int64_t rank) {
  Layout layout;
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

/* static */ Layout LayoutUtil::GetDefaultLayoutForRank(int64_t rank) {
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
    return OkStatus();
  } else if (shape.IsArray()) {
    if (!shape.has_layout()) {
      if (allow_missing_layouts) {
        return OkStatus();
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
    return OkStatus();
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
    return OkStatus();
  }

  if (layout.minor_to_major_size() != shape.rank()) {
    return InvalidArgument(
        "layout minor_to_major field contains %d elements, "
        "but shape is rank %d: {%s}; shape: %s",
        layout.minor_to_major_size(), shape.rank(),
        absl::StrJoin(layout.minor_to_major(), ", "), shape.ShortDebugString());
  }

  std::vector<bool> dimensions_in_layout(shape.rank(), false);
  for (int64_t i = 0; i < shape.rank(); ++i) {
    int64_t dim = layout.minor_to_major(i);
    if (dim < 0 || dim >= shape.rank()) {
      return InvalidArgument(
          "layout minor_to_major field has out-of-bounds value: {%s}; shape: "
          "%s",
          absl::StrJoin(layout.minor_to_major(), ", "),
          shape.ShortDebugString());
    }
    if (dimensions_in_layout[dim]) {
      return InvalidArgument(
          "layout minor_to_major field has duplicate values: {%s}; shape: %s",
          absl::StrJoin(layout.minor_to_major(), ", "),
          shape.ShortDebugString());
    }
    dimensions_in_layout[dim] = true;
  }

  if (!layout.dim_level_types().empty()) {
    if (layout.dim_level_types().size() != shape.rank()) {
      return InvalidArgument(
          "layout dim_level_types field contains %d elements, but shape is "
          "rank %d: {%s}; shape: %s",
          layout.dim_level_types_size(), shape.rank(),
          absl::StrJoin(layout.dim_level_types(), ", ",
                        [](std::string* out, DimLevelType dim_level_type) {
                          absl::StrAppend(out,
                                          DimLevelType_Name(dim_level_type));
                        }),
          shape.ShortDebugString());
    }
  }

  if (!layout.dim_unique().empty()) {
    if (layout.dim_unique().size() != shape.rank()) {
      return InvalidArgument(
          "layout dim_unique field contains %d elements, but shape is "
          "rank %d: {%s}; shape: %s",
          layout.dim_unique_size(), shape.rank(),
          absl::StrJoin(layout.dim_unique(), ", ",
                        [](std::string* out, bool dim_unique) {
                          absl::StrAppend(out, BoolToString(dim_unique));
                        }),
          shape.ShortDebugString());
    }
  }

  if (!layout.dim_ordered().empty()) {
    if (layout.dim_ordered().size() != shape.rank()) {
      return InvalidArgument(
          "layout dim_unique field contains %d elements, but shape is "
          "rank %d: {%s}; shape: %s",
          layout.dim_ordered_size(), shape.rank(),
          absl::StrJoin(layout.dim_unique(), ", ",
                        [](std::string* out, bool dim_unique) {
                          absl::StrAppend(out, BoolToString(dim_unique));
                        }),
          shape.ShortDebugString());
    }
  }

  if (LayoutUtil::IsSparse(layout)) {
    if (layout.tiles_size() > 0) {
      return InvalidArgument(
          "layout has tiles, but the shape is a sparse array: %s",
          shape.ShortDebugString());
    }
    if (layout.has_physical_shape()) {
      TF_RETURN_IF_ERROR(ShapeUtil::ValidateShape(layout.physical_shape()));
      TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
          layout.physical_shape(),
          [&](const Shape& subshape, const ShapeIndex& index) {
            if (subshape.has_layout() &&
                subshape.layout().has_physical_shape()) {
              return InvalidArgument(
                  "layout has a physical_shape, whose layout also has a "
                  "physical shape: %s",
                  shape.ShortDebugString());
            }
            return OkStatus();
          }));
      if (layout.index_primitive_type() != PRIMITIVE_TYPE_INVALID &&
          !primitive_util::IsUnsignedIntegralType(
              layout.index_primitive_type())) {
        return InvalidArgument(
            "index_primitive_type is not an unsigned integer type: %s",
            shape.ShortDebugString());
      }
      if (layout.pointer_primitive_type() != PRIMITIVE_TYPE_INVALID &&
          !primitive_util::IsUnsignedIntegralType(
              layout.pointer_primitive_type())) {
        return InvalidArgument(
            "pointer_primitive_type is not an unsigned integer type: "
            "%s",
            shape.ShortDebugString());
      }
    }
  } else {
    if (layout.index_primitive_type() != PRIMITIVE_TYPE_INVALID) {
      return InvalidArgument(
          "layout has a index_primitive_type, but is not a sparse array: %s",
          shape.ShortDebugString());
    }
    if (layout.pointer_primitive_type() != PRIMITIVE_TYPE_INVALID) {
      return InvalidArgument(
          "layout has a pointer_primitive_type, but is not a sparse array: %s",
          shape.ShortDebugString());
    }
    if (layout.has_physical_shape()) {
      return InvalidArgument(
          "layout has a physical_shape, but is not a sparse array: %s",
          shape.ShortDebugString());
    }
    for (const auto& tile : layout.tiles()) {
      if (tile.dimensions().empty() ||
          absl::c_any_of(tile.dimensions(),
                         [](int64_t dim) { return dim == 0; })) {
        return InvalidArgument("layout has invalid tiles: %s",
                               shape.ShortDebugString());
      }
    }
  }

  for (int64_t dim = 0; dim < shape.rank(); ++dim) {
    DimLevelType dim_level_type = GetDimLevelType(layout, dim);
    bool dim_unique = DimUnique(layout, dim);
    bool dim_ordered = DimOrdered(layout, dim);
    if (!ValidateDimLevel(dim_level_type, dim_unique, dim_ordered)) {
      return InvalidArgument(
          "layout dimension %d has invalid level encoding %s%s%s: %s", dim,
          DimLevelType_Name(dim_level_type), dim_unique ? "" : ", non-unique",
          dim_ordered ? "" : ", non-ordered", shape.ShortDebugString());
    }
  }

  return OkStatus();
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

/* static */ void LayoutUtil::ClearTiles(Shape* shape) {
  ShapeUtil::ForEachMutableSubshape(
      shape, [](Shape* subshape, const ShapeIndex&) {
        if (subshape->has_layout()) {
          if (subshape->has_layout()) {
            subshape->mutable_layout()->clear_tiles();
          }
        }
      });
}

/* static */ bool LayoutUtil::IsDenseArray(const Shape& shape) {
  return shape.IsArray() && (!shape.has_layout() || IsDense(shape.layout()));
}

/* static */ bool LayoutUtil::IsSparseArray(const Shape& shape) {
  return shape.IsArray() && shape.has_layout() && IsSparse(shape.layout());
}

/* static */ bool LayoutUtil::IsCOOArray(const Shape& shape) {
  return shape.IsArray() && shape.has_layout() && IsCOO(shape.layout());
}

/* static */ bool LayoutUtil::IsCSRArray(const Shape& shape) {
  return shape.IsArray() && shape.rank() == 2 && shape.has_layout() &&
         IsCSR(shape.layout());
}

/* static */ bool LayoutUtil::IsCSCArray(const Shape& shape) {
  return shape.IsArray() && shape.rank() == 2 && shape.has_layout() &&
         IsCSC(shape.layout());
}

/* static */ bool LayoutUtil::IsDense(const Layout& layout) {
  return absl::c_all_of(
      layout.dim_level_types(),
      [](DimLevelType dim_level_type) { return dim_level_type == DIM_DENSE; });
}

/* static */ bool LayoutUtil::IsSparse(const Layout& layout) {
  return !IsDense(layout);
}

/* static */ bool LayoutUtil::IsCOO(const Layout& layout) {
  return !layout.dim_level_types().empty() &&
         layout.dim_level_type(0) == DIM_COMPRESSED &&
         absl::c_all_of(layout.dim_level_types().subspan(1),
                        [](DimLevelType dim_level_type) {
                          return dim_level_type == DIM_SINGLETON;
                        });
}

/* static */ bool LayoutUtil::IsCSR(const Layout& layout) {
  return IsMonotonicWithDim0Major(layout) &&
         layout.dim_level_types() ==
             absl::Span<const DimLevelType>{DIM_DENSE, DIM_COMPRESSED};
}

/* static */ bool LayoutUtil::IsCSC(const Layout& layout) {
  return IsMonotonicWithDim0Minor(layout) &&
         layout.dim_level_types() ==
             absl::Span<const DimLevelType>{DIM_DENSE, DIM_COMPRESSED};
}

/* static */ bool LayoutUtil::IsMonotonicWithDim0Minor(const Layout& layout) {
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end());
}

/* static */ bool LayoutUtil::IsMonotonicWithDim0Major(const Layout& layout) {
  return std::is_sorted(layout.minor_to_major().begin(),
                        layout.minor_to_major().end(), std::greater<int64_t>());
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
  return shape.has_layout();
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

/* static */ std::vector<int64_t> LayoutUtil::MakeLogicalToPhysical(
    const Layout& layout) {
  std::vector<int64_t> logical_to_physical(layout.minor_to_major_size());
  for (int64_t physical = 0, end = logical_to_physical.size(); physical < end;
       ++physical) {
    const int64_t logical = Major(layout, physical);
    logical_to_physical[logical] = physical;
  }
  return logical_to_physical;
}

/* static */ void LayoutUtil::PrintHumanString(Printer* printer,
                                               const Layout& layout) {
  layout.Print(printer);
}

/* static */ std::string LayoutUtil::HumanString(const Layout& layout) {
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
    for (int64_t i = 0; i < ShapeUtil::TupleElementCount(src); ++i) {
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
  return OkStatus();
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
  }
  if (lhs.IsArray()) {
    if (lhs.rank() != rhs.rank()) {
      return false;
    }
    if (!lhs.has_layout() && !rhs.has_layout()) {
      return true;
    }
    if (!lhs.has_layout() || !rhs.has_layout()) {
      return false;
    }
    return LayoutUtil::Equal(lhs.layout(), rhs.layout());
  }
  // Layouts of non-array and non-tuple shapes is ignored.
  return true;
}

/* static */ bool LayoutUtil::AreDimensionsConsecutive(
    const Layout& layout, absl::Span<const int64_t> dims) {
  absl::InlinedVector<int64_t, 8> positions_in_layout;
  for (int64_t dim : dims) {
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

/*static*/ Layout LayoutUtil::MoveDimToMajor(const Layout& layout,
                                             int64_t dim) {
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

/*static*/ int64_t LayoutUtil::LinearIndex(const Shape& shape,
                                           absl::Span<const int64_t> indices) {
  CHECK(shape.IsArray());
  CHECK(shape.has_layout());
  const int rank = shape.rank();
  CHECK_EQ(rank, indices.size());

  if (rank == 0) {
    return 0;
  }
  if (rank == 1) {
    return indices[0];
  }

  Tile tile = {};
  if (!shape.layout().tiles().empty()) {
    tile = shape.layout().tiles()[0];
  }

  int64_t linear_index = 0;
  int64_t tile_multiplier = 1;
  // Initialize to number of elements in a tile.
  for (int64_t i : tile.dimensions()) {
    tile_multiplier *= i;
  }
  int64_t within_tile_multiplier = 1;

  // We only look at the top-level tile.
  for (int64_t minor = 0; minor < rank; minor++) {
    int64_t logical_dim = Minor(shape.layout(), minor);
    int64_t shape_dim_size = shape.dimensions(logical_dim);
    int64_t index = indices[logical_dim];

    if (minor < tile.dimensions().size()) {
      int64_t tile_dim_size =
          tile.dimensions()[tile.dimensions().size() - 1 - minor];
      linear_index += tile_multiplier * (index / tile_dim_size) +
                      within_tile_multiplier * (index % tile_dim_size);
      tile_multiplier *= CeilOfRatio(shape_dim_size, tile_dim_size);
      within_tile_multiplier *= tile_dim_size;
    } else {
      linear_index += index * tile_multiplier;
      tile_multiplier *= shape_dim_size;
    }
  }
  return linear_index;
}

/*static*/ int64_t LayoutUtil::MemorySpace(const Shape& shape) {
  return shape.has_layout() ? shape.layout().memory_space()
                            : Layout::kDefaultMemorySpace;
}

/*static*/ DimLevelType LayoutUtil::GetDimLevelType(const Layout& layout,
                                                    int64_t dim) {
  if (layout.dim_level_types_size() == 0) {
    return DIM_DENSE;
  }
  CHECK_LT(dim, layout.dim_level_types_size());
  return layout.dim_level_type(dim);
}

/*static*/ bool LayoutUtil::DimUnique(const Layout& layout, int64_t dim) {
  if (layout.dim_unique_size() == 0) {
    return true;
  }
  CHECK_LT(dim, layout.dim_unique_size());
  return layout.dim_unique(dim);
}

/*static*/ bool LayoutUtil::DimOrdered(const Layout& layout, int64_t dim) {
  if (layout.dim_ordered_size() == 0) {
    return true;
  }
  CHECK_LT(dim, layout.dim_ordered_size());
  return layout.dim_ordered(dim);
}

bool LayoutUtil::ValidateDimLevel(DimLevelType dim_level_type, bool dim_unique,
                                  bool dim_ordered) {
  switch (dim_level_type) {
    case DIM_DENSE:
      return dim_unique && dim_ordered;
    case DIM_COMPRESSED:
    case DIM_SINGLETON:
      return true;
    default:
      return false;
  }
}

}  // namespace xla
