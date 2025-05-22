/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/layout_util.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/primitive_util.h"
#include "xla/printer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"

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

}  // namespace

/* static */ Layout LayoutUtil::MakeLayout(
    absl::Span<const int64_t> minor_to_major, absl::Span<const Tile> tiles,
    int64_t tail_padding_alignment_in_elements,
    PrimitiveType index_primitive_type, PrimitiveType pointer_primitive_type,
    int64_t element_size_in_bits, int64_t memory_space,
    absl::Span<const SplitConfig> split_configs,
    std::optional<Shape> physical_shape,
    int64_t dynamic_shape_metadata_prefix_bytes) {
  Layout layout;
  for (int64_t dimension_number : minor_to_major) {
    layout.add_minor_to_major(dimension_number);
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
  layout.set_tail_padding_alignment_in_elements(
      tail_padding_alignment_in_elements);
  layout.set_index_primitive_type(index_primitive_type);
  layout.set_pointer_primitive_type(pointer_primitive_type);
  layout.set_element_size_in_bits(element_size_in_bits);
  layout.set_memory_space(memory_space);
  for (const SplitConfig& split_config : split_configs) {
    layout.add_split_configs(split_config);
  }
  if (physical_shape != std::nullopt) {
    *layout.mutable_physical_shape() = *std::move(physical_shape);
  }
  layout.set_dynamic_shape_metadata_prefix_bytes(
      dynamic_shape_metadata_prefix_bytes);
  return layout;
}

/* static */ Layout LayoutUtil::MakeDescendingLayout(int64_t num_dims) {
  std::vector<int64_t> layout(num_dims);
  std::iota(layout.rbegin(), layout.rend(), static_cast<int64_t>(0));
  return MakeLayout(layout);
}

/* static */ bool LayoutUtil::HasDescendingLayout(const Layout& layout) {
  return absl::c_is_sorted(layout.minor_to_major(), std::greater<int64_t>());
}

/* static */ Layout LayoutUtil::MakeAscendingLayout(int64_t num_dims) {
  std::vector<int64_t> layout(num_dims);
  std::iota(layout.begin(), layout.end(), static_cast<int64_t>(0));
  return MakeLayout(layout);
}

/* static */ bool LayoutUtil::HasAscendingLayout(const Layout& layout) {
  return absl::c_is_sorted(layout.minor_to_major(), std::less<int64_t>());
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

// Internal helper that creates a default layout for an array of the given
// number of dimensions.
Layout CreateDefaultLayoutForRank(int64_t num_dims) {
  Layout layout;
  auto* minor_to_major = layout.mutable_minor_to_major();
  minor_to_major->resize(num_dims, 0);
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
  return CreateDefaultLayoutForRank(shape.dimensions().size());
}

/* static */ Layout LayoutUtil::GetDefaultLayoutForRank(int64_t num_dims) {
  return CreateDefaultLayoutForRank(num_dims);
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
  } else if (shape->IsArray()) {
    auto* minor_to_major = shape->mutable_layout()->mutable_minor_to_major();
    minor_to_major->resize(shape->dimensions().size(), 0);
    SetDefaultLayoutToContainer(minor_to_major);
  } else {
    // Opaque, token types etc. have no layout.
  }
}

/* static */ Shape LayoutUtil::GetWithDefaultLayout(const Shape& shape) {
  Shape copy(shape);
  LayoutUtil::SetToDefaultLayout(&copy);
  return copy;
}

/* static */ void LayoutUtil::SetToDefaultLayout(ProgramShape* program_shape) {
  for (int i = 0; i < program_shape->parameters_size(); ++i) {
    LayoutUtil::SetToDefaultLayout(program_shape->mutable_parameters(i));
  }
  LayoutUtil::SetToDefaultLayout(program_shape->mutable_result());
}

/* static */ absl::Status LayoutUtil::ValidateLayoutInShape(
    const Shape& shape, bool allow_missing_layouts) {
  if (shape.IsTuple()) {
    for (auto& element_shape : shape.tuple_shapes()) {
      TF_RETURN_IF_ERROR(
          ValidateLayoutInShape(element_shape, allow_missing_layouts));
    }
    return absl::OkStatus();
  } else if (shape.IsArray()) {
    if (!shape.has_layout()) {
      if (allow_missing_layouts) {
        return absl::OkStatus();
      }
      return InvalidArgument("shape %s does not have a layout",
                             ShapeUtil::HumanString(shape));
    }
    return ValidateLayoutForShape(shape.layout(), shape);
  } else {
    // Token, opaque, etc. shape.
    return absl::OkStatus();
  }
}

/* static */ absl::Status LayoutUtil::ValidateLayoutForShape(
    const Layout& layout, const Shape& shape) {
  if (shape.IsTuple()) {
    return InvalidArgument("a single Layout is not valid for tuple shapes");
  }

  if (!shape.IsArray()) {
    return absl::OkStatus();
  }

  if (layout.minor_to_major_size() != shape.dimensions().size()) {
    return InvalidArgument(
        "layout minor_to_major field contains %d elements, "
        "but shape has %d dimensions: {%s}; shape: %s",
        layout.minor_to_major_size(), shape.dimensions().size(),
        absl::StrJoin(layout.minor_to_major(), ", "), shape.ToString());
  }

  absl::InlinedVector<bool, InlineRank()> dimensions_in_layout(
      shape.dimensions().size(), false);
  for (int64_t i = 0; i < shape.dimensions().size(); ++i) {
    int64_t dim = layout.minor_to_major(i);
    if (dim < 0 || dim >= shape.dimensions().size()) {
      return InvalidArgument(
          "layout minor_to_major field has out-of-bounds value: {%s}; shape: "
          "%s",
          absl::StrJoin(layout.minor_to_major(), ", "), shape.ToString());
    }
    if (dimensions_in_layout[dim]) {
      return InvalidArgument(
          "layout minor_to_major field has duplicate values: {%s}; shape: %s",
          absl::StrJoin(layout.minor_to_major(), ", "), shape.ToString());
    }
    dimensions_in_layout[dim] = true;
  }

  if (layout.tail_padding_alignment_in_elements() <= 0) {
    return InvalidArgument(
        "layout tail_padding_alignment_in_elements field is <= 0: {%d}",
        layout.tail_padding_alignment_in_elements());
  }

  if (layout.index_primitive_type() != PRIMITIVE_TYPE_INVALID) {
    return InvalidArgument(
        "layout has a index_primitive_type, but is not a sparse array: %s",
        shape.ToString());
  }
  if (layout.pointer_primitive_type() != PRIMITIVE_TYPE_INVALID) {
    return InvalidArgument(
        "layout has a pointer_primitive_type, but is not a sparse array: %s",
        shape.ToString());
  }
  if (layout.has_physical_shape()) {
    return InvalidArgument(
        "layout has a physical_shape, but is not a sparse array: %s",
        shape.ToString());
  }
  for (const auto& tile : layout.tiles()) {
    if (tile.dimensions().empty() ||
        absl::c_any_of(tile.dimensions(),
                       [](int64_t dim) { return dim == 0; })) {
      return InvalidArgument("layout has invalid tiles: %s", shape.ToString());
    }
  }

  if (layout.element_size_in_bits() < 0) {
    return InvalidArgument("layout element_size_in_bits field is negative: %d",
                           layout.element_size_in_bits());
  }

  return absl::OkStatus();
}

/* static */ void LayoutUtil::ClearLayout(Shape* shape) {
  if (shape->IsArray()) {
    shape->clear_layout();
  } else if (shape->IsTuple()) {
    for (auto& element_shape : *shape->mutable_tuple_shapes()) {
      ClearLayout(&element_shape);
    }
  }
}

/* static */ void LayoutUtil::ClearLayout(ProgramShape* program_shape) {
  for (int i = 0; i < program_shape->parameters_size(); ++i) {
    LayoutUtil::ClearLayout(program_shape->mutable_parameters(i));
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

/* static */ bool LayoutUtil::HasAnyLayout(const Shape& shape) {
  if (shape.IsTuple()) {
    // Tuple shape: all subshapes must have a layout.
    return absl::c_any_of(shape.tuple_shapes(),
                          [](const Shape& s) { return HasAnyLayout(s); });
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

/* static */ bool LayoutUtil::HasCustomElementSizeInBits(const Shape& shape) {
  if (shape.IsTuple()) {
    return absl::c_any_of(shape.tuple_shapes(),
                          LayoutUtil::HasCustomElementSizeInBits);
  } else if (!shape.IsArray()) {
    // Opaque or token types have no custom element size in bits.
    return false;
  }
  return shape.has_layout() && shape.layout().element_size_in_bits() != 0;
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
absl::Status CopyLayoutInternal(const Shape& src, Shape* dst) {
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
  } else if (src.IsArray()) {
    if (src.has_layout()) {
      if (src.dimensions().size() != dst->dimensions().size()) {
        return InvalidArgument("cannot copy layout from shape: ranks differs");
      }
      TF_RETURN_IF_ERROR(
          LayoutUtil::ValidateLayoutForShape(src.layout(), *dst));
      *dst->mutable_layout() = src.layout();
    } else {
      dst->clear_layout();
    }
  }
  return absl::OkStatus();
}

}  // namespace

/* static */
absl::Status LayoutUtil::CopyLayoutBetweenShapes(const Shape& src, Shape* dst) {
  return CopyLayoutInternal(src, dst);
}

/* static */ bool LayoutUtil::LayoutsInShapesEqual(
    const Shape& lhs, const Shape& rhs, std::optional<Layout::Equal> equal) {
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
    if (lhs.dimensions().size() != rhs.dimensions().size()) {
      return false;
    }
    if (!lhs.has_layout() && !rhs.has_layout()) {
      return true;
    }
    if (!lhs.has_layout() || !rhs.has_layout()) {
      return false;
    }

    if (equal.has_value()) {
      return equal.value()(lhs.layout(), rhs.layout());
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
  const int num_dims = shape.dimensions().size();
  CHECK_EQ(num_dims, indices.size());

  if (num_dims == 0) {
    return 0;
  }
  if (num_dims == 1) {
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
  for (int64_t minor = 0; minor < num_dims; minor++) {
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

/*static*/ bool LayoutUtil::ByteStridesIsMajorToMinor(
    absl::Span<const int64_t> byte_strides, absl::Span<const int64_t> dims,
    PrimitiveType element_type) {
  CHECK_EQ(dims.size(), byte_strides.size());
  int64_t stride = ShapeUtil::ByteSizeOfPrimitiveType(element_type);
  for (int i = dims.size() - 1; i >= 0; --i) {
    if (byte_strides[i] != stride) {
      return false;
    }
    stride *= dims[i];
  }
  return true;
}

/*static*/ int64_t LayoutUtil::MaxSplitSize(const Shape& shape, int64_t dim) {
  CHECK(shape.IsArray()) << ShapeUtil::HumanString(shape);
  if (!shape.has_layout()) {
    return shape.dimensions(dim);
  }
  const SplitConfig* split_config = nullptr;
  for (const SplitConfig& config : shape.layout().split_configs()) {
    if (Major(shape.layout(), config.dimension()) == dim) {
      split_config = &config;
      break;
    }
  }
  if (split_config != nullptr) {
    int64_t max_split_size = 0;
    int64_t last_split_index = 0;
    for (int split_index : split_config->split_indices()) {
      int64_t split_size = split_index - last_split_index;
      max_split_size = std::max(split_size, max_split_size);
      last_split_index = split_index;
    }
    max_split_size =
        std::max(max_split_size, shape.dimensions(dim) - last_split_index);
    return max_split_size;
  }
  return shape.dimensions(dim);
}

/*static*/ int64_t LayoutUtil::MaxElementsInPerSplit(const Shape& shape) {
  CHECK(shape.IsArray()) << ShapeUtil::HumanString(shape);
  int64_t max_elements_in = 1;
  for (int dim = 0; dim < shape.dimensions().size(); ++dim) {
    max_elements_in *= MaxSplitSize(shape, dim);
  }
  return max_elements_in;
}

/*static*/ std::optional<SplitConfig> LayoutUtil::GetSplitConfig(
    const Shape& shape) {
  CHECK_LE(shape.layout().split_configs_size(), 1);
  return shape.layout().split_configs_size() > 0
             ? std::make_optional(shape.layout().split_configs(0))
             : std::nullopt;
}

}  // namespace xla
