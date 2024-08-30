/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_LAYOUT_H_
#define XLA_LAYOUT_H_

#include <cstdint>
#include <limits>
#include <memory>
#include <ostream>
#include <string>

#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "xla/printer.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {

class Shape;

// Describes a tile used in tiling-based layout. Refer to
// g3doc/third_party/tensorflow/compiler/xla/g3doc/tiled_layout.md for
// details.
class Tile {
 public:
  Tile() = default;
  
  explicit Tile(absl::Span<const int64_t> dimensions)
      : dimensions_(dimensions.begin(), dimensions.end()) {}

  // De/Serialize a Tile to and from a TileProto.
  static Tile CreateFromProto(const TileProto& tile_proto) {
    return Tile(tile_proto.dimensions());
  }
  
  TileProto ToProto() const;
  void SetProto(TileProto& tile_proto) const;

  bool operator==(const Tile& other) const {
    return dimensions() == other.dimensions();
  }

  bool operator!=(const Tile& other) const {
    return !(*this == other);
  }

  void Print(Printer* printer) const;
  std::string ToString() const;

  // Returns the bound of the tile in the given dimension index.
  int64_t dimension(int i) const {
    return dimensions_[i];
  }

  // Returns the dimensions of the tile.
  absl::Span<const int64_t> dimensions() const {
    return dimensions_;
  }

  Tile& add_dimensions(int64_t value) {
    dimensions_.push_back(value);
    return *this;
  }

  Tile& clear_dimensions() {
    dimensions_.clear();
    return *this;
  }

  // This dimension size means the corresponding dimension in the shape is
  // combined with the next minor dimension before tiling is applied.
  static constexpr int64_t kCombineDimension =
      std::numeric_limits<int64_t>::min();

  template <typename H>
  friend H AbslHashValue(H h, const Tile& t) {
    return H::combine(std::move(h), t.dimensions_);
  }

 private:
  // The bounds of the tile.
  absl::InlinedVector<int64_t, 2> dimensions_;
};

using TileVector = absl::InlinedVector<Tile, 3>;

// Describes how data is split between different memories. Each SplitConfig
// object represents a split in one dimension. Each SplitConfig is associated
// with a vector of split indices which point to the points in the iteration
// where the splits occur. For example, if the dimension contains 1024 elements,
// a split indices value of {512} indicates splitting this dimension into two
// right through the middle. The dimension here refers to the physical dimension
// such that 0 is the majormost dimension and rank-1 is the minormost dimension.
class SplitConfig {
 public:
  SplitConfig(int64_t dimension, absl::Span<const int64_t> split_indices)
      : dimension_(dimension),
        split_indices_(split_indices.begin(), split_indices.end()) {}

  static SplitConfig CreateFromProto(
      const SplitConfigProto& split_config_proto) {
    return SplitConfig(split_config_proto.dimension(),
                       split_config_proto.split_indices());
  }
  
  SplitConfigProto ToProto() const;
  void SetProto(SplitConfigProto& split_config_proto) const;

  bool operator==(const SplitConfig& other) const {
    return dimension() == other.dimension() &&
           split_indices() == other.split_indices();
  }
  
  bool operator!=(const SplitConfig& other) const {
    return !(*this == other);
  }

  std::string ToString() const;

  // Returns the dimension that is split.
  int64_t dimension() const {
    return dimension_;
  }

  SplitConfig& set_dimension(int64_t dimension) {
    dimension_ = dimension;
    return *this;
  }

  // Returns the indices where splits occur.
  absl::Span<const int64_t> split_indices() const {
    return split_indices_;
  }

  int64_t split_indices(int64_t idx) const {
    return split_indices_.at(idx);
  }

  int64_t split_indices_size() const {
    return split_indices_.size();
  }

  SplitConfig& add_split_indices(int64_t split_index) {
    split_indices_.push_back(split_index);
    return *this;
  }

  SplitConfig& clear_split_indices() {
    split_indices_.clear();
    return *this;
  }

  template <typename H>
  friend H AbslHashValue(H h, const SplitConfig& t) {
    return H::combine(std::move(h), t.dimension_, t.split_indices_);
  }

 private:
  int64_t dimension_;
  absl::InlinedVector<int64_t, 1> split_indices_;
};

// TODO: Rename the `dim_level_types` field to `lvl_types`, so that it
// matches `mlir::sparse_tensor::SparseTensorEncodingAttr`.
class Layout {
 public:
  Layout();
  Layout(const Layout& other);
  Layout(Layout&& other);
  ~Layout();

  // Constructs a dense layout with the given minor-to-major order.
  explicit Layout(absl::Span<const int64_t> minor_to_major);

  explicit Layout(absl::Span<const int64_t> minor_to_major,
                  absl::Span<const Tile> tiles, int64_t element_size_in_bits);

  // Constructs a dense tiled layout with the given minor-to-major order, dim
  // level types, and tiles.
  explicit Layout(absl::Span<const int64_t> minor_to_major,
                  absl::Span<const DimLevelType> dim_level_types,
                  absl::Span<const bool> dim_unique,
                  absl::Span<const bool> dim_ordered,
                  absl::Span<const Tile> tiles,
                  int64_t tail_padding_alignment_in_elements = 1,
                  PrimitiveType index_primitive_type = PRIMITIVE_TYPE_INVALID,
                  PrimitiveType pointer_primitive_type = PRIMITIVE_TYPE_INVALID,
                  int64_t element_size_in_bits = 0, int64_t memory_space = 0,
                  absl::Span<const SplitConfig> split_configs = {},
                  std::unique_ptr<Shape> physical_shape = nullptr,
                  int64_t dynamic_shape_metadata_prefix_bytes = 0);

  Layout& operator=(const Layout& other);
  Layout& operator=(Layout&& other);

  // Construct a shape from a LayoutProto.
  static Layout CreateFromProto(const LayoutProto& proto);

  // Returns a LayoutProto representation of the Layout.
  LayoutProto ToProto() const;

  // Sets a LayoutProto to the representation of the Layout.
  void SetProto(LayoutProto& proto) const;

  // Prints a human-readable string that represents this layout.
  void Print(Printer* printer) const;

  // Returns a human-readable string that represents this layout.
  std::string ToString() const;

  // Equal is a configurable functor to check the equality of two layouts.
  //
  // Examples:
  //
  // - Comparing two layouts ignoring their difference in tiles:
  //   Equal().IgnoreTiles()(layout1, layout2);
  class Equal {
   public:
    Equal() = default;

    bool operator()(const Layout& lhs, const Layout& rhs);

    Equal& IgnoreTiles() {
      ignore_tiles_ = true;
      return *this;
    }

    Equal& IgnoreTailPaddingAlignmentInElements() {
      ignore_tail_padding_alignment_in_elements_ = true;
      return *this;
    }

    Equal& IgnoreIndexPrimitiveType() {
      ignore_index_primitive_type_ = true;
      return *this;
    }

    Equal& IgnorePointerPrimitiveType() {
      ignore_pointer_primitive_type_ = true;
      return *this;
    }

    Equal& IgnoreMemorySpace() {
      ignore_memory_space_ = true;
      return *this;
    }

    Equal& IgnoreSplitConfigs() {
      ignore_split_configs_ = true;
      return *this;
    }

    Equal& IgnorePhysicalShape() {
      ignore_physical_shape_ = true;
      return *this;
    }

    Equal& IgnoreElementSize() {
      ignore_element_size_ = true;
      return *this;
    }

    Equal& MinorToMajorOnly() {
      return IgnoreTiles()
          .IgnoreIndexPrimitiveType()
          .IgnorePointerPrimitiveType()
          .IgnoreMemorySpace()
          .IgnorePhysicalShape()
          .IgnoreElementSize()
          .IgnoreTailPaddingAlignmentInElements();
    }

   private:
    bool ignore_tiles_ = false;
    bool ignore_tail_padding_alignment_in_elements_ = false;
    bool ignore_element_size_ = false;
    bool ignore_index_primitive_type_ = false;
    bool ignore_pointer_primitive_type_ = false;
    bool ignore_memory_space_ = false;
    bool ignore_split_configs_ = false;
    bool ignore_physical_shape_ = false;
  };

  bool operator==(const Layout& other) const {
    return Equal()(*this, other);
  }

  bool operator!=(const Layout& other) const {
    return !(*this == other);
  }

  // Accessors and Mutators
  int64_t dimension(int i) const { return minor_to_major_[i]; }
  absl::Span<const int64_t> minor_to_major() const { return minor_to_major_; }
  Layout& set_minor_to_major(absl::Span<const int64_t> minor_to_major) {
    minor_to_major_.assign(minor_to_major.begin(), minor_to_major.end());
    return *this;
  }

  absl::Span<const Tile> tiles() const { return tiles_; }
  Layout& set_tiles(absl::Span<const Tile> tiles) {
    tiles_.assign(tiles.begin(), tiles.end());
    return *this;
  }

  int64_t tail_padding_alignment_in_elements() const {
    return tail_padding_alignment_in_elements_;
  }

  Layout& set_tail_padding_alignment_in_elements(int64_t tail_padding_alignment_in_elements) {
    tail_padding_alignment_in_elements_ = tail_padding_alignment_in_elements;
    return *this;
  }

  PrimitiveType index_primitive_type() const { return index_primitive_type_; }
  Layout& set_index_primitive_type(PrimitiveType index_primitive_type) {
    index_primitive_type_ = index_primitive_type;
    return *this;
  }

  PrimitiveType pointer_primitive_type() const { return pointer_primitive_type_; }
  Layout& set_pointer_primitive_type(PrimitiveType pointer_primitive_type) {
    pointer_primitive_type_ = pointer_primitive_type;
    return *this;
  }

  int64_t memory_space() const { return memory_space_; }
  Layout& set_memory_space(int64_t memory_space) {
    memory_space_ = memory_space;
    return *this;
  }

  int64_t element_size_in_bits() const { return element_size_in_bits_; }
  Layout& set_element_size_in_bits(int64_t element_size_in_bits) {
    element_size_in_bits_ = element_size_in_bits;
    return *this;
  }

  // Returns the DimLevelType for a given dimension.
  DimLevelType dim_level_type(int i) const { return dim_level_types_[i]; }

  // Returns the DimLevelTypes.
  absl::Span<const DimLevelType> dim_level_types() const {
    return dim_level_types_;
  }

  Layout& set_dim_level_types(absl::Span<const DimLevelType> dim_level_types) {
    dim_level_types_.assign(dim_level_types.begin(), dim_level_types.end());
    return *this;
  }

  // Returns true if the dimension is unique.
  bool dim_unique(int i) const { return dim_unique_[i]; }

  // Returns the uniqueness of dimensions.
  absl::Span<const bool> dim_unique() const { return dim_unique_; }

  Layout& set_dim_unique(absl::Span<const bool> dim_unique) {
    dim_unique_.assign(dim_unique.begin(), dim_unique.end());
    return *this;
  }

  // Returns true if the dimension is ordered.
  bool dim_ordered(int i) const { return dim_ordered_[i]; }

  // Returns the order of dimensions.
  absl::Span<const bool> dim_ordered() const { return dim_ordered_; }

  Layout& set_dim_ordered(absl::Span<const bool> dim_ordered) {
    dim_ordered_.assign(dim_ordered.begin(), dim_ordered.end());
    return *this;
  }

  // Returns split configurations.
  absl::Span<const SplitConfig> split_configs() const { return split_configs_; }

  // Sets split configurations.
  Layout& set_split_configs(absl::Span<const SplitConfig> split_configs) {
    split_configs_.assign(split_configs.begin(), split_configs.end());
    return *this;
  }

  // Returns the physical shape.
  const Shape* physical_shape() const { return physical_shape_.get(); }

  // Sets the physical shape.
  Layout& set_physical_shape(std::unique_ptr<Shape> physical_shape) {
    physical_shape_ = std::move(physical_shape);
    return *this;
  }

  // Returns the number of bytes in the dynamic shape metadata prefix.
  int64_t dynamic_shape_metadata_prefix_bytes() const {
    return dynamic_shape_metadata_prefix_bytes_;
  }

  // Sets the number of bytes in the dynamic shape metadata prefix.
  Layout& set_dynamic_shape_metadata_prefix_bytes(int64_t bytes) {
    dynamic_shape_metadata_prefix_bytes_ = bytes;
    return *this;
  }

 private:
  // The minor-to-major order of dimensions in this layout.
  absl::InlinedVector<int64_t, 4> minor_to_major_;

  // The tiles used in the layout.
  TileVector tiles_;

  // Alignment for tail padding in elements.
  int64_t tail_padding_alignment_in_elements_;

  // Primitive types used in indexing and pointers.
  PrimitiveType index_primitive_type_;
  PrimitiveType pointer_primitive_type_;

  // Memory space for the layout.
  int64_t memory_space_;

  // Element size in bits.
  int64_t element_size_in_bits_;

  // DimLevelType for each dimension.
  absl::InlinedVector<DimLevelType, 4> dim_level_types_;

  // Indicates if each dimension is unique.
  absl::InlinedVector<bool, 4> dim_unique_;

  // Indicates if each dimension is ordered.
  absl::InlinedVector<bool, 4> dim_ordered_;

  // Split configurations.
  absl::InlinedVector<SplitConfig, 2> split_configs_;

  // Physical shape if applicable.
  std::unique_ptr<Shape> physical_shape_;

  // Bytes for dynamic shape metadata prefix.
  int64_t dynamic_shape_metadata_prefix_bytes_ = 0;
};

}  // namespace xla

#endif  // XLA_LAYOUT_H_
