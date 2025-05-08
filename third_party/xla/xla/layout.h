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
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/printer.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

class Shape;

// Describes a tile used in tiling-based layout. Refer to
// g3doc/third_party/xla/docs/tiled_layout.md for details.
class Tile {
 public:
  Tile() = default;
  explicit Tile(absl::Span<const int64_t> dimensions)
      : dimensions_(dimensions.begin(), dimensions.end()) {}

  // De/Serialize a Tile to and from a TileProto.
  static absl::StatusOr<Tile> FromProto(const TileProto& tile_proto) {
    Tile tile;
    tile.dimensions_.reserve(tile_proto.dimensions_size());
    for (int64_t dimension : tile_proto.dimensions()) {
      TF_RET_CHECK(dimension >= 0);
      tile.add_dimensions(dimension);
    }
    return tile;
  }
  TileProto ToProto() const;

  bool operator==(const Tile& other) const {
    return dimensions() == other.dimensions();
  }
  bool operator!=(const Tile& other) const { return !(*this == other); }

  // Prints the Tile in the following format:
  // (dimension_1,dimension_2,...).
  // For example, (*,*,2,*,8) means that the tile has 2 dimensions, and the
  // dimension sizes are 2 and 8. '*' means the corresponding dimension in
  // the shape should be combined with the next more minor dimension before
  // tiling.
  void Print(Printer* printer) const;
  std::string ToString() const;

  // Returns the bound of the tile in the given dimension index.
  int64_t dimension(int i) const { return dimensions_[i]; }

  // Returns the dimensions of the tile.
  absl::Span<const int64_t> dimensions() const { return dimensions_; }

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
// such that 0 is the majormost dimension and (number of dimensions - 1) is the
// minormost dimension.
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

  bool operator==(const SplitConfig& other) const {
    return dimension() == other.dimension() &&
           split_indices() == other.split_indices();
  }
  bool operator!=(const SplitConfig& other) const { return !(*this == other); }

  // Formats this SplitConfig as "(dimension:split_indices)".
  // For example, (0:512,1024) means that dimension 0 is split into three
  // parts at indices 512 and 1024.
  std::string ToString() const;

  // Returns the dimension that is split.
  int64_t dimension() const { return dimension_; }
  SplitConfig& set_dimension(int64_t dimension) {
    dimension_ = dimension;
    return *this;
  }

  // Returns the indices where splits occur.
  absl::Span<const int64_t> split_indices() const { return split_indices_; }
  int64_t split_indices(int64_t idx) const { return split_indices_.at(idx); }
  int64_t split_indices_size() const { return split_indices_.size(); }
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

  Layout(absl::Span<const int64_t> minor_to_major, absl::Span<const Tile> tiles,
         int64_t element_size_in_bits);

  // Constructs a dense tiled layout with the given minor-to-major order, dim
  // level types, and tiles.
  Layout(absl::Span<const int64_t> minor_to_major,
         absl::Span<const DimLevelType> dim_level_types,
         absl::Span<const Tile> tiles,
         int64_t tail_padding_alignment_in_elements = 1,
         PrimitiveType index_primitive_type = PRIMITIVE_TYPE_INVALID,
         PrimitiveType element_primitive_type = PRIMITIVE_TYPE_INVALID,
         int64_t element_size_in_bits = 0, int64_t memory_space = 0,
         absl::Span<const SplitConfig> split_configs = {},
         std::unique_ptr<Shape> physical_shape = nullptr,
         int64_t dynamic_shape_metadata_prefix_bytes = 0);

  Layout& operator=(const Layout& other);
  Layout& operator=(Layout&& other);

  // Creates a Layout from a LayoutProto.
  static absl::StatusOr<Layout> FromProto(const LayoutProto& proto);

  ABSL_DEPRECATED("Use FromProto instead.")
  static Layout CreateFromProto(const LayoutProto& proto) {
    return FromProto(proto).value();
  }

  // Returns a LayoutProto representation of the Layout.
  LayoutProto ToProto() const;

  // Prints this layout as human-readable string, in the format
  // "{minor_to_major:properties}", where the fields are:
  //
  //   minor_to_major: Comma-separated minor-to-major order of the dimensions.
  //                   E.g. "{1,0}" means that dimension 1 is the most minor
  //                   dimension, and dimension 0 is the most major dimension.
  //   properties: concatenation of the following, separated by nothing (a
  //               property is ommitted if it is the default):
  //     D(...): Comma-separated list of attributes for each dimension. Each
  //             attribute is a single character abbreviation of the dimension
  //             level type
  //            The  abbreviations can be:
  //               D: DIM_DENSE
  //               C: DIM_COMPRESSED
  //               S: DIM_SINGLETON
  //               H: DIM_LOOSE_COMPRESSED
  //             E.g.
  //               D(D,C): dimension 0 is dense.
  //                       dimension 1 is compressed.
  //             If omitted, all dimensions are dense.
  //     T(...)...(...): The tiling (each (...) is acomma-separated list of
  //                     tile bound sizes). E.g.
  //             T(2,4)(3,5): The shape is tiled with 2x4 and 3x5 tiles.
  //             T(*,*,2,*,4): The dimensions corresponding the '*' are first
  //                 combined with the next more minor dimension, and then the
  //                 result shape is tiled with 2x4 tiles.
  //             If omitted, the shape is not tiled.
  //     L(n): The tail padding alignment in elements. Omitted if n is 1.
  //     #(type): The type of the indices.
  //     *(type): The type of the pointers.
  //     E(n): The element size in bits.
  //     S(n): The numeric value of thememory space. See the definition of
  //           Layout::memory_space() for details.
  //     SC(...)...(...): List of split configs, separated by nothing. Each
  //              (...) is a string of the form "(dimension:split_indices)".
  //              E.g. SC(1:512)(2:1024,2048): dimension 1 is split into 2 parts
  //              at index 512, and dimension 2 is split into 3 parts at index
  //              1024 and 2048.
  //     P(shape): The physical shape.
  //     M(n): The dynamic shape metadata prefix bytes. Omitted if n is 0.
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

  bool operator==(const Layout& other) const;
  bool operator!=(const Layout& other) const { return !(*this == other); }

  // The following methods mirror the protobuf generated code interface for the
  // message LayoutProto. This enabled easy migration of this data structure
  // from a proto to a proper C++ class.
  //
  // TODO(b/29771030): Replace or augment these methods with a more ergonomic
  // interface.

  // Methods for accessing the DimLevelType array.
  int dim_level_types_size() const { return n_dim_level_types_; }
  DimLevelType dim_level_type(int index) const {
    return dim_attributes_[index].dim_level_type;
  }
  Layout& set_dim_level_type(int index, DimLevelType dim_level_type) {
    dim_attributes_[index].dim_level_type = dim_level_type;
    return *this;
  }
  Layout& add_dim_level_type(DimLevelType dim_level_type) {
    while (n_dim_level_types_ >= dim_attributes_.size()) {
      dim_attributes_.push_back(DimInfo());
    }
    dim_attributes_[n_dim_level_types_].dim_level_type = dim_level_type;
    n_dim_level_types_++;
    return *this;
  }
  Layout& clear_dim_level_types() {
    n_dim_level_types_ = 0;
    return *this;
  }

  // Methods for accessing the minor-to-major array.
  int minor_to_major_size() const { return minor_to_major_.size(); }
  int64_t minor_to_major(int index) const { return minor_to_major_[index]; }
  Layout& set_minor_to_major(int index, int64_t value) {
    minor_to_major_[index] = value;
    return *this;
  }
  Layout& add_minor_to_major(int64_t value) {
    minor_to_major_.push_back(value);
    return *this;
  }
  Layout& clear_minor_to_major() {
    minor_to_major_.clear();
    return *this;
  }

  absl::Span<const int64_t> minor_to_major() const { return minor_to_major_; }
  DimensionVector* mutable_minor_to_major() { return &minor_to_major_; }

  // Removes the given dimension from 'minor_to_major_', and adjusts the other
  // dimensions accordingly. Also adjusts 'dim_level_types_'
  // in case it is a sparse layout.
  //
  // Precondition: dim_to_delete is in the range [0, minor_to_major_size()).
  Layout& DeleteDimension(int dim_to_delete);

  // Methods for accessing the tile field.
  int64_t tiles_size() const { return tiles_.size(); }
  const Tile& tiles(int index) const { return tiles_[index]; }
  Tile* mutable_tiles(int index) { return &tiles_[index]; }
  Tile* add_tiles() {
    tiles_.push_back(Tile());
    return &tiles_.back();
  }
  Layout& clear_tiles() {
    tiles_.clear();
    return *this;
  }
  absl::Span<const Tile> tiles() const { return tiles_; }
  TileVector* mutable_tiles() { return &tiles_; }

  int64_t element_size_in_bits() const { return element_size_in_bits_; }
  Layout& set_element_size_in_bits(int64_t value) {
    element_size_in_bits_ = value;
    return *this;
  }

  int64_t tail_padding_alignment_in_elements() const {
    return tail_padding_alignment_in_elements_;
  }

  Layout& set_tail_padding_alignment_in_elements(int64_t value) {
    CHECK_GE(value, 1);
    tail_padding_alignment_in_elements_ = value;
    return *this;
  }

  PrimitiveType index_primitive_type() const { return index_primitive_type_; }
  Layout& set_index_primitive_type(PrimitiveType value) {
    index_primitive_type_ = value;
    return *this;
  }

  PrimitiveType pointer_primitive_type() const {
    return pointer_primitive_type_;
  }
  Layout& set_pointer_primitive_type(PrimitiveType value) {
    pointer_primitive_type_ = value;
    return *this;
  }

  static constexpr int64_t kDefaultMemorySpace = 0;
  static constexpr int64_t kGenericFastMemorySpace = 1;
  static constexpr int64_t kHostMemorySpace = 5;
  int64_t memory_space() const { return memory_space_; }
  Layout& set_memory_space(int64_t value) {
    memory_space_ = value;
    return *this;
  }

  int split_configs_size() const { return split_configs_.size(); }
  const SplitConfig& split_configs(int index) const {
    return split_configs_.at(index);
  }
  SplitConfig* mutable_split_configs(int index) {
    return &split_configs_.at(index);
  }
  Layout& add_split_configs(const SplitConfig& split_config) {
    split_configs_.push_back(split_config);
    return *this;
  }
  void clear_split_configs() { split_configs_.clear(); }
  absl::Span<const SplitConfig> split_configs() const { return split_configs_; }

  // Methods for accessing the physical shape.
  bool has_physical_shape() const { return physical_shape_ != nullptr; }
  const Shape& physical_shape() const {
    CHECK(has_physical_shape());
    return *physical_shape_;
  }
  Shape* mutable_physical_shape();
  void clear_physical_shape();

  int64_t dynamic_shape_metadata_prefix_bytes() const {
    return dynamic_shape_metadata_prefix_bytes_;
  }
  Layout& set_dynamic_shape_metadata_prefix_bytes(int64_t bytes) {
    dynamic_shape_metadata_prefix_bytes_ = bytes;
    return *this;
  }

  void Swap(Layout* other) {
    using std::swap;
    swap(*this, *other);
  }

  void Clear() { *this = Layout(); }

  template <typename H>
  friend H AbslHashValue(H h, const Layout& l) {
    return H::combine(std::move(h), l.minor_to_major_, l.tiles_,
                      l.element_size_in_bits_, l.index_primitive_type_,
                      l.pointer_primitive_type_, l.memory_space_,
                      l.split_configs_, l.tail_padding_alignment_in_elements_);
  }

 private:
  // We store a single inlined vector to hold
  struct DimInfo {
    DimInfo()
        : dim_level_type(DIM_DENSE), dim_unique(true), dim_ordered(true) {}

    DimLevelType dim_level_type : 6;
    bool dim_unique : 1;
    bool dim_ordered : 1;
  };

  absl::InlinedVector<DimInfo, InlineRank()> dim_attributes_;

  uint8_t n_dim_level_types_ = 0;

  // The primitive type to use for sparse array indices and pointers.  Each of
  // these must either be INVALID, or an unsigned integer type.
  PrimitiveType index_primitive_type_ : 8;
  PrimitiveType pointer_primitive_type_ : 8;

  // The assigned memory space.
  int8_t memory_space_ = 0;

  // The number of bits used to store an individual array element.
  // When the value is 0, default to ShapeUtil::ByteSizeOfPrimitiveType.
  int64_t element_size_in_bits_ = 0;

  // A map from physical dimension numbers to logical dimension numbers.
  // The first element is the most minor physical dimension (fastest varying
  // index) and the last the most major (slowest varying index). The contents of
  // the vector are the indices of the *logical* dimensions in the shape.
  //
  // For example, in shape f32[8,100,100,3]{3,0,2,1}, the logical dimensions
  // are [8,100,100,3] and minor_to_major_ is {3,0,2,1}.
  // So, the most minor physical dimension is [8,100,100,3][3], which is size 3.
  // The second most minor is [8,100,100,3][0], which is size 8.
  // The third most minor is [8,100,100,3][2], which is size 100.
  // And the major dim is [8,100,100,3][1], which is size 100.
  DimensionVector minor_to_major_;

  // The tiles used in tiling-based layout.
  TileVector tiles_;

  // The split configurations of the shape, which describes how the storage of
  // the tensor is split between different physical memories.
  absl::InlinedVector<SplitConfig, 1> split_configs_;

  // The shape is padded at the end to a multiple of, in terms of number of
  // elements, this value. This is useful when tiling does not bring the shape
  // to certain desired granules. Tiling effectively pads/reshapes/transposes
  // the shape to another shape. This field pads the total number of elements of
  // that new shape to a multiple of certain number of elements. This is useful
  // such as we want a layout which does not tile the data but still requires it
  // to be padded to certain number of elements.
  //
  // Invariant: this must be >= 1.
  int64_t tail_padding_alignment_in_elements_ = 1;

  // The physical on-device shape used to represent a sparse array.
  std::unique_ptr<Shape> physical_shape_;

  // The dynamic shape metadata size in bytes in front of the shape data. The
  // field may be non-zero for a static shape whose associated buffer is for a
  // dynamic shape, e.g. a result of SliceToDynamic.
  int64_t dynamic_shape_metadata_prefix_bytes_ = 0;
};

std::ostream& operator<<(std::ostream& out, const Tile& Tile);
std::ostream& operator<<(std::ostream& out, const Layout& layout);

}  // namespace xla

#endif  // XLA_LAYOUT_H_
