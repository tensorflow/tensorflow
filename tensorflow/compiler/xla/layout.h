/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_LAYOUT_H_
#define TENSORFLOW_COMPILER_XLA_LAYOUT_H_

#include <vector>

#include "absl/types/span.h"

#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Describes a tile used in tiling-based layout. Refer to
// g3doc/third_party/tensorflow/compiler/xla/g3doc/layout_with_tiling.md for
// details.
class Tile {
 public:
  Tile() = default;
  explicit Tile(absl::Span<const int64> dimensions)
      : dimensions_(dimensions.begin(), dimensions.end()) {}

  // De/Serialize a Tile to and from a TileProto.
  static Tile CreateFromProto(const TileProto& tile_proto) {
    return Tile(AsInt64Slice(tile_proto.dimensions()));
  }
  TileProto ToProto() const;

  bool operator==(const Tile& other) const {
    return dimensions() == other.dimensions();
  }
  bool operator!=(const Tile& other) const { return !(*this == other); }

  string ToString() const;

  // Returns the bound of the tile in the given dimension index.
  int64 dimension(int i) const { return dimensions_.at(i); }

  // Returns the dimensions of the tile.
  const std::vector<int64>& dimensions() const { return dimensions_; }

  Tile& add_dimensions(int64 value) {
    dimensions_.push_back(value);
    return *this;
  }

  Tile& clear_dimensions() {
    dimensions_.clear();
    return *this;
  }

  // This dimension size means the corresponding dimension in the shape is
  // combined with the next minor dimension before tiling is applied.
  static constexpr int64 kCombineDimension = std::numeric_limits<int64>::min();

  template <typename H>
  friend H AbslHashValue(H h, const Tile& t) {
    return H::combine(std::move(h), t.dimensions_);
  }

 private:
  // The bounds of the tile.
  std::vector<int64> dimensions_;
};

class Layout {
 public:
  Layout() = default;

  // Constructs a dense layout with the given minor-to-major order.
  explicit Layout(absl::Span<const int64> minor_to_major)
      : format_(DENSE),
        minor_to_major_(minor_to_major.begin(), minor_to_major.end()) {}

  // Constructs a dense tiled layout with the given minor-to-major order and
  // tiles.
  Layout(absl::Span<const int64> minor_to_major, absl::Span<const Tile> tiles,
         int64 element_size_in_bits = 0)
      : format_(DENSE),
        minor_to_major_(minor_to_major.begin(), minor_to_major.end()),
        tiles_(tiles.begin(), tiles.end()),
        element_size_in_bits_(element_size_in_bits) {}

  // Construct a shape from a LayoutProto.
  static Layout CreateFromProto(const LayoutProto& proto);

  // Returns a LayoutProto representation of the Layout.
  LayoutProto ToProto() const;

  // Returns a human-readable string that represents this layout.
  string ToString() const;

  // Equal is a configurable functor to check the equality of two layouts.
  //
  // Examples:
  //
  // - Comparing two layouts ignoring their difference in tiles:
  //   Equal().IgnoreTiles()(layout1, layout2);
  //
  // - Comparing two layouts ignoring their difference in tiles and element
  //   size:
  //   Equal().IgnoreTiles().IgnoreElementSize()(layout1, layout2);
  class Equal {
   public:
    Equal() = default;

    bool operator()(const Layout& lhs, const Layout& rhs);

    Equal& IgnoreTiles() {
      ignore_tiles_ = true;
      return *this;
    }

    Equal& IgnoreElementSize() {
      ignore_element_size_ = true;
      return *this;
    }

    Equal& MinorToMajorOnly() {
      ignore_tiles_ = true;
      ignore_element_size_ = true;
      return *this;
    }

   private:
    bool ignore_tiles_ = false;
    bool ignore_element_size_ = false;
  };

  bool operator==(const Layout& other) const;
  bool operator!=(const Layout& other) const { return !(*this == other); }

  // The following methods mirror the protobuf generated code interface for the
  // message LayoutProto. This enabled easy migration of this data structure
  // from a proto to a proper C++ class.
  //
  // TODO(b/29771030): Replace or augment these methods with a more ergonomic
  // interface.

  // Methods for accessing the format.
  Format format() const { return format_; }
  Layout& set_format(Format value) {
    format_ = value;
    return *this;
  }

  // Methods for accessing the minor-to-major array.
  int minor_to_major_size() const { return minor_to_major_.size(); }
  int64 minor_to_major(int index) const { return minor_to_major_.at(index); }
  Layout& set_minor_to_major(int index, int64 value) {
    minor_to_major_.at(index) = value;
    return *this;
  }
  Layout& add_minor_to_major(int64 value) {
    minor_to_major_.push_back(value);
    return *this;
  }
  Layout& clear_minor_to_major() {
    minor_to_major_.clear();
    return *this;
  }
  const std::vector<int64>& minor_to_major() const { return minor_to_major_; }
  std::vector<int64>* mutable_minor_to_major() { return &minor_to_major_; }

  // Methods for accessing the tile field.
  int tiles_size() const { return tiles_.size(); }
  const Tile& tiles(int index) const { return tiles_.at(index); }
  Tile* mutable_tiles(int index) { return &tiles_.at(index); }
  Tile* add_tiles() {
    tiles_.push_back(Tile());
    return &tiles_.back();
  }
  Layout& clear_tiles() {
    tiles_.clear();
    return *this;
  }
  const std::vector<Tile>& tiles() const { return tiles_; }
  std::vector<Tile>* mutable_tiles() { return &tiles_; }

  // Methods for accessing the int64 fields.
  int64 max_sparse_elements() const { return max_sparse_elements_; }
  Layout& set_max_sparse_elements(int64 value) {
    max_sparse_elements_ = value;
    return *this;
  }
  int64 element_size_in_bits() const { return element_size_in_bits_; }
  Layout& set_element_size_in_bits(int64 value) {
    element_size_in_bits_ = value;
    return *this;
  }

  void Swap(Layout* other) {
    using std::swap;
    swap(*this, *other);
  }

  void Clear() {
    format_ = INVALID_FORMAT;
    minor_to_major_.clear();
    max_sparse_elements_ = 0;
    element_size_in_bits_ = 0;
  }

  template <typename H>
  friend H AbslHashValue(H h, const Layout& l) {
    return H::combine(std::move(h), l.format_, l.minor_to_major_,
                      l.max_sparse_elements_, l.tiles_,
                      l.element_size_in_bits_);
  }

 private:
  // The format of this layout.
  Format format_ = INVALID_FORMAT;

  // Sequence of dimension numbers, from minor (fastest varying index) to major
  // (slowest varying index).
  std::vector<int64> minor_to_major_;

  // The maximum number of elements that can be stored for SPARSE formats.  This
  // can be used to determine the maximum size in bytes of arrays stored in
  // memory.  This field must be zero unless the format is SPARSE.
  int64 max_sparse_elements_ = 0;

  // The tiles used in tiling-based layout.
  std::vector<Tile> tiles_;

  // The number of bits used to store an individual array element.
  int64 element_size_in_bits_ = 0;
};

std::ostream& operator<<(std::ostream& out, const Tile& Tile);
std::ostream& operator<<(std::ostream& out, const Layout& layout);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LAYOUT_H_
