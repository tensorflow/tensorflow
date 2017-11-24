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

// HLO shardings describe how an HLO instruction is split across multiple
// computations.

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_H_

#include <string>

#include "tensorflow/compiler/xla/array.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/protobuf_util.h"
#include "tensorflow/compiler/xla/shape_tree.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// HLO shardings describe how an HLO instruction is split across multiple
// computations.
class HloSharding {
 public:
  // Creates a trivial sharding that replicates a maximal tile across all
  // devices.
  static HloSharding Replicate() { return HloSharding(); }

  // Creates a sharding that emulates device placement; a tile shape equal to
  // the input shape (one tile) assigned to a single device.
  static HloSharding AssignDevice(int64 device_id);

  // Creates a new sharding which splits a shape into tiles each with shape
  // `tile_shape`. Each tile is assigned to one device, which is specified by
  // `tile_assignment`. Any tensor not a multiple of the tile size in any
  // dimension is implicitly padded to the tile size.
  //
  // e.g. Tile({2, 2}, {0, 1}) on a tensor of shape {3, 2} would look like:
  //      2     1 padding
  //   <------><->
  //   +----+----+
  //   | 0  |  1 |
  //   +----+----+
  //
  // Split into two tiles, one of which is implicitly padded by one.
  static HloSharding Tile(const Shape& tile_shape,
                          const Array<int64>& tile_assignment) {
    return HloSharding(tile_shape, tile_assignment);
  }

  // Creates a new sharding which splits a one-dimensional input shape into
  // `num_tiles` tiles.
  static HloSharding Tile1D(const Shape& input_shape, int64 num_tiles);

  // Creates a new sharding for a tuple type. The given ShapeTree must have
  // elements for every leaf shape contained in the tuple.
  static HloSharding Tuple(const ShapeTree<HloSharding>& sub_shardings) {
    std::vector<HloSharding> flattened_list;
    flattened_list.reserve(
        std::distance(sub_shardings.leaf_begin(), sub_shardings.leaf_end()));
    for (const auto& index_to_sharding : sub_shardings.leaves()) {
      flattened_list.push_back(index_to_sharding.second);
    }
    return HloSharding(flattened_list);
  }

  // Create a new sharding from a protobuf OpSharding.
  static StatusOr<HloSharding> FromProto(const OpSharding& proto);

  OpSharding ToProto() const;
  string ToString() const;

  // Validate that this sharding can be applied to a tensor with shape `shape`.
  Status Validate(const Shape& shape, int64 num_devices) const;

  // Returns true if the sharding has tuple type.
  bool IsTuple() const { return tuple_; }

  // Returns true if the sharding is trivial: replicate on all devices.
  bool IsReplicated() const {
    if (!IsTuple()) {
      return replicated_;
    }
    return std::all_of(tuple_elements_.begin(), tuple_elements_.end(),
                       [](const HloSharding& s) { return s.IsReplicated(); });
  }

  // Returns true if the tile size is the same as the input size.
  bool IsTileMaximal() const {
    if (!IsTuple()) {
      return maximal_;
    }
    return std::all_of(tuple_elements_.begin(), tuple_elements_.end(),
                       [](const HloSharding& s) { return s.IsTileMaximal(); });
  }

  // Returns true if the sharding defines an operation on the given device.
  bool UsesDevice(int64 device) const;

  // Returns the tile that should be executed on the given device.
  // REQUIRES: !IsTuple()
  std::vector<int64> TileIndexForDevice(int64 device) const;

  // Returns the device that should execute the given tile.
  // It is an error to call this if is_replicated() is true.
  // REQUIRES: !IsTuple()
  int64 DeviceForTileIndex(tensorflow::gtl::ArraySlice<int64> index) const;

  // Given a device ID, returns the offset within the input space of the
  // tile that should be executed on the given core. This returns the lower
  // extent of the tile in the input space.
  // REQUIRES: !IsTuple()
  std::vector<int64> TileOffsetForDevice(int64 device) const;

  // Given a device ID, returns the limit within the input space of the
  // tile that should be executed on the given core. This returns the upper
  // extent of the tile in the input space.
  // REQUIRES: !IsTuple()
  std::vector<int64> TileLimitForDevice(int64 device) const;

  // Returns the single device this op operates on.
  // REQUIRES: !IsTuple&& !Replicated() && IsTileMaximal()
  StatusOr<int64> UniqueDevice() const;

  // Returns true if this op only uses a single device.
  bool HasUniqueDevice() const;

  // Returns the ShapeTree containing the shardings for each element of this
  // tuple, if IsTuple, or a ShapeTree with a single element containing this
  // sharding. Only the leaf elements are populated. This creates a new
  // ShapeTree object so is not cheap.
  ShapeTree<HloSharding> GetAsShapeTree(const Shape& shape) const {
    if (IsTuple()) {
      ShapeTree<HloSharding> result(shape, HloSharding::Replicate());
      CHECK_EQ(std::distance(result.leaf_begin(), result.leaf_end()),
               tuple_elements_.size());
      auto it = tuple_elements_.begin();
      for (auto& index_to_sharding : result.leaves()) {
        index_to_sharding.second = *it++;
      }
      return result;
    } else {
      return ShapeTree<HloSharding>(shape, *this);
    }
  }

  bool operator==(const HloSharding& other) const {
    return replicated_ == other.replicated_ && maximal_ == other.maximal_ &&
           protobuf_util::ProtobufEquals(tile_shape_, other.tile_shape_) &&
           tile_assignment_ == other.tile_assignment_ &&
           tuple_elements_ == other.tuple_elements_;
  }
  bool operator!=(const HloSharding& other) const { return !(*this == other); }

  size_t Hash() const {
    if (!tuple_) {
      size_t h = 0;
      for (const auto& element : tuple_elements_) {
        h = tensorflow::Hash64Combine(h, element.Hash());
      }
      return h;
    }
    if (replicated_) {
      return 0;
    }
    size_t h = 0;
    for (uint32 v : tile_assignment_) {
      h = tensorflow::Hash64Combine(h, std::hash<uint32>{}(v));
    }
    for (uint32 v : tile_shape_.dimensions()) {
      h = tensorflow::Hash64Combine(h, std::hash<uint32>{}(v));
    }
    return h;
  }

  // Gets the tile shape.
  // REQUIRES: !IsTileMaximal() && !IsTuple()
  const Shape& tile_shape() const { return tile_shape_; }
  // Gets the tile assignment tensor.
  // REQUIRES: !IsReplicated() && !IsTuple()
  const Array<int64>& tile_assignment() const { return tile_assignment_; }

 private:
  HloSharding()
      : replicated_(true),
        maximal_(true),
        tuple_(false),
        tile_shape_(),
        tile_assignment_({0}) {}
  explicit HloSharding(int64 device_id)
      : replicated_(false),
        maximal_(true),
        tuple_(false),
        tile_shape_(),
        tile_assignment_({1}, device_id) {}
  HloSharding(const Shape& tile_shape, const Array<int64>& tile_assignment)
      : replicated_(false),
        maximal_(false),
        tuple_(false),
        tile_shape_(tile_shape),
        tile_assignment_(tile_assignment) {}
  HloSharding(const std::vector<HloSharding>& tuple_shardings)
      : replicated_(false),
        maximal_(false),
        tuple_(true),
        tile_assignment_({0}),
        tuple_elements_(tuple_shardings) {}

  // Internal helper to validate a tuple sharding.
  Status ValidateTuple(const Shape& shape, int64 num_devices) const;
  // Internal helper to validate a non-tuple (leaf) sharding.
  Status ValidateNonTuple(const Shape& shape, int64 num_devices) const;

  bool replicated_;
  bool maximal_;
  bool tuple_;
  Shape tile_shape_;
  Array<int64> tile_assignment_;
  // Only non-empty when tuple_ is true, but because empty tuples are allowed
  // may also be empty even then. This is a flattened list of all the leaf
  // shardings in a tuple shape, by pre-order walk (ShapeTree iterator order).
  std::vector<HloSharding> tuple_elements_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_H_
