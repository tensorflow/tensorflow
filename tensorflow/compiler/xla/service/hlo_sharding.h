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

  // Create a new sharding from a protobuf OpSharding.
  static StatusOr<HloSharding> FromProto(const OpSharding& proto);

  OpSharding ToProto() const;
  string ToString() const;

  // Validate that this sharding can be applied to a tensor with shape `shape`.
  Status Validate(const Shape& shape, int64 num_devices) const;

  // Returns true if the sharding is trivial: replicate on all devices.
  bool IsReplicated() const { return replicated_; }

  // Returns true if the tile size is the same as the input size.
  bool IsTileMaximal() const { return maximal_; }

  // Returns true if the sharding defines an operation on the given device.
  bool UsesDevice(int64 device) const;

  // Returns the tile that should be executed on the given device.
  std::vector<int64> TileIndexForDevice(int64 device) const;

  // Returns the device that should execute the given tile.
  // It is an error to call this if is_replicated() is true.
  int64 DeviceForTileIndex(tensorflow::gtl::ArraySlice<int64> index) const;

  // Given a device ID, returns the offset within the input space of the
  // tile that should be executed on the given core. This returns the lower
  // extent of the tile in the input space.
  std::vector<int64> TileOffsetForDevice(int64 device) const;

  // Given a device ID, returns the limit within the input space of the
  // tile that should be executed on the given core. This returns the upper
  // extent of the tile in the input space.
  std::vector<int64> TileLimitForDevice(int64 device) const;

  // Returns the single device this op operates on.
  // Requires !Replicated() && IsTileMaximal().
  StatusOr<int64> UniqueDevice() const;

  // Returns true if this op only uses a single device.
  bool HasUniqueDevice() const { return !IsReplicated() && IsTileMaximal(); }

  bool operator==(const HloSharding& other) const {
    return replicated_ == other.replicated_ && maximal_ == other.maximal_ &&
           protobuf_util::ProtobufEquals(tile_shape_, other.tile_shape_) &&
           tile_assignment_ == other.tile_assignment_;
  }
  bool operator!=(const HloSharding& other) const { return !(*this == other); }

  size_t Hash() const {
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
  // It is an error to call this if IsTileMaximal() is true.
  const Shape& tile_shape() const { return tile_shape_; }
  // Gets the tile assignment tensor.
  // It is an error to call this if IsReplicated() is true.
  const Array<int64>& tile_assignment() const { return tile_assignment_; }

 private:
  HloSharding()
      : replicated_(true),
        maximal_(true),
        tile_shape_(),
        tile_assignment_({0}) {}
  explicit HloSharding(int64 device_id)
      : replicated_(false),
        maximal_(true),
        tile_shape_(),
        tile_assignment_({1}, device_id) {}
  HloSharding(const Shape& tile_shape, const Array<int64>& tile_assignment)
      : replicated_(false),
        maximal_(false),
        tile_shape_(tile_shape),
        tile_assignment_(tile_assignment) {}

  bool replicated_;
  bool maximal_;
  Shape tile_shape_;
  Array<int64> tile_assignment_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_HLO_SHARDING_H_
