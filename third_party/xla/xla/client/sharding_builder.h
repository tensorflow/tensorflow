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

#ifndef XLA_CLIENT_SHARDING_BUILDER_H_
#define XLA_CLIENT_SHARDING_BUILDER_H_

#include <vector>

#include "xla/array.h"
#include "xla/shape_tree.h"
#include "xla/shape_util.h"
#include "xla/types.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {
namespace sharding_builder {
// A shaped array used to describe the assignment of tiles to devices.
using TileAssignment = Array<int64_t>;

// Creates a replicated sharding - replicate a tensor on every device.
OpSharding Replicate();

// Creates a manual sharding - the partitioner will not change the shape.
OpSharding Manual();

// Creates a sharding that assigns a tensor to just one device.
OpSharding AssignDevice(int device);

// Creates a tiled sharding with the given tile shape and assignment of tiles
// to devices.
//
// If tile_shape is not evenly divisible by the number of devices in
// tile_assignment, operations behave as if implicit padding had been inserted.
// The value of this padding is undefined.
OpSharding Tile(const Shape& tile_shape, const TileAssignment& tile_assignment);

// Creates a sharding in one dimension, with the given tile shape which must
// be rank 1 and using devices [0..num_tiles).
//
// This is simply a convenience wrapper for Tile().
OpSharding Tile1D(const Shape& tile_shape, int64_t num_tiles);

// Creates a tuple sharding from the given ShapeTree of element shardings.
OpSharding Tuple(const ShapeTree<OpSharding>& shardings);

}  // namespace sharding_builder
}  // namespace xla

#endif  // XLA_CLIENT_SHARDING_BUILDER_H_
