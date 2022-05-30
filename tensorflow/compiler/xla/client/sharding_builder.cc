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

#include "tensorflow/compiler/xla/client/sharding_builder.h"

namespace xla {
namespace sharding_builder {

OpSharding Replicate() {
  OpSharding result;
  result.set_type(OpSharding::REPLICATED);
  return result;
}

OpSharding Manual() {
  OpSharding result;
  result.set_type(OpSharding::MANUAL);
  return result;
}

OpSharding AssignDevice(int device) {
  OpSharding result;
  result.set_type(OpSharding::MAXIMAL);
  result.add_tile_assignment_dimensions(1);
  result.add_tile_assignment_devices(device);
  return result;
}

OpSharding Tile(const Shape& tile_shape,
                const TileAssignment& tile_assignment) {
  OpSharding result;
  result.set_type(OpSharding::OTHER);
  *result.mutable_tile_shape() = tile_shape.ToProto();
  for (int64_t dim : tile_assignment.dimensions()) {
    result.add_tile_assignment_dimensions(dim);
  }
  for (uint32_t device : tile_assignment) {
    result.add_tile_assignment_devices(device);
  }
  return result;
}

OpSharding Tile1D(const Shape& tile_shape, int64_t num_tiles) {
  OpSharding result;
  result.set_type(OpSharding::OTHER);

  CHECK_EQ(tile_shape.rank(), 1);
  std::vector<int64_t> dimensions(1, num_tiles);
  *result.mutable_tile_shape() = tile_shape.ToProto();
  auto& tile_dimension =
      (*result.mutable_tile_shape()->mutable_dimensions())[0];
  tile_dimension = CeilOfRatio(static_cast<int64_t>(tile_dimension), num_tiles);
  result.add_tile_assignment_dimensions(num_tiles);
  for (int64_t i = 0; i < num_tiles; ++i) {
    result.add_tile_assignment_devices(i);
  }
  return result;
}

OpSharding Tuple(const ShapeTree<OpSharding>& shardings) {
  OpSharding result;
  result.set_type(OpSharding::TUPLE);
  for (const auto& index_to_sharding : shardings.leaves()) {
    *result.add_tuple_shardings() = index_to_sharding.second;
  }
  return result;
}

}  // namespace sharding_builder
}  // namespace xla
