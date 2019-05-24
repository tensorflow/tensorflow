/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/tools/mapping_helper.h"

#include <poplar/Graph.hpp>
#include <poplar/Tensor.hpp>
#include <poputil/TileMapping.hpp>

#include <algorithm>

namespace xla {
namespace poplarplugin {
namespace {
template <typename T>
void rotate_right(T& t, std::size_t spaces) {
  std::rotate(t.rbegin(), t.rbegin() + spaces, t.rend());
}

void MapTensorLinearlyImpl(
    LinearMapperState& state, poplar::Graph& graph, poplar::Tensor& tensor,
    std::vector<std::vector<poplar::Interval>>& mapping) {
  uint64& next_tile_to_map_from = state[&graph];

  // The number of tiles the mapping is across.
  auto mapping_tile_count = mapping.size();
  auto tile_count = graph.getTarget().getNumTiles();

  // Move the tile mapping cyclically by the offset.
  mapping.resize(tile_count);
  rotate_right(mapping, next_tile_to_map_from);
  graph.setTileMapping(tensor, mapping);

  // Update offset.
  next_tile_to_map_from += mapping_tile_count;
  next_tile_to_map_from = next_tile_to_map_from % tile_count;
}
}  // namespace

void MappingHelper::MapTensorLinearly(LinearMapperState& state,
                                      poplar::Graph& graph,
                                      poplar::Tensor& tensor) {
  auto mapping = poputil::calcLinearTileMapping(graph, tensor);
  MapTensorLinearlyImpl(state, graph, tensor, mapping);
}

void MappingHelper::MapTensorLinearly(LinearMapperState& state,
                                      poplar::Graph& graph,
                                      poplar::Tensor& tensor,
                                      uint32 min_elements_per_tile,
                                      uint32 grain_size) {
  auto mapping = poputil::calcLinearTileMapping(
      graph, tensor.shape(), min_elements_per_tile, grain_size);
  MapTensorLinearlyImpl(state, graph, tensor, mapping);
}
}  // namespace poplarplugin
}  // namespace xla
