/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "xla/service/gpu/model/tile_analysis.h"

#include <cstdint>
#include <utility>

#include "xla/status_macros.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

StatusOr<Tile> ComposeTiles(const Tile& producer_tile,
                            const Tile& consumer_tile) {
  int64_t producer_tile_rank = producer_tile.getRank();
  TF_RET_CHECK(producer_tile_rank == consumer_tile.getRank())
      << "Tile rank mismatch";

  Tile composed_tile{consumer_tile.offsets(), consumer_tile.sizes()};
  for (int i = 0; i < producer_tile_rank; ++i) {
    composed_tile.offsets()[i] += producer_tile.offsets()[i];
    TF_RET_CHECK(consumer_tile.offsets()[i] + consumer_tile.sizes()[i] <=
                 producer_tile.sizes()[i])
        << "Composition leads to an OOB tile.";
  }
  return {std::move(composed_tile)};
}

}  // namespace gpu
}  // namespace xla
