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

#ifndef XLA_SERVICE_GPU_MODEL_TILE_ANALYSIS_H_
#define XLA_SERVICE_GPU_MODEL_TILE_ANALYSIS_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "xla/statusor.h"

namespace xla {
namespace gpu {

// Stores parameters of a tile/slice.
// Note, that the strides are assumed to be equal to 1s.
class Tile {
 public:
  Tile(std::vector<int64_t>&& offsets, std::vector<int64_t>&& sizes)
      : offsets_(std::move(offsets)), sizes_(std::move(sizes)) {
    CHECK(offsets_.size() == sizes_.size())
        << "len(offsets) should match len(sizes)";
    CHECK(!offsets_.empty()) << "Rank should be >= 1";
  }

  std::vector<int64_t> offsets() const { return offsets_; }
  std::vector<int64_t>& offsets() { return offsets_; }

  std::vector<int64_t> sizes() const { return sizes_; }
  std::vector<int64_t>& sizes() { return sizes_; }

  int64_t getRank() const { return offsets_.size(); }

 private:
  std::vector<int64_t> offsets_;
  std::vector<int64_t> sizes_;
};

// Composes consumer_tile(producer_tile) to get a new tile.
StatusOr<Tile> ComposeTiles(const Tile& producer_tile,
                            const Tile& consumer_tile);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TILE_ANALYSIS_H_
