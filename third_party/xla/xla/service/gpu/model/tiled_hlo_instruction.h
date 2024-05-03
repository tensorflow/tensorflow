/* Copyright 2024 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_SERVICE_GPU_MODEL_TILED_HLO_INSTRUCTION_H_
#define XLA_SERVICE_GPU_MODEL_TILED_HLO_INSTRUCTION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

// A wrapper around HloInstruction that represents a tiled HLO instruction.
//
// The class contains information required to emit this instruction in
// block-level codegen. Tile sizes and strides are constants and do not depend
// on the block id. Tile offsets are computed using an indexing map of form:
// `(block_id) -> (tile_offset0, tile_offset1, ...)`.
class TiledHloInstruction {
 public:
  // PtrHash and PtrEqual are helper classes to use in hash maps and sets that
  // compare values behind the pointers. For example,
  // absl::flat_hash_set<TiledHloInstruction*, PtrHash, PtrEqual> hlo_set;
  struct PtrHash {
    size_t operator()(const TiledHloInstruction* tiled_hlo) const;
  };

  struct PtrEqual {
    bool operator()(const TiledHloInstruction* lhs,
                    const TiledHloInstruction* rhs) const;
  };

  // Creates an instance of TiledHloInstruction. Returns an error if any of the
  // following preconditions is not met:
  // * Number of tile sizes, strides should match HLO shape rank.
  // * Number of result of `block_id_to_tile_offsets_indexing` should match HLO
  //   shape rank.
  // * `block_id_to_tile_offsets_indexing` should have only 1 dimension and 0
  //   symbols.
  static absl::StatusOr<std::unique_ptr<TiledHloInstruction>> Create(
      const HloInstruction* hlo, std::vector<int64_t> tile_sizes,
      std::vector<int64_t> tile_strides,
      IndexingMap block_id_to_tile_offsets_indexing);

  // Returns the original HLO instruction.
  const HloInstruction* hlo() const { return hlo_; }

  // Returns the tile sizes. The number of tile sizes is equal to the rank of
  // the output shape.
  const std::vector<int64_t>& tile_sizes() const { return tile_sizes_; }

  // Returns the tile strides. The number of tile strides is equal to the rank
  // of the output shape.
  const std::vector<int64_t>& tile_strides() const { return tile_strides_; }

  // Returns the indexing map from block_id to tile offsets. The map has a form
  // of `(block_id) -> (tile_offset0, tile_offset1, ...)`. The number of tile
  // offsets is equal to the rank of the output shape.
  const IndexingMap& block_id_to_tile_offsets_indexing() const {
    return block_id_to_tile_offsets_indexing_;
  }

  const TiledHloInstruction* operand(int64_t operand_id) const {
    return operands_[operand_id];
  }

  const std::vector<TiledHloInstruction*>& operands() const {
    return operands_;
  }

  void AppendOperand(TiledHloInstruction* operand) {
    operands_.push_back(operand);
  }

  std::string ToString() const;

 private:
  TiledHloInstruction(const HloInstruction* hlo,
                      std::vector<int64_t> tile_sizes,
                      std::vector<int64_t> tile_strides,
                      IndexingMap block_id_to_tile_offsets_indexing)
      : hlo_(hlo),
        tile_sizes_(std::move(tile_sizes)),
        tile_strides_(std::move(tile_strides)),
        block_id_to_tile_offsets_indexing_(
            std::move(block_id_to_tile_offsets_indexing)) {}

  // Pointer to the original HLO instruction.
  const HloInstruction* hlo_;

  // Tile sizes and strides.
  std::vector<int64_t> tile_sizes_;
  std::vector<int64_t> tile_strides_;

  // Indexing map from block_id to tile offsets.
  IndexingMap block_id_to_tile_offsets_indexing_;

  // Operands of the instruction in the tiled computation graph.
  std::vector<TiledHloInstruction*> operands_;
};

bool operator==(const TiledHloInstruction& lhs, const TiledHloInstruction& rhs);
bool operator!=(const TiledHloInstruction& lhs, const TiledHloInstruction& rhs);

template <typename H>
H AbslHashValue(H h, const TiledHloInstruction& tiled_hlo_instruction) {
  return H::combine(std::move(h), tiled_hlo_instruction.hlo(),
                    tiled_hlo_instruction.tile_sizes(),
                    tiled_hlo_instruction.tile_strides(),
                    tiled_hlo_instruction.block_id_to_tile_offsets_indexing());
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TILED_HLO_INSTRUCTION_H_
