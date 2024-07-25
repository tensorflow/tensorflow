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

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/model/indexing_map.h"

namespace xla {
namespace gpu {

// A wrapper around HloInstruction that represents a tiled HLO instruction.
//
// The class contains information required to emit this instruction in
// block-level codegen. Tile sizes and strides are constants and do not depend
// on the multidimensional tile index. Tile offsets are computed using an
// indexing map of the form:
//   `(d0, d1, ...) -> (tile_offset0, tile_offset1, ...)`
// where (d0, d1, ...) is the tile multi-index.
class TiledHloInstruction {
 public:
  // Creates an instance of TiledHloInstruction. Returns an error if any of the
  // following preconditions is not met:
  // * Number of tile sizes, strides should match HLO shape rank.
  // * Number of results of `tile_offsets_indexing` should match HLO shape rank.
  // * `tile_offsets_indexing` should have the number of dimensions equal to the
  //   rank of the output tile and 0 symbols.
  static absl::StatusOr<std::unique_ptr<TiledHloInstruction>> Create(
      const HloInstruction* hlo,
      llvm::SmallVector<const TiledHloInstruction*> operands,
      llvm::SmallVector<int64_t> tile_sizes,
      llvm::SmallVector<int64_t> tile_strides,
      std::optional<IndexingMap> tile_offsets_indexing);

  // Returns the original HLO instruction.
  const HloInstruction* hlo() const { return hlo_; }

  // Operands of the instruction in the tiled computation graph.
  const TiledHloInstruction* operand(int64_t operand_id) const {
    return operands_[operand_id];
  }

  const llvm::SmallVector<const TiledHloInstruction*>& operands() const {
    return operands_;
  }

  // Returns the tile sizes. The number of tile sizes is equal to the rank of
  // the output shape.
  const llvm::SmallVector<int64_t>& tile_sizes() const { return tile_sizes_; }

  // Returns the tile strides. The number of tile strides is equal to the rank
  // of the output shape.
  const llvm::SmallVector<int64_t>& tile_strides() const {
    return tile_strides_;
  }

  // Returns the indexing map from tile multi-index to tile offsets. The map has
  // a form of `(d0, d1, ...) -> (tile_offset0, tile_offset1, ...)`. The number
  // of input dimensions is equal to the rank of output tile of the computation.
  // The number of tile offsets is equal to the rank of the tiled hlo.
  //
  // The indexing map is not computed by default.
  absl::StatusOr<const IndexingMap> tile_offsets_indexing() const {
    if (!tile_offsets_indexing_.has_value()) {
      return absl::FailedPreconditionError(
          "tile_offsets_indexing was not computed. It is likely that "
          "`compute_all_tile_offset_indexing_maps` should be set to true in "
          "`SymbolicTileAnalysis::ComputeTiledHloInstructions`.");
    }
    return *tile_offsets_indexing_;
  }

  std::string ToString() const;

  // This allows GUnit to print TiledHloInstruction.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TiledHloInstruction& tiled_hlo) {
    sink.Append(tiled_hlo.ToString());
  }

 private:
  TiledHloInstruction(const HloInstruction* hlo,
                      llvm::SmallVector<const TiledHloInstruction*> operands,
                      llvm::SmallVector<int64_t> tile_sizes,
                      llvm::SmallVector<int64_t> tile_strides,
                      std::optional<IndexingMap> tile_offsets_indexing)
      : hlo_(hlo),
        operands_(std::move(operands)),
        tile_sizes_(std::move(tile_sizes)),
        tile_strides_(std::move(tile_strides)),
        tile_offsets_indexing_(std::move(tile_offsets_indexing)) {}

  // Pointer to the original HLO instruction.
  const HloInstruction* hlo_;

  // Operands of the instruction in the tiled computation graph.
  llvm::SmallVector<const TiledHloInstruction*> operands_;

  // Tile sizes and strides.
  llvm::SmallVector<int64_t> tile_sizes_;
  llvm::SmallVector<int64_t> tile_strides_;

  // See comment for `tile_offsets_indexing()`.
  std::optional<IndexingMap> tile_offsets_indexing_;
};

inline bool operator==(const TiledHloInstruction& lhs,
                       const TiledHloInstruction& rhs) {
  if (lhs.hlo() != rhs.hlo() || lhs.tile_sizes() != rhs.tile_sizes() ||
      lhs.tile_strides() != rhs.tile_strides()) {
    return false;
  }

  if (lhs.operands().empty() && rhs.operands().empty()) {
    // Tile offsets indexing is guaranteed to be computed only if tile sizes are
    // different and the instruction has no operands.
    return lhs.tile_offsets_indexing() == rhs.tile_offsets_indexing();
  }

  return lhs.operands() == rhs.operands();
}

inline bool operator!=(const TiledHloInstruction& lhs,
                       const TiledHloInstruction& rhs) {
  return !(lhs == rhs);
}

template <typename H>
H AbslHashValue(H h, const TiledHloInstruction& tiled_hlo_instruction) {
  // There is no default hash implementation for llvm::SmallVector neither in
  // AbslHashValue nor in llvm::hash_value. We can use the available hash
  // implementation for absl::Span instead.
  return H::combine(
      std::move(h), tiled_hlo_instruction.hlo(),
      absl::Span<int64_t const>(tiled_hlo_instruction.tile_sizes()),
      absl::Span<int64_t const>(tiled_hlo_instruction.tile_strides()),
      absl::Span<const TiledHloInstruction* const>(
          tiled_hlo_instruction.operands()));
}

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TILED_HLO_INSTRUCTION_H_
