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
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {
namespace gpu {

// A wrapper around HloInstruction that represents a tiled HLO instruction.
//
// The class contains information required to emit this instruction in
// block-level codegen. Tile sizes and strides are constants and do not depend
// on the block id. Tile offsets are computed using an indexing map of the form:
// `(block_id) -> (tile_offset0, tile_offset1, ...)`
class TiledHloInstruction {
 public:
  virtual ~TiledHloInstruction() = default;

  // Creates an instance of TiledHloInstruction. Returns an error if any of the
  // following preconditions is not met:
  // * Number of tile sizes, strides should match the rank of the HLO.
  // * If `tile_offsets_indexing` is provided then it must satisfy:
  // - The number of results must match the rank of the HLO.
  // - Input should have exactly 1 dimension.
  // - The number of runtime variables must match the number of runtime
  //   variables in tile_offsets_indexing map.
  static absl::StatusOr<std::unique_ptr<TiledHloInstruction>> Create(
      const HloInstruction* hlo,
      llvm::SmallVector<const TiledHloInstruction*> operands,
      llvm::SmallVector<const TiledHloInstruction*> runtime_variables,
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

  const llvm::SmallVector<const TiledHloInstruction*>& runtime_variables()
      const {
    return runtime_variables_;
  }

  // Returns the tile sizes. The number of tile sizes is equal to the rank of
  // the output shape.
  const llvm::SmallVector<int64_t>& tile_sizes() const { return tile_sizes_; }
  int64_t tile_size(int64_t dim_idx) const { return tile_sizes_[dim_idx]; }

  // Returns the tile strides. The number of tile strides is equal to the rank
  // of the output shape.
  const llvm::SmallVector<int64_t>& tile_strides() const {
    return tile_strides_;
  }
  int64_t tile_stride(int64_t dim_idx) const { return tile_strides_[dim_idx]; }

  // Returns the indexing map from block_id to tile offsets. The map has a form
  // of `(block_id) -> (tile_offset0, tile_offset1, ...)`. The number of tile
  // offsets is equal to the rank of the tiled hlo.
  //
  // The indexing map is not computed by default.
  absl::StatusOr<IndexingMap> tile_offsets_indexing() const {
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

 protected:
  TiledHloInstruction(
      const HloInstruction* hlo,
      llvm::SmallVector<const TiledHloInstruction*> operands,
      llvm::SmallVector<const TiledHloInstruction*> runtime_variables,
      llvm::SmallVector<int64_t> tile_sizes,
      llvm::SmallVector<int64_t> tile_strides,
      std::optional<IndexingMap> tile_offsets_indexing)
      : hlo_(hlo),
        operands_(std::move(operands)),
        runtime_variables_(std::move(runtime_variables)),
        tile_sizes_(std::move(tile_sizes)),
        tile_strides_(std::move(tile_strides)),
        tile_offsets_indexing_(std::move(tile_offsets_indexing)) {
    if (tile_offsets_indexing_.has_value()) {
      CHECK_EQ(tile_offsets_indexing_->GetDimVarsCount(), 1);
      CHECK_EQ(tile_offsets_indexing_->GetRTVarsCount(),
               runtime_variables_.size());
    }
  }

 private:
  // Pointer to the original HLO instruction.
  const HloInstruction* hlo_;

  // Operands of the instruction in the tiled computation graph.
  llvm::SmallVector<const TiledHloInstruction*> operands_;
  llvm::SmallVector<const TiledHloInstruction*> runtime_variables_;

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
    // the same and the instruction has no operands.
    return lhs.tile_offsets_indexing() == rhs.tile_offsets_indexing();
  }

  return lhs.operands() == rhs.operands() &&
         lhs.runtime_variables() == rhs.runtime_variables();
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
          tiled_hlo_instruction.operands()),
      absl::Span<const TiledHloInstruction* const>(
          tiled_hlo_instruction.runtime_variables()));
}

class TiledHloComputation;

// `TiledHloFusionInstruction` is to `TiledHloInstruction` what
// `HloFusionInstruction` is to `HloInstruction`.
//
// The main use case for `TiledHloFusionInstruction`s is to support nested
// fusions in block-level codegen.
//
// Similarly to `HloFusionInstruction`, this subclass holds a nested
// `TiledHloComputation` accessible through the `called_computation()` method.
class TiledHloFusionInstruction : public TiledHloInstruction {
 public:
  static absl::StatusOr<std::unique_ptr<TiledHloFusionInstruction>> Create(
      const HloInstruction* hlo,
      llvm::SmallVector<const TiledHloInstruction*> operands,
      llvm::SmallVector<const TiledHloInstruction*> runtime_variables,
      std::unique_ptr<TiledHloComputation> called_computation,
      llvm::SmallVector<int64_t> tile_sizes,
      llvm::SmallVector<int64_t> tile_strides,
      std::optional<IndexingMap> tile_offsets_indexing);

  // The `TiledHloComputation` called by this instruction.
  const TiledHloComputation* called_computation() const {
    return called_computation_.get();
  }

 private:
  TiledHloFusionInstruction(
      const HloInstruction* hlo,
      llvm::SmallVector<const TiledHloInstruction*> operands,
      llvm::SmallVector<const TiledHloInstruction*> runtime_variables,
      std::unique_ptr<TiledHloComputation> called_computation,
      llvm::SmallVector<int64_t> tile_sizes,
      llvm::SmallVector<int64_t> tile_strides,
      std::optional<IndexingMap> tile_offsets_indexing);

  // See comment for `called_computation()`.
  std::unique_ptr<TiledHloComputation> called_computation_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_TILED_HLO_INSTRUCTION_H_
