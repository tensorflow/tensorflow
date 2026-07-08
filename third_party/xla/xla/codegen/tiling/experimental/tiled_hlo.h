/* Copyright 2025 The TensorFlow Authors. All Rights Reserved.

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

#ifndef XLA_CODEGEN_TILING_EXPERIMENTAL_TILED_HLO_H_
#define XLA_CODEGEN_TILING_EXPERIMENTAL_TILED_HLO_H_

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/experimental/tile.h"
#include "xla/codegen/tiling/experimental/tiling_space.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/iterator_util.h"
#include "xla/tsl/lib/gtl/iterator_range.h"
#include "xla/util.h"

namespace xla::gpu::experimental {

class TiledHloInstruction;

// A region is a collection of instructions grouped to represent a nested
// control flow (e.g., loops) or a distinct computation branch.
class TiledHloRegion
    : public std::vector<std::unique_ptr<TiledHloInstruction>> {};

// A node in the symbolic tiled representation of an HLO computation. During
// tiling and codegen an HLO instruction may need to be emitted multiple times
// with different tiling parameters.
class TiledHloInstruction {
 public:
  TiledHloInstruction(const HloInstruction* hlo, Tile tile,
                      const TiledHloInstruction* parent = nullptr,
                      int64_t parent_region_index = -1)
      : hlo_(hlo),
        tile_(std::move(tile)),
        parent_(parent),
        parent_region_index_(parent_region_index) {}

  const HloInstruction* hlo() const { return hlo_; }

  const Tile& tile() const { return tile_; }
  void set_tile(Tile tile) { tile_ = std::move(tile); }

  const TiledHloInstruction* operand(int64_t operand_id) const {
    return operands_[operand_id];
  }
  llvm::ArrayRef<const TiledHloInstruction*> operands() const {
    return operands_;
  }
  void AddOperand(TiledHloInstruction* operand) {
    operands_.push_back(operand);
  }

  llvm::ArrayRef<TiledHloRegion> hlo_regions() const { return regions_; }
  void AddHloRegion(TiledHloRegion region) {
    regions_.push_back(std::move(region));
  }

  const TiledHloInstruction* parent() const { return parent_; }

  int64_t parent_region_index() const { return parent_region_index_; }

  // Returns the TiledHloInstructions that correspond to the runtime variables
  // of the original HLO instruction.
  llvm::SmallVector<const TiledHloInstruction*, 2> runtime_variables() const;

  // Returns a string representation of the instruction. Used only for error
  // messages and debugging.
  std::string ToString(absl::string_view field_separator = "\n\t") const;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TiledHloInstruction& tiled_hlo) {
    sink.Append(tiled_hlo.ToString());
  }

  // Temporary helpers to match API of the old TiledHloInstruction.
  // TODO: b/509505290 -- Remove these once we migrate to this tiling and the
  // old API is removed.
  llvm::SmallVector<int64_t> tile_sizes() const {
    auto tile_sizes = tile_.GetStaticTileSizes();
    CHECK_OK(tile_sizes);
    return *tile_sizes;
  }
  llvm::SmallVector<int64_t> tile_strides() const {
    auto tile_strides = tile_.GetStaticTileStrides();
    CHECK_OK(tile_strides);
    return *tile_strides;
  }

  int64_t tile_size(int64_t dim) const {
    auto tile_sizes = this->tile_sizes();
    CHECK_LT(dim, tile_sizes.size());
    return tile_sizes[dim];
  }

  int64_t tile_stride(int64_t dim) const {
    auto tile_strides = this->tile_strides();
    CHECK_LT(dim, tile_strides.size());
    return tile_strides[dim];
  }

 private:
  // Pointer to the original HLO instruction.
  const HloInstruction* hlo_;

  // Symbolic tile.
  Tile tile_;

  // Operands of the instruction in the tiled computation graph.
  llvm::SmallVector<const TiledHloInstruction*, 2> operands_;

  // Regions of the instruction.
  llvm::SmallVector<TiledHloRegion, 2> regions_;

  // Parent instruction that owns the region containing this instruction.
  const TiledHloInstruction* parent_ = nullptr;

  // Index of the region within the parent instruction.
  int64_t parent_region_index_ = -1;
};

inline bool operator==(const TiledHloInstruction& lhs,
                       const TiledHloInstruction& rhs) {
  return lhs.hlo() == rhs.hlo() && lhs.tile() == rhs.tile() &&
         lhs.operands() == rhs.operands();
}

inline bool operator!=(const TiledHloInstruction& lhs,
                       const TiledHloInstruction& rhs) {
  return !(lhs == rhs);
}

template <typename H>
H AbslHashValue(H h, const TiledHloInstruction& tiled_hlo_instruction) {
  h = H::combine(std::move(h), *tiled_hlo_instruction.hlo(),
                 tiled_hlo_instruction.tile());
  for (const TiledHloInstruction* operand : tiled_hlo_instruction.operands()) {
    h = H::combine(std::move(h), operand);
  }
  return h;
}

class DefinitionCache {
 public:
  // Finds a root-level definition for a given HLO instruction with the given
  // tile.
  TiledHloInstruction* FindRootDef(const HloInstruction& hlo, Tile tile) const;

  // Finds a definition for the given candidate instruction (use).
  //
  // We assume that instructions are processed in the order they show up in the
  // HLO DAG (top-down). If a dominating definition is found in the cache, it
  // means it was already created in a previously processed path. If the cached
  // definition is a sibling of the candidate (shares the same parent and
  // region), it is safe to reuse it because the final post-order sort will
  // ensure the definition is scheduled before the user.
  TiledHloInstruction* FindDef(const TiledHloInstruction& use) const;

  // Inserts a definition into the cache.
  void AddDef(TiledHloInstruction& def) {
    tiled_hlo_cache_[{def.hlo(), def.tile()}].push_back(&def);
  }

 private:
  struct TiledHloKey {
    const HloInstruction* hlo;
    Tile tile;

    bool operator==(const TiledHloKey& other) const {
      return hlo == other.hlo && tile == other.tile;
    }

    template <typename H>
    friend H AbslHashValue(H h, const TiledHloKey& key) {
      return H::combine(std::move(h), key.hlo, key.tile);
    }
  };

  absl::flat_hash_map<TiledHloKey, std::vector<TiledHloInstruction*>>
      tiled_hlo_cache_;
};

// Constructs and holds symbolic tiles for all the instructions within a fusion.
class TiledHloComputation {
 public:
  using InstructionType = TiledHloInstruction;

  static absl::StatusOr<TiledHloComputation> Tile(
      const HloFusionAdaptor& fusion,
      std::unique_ptr<TilingSpace> tiling_space);

  // Returns the symbolic tiled HLO instructions in def-before-use order.
  const TiledHloRegion& tiled_hlo_instructions() const {
    return tiled_hlo_instructions_;
  }

  // Returns an iterator range over the instructions in the computation in
  // def-before-use order.
  tsl::gtl::iterator_range<UnwrappingIterator<TiledHloRegion::const_iterator>>
  instructions() const {
    return {MakeUnwrappingIterator(tiled_hlo_instructions_.begin()),
            MakeUnwrappingIterator(tiled_hlo_instructions_.end())};
  }

  // Return the underlying MLIRContext.
  mlir::MLIRContext* GetMLIRContext() const {
    return tiling_space_->mlir_context();
  };

  // Returns the tiling space.
  const TilingSpace& tiling_space() const { return *tiling_space_; }

  // Returns the root instructions.
  absl::Span<const TiledHloInstruction* const> roots() const { return roots_; }

  // Returns the map from runtime variable symbol to TiledHloInstruction.
  const absl::flat_hash_map<int64_t,
                            std::pair<const TiledHloInstruction*, Interval>>&
  rt_symbol_to_tiled_hlo() const {
    return rt_symbol_to_tiled_hlo_;
  }

  // Returns a string representation of the analysis.
  std::string ToString() const;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink,
                            const TiledHloComputation& tiled_computation) {
    sink.Append(tiled_computation.ToString());
  }

  // Temporary helpers to match API of the old TiledHloComputation.
  // TODO: b/509505290 -- Remove these once we migrate to this tiling and the
  // old API is removed.
  int64_t num_output_tiles() const {
    int64_t res = 1;
    for (const auto& dimension : tiling_space_->dimensions()) {
      if (dimension.type != TilingSpace::DimensionSemantics::kParallel) {
        continue;
      }
      res *= CeilOfRatio(dimension.dimension_size, *dimension.tile_size);
    }
    return res;
  }

 private:
  TiledHloComputation(
      std::unique_ptr<TilingSpace> tiling_space,
      TiledHloRegion tiled_hlo_instructions,
      llvm::SmallVector<const TiledHloInstruction*> roots,
      absl::flat_hash_map<int64_t,
                          std::pair<const TiledHloInstruction*, Interval>>
          rt_symbol_to_tiled_hlo)
      : tiling_space_(std::move(tiling_space)),
        tiled_hlo_instructions_(std::move(tiled_hlo_instructions)),
        roots_(std::move(roots)),
        rt_symbol_to_tiled_hlo_(std::move(rt_symbol_to_tiled_hlo)) {}

  static absl::StatusOr<TiledHloRegion> CreateHloRegion(
      std::unique_ptr<TiledHloInstruction> tiled_root,
      const HloFusionAdaptor& fusion, TilingSpace& tiling_space,
      absl::flat_hash_map<int64_t,
                          std::pair<const TiledHloInstruction*, Interval>>&
          rt_symbol_to_tiled_hlo);

  std::unique_ptr<TilingSpace> tiling_space_;

  // The tiled HLO instructions in def-before-use order.
  TiledHloRegion tiled_hlo_instructions_;

  // Stores pointers to the root instructions. Note that they do not necessarily
  // appear all at the end of `instructions_`.
  llvm::SmallVector<const TiledHloInstruction*> roots_;

  // Map from runtime variable symbol to TiledHloInstruction.
  absl::flat_hash_map<int64_t, std::pair<const TiledHloInstruction*, Interval>>
      rt_symbol_to_tiled_hlo_;
};

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_TILED_HLO_H_
