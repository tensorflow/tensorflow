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

#include "xla/service/gpu/model/symbolic_tile_analysis.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/hash/hash.h"
#include "absl/log/check.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace {

using ::mlir::AffineExpr;
using ::mlir::MLIRContext;

struct OutputTilingInfo {
  // The number of output tiles for each dimension.
  // E.g. if dimensions are [29, 16] and tile size is [4, 8] then
  // `num_output_tiles_per_dim` will be [8, 2].
  llvm::SmallVector<int64_t> num_output_tiles_per_dim;

  // An indexing map from an output tile multi-index to tile offsets.
  //
  // The dimensions of the indexing map correspond to the dimensions passed
  // to `ComputeOutputTilingInfo` and the number of dimensions is equal to the
  // size of `num_output_tiles_per_dim`. For example above it would look like:
  //   `(tid_0, tid_1) -> (<tile 0 offset>, <tile 1 offset>)`.
  IndexingMap output_tile_offset_indexing;
};

OutputTilingInfo ComputeOutputTilingInfo(absl::Span<const int64_t> dimensions,
                                         absl::Span<const int64_t> tile_sizes,
                                         mlir::MLIRContext* mlir_context) {
  CHECK_EQ(dimensions.size(), tile_sizes.size());  // Crash OK

  llvm::SmallVector<int64_t> outer_loop_bounds;
  outer_loop_bounds.reserve(dimensions.size());
  for (auto [dim_size, tile_size] : llvm::zip(dimensions, tile_sizes)) {
    outer_loop_bounds.push_back(CeilOfRatio(dim_size, tile_size));
  }

  llvm::SmallVector<AffineExpr> tiled_dims;
  tiled_dims.reserve(dimensions.size());

  for (auto [dim_id, tile_size] : llvm::enumerate(tile_sizes)) {
    tiled_dims.push_back(tile_size *
                         mlir::getAffineDimExpr(dim_id, mlir_context));
  }

  IndexingMap output_tile_offset_indexing = IndexingMap::FromTensorSizes(
      mlir::AffineMap::get(
          /*dimCount=*/dimensions.size(), /*symbolCount=*/0, tiled_dims,
          mlir_context),
      /*dim_upper_bounds=*/outer_loop_bounds, /*symbol_upper_bounds=*/{});
  for (int i = 0; i < output_tile_offset_indexing.GetDimVarsCount(); ++i) {
    output_tile_offset_indexing.RenameDimVar(i, absl::StrCat("tid_", i));
  }
  return {outer_loop_bounds, output_tile_offset_indexing};
}

absl::StatusOr<IndexingMap> ComputeTileOffsetIndexing(
    const SymbolicTiledHloInstruction& tiled_hlo,
    const IndexingMap& output_tile_offset_indexing,
    mlir::MLIRContext* mlir_context) {
  IndexingMap tile_offset_indexing = ComposeIndexingMaps(
      output_tile_offset_indexing, tiled_hlo.indexing_map());

  // A symbol in an indexing map means that to produce on element of output, we
  // need to read all elements of input in the symbol range. Since this function
  // computes start of the tile, we need to substitute each symbol with its
  // lower bound value. We assume here the iteration order is normalized.
  // TODO(b/330906085): Support cases when tile offsets are not 0.
  if (absl::c_any_of(tile_offset_indexing.GetSymbolBounds(),
                     [](const Interval& symbol_bound) {
                       return symbol_bound.lower != 0;
                     })) {
    return absl::FailedPreconditionError(
        absl::StrCat("Symbol lower bound is not zero. ",
                     ToString(tiled_hlo.indexing_map())));
  }

  std::vector<AffineExpr> symbol_lower_bounds(
      tile_offset_indexing.GetRangeVarsCount(),
      mlir::getAffineConstantExpr(0, mlir_context));
  symbol_lower_bounds.reserve(tile_offset_indexing.GetSymbolCount());
  for (int i = 0; i < tile_offset_indexing.GetRTVarsCount(); ++i) {
    symbol_lower_bounds.push_back(mlir::getAffineSymbolExpr(i, mlir_context));
  }

  mlir::AffineMap simplified_affine_map =
      tile_offset_indexing.GetAffineMap().replaceDimsAndSymbols(
          /*dimReplacements=*/{},
          /*symReplacements=*/symbol_lower_bounds,
          /*numResultDims=*/tile_offset_indexing.GetDimVarsCount(),
          /*numResultSyms=*/tile_offset_indexing.GetRTVarsCount());

  IndexingMap simplified_indexing_map =
      IndexingMap{simplified_affine_map, tile_offset_indexing.GetDimVars(),
                  /*range_vars=*/{}, tile_offset_indexing.GetRTVars()};

  simplified_indexing_map.Simplify();
  simplified_indexing_map.RescaleSymbols();
  simplified_indexing_map.RemoveUnusedSymbols();

  return simplified_indexing_map;
}

// A hash set of unique pointers.
//
// This set add a few key features on top of absl::flat_hash_set<T*>:
// * The set takes ownership of the object and deletes the object if an
//   equivalent element is already in the set.
// * Values are compared by the value behind the pointer, not the pointer
//   itself.
// * This set provides a convenient method to extract the unique pointers into a
//   vector.
// * Values are stored in the order of insertion. This is useful when we have
//   information about the order in which we process elements. For example,
//   during the construction of TiledHloComputation from
//   SymbolicTiledHloInstructions, we know that instruction are already sorted
//   in def-before-use order.
template <typename T>
class OrderedUniquePtrValueHashSet {
 public:
  // Inserts an element into the set.
  // Returns a pair of a non-owning raw pointer to the element that was inserted
  // (or the element that prevented insertion) and a bool indicating whether the
  // element was inserted.
  std::pair<T*, bool> Insert(std::unique_ptr<T> elem) {
    auto [it, inserted] = hash_set_.insert(elem.get());
    if (inserted) {
      data_.push_back(std::move(elem));
    }
    return {*it, inserted};
  }

  void Reserve(int64_t n) {
    hash_set_.reserve(n);
    data_.reserve(n);
  }

  // Moves data out of the set.
  std::vector<std::unique_ptr<T>> ExtractData() { return std::move(data_); }

 private:
  struct PtrHash {
    size_t operator()(const T* v) const { return absl::HashOf(*v); }
  };

  struct PtrEqual {
    bool operator()(const T* lhs, const T* rhs) const {
      return lhs == rhs || *lhs == *rhs;
    }
  };

  // Stores non-owning pointers to the elements in the set. Elements are
  // compared by the value behind the pointer, not the pointer itself.
  absl::flat_hash_set<T*, PtrHash, PtrEqual> hash_set_;

  // Stores owning pointers to the elements in the set.
  std::vector<std::unique_ptr<T>> data_;
};

// Detects pathological cases on which symbolic tile derivation should bail out.
// Note that this function bypasses temporary limitations of the infrastructure,
// and not actual fundamental limitations.
FusionDecision ShouldProceedWithSymbolicTileDerivation(
    const SymbolicTiledHloInstruction& tiled_hlo_instruction) {
  const HloInstruction* hlo = tiled_hlo_instruction.hlo();
  const IndexingMap& indexing_map = tiled_hlo_instruction.indexing_map();

  // Bail out on instructions that are known to cause problems down the
  // line. This is not an inherent limitation of the approach, but simply
  // issues to be resolved in the current implementation.
  if (hlo->opcode() == HloOpcode::kConcatenate) {
    return FusionDecision::Forbid("Bailing out on ") << hlo->ToString();
  }

  // Due to the issue highlighted in b/365727080, and the related workaround
  // deriving a standalone symbolic tile when constructing Triton-specific
  // constraints, reshapes and bitcasts may cause problems down the line.
  // The added check here allows us to bail out early when we reach such a
  // a problematic.
  //
  // TODO(b/365727080): get rid of this filter once the issue is properly
  // fixed.
  if (hlo->opcode() == HloOpcode::kReshape ||
      hlo->opcode() == HloOpcode::kBitcast) {
    mlir::MLIRContext* ctx = indexing_map.GetMLIRContext();

    IndexingMap reshape_indexing_map =
        *ComputeOutputToInputIndexing(hlo, /*output_id=*/0, ctx)
             .indexing_maps[0]
             .begin();

    std::optional<SymbolicTile> reshape_symbolic_tile =
        SymbolicTile::FromIndexingMap(reshape_indexing_map);

    if (!reshape_symbolic_tile.has_value()) {
      return FusionDecision::Forbid("Bailing out on reshape ")
             << hlo->ToString() << " with indexing map "
             << ToString(reshape_indexing_map);
    }
  }

  return FusionDecision::Allow();
}

// Sets a SymbolicTile for each tiled hlo instruction and computes their
// combined constraints. Returns a FusionDecision if a SymbolicTile cannot be
// computed for some instruction or if the constraints are unsatisfiable.
// Returns the combined constraints otherwise.
std::variant<ConstraintExpression, FusionDecision>
SetSymbolicTilesAndComputeConstraints(
    std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
        tiled_hlo_instructions,
    const HloFusionAdaptor& fusion_adaptor) {
  ConstraintExpression constraints = ConstraintExpression::GetAlwaysSatisfied();
  for (const std::unique_ptr<SymbolicTiledHloInstruction>&
           tiled_hlo_instruction : tiled_hlo_instructions) {
    const HloInstruction* hlo = tiled_hlo_instruction->hlo();
    const IndexingMap& indexing_map = tiled_hlo_instruction->indexing_map();

    // We first verify some preconditions on the instructions we intend to
    // codegen. We first check whether an instruction is part of the fusion
    // adaptor, as `tiled_hlo_instructions` may contain instructions that won't
    // be codegen'd (the operands to the fusion computation).
    if (fusion_adaptor.ContainsInstruction(hlo)) {
      FusionDecision should_proceed =
          ShouldProceedWithSymbolicTileDerivation(*tiled_hlo_instruction);
      if (!should_proceed) {
        return should_proceed;
      }
    }

    auto symbolic_tile = SymbolicTile::FromIndexingMap(indexing_map);
    if (!symbolic_tile.has_value()) {
      return FusionDecision::Forbid("Failed to compute symbolic tile for ")
             << ToString(indexing_map) << " for HLO " << hlo->ToString();
    }

    if (!symbolic_tile->is_satisfiable()) {
      return FusionDecision::Forbid("Symbolic tile ")
             << symbolic_tile->ToString() << " is not satisfiable for "
             << ToString(indexing_map) << " for HLO " << hlo->ToString();
    }

    constraints = constraints && symbolic_tile->constraints();
    constraints.Simplify();

    if (!constraints.is_satisfiable()) {
      return FusionDecision::Forbid("Fusion has unsatisfiable constraints");
    }

    tiled_hlo_instruction->set_symbolic_tile(*std::move(symbolic_tile));
  }

  return constraints;
}

// Sorts tiled hlo instructions in def-before-use order.
void SortTiledHloInstructionsInPostOrder(
    std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
        tiled_hlo_instructions,
    const SymbolicTiledHloInstruction* root_tiled_hlo) {
  absl::flat_hash_map<const SymbolicTiledHloInstruction*, int64_t>
      topological_order;

  std::function<void(const SymbolicTiledHloInstruction*)> visit_instruction;
  visit_instruction = [&](const SymbolicTiledHloInstruction* instruction) {
    if (topological_order.contains(instruction)) {
      return;
    }
    for (const SymbolicTiledHloInstruction* operand : instruction->operands()) {
      visit_instruction(operand);
    }
    topological_order[instruction] = topological_order.size();
  };

  visit_instruction(root_tiled_hlo);

  absl::c_sort(tiled_hlo_instructions,
               [&](const std::unique_ptr<SymbolicTiledHloInstruction>& t1,
                   const std::unique_ptr<SymbolicTiledHloInstruction>& t2) {
                 return topological_order.at(t1.get()) <
                        topological_order.at(t2.get());
               });
}

}  // anonymous namespace

/*static*/ SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeComputation(
    const HloComputation& computation, MLIRContext* ctx,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder) {
  auto fusion = HloFusionAdaptor::ForComputation(&computation);
  return SymbolicTileAnalysis::AnalyzeFusion(
      *fusion, ctx, emitter_specific_constraints_builder);
}

/*static*/ SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeFusion(
    const HloFusionAdaptor& fusion, MLIRContext* ctx,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder) {
  OrderedUniquePtrValueHashSet<SymbolicTiledHloInstruction>
      tiled_hlo_instructions_set;

  auto roots = fusion.GetRoots();
  if (roots.size() > 1) {
    return FusionDecision::Forbid("Multi-output fusions are not supported. ")
           << fusion.ToString();
  }
  auto& root = roots[0];

  auto [root_tiled_hlo, _] = tiled_hlo_instructions_set.Insert(
      std::make_unique<SymbolicTiledHloInstruction>(
          &root.instruction(), CreateIdentityMap(root.shape(), ctx)));

  std::vector<SymbolicTiledHloInstruction*> worklist = {root_tiled_hlo};

  while (!worklist.empty()) {
    auto tiled_hlo_instruction = worklist.back();
    worklist.pop_back();
    HloInstructionAdaptor instruction_adaptor(*tiled_hlo_instruction->hlo(),
                                              &fusion);

    if (!fusion.ContainsInstruction(instruction_adaptor)) {
      continue;
    }

    HloInstructionIndexing operands_indexing =
        ComputeOutputToInputIndexing(tiled_hlo_instruction->hlo(),
                                     /*output_id=*/0, ctx);

    for (auto [operand, operand_indexing_map_set] :
         llvm::zip(instruction_adaptor.GetOperands(),
                   operands_indexing.indexing_maps)) {
      CHECK_EQ(operand_indexing_map_set.size(), 1);  // Crash OK

      IndexingMap operand_indexing_map =
          ComposeIndexingMaps(tiled_hlo_instruction->indexing_map(),
                              *operand_indexing_map_set.begin());
      if (operand_indexing_map.IsUndefined()) {
        return FusionDecision::Forbid(
                   "Couldn't derive indexing map for instruction ")
               << tiled_hlo_instruction->hlo()->ToString() << " and operand "
               << operand.instruction().ToString();
      }
      operand_indexing_map.Simplify();
      operand_indexing_map.RescaleSymbols();
      operand_indexing_map.RemoveUnusedSymbols();

      auto [operand_tiled_hlo, inserted] = tiled_hlo_instructions_set.Insert(
          std::make_unique<SymbolicTiledHloInstruction>(
              &operand.instruction(), std::move(operand_indexing_map)));

      tiled_hlo_instruction->AppendOperand(operand_tiled_hlo);

      if (inserted) {
        worklist.push_back(operand_tiled_hlo);
      }
    }
  }

  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
      tiled_hlo_instructions = tiled_hlo_instructions_set.ExtractData();

  // Order instructions in def-before-use order.
  SortTiledHloInstructionsInPostOrder(tiled_hlo_instructions, root_tiled_hlo);

  // Set symbolic tiles for each tiled hlo instruction and compute combined
  // constraints.
  std::variant<ConstraintExpression, FusionDecision> constraints_or =
      SetSymbolicTilesAndComputeConstraints(tiled_hlo_instructions, fusion);
  if (std::holds_alternative<FusionDecision>(constraints_or)) {
    return std::get<FusionDecision>(constraints_or);
  }

  // Create emitter-specific constraints if a builder was provided.
  std::unique_ptr<EmitterSpecificConstraints> emitter_specific_constraints;
  if (emitter_specific_constraints_builder != nullptr) {
    emitter_specific_constraints =
        emitter_specific_constraints_builder(tiled_hlo_instructions, fusion);
  }

  return SymbolicTileAnalysis(
      std::move(tiled_hlo_instructions),
      std::get<ConstraintExpression>(std::move(constraints_or)),
      std::move(emitter_specific_constraints), ctx);
}

absl::StatusOr<bool> SymbolicTileAnalysis::ParametersSatisfyConstraints(
    absl::Span<const int64_t> tile_parameters) const {
  if (!constraints_.is_satisfiable()) {
    return absl::FailedPreconditionError(
        "SymbolicTileAnalysis's constraints are not satisfiable. "
        "This should never happen.");
  }

  if (tile_parameters.size() != num_tile_parameters()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Failed to check if tile parameters satisfy constraints. Number of "
        "provided parameters doesn't match number of expected parameters "
        "(%d != %d)",
        tile_parameters.size(), num_tile_parameters()));
  }

  if (emitter_specific_constraints_ != nullptr) {
    TF_ASSIGN_OR_RETURN(
        bool constraints_are_satisfied,
        emitter_specific_constraints_->ParametersSatisfyConstraints(
            tile_parameters));
    if (!constraints_are_satisfied) {
      return false;
    }
  }

  return constraints_.IsSatisfiedBy(tile_parameters);
}

absl::StatusOr<TiledHloComputation>
SymbolicTileAnalysis::ComputeTiledHloInstructions(
    absl::Span<const int64_t> tile_parameters,
    bool constraints_are_known_satisfied,
    bool compute_all_tile_offset_indexing_maps) const {
  if (!constraints_are_known_satisfied) {
    TF_ASSIGN_OR_RETURN(bool constraints_are_satisfied,
                        ParametersSatisfyConstraints(tile_parameters));
    if (!constraints_are_satisfied) {
      return absl::InvalidArgumentError(
          absl::StrCat("Tile parameters ", absl::StrJoin(tile_parameters, ", "),
                       " do not satisfy constraints."));
    }
  }

  // Offset indexing is needed to emit loads/stores and to deduplicate
  // instructions. In some cases, for example in Cost Model, we need to only
  // deduplicate instructions.
  //
  // Computing tile offset indexing maps is very expensive. This is a
  // performance optimization to avoid computing tile offset indexing maps for
  // instructions that are not needed.
  //
  // Tile offset indexing is only needed when one HLO instruction has no
  // operands and multiple tiles have exactly same sizes and strides. We skip
  // strides in the heuristic below, because they are rarely different.
  //
  // Using `compute_all_tile_offset_indexing_maps` will force to compute tile
  // offset indexing maps for all instructions.
  llvm::SmallPtrSet<const HloInstruction*, 8> parameters_with_offset_indexing;
  absl::flat_hash_map<const SymbolicTiledHloInstruction*,
                      llvm::SmallVector<int64_t>>
      tile_sizes_map;
  if (!compute_all_tile_offset_indexing_maps) {
    absl::flat_hash_set<size_t> hashes;
    for (const std::unique_ptr<SymbolicTiledHloInstruction>&
             symbolic_tiled_hlo : symbolic_tiled_hlo_instructions_) {
      if (!symbolic_tiled_hlo->operands().empty()) {
        continue;
      }

      llvm::SmallVector<int64_t> tile_sizes =
          symbolic_tiled_hlo->TileSizes(tile_parameters);
      size_t hash_value = absl::HashOf(symbolic_tiled_hlo->hlo(),
                                       absl::Span<const int64_t>(tile_sizes));
      tile_sizes_map.emplace(symbolic_tiled_hlo.get(), std::move(tile_sizes));

      auto [it, inserted] = hashes.insert(hash_value);
      // Two SymbolicTiledHloInstructions have identical hash when looking only
      // at HLO instruction pointer and tile sizes. We need to compute tile
      // offset indexing maps for all tiles of this HLO instruction.
      if (!inserted) {
        parameters_with_offset_indexing.insert(symbolic_tiled_hlo->hlo());
      }
    }
  }

  OutputTilingInfo output_tiling_info = ComputeOutputTilingInfo(
      GetRoot()->hlo()->shape().dimensions(), tile_parameters, context_);

  OrderedUniquePtrValueHashSet<TiledHloInstruction> tiled_hlo_instructions_set;
  absl::flat_hash_map<const SymbolicTiledHloInstruction*, TiledHloInstruction*>
      symbolic_to_tiled_hlo_map;
  // The actual number of TiledHloInstructions can be smaller than the number of
  // SymbolicTiledHloInstructions, because some instruction will be
  // deduplicated, but we reserve to the upper bound to avoid reallocations and
  // additional hash calculations.
  tiled_hlo_instructions_set.Reserve(symbolic_tiled_hlo_instructions_.size());

  for (const std::unique_ptr<SymbolicTiledHloInstruction>& symbolic_tiled_hlo :
       symbolic_tiled_hlo_instructions_) {
    llvm::SmallVector<int64_t> tile_sizes;
    auto it = tile_sizes_map.find(symbolic_tiled_hlo.get());
    if (it != tile_sizes_map.end()) {
      tile_sizes = it->second;
    } else {
      tile_sizes = symbolic_tiled_hlo->TileSizes(tile_parameters);
    }

    llvm::SmallVector<int64_t> tile_strides =
        symbolic_tiled_hlo->TileStrides(tile_parameters);

    std::optional<IndexingMap> tile_offset_indexing;
    if (compute_all_tile_offset_indexing_maps ||
        parameters_with_offset_indexing.contains(symbolic_tiled_hlo->hlo()) ||
        symbolic_tiled_hlo->hlo()->opcode() == HloOpcode::kIota) {
      TF_ASSIGN_OR_RETURN(
          tile_offset_indexing,
          ComputeTileOffsetIndexing(
              *symbolic_tiled_hlo,
              output_tiling_info.output_tile_offset_indexing, context_));
    }

    llvm::SmallVector<const TiledHloInstruction*> operands;
    for (const SymbolicTiledHloInstruction* operand :
         symbolic_tiled_hlo->operands()) {
      operands.push_back(symbolic_to_tiled_hlo_map.at(operand));
    }

    TF_ASSIGN_OR_RETURN(std::unique_ptr<TiledHloInstruction> tiled_hlo_holder,
                        TiledHloInstruction::Create(
                            symbolic_tiled_hlo->hlo(), std::move(operands),
                            std::move(tile_sizes), std::move(tile_strides),
                            std::move(tile_offset_indexing)));

    auto [tiled_hlo, inserted] =
        tiled_hlo_instructions_set.Insert(std::move(tiled_hlo_holder));

    symbolic_to_tiled_hlo_map[symbolic_tiled_hlo.get()] = tiled_hlo;
  }
  return TiledHloComputation::FromSortedTiledHloInstructions(
      tiled_hlo_instructions_set.ExtractData(),
      output_tiling_info.num_output_tiles_per_dim);
}

std::string SymbolicTileAnalysis::ToString() const {
  std::stringstream ss;
  NameUniquer name_uniquer("_");
  absl::flat_hash_map<SymbolicTiledHloInstruction*, std::string> tile_names;

  for (const auto& tiled_hlo : symbolic_tiled_hlo_instructions_) {
    std::string tile_name = name_uniquer.GetUniqueName(
        absl::StrCat(tiled_hlo->hlo()->name(), ".tile_0"));
    tile_names[tiled_hlo.get()] = tile_name;

    absl::InlinedVector<std::string, 4> operand_names;
    for (const auto& operand : tiled_hlo->operands()) {
      operand_names.push_back(tile_names.at(operand));
    }

    ss << tile_name << " = " << HloOpcodeString(tiled_hlo->hlo()->opcode())
       << "(" << absl::StrJoin(operand_names, ", ") << ")\n";

    ss << tiled_hlo->ToString();
  }
  return ss.str();
}

namespace {

// The possible tiles sizes for one dimension.
std::vector<int64_t> PossibleTileSizesForOneDimension(int64_t dim_size) {
  CHECK_GE(dim_size, 1);

  std::vector<int64_t> result;
  result.reserve(absl::bit_width(static_cast<uint64_t>(dim_size)));
  for (int64_t tile_size = 1; tile_size < dim_size; tile_size *= 2) {
    result.push_back(tile_size);
  }
  result.push_back(dim_size);
  return result;
}

}  // namespace

namespace detail {
std::vector<SymbolicTileAnalysis::Tiling> GetGoodTilings(
    absl::Span<const int64_t> dim_sizes,
    std::function<bool(absl::Span<const int64_t>)> is_valid) {
  CHECK(is_valid != nullptr);

  std::vector<SymbolicTileAnalysis::Tiling> tilings;
  tilings.push_back({});
  for (int dim_size : dim_sizes) {
    std::vector<int64_t> possible_tile_sizes =
        PossibleTileSizesForOneDimension(dim_size);
    std::vector<SymbolicTileAnalysis::Tiling> extended_tilings;
    extended_tilings.reserve(tilings.size() * possible_tile_sizes.size());
    for (const SymbolicTileAnalysis::Tiling& tiling : tilings) {
      for (int64_t tile_size : possible_tile_sizes) {
        SymbolicTileAnalysis::Tiling extended_tiling = tiling;
        extended_tiling.push_back(tile_size);
        extended_tilings.push_back(extended_tiling);
      }
    }
    tilings = std::move(extended_tilings);
  }

  tilings.erase(
      std::remove_if(tilings.begin(), tilings.end(), std::not_fn(is_valid)),
      tilings.end());

  return tilings;
}
}  // namespace detail

absl::StatusOr<std::vector<SymbolicTileAnalysis::Tiling>>
SymbolicTileAnalysis::GetGoodTilings() const {
  TF_RET_CHECK(!symbolic_tiled_hlo_instructions_.empty());
  TF_RET_CHECK(symbolic_tiled_hlo_instructions_.back() != nullptr);

  const SymbolicTiledHloInstruction& instr =
      *symbolic_tiled_hlo_instructions_.back();
  TF_RET_CHECK(instr.hlo() != nullptr);
  const Shape& shape = instr.hlo()->shape();
  if (!absl::c_all_of(shape.dimensions(),
                      [](int64_t dim_size) { return dim_size >= 1; })) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Shape %s has zero or negative dimensions.", shape.ToString()));
  }

  absl::Status status = absl::OkStatus();
  std::vector<SymbolicTileAnalysis::Tiling> result = detail::GetGoodTilings(
      shape.dimensions(), [&](absl::Span<const int64_t> tile_sizes) {
        absl::StatusOr<bool> is_valid =
            ParametersSatisfyConstraints(tile_sizes);
        if (!is_valid.ok()) {
          status = is_valid.status();
          return false;
        }
        return is_valid.value();
      });

  if (status.ok()) {
    return result;
  }

  return status;
}

}  // namespace gpu
}  // namespace xla
