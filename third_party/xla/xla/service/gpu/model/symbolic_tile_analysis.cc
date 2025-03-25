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
#include <iterator>
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
#include "absl/log/log.h"
#include "absl/numeric/bits.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/raw_ostream.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/model/constraint_expression.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/gpu/model/tiled_hlo_instruction.h"
#include "xla/service/instruction_fusion.h"
#include "xla/service/name_uniquer.h"
#include "xla/shape.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace {

using ::mlir::AffineExpr;
using ::mlir::AffineExprKind;
using ::mlir::MLIRContext;

struct OutputTilingInfo {
  // The number of output tiles for each dimension of the root indexing.
  // E.g. if dimensions are [29, 16] and tile size is [4, 8] then
  // `num_output_tiles_per_dim` will be [8, 2].
  llvm::SmallVector<int64_t> num_output_tiles_per_dim;

  // An indexing map from an output tile multi-index to tile offsets.
  //
  // The dimensions of the indexing map correspond to the dimensions passed
  // to `ComputeOutputTilingInfo` and the number of dimensions is equal to the
  // size of `num_output_tiles_per_dim`. For example above it would look like:
  //   `(pid_0, pid_1) -> (<tile 0 offset>, <tile 1 offset>)`.
  IndexingMap output_tile_offset_indexing;
};

OutputTilingInfo ComputeOutputTilingInfo(const IndexingMap& root_indexing,
                                         absl::Span<const int64_t> tile_sizes,
                                         mlir::MLIRContext* mlir_context) {
  int64_t rank = root_indexing.GetDimVarsCount();
  CHECK_EQ(rank, tile_sizes.size());  // Crash OK

  llvm::SmallVector<int64_t> outer_loop_bounds;
  outer_loop_bounds.reserve(rank);
  for (auto [dim_bounds, tile_size] :
       llvm::zip(root_indexing.GetDimensionBounds(), tile_sizes)) {
    CHECK_EQ(dim_bounds.lower, 0)
        << "Root indexing domain does not start at 0.";
    outer_loop_bounds.push_back(CeilOfRatio(dim_bounds.upper + 1, tile_size));
  }

  llvm::SmallVector<AffineExpr> tiled_dims;
  tiled_dims.reserve(rank);
  for (auto [dim_id, tile_size] : llvm::enumerate(tile_sizes)) {
    tiled_dims.push_back(tile_size *
                         mlir::getAffineDimExpr(dim_id, mlir_context));
  }

  std::vector<IndexingMap::Variable> dim_vars =
      DimVarsFromTensorSizes(outer_loop_bounds);
  // Name the dimension variables for convenience.
  for (auto&& [idx, dim_var] : llvm::enumerate(dim_vars)) {
    dim_var.name = absl::StrCat("pid_", idx);
  }

  IndexingMap output_tile_offset_indexing{
      mlir::AffineMap::get(
          /*dimCount=*/rank, /*symbolCount=*/0, tiled_dims, mlir_context),
      dim_vars, /*range_vars=*/{}, /*rt_vars=*/{}};
  return {outer_loop_bounds, output_tile_offset_indexing};
}

// Extension of SymbolicTiledHloInstruction for fusions that holds the analysis
// of the fusion's computation.
class SymbolicTiledHloFusionInstruction : public SymbolicTiledHloInstruction {
 public:
  SymbolicTiledHloFusionInstruction(const HloInstruction* hlo,
                                    IndexingMap indexing_map,
                                    SymbolicTileAnalysis analysis)
      : SymbolicTiledHloInstruction(hlo, std::move(indexing_map)),
        analysis_(std::move(analysis)) {}

  SymbolicTileAnalysis analysis_;
};

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

  // Here we rely on IndexingMap internals. Symbols are split into range vars
  // and runtime variables. The range vars come first, followed by the runtime
  // variables.
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

// Whether the given HLO instruction is part of a nested GEMM fusion.
bool IsWithinNestedGemmFusion(const HloInstruction* hlo) {
  const HloComputation* computation = hlo->parent();
  if (computation->IsFusionComputation()) {
    const GpuBackendConfig backend_config =
        *computation->FusionInstruction()->backend_config<GpuBackendConfig>();
    absl::string_view fusion_kind =
        backend_config.fusion_backend_config().kind();
    return fusion_kind == kTritonNestedGemmFusionKind;
  }

  return false;
}

// Detects pathological cases on which symbolic tile derivation should bail out.
// Note that this function bypasses temporary limitations of the infrastructure,
// and not actual fundamental limitations.
FusionDecision ShouldProceedWithSymbolicTileDerivation(
    const SymbolicTiledHloInstruction& tiled_hlo_instruction) {
  const HloInstruction* hlo = tiled_hlo_instruction.hlo();
  const IndexingMap& indexing_map = tiled_hlo_instruction.indexing_map();
  // Bail out on concatenates in the general path for now, but allow a
  // restricted form of concatenates for the nested GEMM fusion path.
  //
  // Relaxing this restriction will require making sure that the cost model
  // works well with concatenates, and that we always construct nested fusions
  // for concatenates.
  if (hlo->opcode() == HloOpcode::kConcatenate &&
      !IsWithinNestedGemmFusion(hlo)) {
    return FusionDecision::Forbid("Bailing out on ") << hlo->ToString();
  }

  // Due to the issue highlighted in b/365727080, and the related workaround
  // deriving a standalone symbolic tile when constructing Triton-specific
  // constraints, reshapes and bitcasts may cause problems down the line.
  // The added check here allows us to bail out early when we reach such a
  // a problematic case.
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

// Extracts HloInstructions from a span of HloInstructionAdaptors.
absl::InlinedVector<const HloInstruction*, 2> ToInstructions(
    absl::Span<const HloInstructionAdaptor> instruction_adaptors) {
  absl::InlinedVector<const HloInstruction*, 2> hlo_instructions;
  hlo_instructions.reserve(instruction_adaptors.size());
  absl::c_transform(
      instruction_adaptors, std::back_inserter(hlo_instructions),
      [&](const HloInstructionAdaptor& instr) { return &instr.instruction(); });
  return hlo_instructions;
}

// Returns the index of the single root without any users among the given roots.
// This implies that any other root is an ancestor of the returned root.
// Returns an error if there are multiple or no roots without any users.
absl::StatusOr<int64_t> GetRealRootIndex(
    absl::Span<const HloInstructionAdaptor> fusion_adaptor_roots) {
  auto has_no_users = [](const HloInstructionAdaptor& root) {
    return root.GetUsers().empty();
  };
  auto it = absl::c_find_if(fusion_adaptor_roots, has_no_users);
  if (it == fusion_adaptor_roots.end()) {
    return absl::FailedPreconditionError(
        "Each fusion should have at least one root without users but no root "
        "was found.");
  }
  if (std::find_if(std::next(it), fusion_adaptor_roots.end(), has_no_users) !=
      fusion_adaptor_roots.end()) {
    return absl::FailedPreconditionError(
        "Only simple multi-output fusions with one real root are supported but "
        "multiple roots were found.");
  }
  return it - fusion_adaptor_roots.begin();
}

// Computes the indexing information for the roots of the 'fusion'.
/*static*/ absl::StatusOr<RootIndexing> SymbolicTileAnalysis::GetRootIndexing(
    const HloFusionAdaptor& fusion, MLIRContext* ctx) {
  auto fusion_adaptor_roots = fusion.GetRoots();

  TF_ASSIGN_OR_RETURN(int64_t real_root_index,
                      GetRealRootIndex(fusion_adaptor_roots));

  // Keep track of the roots separately. If there is just a single root, we
  // don't need that, as it will necessarily appear last in def-before-use
  // order. But with multiple roots, we can have roots that are also ancestors
  // of another root.
  absl::InlinedVector<const HloInstruction*, 2> roots =
      ToInstructions(fusion_adaptor_roots);

  auto indexing_map = CreateIdentityMap(roots[real_root_index]->shape(), ctx);
  return RootIndexing{real_root_index, std::move(roots),
                      /*real_root_indexing=*/std::move(indexing_map)};
}

/*static*/ SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeComputation(
    const HloComputation& computation, MLIRContext* ctx,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder) {
  auto fusion = HloFusionAdaptor::ForComputation(&computation);
  return SymbolicTileAnalysis::AnalyzeFusion(
      *fusion, ctx, emitter_specific_constraints_builder);
}

/*static*/ SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeFusionImpl(
    const HloFusionAdaptor& fusion, MLIRContext* ctx,
    const RootIndexing& root_indexing,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder) {
  OrderedUniquePtrValueHashSet<SymbolicTiledHloInstruction>
      tiled_hlo_instructions_set;

  // TODO(b/372454662): Once we get rid of the restriction of only one real
  // root, this needs to be adapted.
  auto [root_tiled_hlo, _] = tiled_hlo_instructions_set.Insert(
      std::make_unique<SymbolicTiledHloInstruction>(
          root_indexing.GetRealRoot(), root_indexing.real_root_indexing));

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

      std::unique_ptr<SymbolicTiledHloInstruction> tiled_operand;
      if (operand.opcode() == HloOpcode::kFusion &&
          fusion.ContainsInstruction(&operand.instruction())) {
        // The operand is a nested fusion, analyze it recursively.
        auto nested_fusion_adaptor = HloFusionAdaptor::ForComputation(
            operand.instruction().fused_instructions_computation());

        // Construct a root indexing for the nested fusion by turning the range
        // variables into dimensions.
        llvm::SmallVector<int64_t, 1> range_var_indices(
            operand_indexing_map.GetRangeVarsCount());
        absl::c_iota(range_var_indices, 0);
        auto nested_root_map = ConvertRangeVariablesToDimensions(
            operand_indexing_map, range_var_indices);
        auto nested_roots = ToInstructions(nested_fusion_adaptor->GetRoots());
        // Nested fusions can be empty. Walk up to the parent parameter. This
        // avoids touching the delicate HloFusionAdaptor logic.
        for (auto& root : nested_roots) {
          if (root->opcode() == HloOpcode::kParameter) {
            root = root->parent()->FusionInstruction()->operand(
                root->parameter_number());
          }
        }
        RootIndexing nested_root_indexing{
            /*real_root_index=*/0,
            /*roots=*/nested_roots,
            /*real_root_indexing=*/nested_root_map};

        auto analysis_or = SymbolicTileAnalysis::AnalyzeFusionImpl(
            *nested_fusion_adaptor, ctx, nested_root_indexing,
            emitter_specific_constraints_builder);
        if (std::holds_alternative<FusionDecision>(analysis_or)) {
          return analysis_or;
        }
        tiled_operand = std::make_unique<SymbolicTiledHloFusionInstruction>(
            &operand.instruction(), std::move(operand_indexing_map),
            std::get<SymbolicTileAnalysis>(std::move(analysis_or)));

      } else {
        tiled_operand = std::make_unique<SymbolicTiledHloInstruction>(
            &operand.instruction(), std::move(operand_indexing_map));
      }
      auto [operand_tiled_hlo, inserted] =
          tiled_hlo_instructions_set.Insert(std::move(tiled_operand));
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
      std::move(tiled_hlo_instructions), root_indexing,
      std::get<ConstraintExpression>(std::move(constraints_or)),
      std::move(emitter_specific_constraints), ctx);
}

/*static*/ SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeFusion(
    const HloFusionAdaptor& fusion, MLIRContext* ctx,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder) {
  auto root_indexing_or = GetRootIndexing(fusion, ctx);
  if (!root_indexing_or.ok()) {
    return FusionDecision::Forbid(root_indexing_or.status().message());
  }
  return AnalyzeFusionImpl(fusion, ctx, *root_indexing_or,
                           emitter_specific_constraints_builder);
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

namespace {
void CollectUsedDimIds(AffineExpr expr, std::vector<bool>& dim_var_used) {
  switch (expr.getKind()) {
    case AffineExprKind::DimId: {
      const uint32_t dim_id =
          mlir::cast<mlir::AffineDimExpr>(expr).getPosition();
      dim_var_used[dim_id] = true;
      break;
    }
    case AffineExprKind::Add:
    case AffineExprKind::Mul:
    case AffineExprKind::FloorDiv:
    case AffineExprKind::Mod: {
      auto bin_op = mlir::cast<mlir::AffineBinaryOpExpr>(expr);
      CollectUsedDimIds(bin_op.getLHS(), dim_var_used);
      CollectUsedDimIds(bin_op.getRHS(), dim_var_used);
      break;
    }
    case AffineExprKind::CeilDiv: {
      std::string expr_str;
      llvm::raw_string_ostream string_stream(expr_str);
      expr.print(string_stream);
      CHECK(false)
          << "We do not expect CeilDiv in our indexing expressions, got "
          << expr_str;
      break;
    }
    case AffineExprKind::Constant:
    case AffineExprKind::SymbolId:
      break;
  }
}

bool AllDimIdsAreUsedOrHaveDomainSize1(const IndexingMap& tile_offsets) {
  std::vector<bool> dim_var_used(tile_offsets.GetDimVarsCount(), false);
  for (int64_t i = 0; i < dim_var_used.size(); ++i) {
    if (tile_offsets.GetDimensionBound(i).IsPoint()) {
      dim_var_used[i] = true;
    }
  }
  auto offset_expressions = tile_offsets.GetAffineMap().getResults();
  for (auto offset_expr : offset_expressions) {
    CollectUsedDimIds(offset_expr, dim_var_used);
  }
  return absl::c_all_of(dim_var_used, [](bool value) { return value; });
}

// Returns true when we can determine that the tiling attached to
// `tiled_hlo_instr` covers the whole shape of the corresponding hlo instruction
// uniquely. For a tiling to cover a whole shape uniquely, we need to prove that
// every output index is covered by the tiling function (*surjectivity*), and
// that tiles do not overlap (*injectivity*).
bool TilingCoversWholeShapeUniquely(TiledHloInstruction* tiled_hlo_instr) {
  // Check whether we can use `tiled_hlo_instr`.
  Shape output_shape = tiled_hlo_instr->hlo()->shape();
  auto maybe_tile_offset_indexing = tiled_hlo_instr->tile_offsets_indexing();
  if (!maybe_tile_offset_indexing.ok()) {
    return false;
  }
  auto tile_offset_indexing = maybe_tile_offset_indexing.value();
  // We first check *injectivity*. `tile_offsets_indexing` is derived from a map
  // from the "real root"'s output to the output of the given instruction, and
  // gives us the tile offsets of each tile. By construction, we know that the
  // mapping can be decomposed into offset + stride * index, so it is a linear
  // expression. Therefore it should hold that also the composed indexing map
  // `tile_offset_indexing` (where we have inserted 0, tile_size, tile_size * 2,
  // ... instead of 0, 1, 2, ...) can be decomposed to an expression
  // offset + stride * tile_size * index. This implies that for each dimension
  // `tile_offset_indexing` has an expression for the tile offsets that ensures
  // there is a gap of tile_size between different tile offsets. Therefore
  // injectivity can be checked by verifying that for each combination of input
  // variables, a distinct result is produced. We can check this by ensuring
  // that each input variable is used at least once in the result expression of
  // `tile_offsets_indexing`.

  // This is a slightly handwavy claim, but holds because expressions of the
  // form `d0 floordiv c` never initially appear in a symbolic tile without an
  // associated `d0 mod c`, and operations of the form `d0 * d1` never appear.
  // This should leave us with a guarantee that any combination involving
  // several independent variables won't ever produce duplicate indices. (And
  // trivially, if a parameter `d0` doesn't appear in the output expressions, it
  // means that there are at least `range(d0)` identical outputs for the
  // function. In that case, injectivity only holds if `range(d0) = 1`.
  // TODO(b/390559452): This logic may not work out for ops like ReduceWindow or
  // Convolutions, where we might have overlapping tiles "by design"
  // (recognizable with symbol `s_i` with `range(s_i) > 1`). For now, just
  // disallow any symbols.
  if (tile_offset_indexing.GetSymbolCount() > 0) {
    return false;
  }
  if (!AllDimIdsAreUsedOrHaveDomainSize1(tile_offset_indexing)) {
    return false;
  }
  auto range_evaluator = tile_offset_indexing.GetRangeEvaluator();
  for (int64_t i = 0; i < output_shape.dimensions_size(); ++i) {
    // For now, all strides need to be 0 or 1. With stride 0, we also need to
    // check whether the tile covers the whole dimension.
    // TODO(b/390559452): If we allow strides with absolute value > 1, we
    // need to make sure that the tile offset expression has an additive
    // component with domain [0, stride - 1]. Also we don't handle negative
    // strides yet.
    if (tiled_hlo_instr->tile_stride(i) < 0 ||
        tiled_hlo_instr->tile_stride(i) > 1) {
      return false;
    }
    // Below we check *surjectivity*, which amounts to checking that
    // `tile_offsets_indexing` yields contiguous tiles of contiguous
    // elements (`stride = 1`), and that the first tile and last tile
    // respectively map to the start and end of the array.
    // Given that we restrict the stride to 0 or 1, it is enough to check the
    // range of the tile offsets whether the largest tile offset plus the tile
    // cover the dimension size, and the smallest tile offset is 0.
    auto interval = range_evaluator.ComputeExpressionRange(
        tile_offset_indexing.GetAffineMap().getResult(i));
    if (interval.lower != 0) {
      return false;
    }
    // We can allow that the last tile extends into out of bounds, we add
    // proper masking during codegen to make sure that we don't
    // read/write out of bounds.
    if ((interval.upper + tiled_hlo_instr->tile_size(i) <
         output_shape.dimensions(i))) {
      return false;
    }
  }
  return true;
}

absl::StatusOr<std::vector<const TiledHloInstruction*>> InitializeTiledRoots(
    absl::Span<const HloInstruction* const> roots,
    const std::vector<std::unique_ptr<TiledHloInstruction>>&
        tiled_hlo_instructions) {
  absl::flat_hash_map<const HloInstruction*, int64_t> roots_to_output_index;
  roots_to_output_index.reserve(roots.size());
  int64_t output_index = 0;
  for (auto* root : roots) {
    roots_to_output_index[root] = output_index;
    ++output_index;
  }

  // Collect a tiled hlo instruction for each root. The roots which are extra
  // outputs can reference "internal" tiled hlo instructions and may appear
  // multiple times in `instructions_`.
  std::vector<const TiledHloInstruction*> tiled_roots(roots.size(), nullptr);
  // Handle the real root as special case.
  tiled_roots[roots_to_output_index[tiled_hlo_instructions.back()->hlo()]] =
      tiled_hlo_instructions.back().get();
  for (const auto& tiled_hlo_instr : llvm::drop_end(tiled_hlo_instructions)) {
    auto it = roots_to_output_index.find(tiled_hlo_instr->hlo());
    if (it != roots_to_output_index.end() &&
        TilingCoversWholeShapeUniquely(tiled_hlo_instr.get())) {
      // We may overwrite a previous value, but in case there are multiple
      // tiled hlo instructions for the root, we arbitrarily prefer the last one
      // in def-before-use order.
      tiled_roots[it->second] = tiled_hlo_instr.get();
    }
  }
  // We expect that we found at least one tiled hlo instruction for each root.
  // If not, return an error.
  for (auto [tiled_root, root] : llvm::zip(tiled_roots, roots)) {
    if (tiled_root == nullptr) {
      return absl::UnimplementedError(
          absl::StrCat("Unsupported case of multi-output fusion, we found no "
                       "tiling to reuse for ",
                       root->ToString()));
    }
  }
  return tiled_roots;
}

// Returns the reduction tile size of the given HLO. At the moment, we
// only support fusions with a single reduction dimension. This restriction can
// be lifted in the future.
absl::StatusOr<int64_t> GetReductionTileSize(
    const SymbolicTiledHloFusionInstruction& symbolic_fusion_tiling) {
  const HloInstruction* hlo = symbolic_fusion_tiling.hlo();
  auto backend_config = hlo->backend_config<GpuBackendConfig>();
  if (!backend_config.ok()) {
    return absl::FailedPreconditionError(
        absl::StrCat("No gpu_backend_config in ", hlo->ToString()));
  }
  auto output_tile_sizes =
      BlockLevelParameters::FromBlockLevelFusionConfig(
          backend_config->fusion_backend_config().block_level_fusion_config())
          .output_tile_sizes;
  if (output_tile_sizes.size() != 1) {
    return absl::FailedPreconditionError(
        "Nested fusions should only have one root.");
  }
  const auto& indexing_map = symbolic_fusion_tiling.indexing_map();
  auto symbol_expr =
      mlir::getAffineSymbolExpr(0, indexing_map.GetMLIRContext());
  const auto& results = indexing_map.GetAffineMap().getResults();
  auto it = absl::c_find(results, symbol_expr);
  if (it == results.end()) {
    return absl::FailedPreconditionError("No symbol in indexing map results.");
  }
  return output_tile_sizes.front()[it - results.begin()];
}

}  // namespace

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

  // Check that all strides are >= 0. Our codegen doesn't support negative
  // strides at the moment if padding is required. Also, for the Reverse op it
  // might make sense to emit code for it, and normalizing strides to >= 0.
  for (const std::unique_ptr<SymbolicTiledHloInstruction>& symbolic_tiled_hlo :
       symbolic_tiled_hlo_instructions_) {
    llvm::SmallVector<int64_t> tile_strides = EvaluateTileStrides(
        symbolic_tiled_hlo->symbolic_tile(), tile_parameters);
    if (absl::c_any_of(tile_strides,
                       [](int64_t stride) { return stride < 0; })) {
      return absl::UnimplementedError(
          absl::StrCat("Full support for negative strides is not implemented ",
                       symbolic_tiled_hlo->ToString()));
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
    for (const std::unique_ptr<SymbolicTiledHloInstruction>& symbolic_tiling :
         symbolic_tiled_hlo_instructions_) {
      if (!symbolic_tiling->operands().empty()) {
        continue;
      }

      llvm::SmallVector<int64_t> tile_sizes =
          EvaluateTileSizes(symbolic_tiling->symbolic_tile(), tile_parameters);
      size_t hash_value = absl::HashOf(symbolic_tiling->hlo(),
                                       absl::Span<const int64_t>(tile_sizes));
      tile_sizes_map.emplace(symbolic_tiling.get(), std::move(tile_sizes));

      auto [it, inserted] = hashes.insert(hash_value);
      // Two SymbolicTiledHloInstructions have identical hash when looking only
      // at HLO instruction pointer and tile sizes. We need to compute tile
      // offset indexing maps for all tiles of this HLO instruction.
      if (!inserted) {
        parameters_with_offset_indexing.insert(symbolic_tiling->hlo());
      }
    }
    if (root_indexing_.roots.size() > 1) {
      // We need tile_offset_indexing to check whether we can reuse a tile for
      // another root.
      parameters_with_offset_indexing.insert(root_indexing_.roots.begin(),
                                             root_indexing_.roots.end());
    }
  }

  // TODO(b/390569102): This assumes that there is only one root that matters
  // for computing the tiling, and that it is the last symbolic tiled hlo
  // instruction in the list.
  OutputTilingInfo output_tiling_info = ComputeOutputTilingInfo(
      root_indexing_.real_root_indexing, tile_parameters, context_);

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
      tile_sizes = EvaluateTileSizes(symbolic_tiled_hlo->symbolic_tile(),
                                     tile_parameters);
    }

    llvm::SmallVector<int64_t> tile_strides = EvaluateTileStrides(
        symbolic_tiled_hlo->symbolic_tile(), tile_parameters);

    std::optional<IndexingMap> tile_offset_indexing;
    const HloInstruction* hlo = symbolic_tiled_hlo->hlo();
    if (compute_all_tile_offset_indexing_maps ||
        parameters_with_offset_indexing.contains(hlo) ||
        hlo->opcode() == HloOpcode::kIota) {
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

    std::unique_ptr<TiledHloInstruction> tiled_instruction;
    if (const auto* symbolic_fusion_tiling =
            dynamic_cast<const SymbolicTiledHloFusionInstruction*>(
                symbolic_tiled_hlo.get())) {
      // Instruction is a nested fusion, compute tiled instructions recursively.
      std::vector<int64_t> nested_tiling_parameters(tile_parameters.begin(),
                                                    tile_parameters.end());
      TF_ASSIGN_OR_RETURN(int64_t reduction_tile_size,
                          GetReductionTileSize(*symbolic_fusion_tiling));
      nested_tiling_parameters.push_back(reduction_tile_size);

      TF_ASSIGN_OR_RETURN(
          auto tiled_hlo_computation,
          symbolic_fusion_tiling->analysis_.ComputeTiledHloInstructions(
              nested_tiling_parameters, constraints_are_known_satisfied,
              compute_all_tile_offset_indexing_maps));

      TF_ASSIGN_OR_RETURN(tiled_instruction,
                          TiledHloFusionInstruction::Create(
                              hlo, std::move(operands),
                              std::make_unique<TiledHloComputation>(
                                  std::move(tiled_hlo_computation)),
                              std::move(tile_sizes), std::move(tile_strides),
                              std::move(tile_offset_indexing)));
    } else {
      TF_ASSIGN_OR_RETURN(
          tiled_instruction,
          TiledHloInstruction::Create(
              hlo, std::move(operands), std::move(tile_sizes),
              std::move(tile_strides), std::move(tile_offset_indexing)));
    }

    auto [tiled_hlo, inserted] =
        tiled_hlo_instructions_set.Insert(std::move(tiled_instruction));

    symbolic_to_tiled_hlo_map[symbolic_tiled_hlo.get()] = tiled_hlo;
  }
  auto tiled_hlo_instructions = tiled_hlo_instructions_set.ExtractData();
  TF_ASSIGN_OR_RETURN(
      auto tiled_roots,
      InitializeTiledRoots(root_indexing_.roots, tiled_hlo_instructions));
  return TiledHloComputation::FromSortedTiledHloInstructions(
      std::move(tiled_hlo_instructions), tiled_roots,
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
