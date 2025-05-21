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
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Support/LLVM.h"
#include "xla/hlo/analysis/indexing_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/analysis/indexing_map_serialization.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
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
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

namespace {

using ::mlir::AffineExpr;
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

  // Same as above, but linearized in row major order, so for the example above
  // it would look like:
  //   `(pid) -> (<tile 0 offset>, <tile 1 offset>)`.
  // with tile offset expressions where pid_0 is replaced by (pid floordiv 2),
  // and pid_1 is replaced by (pid mod 2).
  IndexingMap linear_output_tile_offset_indexing;
};

llvm::SmallVector<int64_t> GetNumberOfTilesPerDimension(
    const TiledHloInstruction& tiled_hlo_instr) {
  llvm::SmallVector<int64_t> result;
  absl::Span<const int64_t> dimensions =
      tiled_hlo_instr.hlo()->shape().dimensions();
  result.reserve(dimensions.size());
  for (auto [dim_size, tile_size] :
       llvm::zip(dimensions, tiled_hlo_instr.tile_sizes())) {
    result.push_back(CeilOfRatio(dim_size, tile_size));
  }
  return result;
}

IndexingMap LinearizeTileOffsets(
    const IndexingMap& tile_offsets_indexing,
    absl::Span<const int64_t> num_output_tiles_per_dim,
    mlir::MLIRContext* mlir_context) {
  int64_t num_tiles = Product(num_output_tiles_per_dim);
  mlir::AffineExpr program_id = mlir::getAffineDimExpr(0, mlir_context);
  auto tile_exprs =
      DelinearizeIndex(num_output_tiles_per_dim, program_id, mlir_context);
  std::vector<IndexingMap::Variable> dim_vars{{0, num_tiles - 1, "pid_0"}};
  IndexingMap program_id_to_output_dims{
      mlir::AffineMap::get(
          /*dimCount=*/1, /*symbolCount=*/0, tile_exprs, mlir_context),
      dim_vars, /*range_vars=*/{}, /*rt_vars=*/{}};
  auto linearized_tile_offsets_indexing =
      ComposeIndexingMaps(program_id_to_output_dims, tile_offsets_indexing);
  linearized_tile_offsets_indexing.Simplify();
  linearized_tile_offsets_indexing.RescaleSymbols();
  linearized_tile_offsets_indexing.RemoveUnusedSymbols();
  return linearized_tile_offsets_indexing;
}

absl::StatusOr<OutputTilingInfo> ComputeOutputTilingInfo(
    const IndexingMap& root_indexing, absl::Span<const int64_t> tile_sizes,
    mlir::MLIRContext* mlir_context,
    const std::optional<absl::Span<const Interval>>&
        parent_output_tile_dim_bounds = std::nullopt) {
  int64_t rank = root_indexing.GetDimVarsCount();
  CHECK_EQ(rank, tile_sizes.size());  // Crash OK

  llvm::SmallVector<int64_t> outer_loop_bounds;
  std::vector<IndexingMap::Variable> dim_vars;
  outer_loop_bounds.reserve(rank);
  dim_vars.reserve(rank);
  llvm::SmallVector<AffineExpr> tiled_dims;
  tiled_dims.reserve(rank);
  llvm::SmallVector<std::pair<mlir::AffineExpr, Interval>> constraints;
  for (auto [dim_bounds, tile_size] :
       llvm::zip(root_indexing.GetDimensionBounds(), tile_sizes)) {
    // Start out by making the assumption that we only get tiles with offsets
    // divisible by the tile size. This is true for our initial support of
    // concatenates, but is not a given in the future.
    if (dim_bounds.lower % tile_size != 0) {
      return absl::UnimplementedError(
          absl::StrCat("Dimension bounds are not divisible by tile size: ",
                       ToString(root_indexing)));
    }

    int64_t dim_id = dim_vars.size();
    int64_t upper_bound = CeilOfRatio(dim_bounds.upper + 1, tile_size);
    if (parent_output_tile_dim_bounds) {
      // This is used to handle cases where the iteration space of a nested
      // fusion is smaller than the iteration space of the parent fusion
      // instruction. When we want to linearize that, we would not model the
      // "gap" in the linear iteration space correctly, as that can only be
      // modelled with a constraint instead of a dimension bound. Therefore,
      // we replace the dimension bound with the full range (taken from the
      // parent) and add a dimension constraint to describe it equivalently in a
      // way where linearization will work correctly.
      CHECK_EQ(parent_output_tile_dim_bounds->size(),
               rank);  // Crash OK
      constraints.push_back(
          {mlir::getAffineDimExpr(dim_id, mlir_context),
           Interval{dim_bounds.lower / tile_size, upper_bound - 1}});
      upper_bound = parent_output_tile_dim_bounds.value()[dim_id].upper + 1;
    } else if (dim_bounds.lower != 0) {
      // Dimension lower bounds != 0 currently only happen for Concatenate ops,
      // and for those we should have passed `parent_output_tile_dim_bounds`.
      return absl::UnimplementedError(
          absl::StrCat("Dimension lower bound is not equal to 0: ",
                       ToString(root_indexing)));
    }
    outer_loop_bounds.push_back(upper_bound);

    dim_vars.push_back({0, outer_loop_bounds.back() - 1});
    dim_vars.back().name = absl::StrCat("pid_", dim_id);
    tiled_dims.push_back(tile_size *
                         mlir::getAffineDimExpr(dim_id, mlir_context));
  }

  IndexingMap output_tile_offset_indexing{
      mlir::AffineMap::get(
          /*dimCount=*/rank, /*symbolCount=*/0, tiled_dims, mlir_context),
      dim_vars, /*range_vars=*/{}, /*rt_vars=*/{}, constraints};
  IndexingMap linear_output_tile_offset_indexing = LinearizeTileOffsets(
      output_tile_offset_indexing, outer_loop_bounds, mlir_context);
  return OutputTilingInfo{outer_loop_bounds, output_tile_offset_indexing,
                          linear_output_tile_offset_indexing};
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
                  /*range_vars=*/{}, tile_offset_indexing.GetRTVars(),
                  tile_offset_indexing.GetConstraints()};

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

// Returns `true` if `SymbolicTileAnalysis` should simplify point dimensions
// away when deriving indexing maps.
//
// Simplifying point dimensions away is helpful as it allows symbolic tile
// derivation to succeed in more cases. However, it can lead to generating
// ill-typed programs when we need to propagate a larger (padded) tile through
// the program. In that case, simplifying the point dimension away prevents
// propagation, and leads to the downstream generation of an incorrect program.
//
// This is typically the case when trying to feed a vector-matrix or
// matrix-vector dot product into NVIDIA GPU tensor cores---which expect their
// inputs to have specific dimensions. In that case, we usually want to pretend
// to tile the vector with a tile size appropriate for the tensor core, even
// though one of its dimensions is 1.
//
// Adding this here is a slight abstraction leak, since it slightly specializes
// symbolic tile analysis to NVIDIA GPUs. This is not totally unreasonable
// though: given sufficient analytical capabilities for symbolic tile
// derivation, preventing the simplification of point dimensions should not
// cause us to fail to tile more programs, and would better track the
// propagation of tiles throughout the program. As a result, a mode that does
// not perform this simplification is actually "more correct"---but currently
// leads to more fusions being untileable.
bool ShouldDerivationSimplifyPointDimensions(const HloFusionAdaptor& fusion) {
  for (const HloInstructionAdaptor& instruction_adaptor :
       fusion.MakeInstructionPostOrder()) {
    if (!fusion.ContainsInstruction(&instruction_adaptor.instruction())) {
      continue;
    }

    if (instruction_adaptor.opcode() == HloOpcode::kDot) {
      return false;
    }

    if (instruction_adaptor.opcode() == HloOpcode::kFusion) {
      auto nested_fusion_adaptor = HloFusionAdaptor::ForComputation(
          instruction_adaptor.instruction().fused_instructions_computation());
      if (!ShouldDerivationSimplifyPointDimensions(*nested_fusion_adaptor)) {
        return false;
      }
    }
  }
  return true;
}

// Helper to handle nested parameters for `TilingSpecification::FromFusion`.
// It is assumed that `num_tile_sizes_by_instruction` does not contain any
// information regarding the nested tiling parameters of the fusion.
//
// `num_tile_sizes_by_instruction` is however allowed to contain information
// regarding the tiling parameters of the fusion that are visible at the output.
absl::Status PopulateNestedParameters(
    const HloFusionAdaptor& fusion,
    absl::flat_hash_map<const HloInstruction*, int64_t>&
        num_tile_sizes_by_instruction) {
  auto set_num_tile_sizes_for_instruction =
      [&](const HloInstruction& instruction, int64_t num_parameters) {
        // This should never happen if our outer logic is correct, but we check
        // it just in case.
        if (!instruction.shape().IsArray()) {
          return absl::FailedPreconditionError(absl::StrCat(
              "Instruction ", instruction.ToString(),
              " has non-array shape: ", instruction.shape().ToString()));
        }
        // If the instruction is already in the specification, update it. This
        // should in principle only occur if the instruction defines both tiling
        // parameters visible at its output as well as hidden tiling parameters.
        // A `dot` that is the root of a fusion will model this case, for
        // example.
        num_tile_sizes_by_instruction[&instruction] += num_parameters;
        return absl::OkStatus();
      };

  for (auto& instruction_adaptor : fusion.MakeInstructionPostOrder()) {
    if (!fusion.ContainsInstruction(instruction_adaptor)) {
      continue;
    }

    if (instruction_adaptor.opcode() == HloOpcode::kFusion) {
      std::unique_ptr<HloFusionAdaptor> nested_fusion_adaptor =
          HloFusionAdaptor::ForComputation(
              instruction_adaptor.instruction()
                  .fused_instructions_computation());
      TF_RETURN_IF_ERROR(PopulateNestedParameters(
          *nested_fusion_adaptor, num_tile_sizes_by_instruction));
      continue;
    }

    if (instruction_adaptor.opcode() == HloOpcode::kDot) {
      int64_t num_parameters = instruction_adaptor.instruction()
                                   .dot_dimension_numbers()
                                   .lhs_contracting_dimensions()
                                   .size();
      TF_RETURN_IF_ERROR(set_num_tile_sizes_for_instruction(
          instruction_adaptor.instruction(), num_parameters));
    }
  }
  return absl::OkStatus();
}

}  // anonymous namespace

/*static*/ absl::StatusOr<TilingSpecification>
TilingSpecification::FromFusionAdaptor(const HloFusionAdaptor& fusion_adaptor) {
  absl::flat_hash_map<const HloInstruction*, int64_t>
      num_tile_sizes_by_instruction;

  for (const HloInstructionAdaptor& root : fusion_adaptor.GetRoots()) {
    const HloInstruction& instruction = root.instruction();
    if (!instruction.shape().IsArray()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Instruction ", instruction.ToString(),
          " has non-array shape: ", instruction.shape().ToString()));
    }
    num_tile_sizes_by_instruction[&instruction] =
        instruction.shape().dimensions().size();
  }

  TF_RETURN_IF_ERROR(
      PopulateNestedParameters(fusion_adaptor, num_tile_sizes_by_instruction));
  return TilingSpecification(std::move(num_tile_sizes_by_instruction));
}

/*static*/ absl::StatusOr<TilingSpecification> TilingSpecification::FromFusion(
    const HloFusionInstruction& fusion) {
  std::unique_ptr<HloFusionAdaptor> fusion_adaptor =
      HloFusionAdaptor::ForComputation(fusion.fused_instructions_computation());
  return TilingSpecification::FromFusionAdaptor(*fusion_adaptor);
}

absl::StatusOr<absl::Span<const int64_t>> Tiling::TileSizesForInstruction(
    const HloInstruction* hlo) const {
  if (auto it = tile_sizes_.find(hlo); it != tile_sizes_.end()) {
    return it->second;
  }

  return absl::NotFoundError(
      absl::StrCat("No tile sizes found for instruction: ", hlo->ToString()));
}

bool Tiling::ConformsTo(const TilingSpecification& tiling_specification) const {
  const absl::flat_hash_map<const HloInstruction*, int64_t>&
      num_tile_sizes_by_instruction =
          tiling_specification.num_tile_sizes_by_instruction();
  if (tile_sizes_.size() != num_tile_sizes_by_instruction.size()) {
    return false;
  }
  for (const auto& [hlo, num_parameters] : num_tile_sizes_by_instruction) {
    auto it = tile_sizes_.find(hlo);
    if (it == tile_sizes_.end() || it->second.size() != num_parameters) {
      return false;
    }
  }
  return true;
}

// Extracts `HloInstruction`s from a span of `HloInstructionAdaptor`s.
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

  IndexingMap::SimplifyPointDimensions simplification_mode =
      IndexingMap::SimplifyPointDimensions::kPreserve;
  if (ShouldDerivationSimplifyPointDimensions(fusion)) {
    simplification_mode = IndexingMap::SimplifyPointDimensions::kReplace;
  }

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
      operand_indexing_map.Simplify(simplification_mode);
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

      // TODO(b/393299275): propagation to operands is not correct when nesting,
      // because we derive something all the way to the parameters that are
      // outside the fusion. We should not derive anything for those operands.
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

// Returns whether the tiling from `output` can be used by the emitter for
// producing a fusion output without causing issues in case a buffer is shared
// between a fusion operand and a fusion output. Buffer sharing is (as of May
// 2025) allowed if there is a path from the fusion operand to the fusion output
// with only elementwise or bitcast ops. To avoid race conditions where we
// overwrite an input value that is still required to compute another output
// value, we need to make sure that we use an input tile only in the iteration
// in which we overwrite it. Just using the propagated tile offsets of `output`
// does not ensure this, as a tile size may not divide the dimension size evenly
// and padding will be used. We might have different padding for different
// output shapes. For example consider the following triton fusion
// fused_computation {
//   param_0 = f32[36] parameter(0)
//   abs = f32[36] abs(param_0)
//   reshape = f32[3,12] reshape(abs)
//   ROOT tuple = (f32[3,12], f32[36]) tuple(reshape, abs)
// }
// with tiling parameters {1, 16} and {16}, respectively. With the f32[36]
// shape, we would only pad the last tile, while for the shape f32[3,12] we
// would pad all tiles. By ensuring the equality of the propagated tile offsets
// indexing map with a tile offsets indexing map computed directly for this root
// using the propagated tile size parameter, we ensure that there will be no
// difference regarding which of the output tiles is padded. In our example
// above, the propagated tile offsets on the `abs` instruction would be [0, 12,
// 24], while the directly computed tile offsets would be [0, 16, 32].
// The equality of the tile offsets maps implies that we would be able to CSE
// all producer instructions of `output` with the TiledHloInstructions computed
// by using `output` as a tiling root using the propagated tile sizes as tiling
// parameters. As a side effect, this check will also make sure that by using
// the tiling from `output` we will produce the full output.
// `reference_num_output_tiles` is the number of tiles of the root from which
// the tiling was propagated. We need to ensure that the iteration spaces for
// tiles match for all outputs. This is a restriction we may lift later in case
// the buffer sharing logic is adapted.
// This method assumes that `output` has tile_offset_indexing computed, and
// returns a FailedPrecondition error if not.
absl::StatusOr<bool> IsSafeForBufferSharing(const TiledHloInstruction& output,
                                            int64_t reference_num_output_tiles,
                                            mlir::MLIRContext* mlir_context) {
  // For expanding reshapes, we can have the case that the number of
  // blocks are different. This is not supported by the triton emitter.
  llvm::SmallVector<int64_t> num_tiles_per_dim =
      GetNumberOfTilesPerDimension(output);
  if (Product(num_tiles_per_dim) != reference_num_output_tiles) {
    return false;
  }
  // Compute the tile offset indexing directly for `output`. We use default
  // iteration order and tile stride of 1, which means we can take the identity
  // map.
  auto identity_indexing_map =
      CreateIdentityMap(output.hlo()->shape(), mlir_context);
  TF_ASSIGN_OR_RETURN(auto tiling_info, ComputeOutputTilingInfo(
                                            identity_indexing_map,
                                            output.tile_sizes(), mlir_context));

  // Check whether the tile_offsets_indexing expression is the same as one
  // computed directly for this root.
  absl::StatusOr<IndexingMap> maybe_tile_offset_indexing =
      output.tile_offsets_indexing();
  if (!maybe_tile_offset_indexing.ok()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected ", output.ToString(),
                     " to have a tile_offsets_indexing value"));
  }
  return maybe_tile_offset_indexing.value() ==
         tiling_info.linear_output_tile_offset_indexing;
}

absl::StatusOr<std::vector<const TiledHloInstruction*>> InitializeTiledRoots(
    absl::Span<const HloInstruction* const> roots,
    const std::vector<std::unique_ptr<TiledHloInstruction>>&
        tiled_hlo_instructions,
    absl::Span<const int64_t> num_output_tiles_per_dim,
    mlir::MLIRContext* mlir_context) {
  // TODO(b/390559452): Investigate whether it is faster to use linear lookup.
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
  // Handle the real root as special case. Then we don't need to do any extra
  // work in case we are not dealing with a multi-output fusion.
  auto real_root = tiled_hlo_instructions.back().get();
  tiled_roots[roots_to_output_index[real_root->hlo()]] = real_root;

  for (const auto& tiled_hlo_instr : llvm::drop_end(tiled_hlo_instructions)) {
    auto it = roots_to_output_index.find(tiled_hlo_instr->hlo());
    if (it == roots_to_output_index.end()) {
      continue;
    }
    // We potentially allow sharing an input buffer with an output buffer.
    // Therefore we need to make sure that we use an input tile only in the
    // iteration in which we overwrite it.
    TF_ASSIGN_OR_RETURN(
        bool valid, IsSafeForBufferSharing(*tiled_hlo_instr,
                                           /*reference_num_output_tiles=*/
                                           Product(num_output_tiles_per_dim),
                                           mlir_context));
    if (!valid) {
      continue;
    }
    // We may overwrite a previous value, but in case there are multiple
    // tiled hlo instructions for the root, we arbitrarily prefer the last one
    // in def-before-use order.
    tiled_roots[it->second] = tiled_hlo_instr.get();
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
  // TODO(b/393299275): this is hacky, and will fail if the fusion involves e.g.
  // another reduction. This'll need fixing before this can be generalized to
  // arbitrary fusions beyond the dot emitter.
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

// TODO(b/406244630): this function is too long. We should chunk it up.
absl::StatusOr<TiledHloComputation> ComputeTiledHloInstructionsImpl(
    const SymbolicTileAnalysis& analysis,
    absl::Span<const int64_t> tile_parameters,
    bool constraints_are_known_satisfied,
    bool compute_all_tile_offset_indexing_maps,
    const std::optional<absl::Span<const Interval>>&
        parent_output_tile_dim_bounds,
    MLIRContext* context) {
  if (!constraints_are_known_satisfied) {
    TF_ASSIGN_OR_RETURN(bool constraints_are_satisfied,
                        analysis.ParametersSatisfyConstraints(tile_parameters));
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
       analysis.GetSymbolicTiledHloComputation()) {
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
         analysis.GetSymbolicTiledHloComputation()) {
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
    if (analysis.GetRoots().size() > 1) {
      // We need tile_offset_indexing to check whether we can reuse a tile for
      // another root.
      parameters_with_offset_indexing.insert(analysis.GetRoots().begin(),
                                             analysis.GetRoots().end());
    }
  }

  // TODO(b/390569102): This assumes that there is only one root that matters
  // for computing the tiling, and that it is the last symbolic tiled hlo
  // instruction in the list.
  TF_ASSIGN_OR_RETURN(
      OutputTilingInfo output_tiling_info,
      ComputeOutputTilingInfo(analysis.GetRealRootIndexing(), tile_parameters,
                              context, parent_output_tile_dim_bounds));

  OrderedUniquePtrValueHashSet<TiledHloInstruction> tiled_hlo_instructions_set;
  absl::flat_hash_map<const SymbolicTiledHloInstruction*, TiledHloInstruction*>
      symbolic_to_tiled_hlo_map;
  // The actual number of `TiledHloInstruction`s can be smaller than the number
  // of `SymbolicTiledHloInstruction`s, because some instruction will be
  // deduplicated, but we reserve to the upper bound to avoid reallocations and
  // additional hash calculations.
  tiled_hlo_instructions_set.Reserve(
      analysis.GetSymbolicTiledHloComputation().size());

  for (const std::unique_ptr<SymbolicTiledHloInstruction>& symbolic_tiled_hlo :
       analysis.GetSymbolicTiledHloComputation()) {
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
              output_tiling_info.linear_output_tile_offset_indexing, context));
    }
    std::optional<std::vector<Interval>> fusion_tile_dim_bounds;
    if (hlo->opcode() == HloOpcode::kFusion && !hlo->users().empty() &&
        hlo->users().front()->opcode() == HloOpcode::kConcatenate) {
      fusion_tile_dim_bounds =
          output_tiling_info.output_tile_offset_indexing.GetDimensionBounds();
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
      std::vector<int64_t> nested_tiling_parameters(tile_parameters.begin(),
                                                    tile_parameters.end());
      if (hlo->users().empty()) {
        return absl::FailedPreconditionError(
            absl::StrCat("Expected the nested fusion instruction ",
                         hlo->ToString(), " to have a user"));
      }
      const HloInstruction* user = hlo->users().front();
      // Nested fusions materialize regions of control flow delineated rooted
      // in their user. If the user has a contracting dimension, we need to
      // derive a tile size along the reduction dimension as well, and therefore
      // add a dimension to the tile parameters.
      if (user->opcode() == HloOpcode::kDot) {
        // TODO(b/393299275): reductions will also fall through this branch,
        // so the check will have to be extended.
        TF_ASSIGN_OR_RETURN(int64_t reduction_tile_size,
                            GetReductionTileSize(*symbolic_fusion_tiling));
        nested_tiling_parameters.push_back(reduction_tile_size);
      } else if (user->opcode() != HloOpcode::kConcatenate) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Expected the user of a nested fusion to be a dot or concatenate, "
            "but got ",
            user->ToString()));
      }

      // Compute tiled instructions recursively.
      TF_ASSIGN_OR_RETURN(
          auto tiled_hlo_computation,
          ComputeTiledHloInstructionsImpl(symbolic_fusion_tiling->analysis_,
                                          nested_tiling_parameters,
                                          constraints_are_known_satisfied,
                                          compute_all_tile_offset_indexing_maps,
                                          fusion_tile_dim_bounds, context));

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
      InitializeTiledRoots(analysis.GetRoots(), tiled_hlo_instructions,
                           output_tiling_info.num_output_tiles_per_dim,
                           context));
  return TiledHloComputation::FromSortedTiledHloInstructions(
      std::move(tiled_hlo_instructions), tiled_roots,
      output_tiling_info.num_output_tiles_per_dim);
}

absl::StatusOr<TiledHloComputation>
SymbolicTileAnalysis::ComputeTiledHloInstructions(
    absl::Span<const int64_t> tile_parameters,
    bool constraints_are_known_satisfied,
    bool compute_all_tile_offset_indexing_maps) const {
  return ComputeTiledHloInstructionsImpl(
      *this, tile_parameters, constraints_are_known_satisfied,
      compute_all_tile_offset_indexing_maps,
      /*parent_output_tile_dim_bounds=*/std::nullopt, context_);
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
