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
#include <tuple>
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
#include "llvm/ADT/Sequence.h"
#include "llvm/ADT/SmallBitVector.h"
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

// Tiling of the output of a fusion (computation).
// It is a mapping from the tile multi index to the index of the output tensor
// (i.e. output of the the top level fusion, not the nested one). In case of
// multi-output fusions that will be the tiling of the "real root" (see
// symbolic_tile_analysis.h).
//
// Tiling might also include runtime variables.
//
// `ComputeOutputTilingInfo` creates a new instance of this struct.
struct OutputTilingInfo {
  // The number of output tiles for each dimension of the root indexing.
  // For example, ,if dimensions are [29, 16] and tile size is [4, 8] then
  // `num_output_tiles_per_dim` will be [8, 2] = [29 ceildiv 4, 16 ceildiv 8].
  llvm::SmallVector<int64_t> num_output_tiles_per_dim;

  // An indexing map to compute tile offsets from the root index and runtime
  // variables.
  //
  // The dimensions of the indexing map correspond to the dimensions passed
  // to `ComputeOutputTilingInfo` and the number of dimensions is equal to the
  // size of `num_output_tiles_per_dim`. For example above it would look like:
  //   `(pid_0, pid_1){rt0, rt1, ..} -> (<tile 0 offset>, <tile 1 offset>)`.
  IndexingMap output_tile_offset_indexing;

  // The subset of tiling parameters that are active for this tiling, in
  // major-to-minor order.
  llvm::SmallVector<int64_t> active_tiling_parameters;

  // Same as `output_tile_offset_indexing`, but linearized according to the
  // order specified in `active_tiling_parameters`. For the example in
  // `output_tile_offset_indexing`, and with `active_tiling_parameters` set to
  // {0, 1} (row-major order), it would look like:
  //   `(d0){rt0, rt1, ..} -> (<tile 0 offset>, <tile 1 offset>)`.
  // where pid_0 is replaced by (d0 floordiv 2) and pid_1 is replaced by
  // (d0 mod 2) in tile offset expressions.
  IndexingMap linear_output_tile_offset_indexing;

  std::string ToString(const absl::string_view field_separator = "\n") {
    return absl::StrCat(
        "num_output_tiles_per_dim: ", num_output_tiles_per_dim.size(),
        field_separator, absl::StrJoin(num_output_tiles_per_dim, ", "),
        field_separator, "output_tile_offset_indexing: ",
        xla::ToString(output_tile_offset_indexing), field_separator,
        "linear_output_tile_offset_indexing: ",
        xla::ToString(linear_output_tile_offset_indexing));
  }
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

// Helper to produce a map from a program id to tile offsets.
//
// `tile_offsets_indexing` takes in as many parameters as there are tiling
// parameters in the whole fusion, but we don't always want to linearize
// indexing into the whole parameter space.
//
// In order to linearize indices over only a subset of the axes, we provide a
// vector `major_to_minor_active_tiling_parameters` that indicates which
// parameters are "active" (i.e. where the relevant axis should be taken into
// account), and in which order (major-to-minor) the axes should be processed.
IndexingMap LinearizeTileOffsets(
    const IndexingMap& tile_offsets_indexing,
    absl::Span<const int64_t> num_output_tiles_per_dim,
    absl::Span<const int64_t> major_to_minor_active_tiling_parameters,
    mlir::MLIRContext* mlir_context) {
  // Gather the active output tile sizes in major-to-minor order so as to
  // produce the right delinearized index.
  std::vector<int64_t> active_num_output_tiles_per_dim;
  active_num_output_tiles_per_dim.reserve(
      major_to_minor_active_tiling_parameters.size());
  for (int64_t dim_id : major_to_minor_active_tiling_parameters) {
    active_num_output_tiles_per_dim.push_back(num_output_tiles_per_dim[dim_id]);
  }
  int64_t num_tiles = Product(num_output_tiles_per_dim);
  CHECK_EQ(num_tiles, Product(active_num_output_tiles_per_dim));
  mlir::AffineExpr program_id = mlir::getAffineDimExpr(0, mlir_context);
  std::vector<mlir::AffineExpr> tile_exprs(
      num_output_tiles_per_dim.size(),
      mlir::getAffineConstantExpr(0, mlir_context));
  for (auto [dim_id, tile_expr] :
       llvm::zip(major_to_minor_active_tiling_parameters,
                 DelinearizeIndex(active_num_output_tiles_per_dim, program_id,
                                  mlir_context))) {
    tile_exprs[dim_id] = tile_expr;
  }
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

// Creates the concrete tiling of an output of the computation from the
// indexing map of the computation's root and the tile sizes.
absl::StatusOr<OutputTilingInfo> ComputeOutputTilingInfo(
    const IndexingMap& root_indexing, absl::Span<const int64_t> tile_sizes,
    absl::Span<const int64_t> major_to_minor_active_tiling_parameters,
    mlir::MLIRContext* mlir_context,
    const std::optional<absl::Span<const Interval>>&
        parent_output_tile_dim_bounds = std::nullopt) {
  int64_t num_tiling_parameters = root_indexing.GetDimVarsCount();
  CHECK_EQ(num_tiling_parameters, tile_sizes.size());  // Crash OK
  CHECK_EQ(0, root_indexing.GetRangeVarsCount())
      << "Range variables must be converted to dimensions";

  const IndexingMap::Variable ignore_variable{0, 0, "ignore"};
  llvm::SmallVector<int64_t> outer_loop_bounds(num_tiling_parameters, 1);
  std::vector<IndexingMap::Variable> dim_vars(num_tiling_parameters,
                                              ignore_variable);
  llvm::SmallVector<AffineExpr> tiled_dims(
      num_tiling_parameters, mlir::getAffineConstantExpr(0, mlir_context));
  llvm::SmallVector<std::pair<mlir::AffineExpr, Interval>> constraints;

  std::vector<Interval> all_dim_bounds = root_indexing.GetDimensionBounds();
  for (int64_t dim_id : major_to_minor_active_tiling_parameters) {
    const Interval& dim_bounds = all_dim_bounds[dim_id];
    int64_t tile_size = tile_sizes[dim_id];

    // Start out by making the assumption that we only get tiles with offsets
    // divisible by the tile size. This is true for our initial support of
    // concatenates, but is not a given in the future.
    if (dim_bounds.lower % tile_size != 0) {
      return absl::UnimplementedError(
          absl::StrCat("Dimension bounds are not divisible by tile size: ",
                       ToString(root_indexing)));
    }

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
               num_tiling_parameters);  // Crash OK
      constraints.push_back(
          {mlir::getAffineDimExpr(dim_id, mlir_context),
           Interval{dim_bounds.lower / tile_size, upper_bound - 1}});
      upper_bound = parent_output_tile_dim_bounds.value()[dim_id].upper + 1;
    } else if (dim_bounds.lower != 0) {
      // Dimension lower bounds != 0 currently only happen for `concatenate`s.
      // For those, we expect to have passed a `parent_output_tile_dim_bounds`
      // parameter.
      return absl::UnimplementedError(
          absl::StrCat("Dimension lower bound is not equal to 0: ",
                       ToString(root_indexing)));
    }
    outer_loop_bounds[dim_id] = upper_bound;

    // TODO(b/393299275): naming is not correct as that might also be a nested
    // tile parameter.
    dim_vars[dim_id] = {0, upper_bound - 1, absl::StrCat("pid_", dim_id)};
    tiled_dims[dim_id] =
        tile_size * mlir::getAffineDimExpr(dim_id, mlir_context);
  }

  IndexingMap output_tile_offset_indexing{
      mlir::AffineMap::get(
          /*dimCount=*/num_tiling_parameters,
          /*symbolCount=*/root_indexing.GetRTVarsCount(),
          /*results=*/tiled_dims, mlir_context),
      dim_vars, /*range_vars=*/{}, /*rt_vars=*/root_indexing.GetRTVars(),
      constraints};

  // TODO(b/417977182): revisit linearization. This makes it hard to do things
  // like grid tiling, for instance.
  IndexingMap linear_output_tile_offset_indexing = LinearizeTileOffsets(
      output_tile_offset_indexing, outer_loop_bounds,
      major_to_minor_active_tiling_parameters, mlir_context);
  return OutputTilingInfo{outer_loop_bounds,
                          output_tile_offset_indexing,
                          {major_to_minor_active_tiling_parameters.begin(),
                           major_to_minor_active_tiling_parameters.end()},
                          linear_output_tile_offset_indexing};
}

// Extension of SymbolicTiledHloInstruction for fusions that holds the analysis
// of the fusion's computation.
class SymbolicTiledHloFusionInstruction : public SymbolicTiledHloInstruction {
 public:
  SymbolicTiledHloFusionInstruction(
      const HloInstruction* hlo, IndexingMap indexing_map,
      SymbolicTileAnalysis analysis,
      std::vector<SymbolicTiledHloInstruction*> runtime_variables)
      : SymbolicTiledHloInstruction(hlo, std::move(indexing_map),
                                    std::move(runtime_variables)),
        analysis_(std::move(analysis)) {}

  SymbolicTileAnalysis analysis_;
};

// Computes the tile offset indexing map of concrete tiling from a symbolically
// tiled instruction and and an offset indexing map into from the root to its
// output.
absl::StatusOr<IndexingMap> ComputeTileOffsetIndexing(
    const SymbolicTiledHloInstruction& tiled_hlo,
    const IndexingMap& output_tile_offset_indexing,
    mlir::MLIRContext* mlir_context) {
  VLOG(4) << "ComputeTileOffsetIndexing, combining output "
          << ToString(output_tile_offset_indexing) << " with operation "
          << tiled_hlo.ToString();
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
  // Do not remove symbols yet as we need to track runtime variables that were
  // removed.
  return std::move(simplified_indexing_map);
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
  if ((hlo->opcode() == HloOpcode::kConcatenate ||
       hlo->opcode() == HloOpcode::kPad) &&
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
        ComputeOutputToInputIndexing(hlo, /*output_id=*/0, ctx)
            .indexing_maps[0]
            .begin()
            ->map();

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

// Sorts tiled hlo instructions in def-before-use order, starting from
// `root_tiled_hlo`. If instruction is not reachable from the root then it might
// be put in an arbitrary position.
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
    for (const SymbolicTiledHloInstruction* rt_operand :
         instruction->runtime_variables()) {
      visit_instruction(rt_operand);
    }
    topological_order[instruction] = topological_order.size();
  };

  visit_instruction(root_tiled_hlo);

  absl::c_sort(tiled_hlo_instructions,
               [&](const std::unique_ptr<SymbolicTiledHloInstruction>& t1,
                   const std::unique_ptr<SymbolicTiledHloInstruction>& t2) {
                 return topological_order[t1.get()] <
                        topological_order[t2.get()];
               });
  if (VLOG_IS_ON(4)) {
    VLOG(4)
        << "Sorted symbolic tiled HLO instructions in def-before-use order:\n"
        << absl::StrJoin(tiled_hlo_instructions, "\n",
                         [](std::string* out,
                            const std::unique_ptr<SymbolicTiledHloInstruction>&
                                instruction) {
                           absl::StrAppend(out, instruction->ToString("; "));
                         });
  }
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
    TilingSpecification::ParameterMapping& parameter_mapping) {
  auto set_mapping_for_instruction = [&](const HloInstruction& instruction,
                                         int64_t num_parameters) {
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
    for (TilingSpecification::InstructionAndNumTilingParameters& mapping :
         parameter_mapping) {
      if (mapping.instruction == &instruction) {
        mapping.num_tiling_parameters += num_parameters;
        return absl::OkStatus();
      }
    }
    parameter_mapping.push_back({&instruction, num_parameters});
    return absl::OkStatus();
  };

  const auto& instructions = fusion.MakeInstructionPostOrder();
  for (int64_t i = instructions.size() - 1; i >= 0; --i) {
    const HloInstructionAdaptor& instruction_adaptor = instructions[i];
    if (!fusion.ContainsInstruction(instruction_adaptor)) {
      continue;
    }

    if (instruction_adaptor.opcode() == HloOpcode::kFusion) {
      std::unique_ptr<HloFusionAdaptor> nested_fusion_adaptor =
          HloFusionAdaptor::ForComputation(
              instruction_adaptor.instruction()
                  .fused_instructions_computation());
      TF_RETURN_IF_ERROR(
          PopulateNestedParameters(*nested_fusion_adaptor, parameter_mapping));
      continue;
    }

    if (instruction_adaptor.opcode() == HloOpcode::kDot) {
      int64_t num_parameters = instruction_adaptor.instruction()
                                   .dot_dimension_numbers()
                                   .lhs_contracting_dimensions()
                                   .size();
      TF_RETURN_IF_ERROR(set_mapping_for_instruction(
          instruction_adaptor.instruction(), num_parameters));
    }
  }
  return absl::OkStatus();
}

// Derives a `TilingSpecification::ParameterMapping` from a `HloFusionAdaptor`.
//
// The resulting mapping guarantees that:
//  1. instructions are sorted in use-before-def order;
//  2. only instructions introducing new tiling parameters are present---with
//     the exception of 0D output instructions, which will be present but
//     introduce 0 new tiling parameters.
//
// Currently, we require a single real root index in order to allow this to
// work with our restricted support for multi-output fusions.
absl::StatusOr<TilingSpecification::ParameterMapping>
ParameterMappingFromFusionAdaptor(const HloFusionAdaptor& fusion_adaptor,
                                  int64_t real_root_index) {
  TilingSpecification::ParameterMapping parameter_mapping;

  const HloInstruction& real_root =
      fusion_adaptor.GetRoots()[real_root_index].instruction();
  if (!real_root.shape().IsArray()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Instruction ", real_root.ToString(),
                     " has non-array shape: ", real_root.shape().ToString()));
  }
  int64_t num_tile_sizes = real_root.shape().dimensions().size();
  parameter_mapping.push_back({&real_root, num_tile_sizes});

  TF_RETURN_IF_ERROR(
      PopulateNestedParameters(fusion_adaptor, parameter_mapping));

  return parameter_mapping;
}

// Helper to implement `TilingSpecification::DimensionIndexForParameter`.
absl::StatusOr<int64_t> ParameterIndexImpl(
    const TilingSpecification::ParameterMapping& parameter_mapping,
    const HloInstruction* hlo, int64_t index) {
  int64_t offset = 0;
  for (const auto& [instruction, num_parameters] : parameter_mapping) {
    if (instruction == hlo) {
      if (index >= num_parameters) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Index ", index, " is out of bounds for instruction ",
            hlo->ToString(), " with num parameters ", num_parameters));
      }
      return offset + index;
    }

    offset += num_parameters;
  }
  return absl::NotFoundError(
      absl::StrCat("No tile sizes found for instruction: ", hlo->ToString()));
}

// Drop instructions from a vector for which the corresponding bit in `to_drop`
// is set. `to_drop` might be smaller than `elements` in which case the
// remaining elements are assumed to be `false`.
llvm::SmallVector<const TiledHloInstruction*> RemoveInstructionByMask(
    const llvm::SmallVector<const TiledHloInstruction*>& elements,
    const llvm::SmallBitVector& to_drop) {
  llvm::SmallVector<const TiledHloInstruction*> result;
  result.reserve(elements.size() - to_drop.count());
  for (const auto& [i, v] : llvm::enumerate(elements)) {
    if (i < to_drop.size() && to_drop[i]) {
      continue;
    }
    result.push_back(v);
  }
  return result;
}

llvm::SmallVector<const TiledHloInstruction*> MapToTiledInstructions(
    const std::vector<SymbolicTiledHloInstruction*>& symbolic_instructions,
    absl::flat_hash_map<const SymbolicTiledHloInstruction*,
                        TiledHloInstruction*>& symbolic_to_tiled_hlo_map) {
  llvm::SmallVector<const TiledHloInstruction*> result;
  result.reserve(symbolic_instructions.size());
  for (const auto& value : symbolic_instructions) {
    CHECK(symbolic_to_tiled_hlo_map.contains(value)) << value->ToString();
    result.push_back(symbolic_to_tiled_hlo_map.at(value));
  }
  return result;
}

}  // anonymous namespace

absl::StatusOr<absl::Span<const int64_t>> Tiling::TileSizesForInstruction(
    const HloInstruction* hlo) const {
  if (auto it = tile_sizes_.find(hlo); it != tile_sizes_.end()) {
    return it->second;
  }

  return absl::NotFoundError(
      absl::StrCat("No tile sizes found for instruction: ", hlo->ToString()));
}

absl::StatusOr<std::vector<int64_t>> Tiling::Flatten(
    const TilingSpecification& tiling_specification) const {
  std::vector<int64_t> flat_tile_sizes;
  flat_tile_sizes.reserve(tiling_specification.num_parameters());
  for (const auto& mapping : tiling_specification.parameter_mapping()) {
    TF_ASSIGN_OR_RETURN(absl::Span<const int64_t> tile_sizes,
                        TileSizesForInstruction(mapping.instruction));
    if (tile_sizes.size() != mapping.num_tiling_parameters) {
      return absl::FailedPreconditionError(
          absl::StrCat("Instruction ", mapping.instruction->ToString(),
                       " was expected to have ", mapping.num_tiling_parameters,
                       " tile sizes but had ", tile_sizes.size(), "."));
    }
    flat_tile_sizes.insert(flat_tile_sizes.end(), tile_sizes.begin(),
                           tile_sizes.end());
  }

  return flat_tile_sizes;
}

absl::StatusOr<int64_t> TilingSpecification::ParameterIndex(
    const HloInstruction* hlo, int64_t index) const {
  return ParameterIndexImpl(parameter_mapping_, hlo, index);
}

bool Tiling::ConformsTo(const TilingSpecification& tiling_specification) const {
  int64_t num_instructions = tile_sizes_.size();
  int64_t expected_num_instructions =
      tiling_specification.parameter_mapping().size();
  if (num_instructions != expected_num_instructions) {
    VLOG(1) << "Tiling tiles " << num_instructions << " instructions, but "
            << expected_num_instructions
            << " instructions were expected to be "
               "tiled.";
    return false;
  }

  // Linearization takes care of checking that we have the right number of
  // tile sizes specified for each instruction.
  absl::StatusOr<std::vector<int64_t>> flat_tile_sizes_or =
      Flatten(tiling_specification);
  if (!flat_tile_sizes_or.ok()) {
    return false;
  }

  return tiling_specification.constraints().IsSatisfiedBy(*flat_tile_sizes_or);
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

namespace {

// Given a parameter mapping, produces its "input" (or indexing) space, i.e.,
// for each parameter, the length of the dimension it abstracts over.
std::vector<int64_t> InputSpaceForParameterMapping(
    const TilingSpecification::ParameterMapping& parameter_mapping) {
  int64_t num_parameters = absl::c_accumulate(
      parameter_mapping, 0, [](int64_t sum, const auto& mapping) {
        return sum + mapping.num_tiling_parameters;
      });
  std::vector<int64_t> input_space;
  input_space.reserve(num_parameters);

  for (const auto& [hlo, num_parameters] : parameter_mapping) {
    // TODO(b/419026602): handle reductions.
    if (hlo->opcode() == HloOpcode::kDot) {
      auto contracting_dimensions =
          hlo->dot_dimension_numbers().lhs_contracting_dimensions();
      // First, we need to add the contracting dimensions of the `dot`
      // instruction to the input space.
      for (int64_t contracting_dimension : contracting_dimensions) {
        input_space.push_back(
            hlo->operand(0)->shape().dimensions(contracting_dimension));
      }
      int64_t num_contracting_dimensions = contracting_dimensions.size();
      // Optionally, we also add the output dimensions of the `dot` instruction,
      // if they are actual parameters.
      if (num_parameters != num_contracting_dimensions) {
        CHECK_EQ(num_parameters,
                 num_contracting_dimensions + hlo->shape().dimensions().size());
        for (int64_t output_dimension : hlo->shape().dimensions()) {
          input_space.push_back(output_dimension);
        }
      }
      continue;
    }

    CHECK_EQ(hlo->shape().dimensions().size(), num_parameters);
    for (int64_t dimension : hlo->shape().dimensions()) {
      input_space.push_back(dimension);
    }
  }

  return input_space;
}

// Produces an indexing map from the parameter input space of the entire
// fusion to the specified root instruction's output space.
absl::StatusOr<IndexingMap> IndexingMapForRootInstruction(
    const HloInstruction* root,
    const TilingSpecification::ParameterMapping& parameter_mapping,
    MLIRContext* ctx) {
  std::vector<int64_t> input_space =
      InputSpaceForParameterMapping(parameter_mapping);
  int64_t num_output_parameters = root->shape().dimensions().size();

  std::vector<AffineExpr> result_exprs;
  result_exprs.reserve(num_output_parameters);

  int64_t dim_offset = 0;
  for (const auto& [hlo, num_tiling_parameters] : parameter_mapping) {
    if (hlo == root) {
      int64_t num_hidden_parameters =
          num_tiling_parameters - num_output_parameters;
      for (int64_t parameter_index = num_hidden_parameters;
           parameter_index < num_tiling_parameters; ++parameter_index) {
        result_exprs.push_back(
            mlir::getAffineDimExpr(dim_offset + parameter_index, ctx));
      }
      CHECK_EQ(result_exprs.size(), num_output_parameters);

      mlir::AffineMap affine_map = mlir::AffineMap::get(
          input_space.size(), /*symbolCount=*/0, result_exprs, ctx);

      return IndexingMap::FromTensorSizes(affine_map, std::move(input_space),
                                          /*symbol_upper_bounds=*/{});
    }
    dim_offset += num_tiling_parameters;
  }
  return absl::NotFoundError(absl::StrCat(
      "No mapping found for root instruction: ", root->ToString()));
}

}  // namespace

// Computes the indexing information for the roots of the 'fusion'.
/*static*/ absl::StatusOr<RootIndexing> SymbolicTileAnalysis::GetRootIndexing(
    const HloFusionAdaptor& fusion,
    const TilingSpecification::ParameterMapping& parameter_mapping,
    MLIRContext* ctx) {
  auto fusion_adaptor_roots = fusion.GetRoots();

  TF_ASSIGN_OR_RETURN(int64_t real_root_index,
                      GetRealRootIndex(fusion_adaptor_roots));

  // Keep track of the roots separately. If there is just a single root, we
  // don't need that, as it will necessarily appear last in def-before-use
  // order. But with multiple roots, we can have roots that are also ancestors
  // of another root.
  absl::InlinedVector<const HloInstruction*, 2> roots =
      ToInstructions(fusion_adaptor_roots);

  TF_ASSIGN_OR_RETURN(IndexingMap indexing_map,
                      IndexingMapForRootInstruction(roots[real_root_index],
                                                    parameter_mapping, ctx));

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

/*static*/ SymbolicTileAnalysisOrError
SymbolicTileAnalysis::AnalyzeNestedFusion(
    const HloFusionAdaptor& fusion_adaptor,
    const TilingSpecification::ParameterMapping& parameter_mapping,
    MLIRContext* ctx, const IndexingMap& indexing_map,
    IndexingMap::SimplifyPointDimensions simplification_mode,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder,
    std::vector<SymbolicTiledHloInstruction*> root_runtime_variables) {
  auto nested_roots = ToInstructions(fusion_adaptor.GetRoots());
  // Nested fusions can be empty. Walk up to the parent parameter. This
  // avoids touching the delicate HloFusionAdaptor logic.
  for (auto& root : nested_roots) {
    if (root->opcode() == HloOpcode::kParameter) {
      root = root->parent()->FusionInstruction()->operand(
          root->parameter_number());
    }
  }
  RootIndexing nested_root_indexing{/*real_root_index=*/0,
                                    /*roots=*/nested_roots,
                                    /*real_root_indexing=*/indexing_map};

  return SymbolicTileAnalysis::AnalyzeFusionImpl(
      fusion_adaptor, parameter_mapping, ctx, nested_root_indexing,
      simplification_mode, emitter_specific_constraints_builder,
      root_runtime_variables);
}

namespace {

// Given a consumer with contracting dimensions, an operand index, and a map
// going from the outermost fusion root to the `operand_index`-th operand of the
// consumer, replaces the range variables corresponding to contracting
// dimensions of the consumer with tiling parameters.
IndexingMap InsertTilingParameterForContractingDimensions(
    const HloInstruction* consumer, int64_t operand_index,
    const TilingSpecification::ParameterMapping& parameter_mapping,
    IndexingMap outermost_fusion_root_to_operand) {
  if (outermost_fusion_root_to_operand.GetRangeVarsCount() == 0) {
    return outermost_fusion_root_to_operand;
  }

  // At this point, we expect `consumer_to_operand` to contain range variables,
  // and for these range variables to represent dimensions being contracted by
  // the action of `consumer` on `operand`.
  // TODO(b/419026602): handle reductions here as well once priority fusion can
  // handle it. By adding a special path for reductions, we can handle them
  // here as well, even without nests.
  if (consumer->opcode() == HloOpcode::kDot) {
    CHECK(operand_index == 0 || operand_index == 1);
    absl::Span<const int64_t> contracting_dimensions =
        operand_index == 0
            ? consumer->dot_dimension_numbers().lhs_contracting_dimensions()
            : consumer->dot_dimension_numbers().rhs_contracting_dimensions();

    absl::flat_hash_map<int64_t, int64_t> parameter_index_by_symbol_position;
    std::vector<int64_t> symbols_to_remove;
    parameter_index_by_symbol_position.reserve(contracting_dimensions.size());
    symbols_to_remove.reserve(contracting_dimensions.size());
    for (auto [parameter_index, contracting_dimension] :
         llvm::enumerate(contracting_dimensions)) {
      auto symbol = mlir::dyn_cast<mlir::AffineSymbolExpr>(
          outermost_fusion_root_to_operand.GetAffineMap().getResult(
              contracting_dimension));
      // This can only occur if the wrong arguments were passed to this
      // function, and our traversal logic is broken.
      CHECK(symbol);  // Crash OK
      // Replace range variable at index contracting_dimension in the indexing
      // map with the parameter at (hlo, parameter_index).
      absl::StatusOr<int64_t> dim_index =
          ParameterIndexImpl(parameter_mapping, consumer, parameter_index);
      // This also can only fail if our traversal logic is broken.
      CHECK_OK(dim_index);  // Crash OK
      parameter_index_by_symbol_position.insert(
          {symbol.getPosition(), *dim_index});
      symbols_to_remove.push_back(symbol.getPosition());
    }

    std::sort(symbols_to_remove.begin(), symbols_to_remove.end());

    IndexingMap map_without_range_variables = ConvertRangeVariablesToDimensions(
        outermost_fusion_root_to_operand, symbols_to_remove);

    // At this point, `map_without_range_variables` has transformed
    // `outermost_fusion_root_to_operand` from a map of the form
    //
    //   (d0, ... d{C0}, ... d{N-1})[s0, ..., s{M-1}] -> ...
    //
    // into a map of the form
    //   (d0, ... d{C0}, ... d{N-1}, d{N}, ..., d{N+M-1}) -> ...
    //
    // where the newly added parameters correspond to the contracting
    // dimensions. This is not exactly what we want: the parameters
    // corresponding to the contracting dimensions already appeared in the
    // left-hand side of the initial map (annotated as d{C0}, ... above). We
    // compose the resulting map with a new map of the form
    //
    //    (d0, ... d{C0}, ... d{N-1}) -> (d0, ... d{C0}, ... d{N-1}, d{C0}, ...)
    //
    // in order to finish building the new map.
    MLIRContext* ctx = outermost_fusion_root_to_operand.GetMLIRContext();
    int64_t num_inputs = outermost_fusion_root_to_operand.GetDimVarsCount();
    int64_t num_outputs = map_without_range_variables.GetDimVarsCount();
    std::vector<int64_t> tileable_sizes;
    std::vector<mlir::AffineExpr> results;
    tileable_sizes.reserve(num_inputs);
    results.reserve(num_outputs);

    for (const auto& [i, dim_var] :
         llvm::enumerate(outermost_fusion_root_to_operand.GetDimVars())) {
      const Interval& bounds = dim_var.bounds;
      tileable_sizes.push_back(bounds.upper + 1);
      results.push_back(mlir::getAffineDimExpr(i, ctx));
    }

    for (const int64_t symbol_id : symbols_to_remove) {
      mlir::AffineExpr new_result = mlir::getAffineDimExpr(
          parameter_index_by_symbol_position.at(symbol_id), ctx);
      results.push_back(new_result);
    }

    mlir::AffineMap first_affine_map =
        mlir::AffineMap::get(num_inputs, /*symbolCount=*/0, results, ctx);

    IndexingMap first_indexing_map =
        IndexingMap::FromTensorSizes(first_affine_map, tileable_sizes, {});

    return ComposeIndexingMaps(first_indexing_map, map_without_range_variables);
  }

  return outermost_fusion_root_to_operand;
}

// The result of composing the indexing of an operand of an instruction.
struct ComposeIndexingResult {
  // Indexing map of the operand's instruction.
  IndexingMap indexing_map;
  // Runtime variables in the `indexing_map`.
  std::vector<SymbolicTiledHloInstruction*> rt_operands;
  // New instructions that were added to `tiled_hlo_instructions_set` as
  // part of composing the indexing. The caller must process them to ensure that
  // all participating instructions are tiled.
  std::vector<SymbolicTiledHloInstruction*> new_instructions;
};

// Composes the indexing of the operand's instruction starting from indexing of
// of the instruction.
// That is, given HLO
// B = foo(..)
// C = bar(B)
// and having mapping of C, we want to find the mapping of `foo` as operand B
// operand in `bar`.
// See comments in `ComposeIndexingMaps` for the details of the result.
ComposeIndexingResult ComposeInstructionIndexing(
    SymbolicTiledHloInstruction* tiled_hlo_instruction,
    const OperandIndexing& operand_indexing,
    IndexingMap::SimplifyPointDimensions simplification_mode,
    OrderedUniquePtrValueHashSet<SymbolicTiledHloInstruction>&
        tiled_hlo_instructions_set,
    HloInstructionAdaptor operand, HloInstructionAdaptor& instruction_adaptor,
    int64_t operand_pos,
    const TilingSpecification::ParameterMapping& parameter_mapping) {
  const HloInstruction* hlo = tiled_hlo_instruction->hlo();
  // Create an indexing of the instruction that corresponds to the operand.
  IndexingMap composed_indexing = ComposeIndexingMaps(
      tiled_hlo_instruction->indexing_map(), operand_indexing.map());
  if (composed_indexing.IsUndefined()) {
    return ComposeIndexingResult{composed_indexing, /*rt_operands=*/{},
                                 /*new_instructions=*/{}};
  }

  const size_t range_vars_count = composed_indexing.GetRangeVarsCount();
  QCHECK_EQ(range_vars_count +
                tiled_hlo_instruction->runtime_variables().size() +
                operand_indexing.runtime_variables().size(),
            composed_indexing.GetSymbolCount());
  composed_indexing.Simplify(simplification_mode);
  composed_indexing.RescaleSymbols();

  // Removal of unused symbols can drop some of the runtime variables so we
  // need to only add some of them from the instruction and operand itself.
  llvm::SmallBitVector removed = composed_indexing.RemoveUnusedSymbols();
  std::vector<SymbolicTiledHloInstruction*> rt_operands;
  rt_operands.reserve(tiled_hlo_instruction->runtime_variables().size() +
                      operand_indexing.runtime_variables().size());
  for (auto [i, rt] :
       llvm::enumerate(tiled_hlo_instruction->runtime_variables())) {
    size_t idx = i + range_vars_count;
    if (removed.size() > idx && removed[idx]) {
      continue;
    }
    VLOG(2) << "adding runtime variable from instruction " << rt;
    rt_operands.push_back(rt);
  }

  std::vector<SymbolicTiledHloInstruction*> new_instructions;
  for (const auto& [i, rt_var] :
       llvm::enumerate(operand_indexing.runtime_variables())) {
    size_t idx = i + tiled_hlo_instruction->runtime_variables().size() +
                 range_vars_count;
    if (removed.size() > idx && removed[idx]) {
      continue;
    }
    QCHECK_EQ(rt_var.map.GetRTVarsCount(), 0);
    IndexingMap rt_map =
        ComposeIndexingMaps(tiled_hlo_instruction->indexing_map(), rt_var.map);
    HloInstructionAdaptor hlo_adaptor =
        instruction_adaptor.parent().GetInstruction(rt_var.hlo);
    auto tiled_runtime_var = std::make_unique<SymbolicTiledHloInstruction>(
        &hlo_adaptor.instruction(), rt_map,
        tiled_hlo_instruction->runtime_variables());
    auto [tiled_hlo, inserted] =
        tiled_hlo_instructions_set.Insert(std::move(tiled_runtime_var));
    rt_operands.push_back(tiled_hlo);
    if (inserted) {
      new_instructions.push_back(tiled_hlo);
    }
  }

  // Whenever a range variable is introduced in our indexing map, we have
  // introduced a dimension that is collapsed in the output of the fusion.
  // Such dimensions need to be tiled, so we need to map it to one of the
  // fusion's tiling parameters.
  if (composed_indexing.GetRangeVarsCount() != 0) {
    composed_indexing = InsertTilingParameterForContractingDimensions(
        hlo, operand_pos, parameter_mapping, composed_indexing);
  }
  return ComposeIndexingResult{std::move(composed_indexing),
                               std::move(rt_operands),
                               std::move(new_instructions)};
}

std::vector<OperandIndexingSet> GetOperandIndexingMaps(
    const HloInstruction* hlo, MLIRContext* ctx) {
  std::vector<OperandIndexingSet> indexing_maps;
  HloInstructionIndexing operands_indexing =
      ComputeOutputToInputIndexing(hlo, /*output_id=*/0, ctx);
  if (hlo->opcode() == HloOpcode::kPad) {
    OperandIndexing pad_indexing_map =
        *operands_indexing.indexing_maps[0].begin();
    indexing_maps.push_back({OperandIndexing{
        IndexingMap{pad_indexing_map.map().GetAffineMap(),
                    DimVarsFromTensorSizes(hlo->shape().dimensions()),
                    pad_indexing_map.map().GetRangeVars(),
                    pad_indexing_map.map().GetRTVars()}}});
    indexing_maps.push_back({*operands_indexing.indexing_maps[1].begin()});
  } else {
    for (const auto& map_set : operands_indexing.indexing_maps) {
      CHECK_EQ(map_set.size(), 1);  // Crash OK
      indexing_maps.push_back(map_set);
    }
  }
  return indexing_maps;
}

}  // namespace

/*static*/ SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeFusionImpl(
    const HloFusionAdaptor& fusion,
    const TilingSpecification::ParameterMapping& parameter_mapping,
    MLIRContext* ctx, const RootIndexing& root_indexing,
    IndexingMap::SimplifyPointDimensions simplification_mode,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder,
    std::vector<SymbolicTiledHloInstruction*> root_runtime_variables) {
  OrderedUniquePtrValueHashSet<SymbolicTiledHloInstruction>
      tiled_hlo_instructions_set;

  // TODO(b/372454662): Once we get rid of the restriction of only one real
  // root, this needs to be adapted.
  auto [root_tiled_hlo, _] = tiled_hlo_instructions_set.Insert(
      std::make_unique<SymbolicTiledHloInstruction>(
          root_indexing.GetRealRoot(), root_indexing.real_root_indexing,
          std::move(root_runtime_variables)));
  if (root_tiled_hlo->hlo()->opcode() == HloOpcode::kFusion) {
    // This is an acceptable restriction because we expect the user of a nested
    // fusion to be a dot or concatenate, which prevents it from being a root.
    return FusionDecision::Forbid("Root fusion instruction is not supported.");
  }

  std::vector<SymbolicTiledHloInstruction*> worklist = {root_tiled_hlo};
  ConstraintExpression constraints = ConstraintExpression::GetAlwaysSatisfied();

  while (!worklist.empty()) {
    SymbolicTiledHloInstruction* tiled_hlo_instruction = worklist.back();
    worklist.pop_back();
    const HloInstruction* hlo = tiled_hlo_instruction->hlo();

    if (!fusion.ContainsInstruction(hlo)) {
      continue;
    }
    if (tiled_hlo_instruction->hlo()->opcode() == HloOpcode::kFusion) {
      continue;  // Don't analyze parameter operands of nested fusions.
    }

    auto operands_indexing =
        GetOperandIndexingMaps(tiled_hlo_instruction->hlo(), ctx);

    HloInstructionAdaptor instruction_adaptor(*hlo, &fusion);
    for (auto [operand_pos, operand_and_indexing_map_set] : llvm::enumerate(
             llvm::zip(instruction_adaptor.GetOperands(), operands_indexing))) {
      auto& [operand, operand_indexing] = operand_and_indexing_map_set;

      ComposeIndexingResult composed_indexing = ComposeInstructionIndexing(
          tiled_hlo_instruction, *operand_indexing.begin(), simplification_mode,
          tiled_hlo_instructions_set, operand, instruction_adaptor, operand_pos,
          parameter_mapping);

      if (composed_indexing.indexing_map.IsUndefined()) {
        return FusionDecision::Forbid(
                   "Couldn't compose indexing of instruction ")
               << hlo->ToString() << " and operand "
               << operand.instruction().ToString();
      }

      // New instruction indexing might use new instructions that were not
      // previously in the worklist.
      for (const auto& add : composed_indexing.new_instructions) {
        worklist.push_back(add);
      }

      std::unique_ptr<SymbolicTiledHloInstruction> tiled_operand;
      if (operand.opcode() == HloOpcode::kFusion &&
          fusion.ContainsInstruction(&operand.instruction())) {
        // The operand is a nested fusion, analyze it recursively.
        auto nested_fusion_adaptor = HloFusionAdaptor::ForComputation(
            operand.instruction().fused_instructions_computation());

        auto analysis_or = SymbolicTileAnalysis::AnalyzeNestedFusion(
            *nested_fusion_adaptor, parameter_mapping, ctx,
            composed_indexing.indexing_map, simplification_mode,
            emitter_specific_constraints_builder,
            composed_indexing.rt_operands);
        if (std::holds_alternative<FusionDecision>(analysis_or)) {
          return analysis_or;
        }
        SymbolicTileAnalysis analysis =
            std::get<SymbolicTileAnalysis>(std::move(analysis_or));
        constraints =
            constraints && analysis.GetTilingSpecification().constraints();
        constraints.Simplify();
        tiled_operand = std::make_unique<SymbolicTiledHloFusionInstruction>(
            &operand.instruction(), std::move(composed_indexing.indexing_map),
            std::move(analysis), std::move(composed_indexing.rt_operands));
      } else {
        tiled_operand = std::make_unique<SymbolicTiledHloInstruction>(
            &operand.instruction(), std::move(composed_indexing.indexing_map),
            std::move(composed_indexing.rt_operands));
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

  TilingSpecification tiling_specification = TilingSpecification(
      std::move(parameter_mapping),
      constraints && std::get<ConstraintExpression>(std::move(constraints_or)));

  return SymbolicTileAnalysis(std::move(tiled_hlo_instructions), root_indexing,
                              std::move(tiling_specification),
                              std::move(emitter_specific_constraints), ctx);
}

/*static*/ SymbolicTileAnalysisOrError SymbolicTileAnalysis::AnalyzeFusion(
    const HloFusionAdaptor& fusion, MLIRContext* ctx,
    EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder) {
  auto real_root_index_or = GetRealRootIndex(fusion.GetRoots());
  if (!real_root_index_or.ok()) {
    return FusionDecision(real_root_index_or.status());
  }

  auto parameter_mapping_or =
      ParameterMappingFromFusionAdaptor(fusion, *real_root_index_or);
  if (!parameter_mapping_or.ok()) {
    return FusionDecision(parameter_mapping_or.status());
  }

  auto root_indexing_or = GetRootIndexing(fusion, *parameter_mapping_or, ctx);
  if (!root_indexing_or.ok()) {
    return FusionDecision(root_indexing_or.status());
  }
  IndexingMap::SimplifyPointDimensions simplification_mode =
      ShouldDerivationSimplifyPointDimensions(fusion)
          ? IndexingMap::SimplifyPointDimensions::kReplace
          : IndexingMap::SimplifyPointDimensions::kPreserve;

  return AnalyzeFusionImpl(fusion, std::move(*parameter_mapping_or), ctx,
                           std::move(*root_indexing_or), simplification_mode,
                           emitter_specific_constraints_builder,
                           /*root_runtime_variables=*/{});
}

absl::StatusOr<bool> SymbolicTileAnalysis::ParametersSatisfyConstraints(
    absl::Span<const int64_t> tile_parameters) const {
  if (!tiling_specification_.constraints().is_satisfiable()) {
    return absl::FailedPreconditionError(
        "SymbolicTileAnalysis's constraints are not satisfiable. "
        "This should never happen.");
  }

  const HloInstruction* real_root = root_indexing_.GetRealRoot();
  ::xla::gpu::Tiling::TileMapping tile_mapping(
      {{real_root, absl::InlinedVector<int64_t, 4>(tile_parameters.begin(),
                                                   tile_parameters.end())}});
  ::xla::gpu::Tiling tiling(std::move(tile_mapping));
  return ParametersSatisfyConstraints(tiling);
}

absl::StatusOr<bool> SymbolicTileAnalysis::ParametersSatisfyConstraints(
    const ::xla::gpu::Tiling& tiling) const {
  const ConstraintExpression& constraints = tiling_specification_.constraints();
  CHECK(constraints.is_satisfiable());  // Crash OK

  TF_ASSIGN_OR_RETURN(std::vector<int64_t> flat_tiling_parameters,
                      tiling.Flatten(tiling_specification_));

  if (emitter_specific_constraints_ != nullptr) {
    TF_ASSIGN_OR_RETURN(
        bool constraints_are_satisfied,
        emitter_specific_constraints_->ParametersSatisfyConstraints(
            flat_tiling_parameters));
    if (!constraints_are_satisfied) {
      return false;
    }
  }

  return tiling.ConformsTo(tiling_specification_);
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
  // iteration order on output tile sizes and a tile stride of 1, which means
  // that we can take the identity map.
  auto identity_indexing_map =
      CreateIdentityMap(output.hlo()->shape(), mlir_context);
  auto iota = llvm::seq<int64_t>(0, output.hlo()->shape().dimensions().size());
  std::vector<int64_t> major_to_minor_active_tiling_parameters(iota.begin(),
                                                               iota.end());

  TF_ASSIGN_OR_RETURN(
      auto tiling_info,
      ComputeOutputTilingInfo(identity_indexing_map, output.tile_sizes(),
                              major_to_minor_active_tiling_parameters,
                              mlir_context));

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

// Returns the list of positions of dimensions expressions that appear in
// `expr`. No guarantee is made about their order in the resulting vector.
std::vector<int64_t> ExtractDimensionIds(AffineExpr expr) {
  std::vector<int64_t> dim_ids;
  expr.walk([&](mlir::AffineExpr expr) {
    if (auto dim_expr = mlir::dyn_cast<mlir::AffineDimExpr>(expr)) {
      dim_ids.push_back(dim_expr.getPosition());
    }
  });
  return dim_ids;
}

// TODO(b/406244630): this function is too long. We should chunk it up.
// Creates a concrete tiling of HLO computation with provided
// `flat_tiling_parameters` sizes based on the given symbolic tiling `analysis`.
absl::StatusOr<TiledHloComputation> ComputeTiledHloInstructionsImpl(
    const SymbolicTileAnalysis& analysis,
    const std::vector<int64_t>& flat_tiling_parameters,
    std::vector<int64_t> major_to_minor_active_tiling_parameters,
    bool compute_all_tile_offset_indexing_maps,
    const std::optional<absl::Span<const Interval>>&
        parent_output_tile_dim_bounds,
    MLIRContext* context,
    absl::flat_hash_map<const SymbolicTiledHloInstruction*,
                        TiledHloInstruction*>
        symbolic_to_tiled_hlo_map) {
  const IndexingMap& real_root_indexing = analysis.GetRealRootIndexing();
  for (mlir::AffineExpr expr : real_root_indexing.GetAffineMap().getResults()) {
    for (int64_t dim_id : ExtractDimensionIds(expr)) {
      if (absl::c_find(major_to_minor_active_tiling_parameters, dim_id) ==
          major_to_minor_active_tiling_parameters.end()) {
        major_to_minor_active_tiling_parameters.push_back(dim_id);
      }
    }
  }
  // Check that all strides are >= 0. Our codegen doesn't support negative
  // strides at the moment if padding is required. Also, for the Reverse op it
  // might make sense to emit code for it, and normalizing strides to >= 0.
  for (const std::unique_ptr<SymbolicTiledHloInstruction>& symbolic_tiled_hlo :
       analysis.GetSymbolicTiledHloComputation()) {
    llvm::SmallVector<int64_t> tile_strides = EvaluateTileStrides(
        symbolic_tiled_hlo->symbolic_tile(), flat_tiling_parameters);
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

      llvm::SmallVector<int64_t> tile_sizes = EvaluateTileSizes(
          symbolic_tiling->symbolic_tile(), flat_tiling_parameters);
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
      ComputeOutputTilingInfo(real_root_indexing, flat_tiling_parameters,
                              major_to_minor_active_tiling_parameters, context,
                              parent_output_tile_dim_bounds));

  VLOG(3) << "output_tiling_info: " << output_tiling_info.ToString("; ");

  OrderedUniquePtrValueHashSet<TiledHloInstruction> tiled_hlo_instructions_set;
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
                                     flat_tiling_parameters);
    }

    llvm::SmallVector<int64_t> tile_strides = EvaluateTileStrides(
        symbolic_tiled_hlo->symbolic_tile(), flat_tiling_parameters);

    std::optional<IndexingMap> tile_offset_indexing;
    llvm::SmallVector<const TiledHloInstruction*> runtime_variables;
    const HloInstruction* hlo = symbolic_tiled_hlo->hlo();
    if (compute_all_tile_offset_indexing_maps ||
        parameters_with_offset_indexing.contains(hlo) ||
        hlo->opcode() == HloOpcode::kIota) {
      CHECK_EQ(output_tiling_info.linear_output_tile_offset_indexing
                   .GetRTVarsCount(),
               0)
          << "runtime variables for output tiling are not supported";
      TF_ASSIGN_OR_RETURN(
          tile_offset_indexing,
          ComputeTileOffsetIndexing(
              *symbolic_tiled_hlo,
              output_tiling_info.linear_output_tile_offset_indexing, context));
      runtime_variables = MapToTiledInstructions(
          symbolic_tiled_hlo->runtime_variables(), symbolic_to_tiled_hlo_map);
      // Symbols here can only be runtime variables.
      llvm::SmallBitVector removed =
          tile_offset_indexing->RemoveUnusedSymbols();
      runtime_variables = RemoveInstructionByMask(runtime_variables, removed);
    }
    std::optional<std::vector<Interval>> fusion_tile_dim_bounds;
    if (hlo->opcode() == HloOpcode::kFusion && !hlo->users().empty() &&
        hlo->users().front()->opcode() == HloOpcode::kConcatenate) {
      fusion_tile_dim_bounds =
          output_tiling_info.output_tile_offset_indexing.GetDimensionBounds();
    }

    llvm::SmallVector<const TiledHloInstruction*> operands =
        MapToTiledInstructions(symbolic_tiled_hlo->operands(),
                               symbolic_to_tiled_hlo_map);

    std::unique_ptr<TiledHloInstruction> tiled_instruction;
    if (const auto* symbolic_fusion_tiling =
            dynamic_cast<const SymbolicTiledHloFusionInstruction*>(
                symbolic_tiled_hlo.get())) {
      // Compute tiled instructions recursively.
      TF_ASSIGN_OR_RETURN(
          auto tiled_hlo_computation,
          ComputeTiledHloInstructionsImpl(
              symbolic_fusion_tiling->analysis_, flat_tiling_parameters,
              major_to_minor_active_tiling_parameters,
              compute_all_tile_offset_indexing_maps, fusion_tile_dim_bounds,
              context, symbolic_to_tiled_hlo_map));

      TF_ASSIGN_OR_RETURN(
          tiled_instruction,
          TiledHloFusionInstruction::Create(
              hlo, std::move(operands), std::move(runtime_variables),
              std::make_unique<TiledHloComputation>(
                  std::move(tiled_hlo_computation)),
              std::move(tile_sizes), std::move(tile_strides),
              std::move(tile_offset_indexing)));
    } else {
      TF_ASSIGN_OR_RETURN(
          tiled_instruction,
          TiledHloInstruction::Create(
              hlo, std::move(operands), std::move(runtime_variables),
              std::move(tile_sizes), std::move(tile_strides),
              std::move(tile_offset_indexing)));
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

}  // namespace

absl::StatusOr<TiledHloComputation>
SymbolicTileAnalysis::ComputeTiledHloInstructions(
    const ::xla::gpu::Tiling& tiling, bool constraints_are_known_satisfied,
    bool compute_all_tile_offset_indexing_maps) const {
  // We first check that the provided tiling satisfies the constraints, if
  // necessary. We do this here instead of in `ComputeTiledHloInstructionsImpl`
  // because the latter is called recursively, and we don't want to perform
  // this check multiple times.
  if (!constraints_are_known_satisfied) {
    TF_ASSIGN_OR_RETURN(bool parameters_satisfy_constraints,
                        ParametersSatisfyConstraints(tiling));
    if (!parameters_satisfy_constraints) {
      return absl::InvalidArgumentError("Tiling does not satisfy constraints.");
    }
  }

  TF_ASSIGN_OR_RETURN(std::vector<int64_t> flat_tiling_parameters,
                      tiling.Flatten(GetTilingSpecification()));

  return ComputeTiledHloInstructionsImpl(
      *this, flat_tiling_parameters,
      /*major_to_minor_active_tiling_parameters=*/{},
      compute_all_tile_offset_indexing_maps,
      /*parent_output_tile_dim_bounds=*/std::nullopt, context_,
      /*symbolic_to_tiled_hlo_map=*/{});
}

absl::StatusOr<TiledHloComputation>
SymbolicTileAnalysis::ComputeTiledHloInstructions(
    absl::Span<const int64_t> output_tile_sizes,
    bool constraints_are_known_satisfied,
    bool compute_all_tile_offset_indexing_maps) const {
  ::xla::gpu::Tiling::TileMapping tile_mapping(
      {{tiling_specification_.parameter_mapping().begin()->instruction,
        absl::InlinedVector<int64_t, 4>(output_tile_sizes.begin(),
                                        output_tile_sizes.end())}});
  return ComputeTiledHloInstructions(::xla::gpu::Tiling(tile_mapping),
                                     constraints_are_known_satisfied,
                                     compute_all_tile_offset_indexing_maps);
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
