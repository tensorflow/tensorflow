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

#ifndef XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_ANALYSIS_H_
#define XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_ANALYSIS_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/symbolic_tile.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/instruction_fusion.h"

namespace xla {
namespace gpu {

class SymbolicTileAnalysis;
using SymbolicTileAnalysisOrError =
    std::variant<SymbolicTileAnalysis, FusionDecision>;

// Holds the indexing information for the roots of the computation.
struct RootIndexing {
  RootIndexing(int64_t real_root_index,
               absl::Span<const HloInstruction* const> roots,
               IndexingMap real_root_indexing)
      : real_root_index(real_root_index),
        roots(roots.begin(), roots.end()),
        real_root_indexing(std::move(real_root_indexing)) {}

  const HloInstruction* GetRealRoot() const { return roots[real_root_index]; }

  // ID of the root that defines the indexing for other roots.
  int64_t real_root_index;

  // `roots` contains the computation roots in increasing order of their
  // output index.
  absl::InlinedVector<const HloInstruction*, 2> roots;

  // Indexing map to the "real" root.
  IndexingMap real_root_indexing;
};

// An interface to implement additional emitter-specific constraints. This
// interface can be used as an extension point to further constrain the set of
// given limitations of a particular codegen solution.
class EmitterSpecificConstraints {
 public:
  virtual ~EmitterSpecificConstraints() = default;

  virtual absl::StatusOr<bool> ParametersSatisfyConstraints(
      absl::Span<const int64_t> tile_parameters) const = 0;
};

// TODO(b/367306544): get rid of the HloFusionAdaptor parameter once the
// abstraction exists.
using EmitterSpecificConstraintsBuilder =
    std::function<std::unique_ptr<EmitterSpecificConstraints>(
        const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&,
        const HloFusionAdaptor&)>;

// Constructs and holds symbolic tiles for all the instructions within a
// computation. We may hold several different symbolic tiles for the same
// instruction if the instruction is indexed in several different ways in order
// to produce a single chunk of the output. In order to handle this properly,
// we store a symbolic tile for each possible path starting from the root
// instruction of the computation to the relevant instruction.
// We support a simple form of multi-output fusion, where the computation has a
// single "real" root, and the other roots appear in the chain of producers of
// the real root.
class SymbolicTileAnalysis {
 public:
  // A tile size for each dimension.
  //
  // This is an inlined vector to avoid too many heap allocations.
  using Tiling = absl::InlinedVector<int64_t, 4>;

  // Tries to construct a symbolic tile analysis from a computation. Returns
  // a diagnostic if the construction fails for any reason.
  //
  // If `emitter_specific_constraints_builder` is provided, it will be used to
  // construct emitter-specific constraints for the analysis.
  static SymbolicTileAnalysisOrError AnalyzeComputation(
      const HloComputation& computation, mlir::MLIRContext* ctx,
      EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder =
          nullptr);
  static SymbolicTileAnalysisOrError AnalyzeFusion(
      const HloFusionAdaptor& fusion, mlir::MLIRContext* ctx,
      EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder =
          nullptr);

  // Returns a graph of HLO instructions tiled with the given tile parameters.
  // The provided tile parameters must satisfy the analysis's constraints.
  // By default, `ComputeTiledHloInstructions` performs a check that the
  // constraints are satisfied by the chosen tiled parameters. Setting
  // `constraints_are_known_satisfied` to true bypasses this check.
  //
  // If `compute_all_tile_offset_indexing_maps == true`, all
  // TiledHloInstructions will have tile offset indexing maps set. Otherwise,
  // the indexing maps will be set only for instructions that have equal hash to
  // deduplicate them.
  absl::StatusOr<TiledHloComputation> ComputeTiledHloInstructions(
      absl::Span<const int64_t> tile_parameters,
      bool constraints_are_known_satisfied = false,
      bool compute_all_tile_offset_indexing_maps = false) const;

  // Returns the roots of the computation in increasing order of their output
  // index.
  absl::Span<const HloInstruction* const> GetRoots() const {
    return root_indexing_.roots;
  }

  // Returns the root of the computation at output index `idx`.
  const HloInstruction* GetRoot(int64_t idx) const {
    return root_indexing_.roots[idx];
  }

  // Returns the output index of the real root.
  int64_t real_root_index() const { return root_indexing_.real_root_index; }

  // Returns the number of tile parameters in this symbolic analysis.
  // TODO(b/390569102): This assumes that there is only one root that matters
  // for computing the tiling, and that it is the last symbolic tiled hlo
  // instruction in the list.
  int64_t num_tile_parameters() const {
    return root_indexing_.real_root_indexing.GetDimVarsCount();
  }

  // Returns the symbolic tiled HLO instructions in def-before-use order.
  const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
  GetSymbolicTiledHloComputation() const {
    return symbolic_tiled_hlo_instructions_;
  }

  // Returns the constraints for the parameters of the symbolic tiled HLO
  // computation. This is the intersection of the constraints of all the
  // symbolic tiles encountered throughout the computation.
  const ConstraintExpression& GetConstraints() const { return constraints_; }

  // Returns true if a list of tile parameters satisfies the symbolic tile
  // analysis's constraints. If provided, also checks the emitter-specific
  // constraints.
  //
  // Returns false if the constraints are not satisfied but can be evaluated
  // correctly. Returns an error if the constraints cannot be evaluated
  // correctly. This is typically the case if too few tile parameters are
  // provided to fully reduce the constraint expressions to constants.
  absl::StatusOr<bool> ParametersSatisfyConstraints(
      absl::Span<const int64_t> tile_parameters) const;

  // Return the underlying MLIRContext.
  mlir::MLIRContext* GetMLIRContext() const { return context_; };

  // Returns a string representation of the analysis. Used only for error
  // messages and debugging.
  std::string ToString() const;

  // Returns a list of tilings for the symbolic tiled HLO computation of the
  // analysis that are expected to perform well.
  //
  // Note: This is an initial implementation where the results may not perform
  // that well, and now we're filtering the tilings with Triton in mind
  // (allowing only powers of 2 or the full dimension size).
  absl::StatusOr<std::vector<Tiling>> GetGoodTilings() const;

 private:
  SymbolicTileAnalysis(
      std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
          symbolic_tiled_hlo_instructions,
      const RootIndexing& root_indexing, ConstraintExpression constraints,
      std::unique_ptr<EmitterSpecificConstraints> emitter_specific_constraints,
      mlir::MLIRContext* context)
      : symbolic_tiled_hlo_instructions_(
            std::move(symbolic_tiled_hlo_instructions)),
        root_indexing_(std::move(root_indexing)),
        constraints_(std::move(constraints)),
        emitter_specific_constraints_(std::move(emitter_specific_constraints)),
        context_(context) {}

  // Computes indexing information for the roots of the computation.
  static absl::StatusOr<RootIndexing> GetRootIndexing(
      const HloFusionAdaptor& fusion, mlir::MLIRContext* ctx);

  static SymbolicTileAnalysisOrError AnalyzeFusionImpl(
      const HloFusionAdaptor& fusion, mlir::MLIRContext* ctx,
      const RootIndexing& root_indexing,
      EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder =
          nullptr);

  // The tiled HLO instructions in def-before-use order.
  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
      symbolic_tiled_hlo_instructions_;

  // Indexing information for the root of the computation.
  RootIndexing root_indexing_;

  // See the documentation of GetConstraints().
  ConstraintExpression constraints_;

  // Additional emitter-specific constraints on tile parameters. May be null if
  // no builder was provided when constructing the analysis.
  std::unique_ptr<EmitterSpecificConstraints> emitter_specific_constraints_;

  mlir::MLIRContext* context_;
};

namespace detail {
// Only exposed for testing, please use SymbolicTileAnalysis::GetGoodTilings()
// instead.
std::vector<SymbolicTileAnalysis::Tiling> GetGoodTilings(
    absl::Span<const int64_t> dim_sizes,
    std::function<bool(absl::Span<const int64_t>)> is_valid);
}  // namespace detail
}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_SYMBOLIC_TILE_ANALYSIS_H_
