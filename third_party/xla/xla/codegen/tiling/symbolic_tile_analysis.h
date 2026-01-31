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

#ifndef XLA_CODEGEN_TILING_SYMBOLIC_TILE_ANALYSIS_H_
#define XLA_CODEGEN_TILING_SYMBOLIC_TILE_ANALYSIS_H_

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
#include "xla/codegen/tiling/constraint_expression.h"
#include "xla/codegen/tiling/symbolic_tiled_hlo_instruction.h"
#include "xla/codegen/tiling/tiled_hlo_computation.h"
#include "xla/codegen/tiling/tiled_hlo_schedule.h"
#include "xla/codegen/tiling/tiling_specification.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/instruction_fusion.h"

namespace xla {

// "Real root"

// SymbolicTileAnalysis supports multi-output fusions where the root is of the
// form tuple(A, B, C, foo(A, B, C)), i.e. one of the root is a (transitive)
// consumer of all the other roots (here foo(A, B, C)). In such cases, the
// consumer root is the only root that requires defining tiling parameters, and
// we call it the "real root" of the computation.

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

  // Indexing map from the tiling parameters to the "real root"'s output
  // indexing space.
  IndexingMap real_root_indexing;
};

// An interface to implement additional emitter-specific constraints. This
// interface can be used as an extension point to further constrain the set of
// given limitations of a particular codegen solution.
class EmitterSpecificConstraints {
 public:
  virtual ~EmitterSpecificConstraints() = default;

  // Returns `true` if the given tiling parameters satisfy the constraints.
  //
  // The tiling parameters are expected to be flattened as per the parameter
  // mapping defined by the `TilingSpecification` underlying the
  // `SymbolicTileAnalysis`'s of interest.
  virtual absl::StatusOr<bool> ParametersSatisfyConstraints(
      absl::Span<const int64_t> tile_parameters) const = 0;
};

// TODO(b/367306544): get rid of the HloFusionAdaptor parameter once the
// abstraction exists.
using EmitterSpecificConstraintsBuilder =
    std::function<absl::StatusOr<std::unique_ptr<EmitterSpecificConstraints>>(
        const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&,
        const HloFusionAdaptor&)>;

using TiledHloScheduleBuilder =
    std::function<absl::StatusOr<std::unique_ptr<TiledHloSchedule>>(
        const TilingSpecification&)>;

// Constructs and holds symbolic tiles for all the instructions within a
// computation. The analysis may hold several different symbolic tiles for the
// same instruction if the instruction is indexed in several different ways in
// order to produce a single chunk of the output. In order to handle this
// properly, we store a symbolic tile for each possible path starting from the
// root instruction of the computation to the relevant instruction.
//
// We support a simple form of multi-output fusion, where the computation has a
// single "real" root, and the other roots appear in the chain of producers of
// the real root.
//
// Use `AnalyzeComputation` or `AnalyzeFusion` to construct a new analysis.
class SymbolicTileAnalysis {
 public:
  // Tries to construct a symbolic tile analysis from a computation. Returns
  // a diagnostic if the construction fails for any reason.
  //
  // If `emitter_specific_constraints_builder` is provided, it will be used to
  // construct emitter-specific constraints for the analysis.
  //
  // Nested fusions are analyzed recursively, but operands of nested fusions
  // (which are parameter ops) are not analyzed. This is because the symbolic
  // tiles of these operands may contain expressions with symbols which would
  // fail to be tiled.
  static SymbolicTileAnalysisOrError AnalyzeComputation(
      const HloComputation& computation, mlir::MLIRContext* mlir_context,
      EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder =
          nullptr);
  static SymbolicTileAnalysisOrError AnalyzeFusion(
      const HloFusionAdaptor& fusion, mlir::MLIRContext* ctx,
      EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder =
          nullptr);

  // Returns a graph of HLO instructions tiled with the given tiling parameters.
  // The provided tiling parameters must satisfy the analysis's constraints.
  // By default, `ComputeTiledHloInstructions` performs a check that the
  // constraints are satisfied by the chosen tiling parameters. Setting
  // `constraints_are_known_satisfied` to true bypasses this check.
  //
  // `TiledHloInstruction`s will have their `tile_offset_indexing_map` set if
  // either:
  // - `compute_all_tile_offset_indexing_maps` is set, or
  // - `compute_all_tile_offset_indexing_maps` is not set, but there are at
  //   least two `TiledHloInstruction`s with the same hash. In that case,
  //   `tile_offset_indexing_map`s are necessary to deduplicate operations.
  // In either case, the iteration pattern for the `TiledHloInstruction`s will
  // be dictated by the schedule derived from the provided schedule builder.
  absl::StatusOr<TiledHloComputation> ComputeTiledHloInstructions(
      const Tiling& tiling,
      const TiledHloScheduleBuilder& schedule_builder =
          CreateMajorToMinorTiledHloSchedule,
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

  // Returns the indexing for the real root.
  const IndexingMap& GetRealRootIndexing() const {
    return root_indexing_.real_root_indexing;
  }

  // Returns the symbolic tiled HLO instructions in def-before-use order.
  const std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>&
  GetSymbolicTiledHloComputation() const {
    return symbolic_tiled_hlo_instructions_;
  }

  // Returns the `TilingSpecification` for the fusion that was used to construct
  // the analysis.
  //
  // The `TilingSpecification` wraps the shape of the tiling parameters, and
  // their associated constraints.
  const TilingSpecification& GetTilingSpecification() const {
    return tiling_specification_;
  }

  // Returns `true` if a `Tiling` conforms to the symbolic tile analysis's
  // `TilingSpecification`. If provided, also checks the emitter-specific
  // constraints.
  //
  // Returns `false` if the tiling does not conform to the tiling
  // specification.
  absl::StatusOr<bool> ParametersSatisfyConstraints(const Tiling& tiling) const;

  // Return the underlying mlir::MLIRContext.
  mlir::MLIRContext* GetMLIRContext() const { return mlir_context_; };

  // Returns a string representation of the analysis. Used only for error
  // messages and debugging.
  std::string ToString() const;

  // Returns a list of valid tilings for the `SymbolicTiledHloComputation`
  // produced by this analysis.
  absl::StatusOr<std::vector<Tiling>> GetValidTilings() const;

 private:
  SymbolicTileAnalysis(
      std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
          symbolic_tiled_hlo_instructions,
      const RootIndexing& root_indexing,
      TilingSpecification tiling_specification,
      std::unique_ptr<EmitterSpecificConstraints> emitter_specific_constraints,
      mlir::MLIRContext* mlir_context)
      : symbolic_tiled_hlo_instructions_(
            std::move(symbolic_tiled_hlo_instructions)),
        root_indexing_(std::move(root_indexing)),
        tiling_specification_(std::move(tiling_specification)),
        emitter_specific_constraints_(std::move(emitter_specific_constraints)),
        mlir_context_(mlir_context) {}

  // Computes indexing information for the roots of the computation.
  static absl::StatusOr<RootIndexing> GetRootIndexing(
      const HloFusionAdaptor& fusion,
      const TilingSpecification::ParameterMapping& parameter_mapping,
      mlir::MLIRContext* mlir_context);

  static SymbolicTileAnalysisOrError AnalyzeFusionImpl(
      const HloFusionAdaptor& fusion,
      const TilingSpecification::ParameterMapping& parameter_mapping,
      mlir::MLIRContext* mlir_context, const RootIndexing& root_indexing,
      IndexingMap::SimplifyPointDimensions simplification_mode,
      EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder,
      std::vector<SymbolicTiledHloInstruction*> root_runtime_variables);

  // Helper for `AnalyzeFusion` to handle nested fusions.
  static SymbolicTileAnalysisOrError AnalyzeNestedFusion(
      const HloFusionAdaptor& fusion,
      const TilingSpecification::ParameterMapping& parameter_mapping,
      mlir::MLIRContext* mlir_context, const IndexingMap& indexing_map,
      IndexingMap::SimplifyPointDimensions simplification_mode,
      EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder,
      std::vector<SymbolicTiledHloInstruction*> root_runtime_variables);

  static std::variant<std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>,
                      FusionDecision>
  AnalyzeFromInstruction(
      std::unique_ptr<SymbolicTiledHloInstruction> instruction,
      const HloFusionAdaptor& fusion,
      const TilingSpecification::ParameterMapping& parameter_mapping,
      mlir::MLIRContext* mlir_context,
      IndexingMap::SimplifyPointDimensions simplification_mode,
      EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder,
      ConstraintExpression& constraints);

  // The tiled HLO instructions in def-before-use order.
  std::vector<std::unique_ptr<SymbolicTiledHloInstruction>>
      symbolic_tiled_hlo_instructions_;

  // Indexing information for the root of the computation.
  RootIndexing root_indexing_;

  // The tiling specification for the fusion that was used to construct the
  // analysis.
  TilingSpecification tiling_specification_;

  // Additional emitter-specific constraints on tile parameters. May be null if
  // no builder was provided when constructing the analysis.
  std::unique_ptr<EmitterSpecificConstraints> emitter_specific_constraints_;

  mlir::MLIRContext* mlir_context_;
};

namespace detail {

// Only exposed for testing.
absl::StatusOr<std::vector<FlatTiling>> GetFlatTilingsForInputSpace(
    absl::Span<const int64_t> input_space);

}  // namespace detail
}  // namespace xla

#endif  // XLA_CODEGEN_TILING_SYMBOLIC_TILE_ANALYSIS_H_
