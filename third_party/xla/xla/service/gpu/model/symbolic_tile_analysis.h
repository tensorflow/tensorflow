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

#include "absl/container/flat_hash_map.h"
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
#include "xla/service/gpu/model/constraint_expression.h"
#include "xla/service/gpu/model/symbolic_tiled_hlo_instruction.h"
#include "xla/service/gpu/model/tiled_hlo_computation.h"
#include "xla/service/instruction_fusion.h"

namespace xla {
namespace gpu {

class SymbolicTileAnalysis;
using SymbolicTileAnalysisOrError =
    std::variant<SymbolicTileAnalysis, FusionDecision>;

// A `TilingSpecification` describes the structure of a set of expected tile
// sizes, by indicating how many tile sizes must be specified for each
// instruction within a set. When creating a `TilingSpecification` from a
// fusion, we're interested in finding out how many tile sizes need to be
// defined in aggregate in order to tile the whole fusion.
//
// To support that use case, `TilingSpecification`s derived from fusions only
// map instructions to the new tile sizes that they require we define. At the
// output of the fusion, we must define a tile size for each axis of the output;
// on an instruction that is not an output of the fusion, we must define a tile
// size iff that instruction contracts a dimension---i.e. if it introduces a new
// dimension to tile that is not visible at the output of the fusion. For
// example, take the following fusion computation:
//
//   fusion_computation {
//     p0 = f32[128,256] parameter(0)
//     p1 = f32[256,128] parameter(1)
//     dot = f32[128,128] dot(p0, p1),
//       lhs_contracting_dimensions={1}, rhs_contracting_dimensions={0}
//     ROOT abs = f32[128,128] abs(dot)
//   }.
//
// The tiling specification for this fusion would require that we define two
// tile sizes at the output, as well as one additional tile size for the
// contracting dimension introduced by the `dot` instruction. Ergo, the
// specification would be:
//   {
//     abs: 2
//     dot: 1
//   }
//
// The intent is for `TilingSpecification`s to be used in order to construct
// `Tiling`s for a fusion, with the guarantee that if the `Tiling`
// satisfies the `TilingSpecification`, then the `Tiling` contains exactly
// as many parameters as necessary to tile the whole fusion.
//
// TODO(b/419026602): reductions are ignored for now. This will need to handle
// them.
class TilingSpecification {
 public:
  // Associates a number of tiling parameters to an instruction. Since there is
  // never any ambiguity about the ordering of tiling parameters within a single
  // instruction, this is sufficient information to describe both how many
  // tiling parameters are introduced for the given instruction, as well as
  // their semantics (i.e. what contracting dimension they correspond to, or
  // what output dimension they correspond to).
  struct InstructionAndNumTilingParameters {
    // The instruction that this `InstructionAndNumTilingParameters` is
    // associated with.
    const HloInstruction* instruction;
    // The number of tile sizes that must be specified for the instruction.
    int64_t num_tiling_parameters;
  };

  // An ordered sequence of `InstructionAndNumTilingParameters`s. Since
  // parameter mapping within a single `InstructionAndNumTilingParameters`
  // is unambiguous, this abstraction provides enough information to describe
  // the ordering of a set of tiling parameters---e.g. for a whole fusion.
  using ParameterMapping = std::vector<InstructionAndNumTilingParameters>;

  // Returns the parameter mapping for the entire fusion the specification is
  // derived from.
  //
  // The instructions are guaranteed to be in use-before-def order, and it is
  // guaranteed that an instruction will only ever appear at most once.
  const ParameterMapping& parameter_mapping() const {
    return parameter_mapping_;
  }

  // Returns the constraints for the parameters of the tiling specification.
  const ConstraintExpression& constraints() const { return constraints_; }

  // Given the index of a tile size parameter for the given HLO instruction,
  // returns its overall parameter index within the `TilingSpecification`'s
  // parameter mapping.
  absl::StatusOr<int64_t> ParameterIndex(const HloInstruction* hlo,
                                         int64_t index) const;

  // Returns the total number of parameters in the tiling specification.
  int64_t num_parameters() const { return num_parameters_; }

 private:
  // `SymbolicTileAnalysis` is the only class allowed to construct
  // `TilingSpecification`s.
  friend class SymbolicTileAnalysis;
  explicit TilingSpecification(ParameterMapping parameter_mapping,
                               ConstraintExpression constraints)
      : parameter_mapping_(std::move(parameter_mapping)),
        constraints_(std::move(constraints)) {
    num_parameters_ = 0;
    for (const auto& [_, num_tiling_parameters] : parameter_mapping_) {
      num_parameters_ += num_tiling_parameters;
    }
  };

  ParameterMapping parameter_mapping_;
  ConstraintExpression constraints_;
  int64_t num_parameters_;
};

// `Tiling`s are instantiations of `TilingSpecification`s, and the conformance
// of a `Tiling` `t` to a `TilingSpecification` `spec` can be checked by calling
// `t.ConformsTo(spec)`.
//
// A given instruction may be mapped to either
//  1. a sequence of "output" tile sizes, corresponding to tiling of its output
//     shape;
//  2. a sequence of "hidden" tile sizes, corresponding to tiling of its
//     contraction dimensions;
//  3. both of the above.
//
// In the case of 3., the parameters are ordered such that the "hidden" tile
// sizes are listed first.
//
// Given a HLO opcode in isolation, there is never any ambiguity about which
// tile sizes are "output" or "hidden": if an opcode can be assigned "hidden"
// tile sizes, then we can always expect them to have a mapping---while "output"
// tile sizes only appear optionally.
//
// TODO(b/419026602): reductions are ignored for now. This will need to handle
// them.
class Tiling {
 public:
  using TileMapping = absl::flat_hash_map<const HloInstruction*,
                                          absl::InlinedVector<int64_t, 4>>;
  explicit Tiling(TileMapping tile_sizes)
      : tile_sizes_(std::move(tile_sizes)) {}

  // Returns `true` if the tiling conforms to the given tiling specification.
  // To conform to a tiling specification, the tiling must specify exactly the
  // right number of tile sizes for each exposed parameter in the tiling
  // specification.
  bool ConformsTo(const TilingSpecification& tiling_specification) const;

  // Returns the tile sizes for the given instruction. Raises an error if the
  // queried instruction should not be assigned tile sizes.
  absl::StatusOr<absl::Span<const int64_t>> TileSizesForInstruction(
      const HloInstruction* hlo) const;

  // Returns the underlying mapping from instructions to tile sizes.
  const TileMapping& tile_sizes() const { return tile_sizes_; }

  // Returns a flattened list of tile sizes that conforms to the parameter
  // mapping defined by the parameter `TilingSpecification`.
  //
  // Note that `Flatten` does not check whether this tiling conforms to the
  // parameter `TilingSpecification`, and it is the caller's responsibility to
  // ensure that this is the case.
  absl::StatusOr<std::vector<int64_t>> Flatten(
      const TilingSpecification& tiling_specification) const;

 private:
  TileMapping tile_sizes_;
};

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
  //
  // Nested fusions are analyzed recursively, but operands of nested fusions
  // (which are parameter ops) are not analyzed. This is because the symbolic
  // tiles of these operands may contain expressions with symbols which would
  // fail to be tiled.
  static SymbolicTileAnalysisOrError AnalyzeComputation(
      const HloComputation& computation, mlir::MLIRContext* ctx,
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
  // If `compute_all_tile_offset_indexing_maps == true`, all
  // `TiledHloInstruction`s will have tile offset indexing maps set. Otherwise,
  // the indexing maps will be set only for instructions that have equal hash to
  // deduplicate them.
  absl::StatusOr<TiledHloComputation> ComputeTiledHloInstructions(
      const ::xla::gpu::Tiling& tiling,
      bool constraints_are_known_satisfied = false,
      bool compute_all_tile_offset_indexing_maps = false) const;

  // Returns a graph of HLO instructions tiled with the given tiling parameters.
  // The provided tiling parameters must satisfy the analysis's constraints.
  // By default, `ComputeTiledHloInstructions` performs a check that the
  // constraints are satisfied by the chosen tiling parameters. Setting
  // `constraints_are_known_satisfied` to true bypasses this check.
  //
  // If `compute_all_tile_offset_indexing_maps == true`, all
  // `TiledHloInstruction`s will have tile offset indexing maps set. Otherwise,
  // the indexing maps will be set only for instructions that have equal hash to
  // deduplicate them.
  //
  // This variant can only be used for fusions with no hidden nested parameters.
  [[deprecated]] absl::StatusOr<TiledHloComputation>
  ComputeTiledHloInstructions(
      absl::Span<const int64_t> output_tile_sizes,
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

  // Returns true if a list of tile parameters satisfies the symbolic tile
  // analysis's constraints. If provided, also checks the emitter-specific
  // constraints.
  //
  // Returns false if the constraints are not satisfied but can be evaluated
  // correctly. Returns an error if the constraints cannot be evaluated
  // correctly. This is typically the case if too few tile parameters are
  // provided to fully reduce the constraint expressions to constants.
  //
  // This is a convenience overload for the case when only output tile sizes
  // need to be set.
  //
  // DEPRECATED: Use `ParametersSatisfyConstraints(const Tiling& tiling)`
  // instead. This is not safe for fusions involving hidden parameters.
  //
  // TODO(b/421837868): deprecate `SymbolicTileAnalysis::Tiling` everywhere to
  // use logic that supports nests everywhere.
  [[deprecated]] absl::StatusOr<bool> ParametersSatisfyConstraints(
      absl::Span<const int64_t> tile_parameters) const;

  // Returns `true` if a `Tiling` conforms to the symbolic tile analysis's
  // `TilingSpecification`. If provided, also checks the emitter-specific
  // constraints.
  //
  // Returns `false` if the tiling does not conform to the tiling
  // specification.
  absl::StatusOr<bool> ParametersSatisfyConstraints(
      const ::xla::gpu::Tiling& tiling) const;

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
      const RootIndexing& root_indexing,
      TilingSpecification tiling_specification,
      std::unique_ptr<EmitterSpecificConstraints> emitter_specific_constraints,
      mlir::MLIRContext* context)
      : symbolic_tiled_hlo_instructions_(
            std::move(symbolic_tiled_hlo_instructions)),
        root_indexing_(std::move(root_indexing)),
        tiling_specification_(std::move(tiling_specification)),
        emitter_specific_constraints_(std::move(emitter_specific_constraints)),
        context_(context) {}

  // Computes indexing information for the roots of the computation.
  static absl::StatusOr<RootIndexing> GetRootIndexing(
      const HloFusionAdaptor& fusion,
      const TilingSpecification::ParameterMapping& parameter_mapping,
      mlir::MLIRContext* ctx);

  static SymbolicTileAnalysisOrError AnalyzeFusionImpl(
      const HloFusionAdaptor& fusion,
      const TilingSpecification::ParameterMapping& parameter_mapping,
      mlir::MLIRContext* ctx, const RootIndexing& root_indexing,
      IndexingMap::SimplifyPointDimensions simplification_mode,
      EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder);

  // Helper for `AnalyzeFusion` to handle nested fusions.
  static SymbolicTileAnalysisOrError AnalyzeNestedFusion(
      const HloFusionAdaptor& fusion,
      const TilingSpecification::ParameterMapping& parameter_mapping,
      mlir::MLIRContext* ctx, const IndexingMap& indexing_map,
      IndexingMap::SimplifyPointDimensions simplification_mode,
      EmitterSpecificConstraintsBuilder emitter_specific_constraints_builder);

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
