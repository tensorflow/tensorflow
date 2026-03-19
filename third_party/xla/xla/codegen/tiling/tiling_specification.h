/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_CODEGEN_TILING_TILING_SPECIFICATION_H_
#define XLA_CODEGEN_TILING_TILING_SPECIFICATION_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/codegen/tiling/constraint_expression.h"
#include "xla/hlo/ir/hlo_instruction.h"

namespace xla {

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
                                         int64_t index) const {
    return ParameterIndex(parameter_mapping_, hlo, index);
  };

  // Given the index of a tile size parameter for a given HLO instruction,
  // returns its overall parameter index within the given parameter mapping.
  static absl::StatusOr<int64_t> ParameterIndex(
      const TilingSpecification::ParameterMapping& parameter_mapping,
      const HloInstruction* hlo, int64_t index);

  // Returns the total number of parameters in the tiling specification.
  int64_t num_parameters() const { return num_parameters_; }

  // Returns a string representation of the tiling specification.
  std::string ToString() const;

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

// A sequence of tile sizes.
//
// This is an inlined vector to avoid too many heap allocations.
using FlatTiling = absl::InlinedVector<int64_t, 4>;

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
  using TileMapping = absl::flat_hash_map<const HloInstruction*, FlatTiling>;
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
  absl::StatusOr<FlatTiling> Flatten(
      const TilingSpecification& tiling_specification) const;

  // Returns a `Tiling` that conforms to the parameter `TilingSpecification`
  // from the given flattened list of tile sizes.
  //
  // `Unflatten` is the dual of `Flatten`.
  static absl::StatusOr<Tiling> Unflatten(
      absl::Span<const int64_t> flat_tile_sizes,
      const TilingSpecification& tiling_specification);

 private:
  TileMapping tile_sizes_;
};

}  // namespace xla

#endif  // XLA_CODEGEN_TILING_TILING_SPECIFICATION_H_
