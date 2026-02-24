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

#ifndef XLA_CODEGEN_TILING_EXPERIMENTAL_TILING_SPACE_H_
#define XLA_CODEGEN_TILING_EXPERIMENTAL_TILING_SPACE_H_

#include <cstdint>
#include <deque>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/codegen/tiling/constraint_expression.h"
#include "xla/codegen/tiling/experimental/symbolic_tile.h"
#include "xla/hlo/analysis/interval.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"

namespace xla::gpu::experimental {

// TilingSpace holds information about all tiling parameters of a fusion.
//
// It defines symbolic tiles for the fusion roots as symbolic expressions and
// constraints of possible tile "variables":
// * parallel dimensions - output dimensions of the fusion;
// * sequential dimensions - contraction/reduction dimensions of operations in
//   the fusion;
// * runtime variables - for example, offsets of the dynamic slices.
//
// This information allows us later to explore the space of all possible tilings
// and assign concrete tilings for every instruction of the fusion with
// SymbolicTilePropagation.
class TilingSpace {
 public:
  TilingSpace() : constraints_(ConstraintExpression::GetAlwaysSatisfied()) {}

  // Unique ID for the dimension or runtime variable.
  using ID = int64_t;

  enum class DimensionSemantics { kParallel, kSequential };
  struct DimensionInfo {
    // Unique ID for the dimension within the tiling space.
    ID id;
    // Size of the dimension.
    int64_t dimension_size;
    // Type of the dimension.
    DimensionSemantics type;
    // HLO instruction that defines (introduces) the dimension. For example
    // fusion root instruction defines the parallel dimensions. Dot/reduce
    // defines the sequential (contraction) dimensions.
    const HloInstruction* hlo;
    // Index into the ordered list of dimensions of the HLO instruction `hlo`
    // that defines the dimension.
    // All dimensions in the HLO instruction are ordered as
    // [all parallel dims of the output, all reduction/contraction dims].
    //
    // Example, for `[a,b,c] = dot(lhs, rhs, lhs_contracting_dims={d,e}, ...)`.
    // The ordered list of dimensions is [a,b,c,d,e].
    int64_t dim_position;
  };

  // Information about a runtime variable.
  // For example:
  //
  // off = s32[] parameter(0)
  // ds = dynamic-slice(tensor, off), ...
  //
  // `off = s32[] parameter(0)` instruction (`hlo`) defines the runtime
  // variable.
  // User's (dynamic-slice) semantics sets the `bounds` of possible values.
  //
  // If the same hlo is used as runtime variable multiple times, there will be
  // multiple entries in the `rt_vars_` with different IDs.
  //
  // RTVarInfo are accessed by (user_hlo, operand_id), in this case it is
  // (dynamic-slice, 1).
  struct RTVarInfo {
    // Unique ID for the runtime variable within the tiling space.
    ID id;
    // Feasible bounds of the runtime variable.
    // The values outside of the bounds will be clamped.
    Interval bounds;
    // HLO instruction that defines the runtime variable.
    const HloInstruction* hlo;
  };

  static std::unique_ptr<TilingSpace> Create(const HloFusionAdaptor& fusion,
                                             mlir::MLIRContext* ctx);

  std::string ToString() const;

  // This allows GUnit to print the tile.
  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TilingSpace& space) {
    sink.Append(space.ToString());
  }

  // Returns the dimension info for the given `hlo` and `dim_position`.
  // `dim_position` is the index into the ordered list of dimensions of the HLO
  // instruction `hlo` that defines the dimension. The dimension info must
  // exist.
  const DimensionInfo& GetDimensionInfo(const HloInstruction& hlo,
                                        int64_t dim_position) const;

  // Returns the runtime variable info for `hlo` that uses it and its
  // `operand_id`. This runtime variable info must exist.
  const RTVarInfo& GetRTVarInfo(const HloInstruction& hlo,
                                int64_t operand_id) const;

  ConstraintExpression& mutable_constraint() { return constraints_; }
  const ConstraintExpression& constraint() const { return constraints_; }

  mlir::MLIRContext* mlir_context() const { return mlir_context_; }

  llvm::ArrayRef<SymbolicTile> tiled_roots() const { return tiled_roots_; }

  int64_t num_dimensions() const { return dimensions_.size(); }
  int64_t num_rt_vars() const { return rt_vars_.size(); }

  void AppendDimension(const HloInstruction* hlo, int64_t dim_position,
                       int64_t dim_size, DimensionSemantics dim_type);
  void AppendRTVar(const HloInstruction* hlo, int64_t operand_id,
                   const HloInstruction* rt_var, int64_t upper_bound);

 private:
  void ProcessDotLike(const HloInstruction& hlo);
  void ProcessReduce(const HloInstruction& hlo);
  void ProcessDynamicSlice(const HloInstruction& hlo);
  void ProcessInstruction(const HloInstruction& hlo);

  // Maps from (hlo, dim_position) to the dimension info.
  absl::flat_hash_map<std::pair<const HloInstruction*, int64_t>,
                      const DimensionInfo*>
      hlo_to_dimension_;
  // The deque is used to guarantee the pointer stability.
  std::deque<DimensionInfo> dimensions_;

  // Maps from (hlo, operand_id) to the runtime variable info.
  absl::flat_hash_map<std::pair<const HloInstruction*, int64_t>,
                      const RTVarInfo*>
      hlo_to_rt_var_;
  // The deque is used to guarantee the pointer stability.
  std::deque<RTVarInfo> rt_vars_;

  // Symbolic tiles for the fusion roots.
  // For tuple roots, there will be one tile per tuple element. Otherwise,
  // there will be only one symbolic tile.
  llvm::SmallVector<SymbolicTile, 2> tiled_roots_;

  // Constraint expression for the tiling space.
  ConstraintExpression constraints_;

  mlir::MLIRContext* mlir_context_;
};

// If the shape is a tuple, return the shape at the given index.
// Otherwise, return the shape itself.
const Shape& GetFirstShape(const HloInstruction* instr, int64_t index = 0);

}  // namespace xla::gpu::experimental

#endif  // XLA_CODEGEN_TILING_EXPERIMENTAL_TILING_SPACE_H_
