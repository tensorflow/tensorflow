/* Copyright 2023 The OpenXLA Authors.

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

#ifndef XLA_HLO_ANALYSIS_INDEXING_ANALYSIS_H_
#define XLA_HLO_ANALYSIS_INDEXING_ANALYSIS_H_

#include <cstdint>
#include <optional>
#include <ostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/shape.h"

namespace xla {

class OperandIndexing;
using OperandIndexingSet = absl::flat_hash_set<OperandIndexing>;

// Contains indexing maps for all input operands.
// Indexing is computed for one particular output.
struct HloInstructionIndexing {
  std::string ToString(absl::string_view separator = "\n") const;

  // Returns true if the indexing was simplified.
  bool Simplify();

  // Creates a HloInstructionIndexing from a list of indexing maps for all
  // operands and sorted w.r.t. operand index, i.e. indexing_maps[i] corresponds
  // to operand[i] of the instruction.
  static HloInstructionIndexing FromIndexingMaps(
      absl::Span<const IndexingMap> indexing_maps);

  static HloInstructionIndexing FromOperandIndexing(
      absl::Span<const OperandIndexing> operand_indexing);

  // Each element is a set of indexing of the corresponding operand at the same
  // position.
  // TODO(b/417172838): rename?
  // TODO(b/417172838): indexing_maps are often used as
  // indexing_maps[X].begin()->map(), provide a method to access the map
  // directly and also check that there is only one element in the set.
  std::vector<OperandIndexingSet> indexing_maps;
};
std::ostream& operator<<(std::ostream& out,
                         const HloInstructionIndexing& instr_indexing);

// Computes indexing maps for all input operands necessary to compute an element
// of the `output_id` instruction output.
HloInstructionIndexing ComputeOutputToInputIndexing(const HloInstruction* instr,
                                                    int output_id,
                                                    mlir::MLIRContext* ctx);

// Computes indexing maps for all output operands that the element of the
// `input_id` instruction input will participate in.
HloInstructionIndexing ComputeInputToOutputIndexing(const HloInstruction* instr,
                                                    int input_id,
                                                    mlir::MLIRContext* ctx);

// Computes the indexing for `epilogue_parent`'s epilogue. For example, if
// `epilogue_parent` is a transpose, computes the input to output indexing for
// the path from the transpose's output to the root's output.
//
//   transpose
//       |
//     bitcast
//       |
//      ROOT
//
// The root must be specified because in HLO, an instruction can both be a hero
// and part of a side output:
//
//          reduce
//         /      \
//   broadcast    log
//        |        |
//       neg    bitcast
//         \      /
//           ROOT
//
// Here, the we must use the path through the `log` for the epilogue indexing,
// since the other path is not actually an epilogue (it's a side output). This
// fusion does not make much sense, but they are created sometimes.
IndexingMap ComputeEpilogueInputToOutputIndexing(
    HloInstructionAdaptor epilogue_parent, HloInstructionAdaptor epilogue_root,
    mlir::MLIRContext* mlir_context);

// Indexing of the runtime variable of the HLO instruction.
struct RuntimeVarIndexing {
  // Instruction of the runtime variable. Note that while in trivial cases it
  // points to one of the operands of the instruction, with multiple
  // instructions and fusions it may point to an arbitrary instruction in the
  // computation.
  const HloInstruction* hlo;
  // Output-to-input indexing map from the instruction to the output of `hlo`.
  IndexingMap map;
  std::string ToString() const;
};

std::ostream& operator<<(std::ostream& os, const RuntimeVarIndexing& var);
bool operator==(const RuntimeVarIndexing& lhs, const RuntimeVarIndexing& rhs);

// Indexing of an operand of a HLO instruction with information about runtime
// variables.
// Keeps invariant that the number of range variables in the map is the same as
// the number of runtime variables it holds.
class OperandIndexing {
 public:
  OperandIndexing(IndexingMap map, std::vector<RuntimeVarIndexing> rt_vars,
                  std::optional<IndexingMap> replica_id_map = std::nullopt)
      : map_(map), rt_vars_(rt_vars), replica_id_map_(replica_id_map) {
    VerifyOrDie();
  }

  explicit OperandIndexing(IndexingMap map) : map_(map) { VerifyOrDie(); }

  const IndexingMap& map() const { return map_; }
  const std::vector<RuntimeVarIndexing>& runtime_variables() const {
    return rt_vars_;
  }

  const std::optional<IndexingMap>& replica_id_map() const {
    return replica_id_map_;
  }

  std::string ToString() const;
  // Removes unused symbols from the indexing map and updates the runtime
  // variables accordingly.
  // Returns a bit vector of symbols (i.e. range and runtime variables) that
  // were removed. If none of the symbols were removed, returns an empty bit
  // vector.
  llvm::SmallBitVector RemoveUnusedSymbols();
  // Few proxy methods to IndexingMap to simplify the migration.
  bool Simplify() { return map_.Simplify(); }
  bool IsUndefined() const { return map_.IsUndefined(); }

  bool Verify(std::ostream& out) const;
  // Checks invariants and crashes if they are not satisfied.
  void VerifyOrDie() const;

  friend bool operator==(const OperandIndexing& lhs,
                         const OperandIndexing& rhs);

 private:
  IndexingMap map_;
  std::vector<RuntimeVarIndexing> rt_vars_;

  // Replica id map is only set for indexings that involve collective
  // operations. Replica id map has the same inputs as the main map and one
  // result. The result tells from which replica to read the data.
  std::optional<IndexingMap> replica_id_map_;
};

// Compose two operand indexings.
OperandIndexing ComposeOperandIndexing(const OperandIndexing& first,
                                       const OperandIndexing& second);

std::ostream& operator<<(std::ostream& os, const OperandIndexing& var);

template <typename H>
H AbslHashValue(H h, const OperandIndexing& indexing_map) {
  h = H::combine(std::move(h), indexing_map.map());
  for (const auto& rt_var : indexing_map.runtime_variables()) {
    h = H::combine(std::move(h), rt_var.map);
  }
  return h;
}

// TODO(b/417172838): some routines still use IndexingMapSet and
// GroupedByOpIndexingMap. We should convert them to use OperandIndexing
// equivalents.
using IndexingMapSet = absl::flat_hash_set<IndexingMap>;
IndexingMapSet ToIndexingMapSet(const OperandIndexingSet& operand_indexing_set);

using GroupedByOpIndexingMap =
    absl::flat_hash_map<const HloInstruction*, IndexingMapSet>;

using GroupedByOpIndexing =
    absl::flat_hash_map<const HloInstruction*, OperandIndexingSet>;

// Computes output-to-input indexing for every instruction within a fusion
// cluster starting with `target_instr` and going from def to use.
GroupedByOpIndexing ComputeGroupedOutputToInputIndexing(
    const HloFusionAdaptor& fusion_adaptor, HloInstructionAdaptor target_instr,
    mlir::MLIRContext* ctx);

// Groups indexing maps by instructions.
GroupedByOpIndexing GroupIndexingMapsByProducers(
    const HloInstructionIndexing& indexing, const HloInstruction* instr);

// Creates an indexing map for bitcasting from `input_shape` to `output_shape`.
// Equivalent to linearizing the input_shape index and then delinearizing it
// to output_shape.
IndexingMap GetBitcastMap(const Shape& input_shape, const Shape& output_shape,
                          mlir::MLIRContext* mlir_context);
IndexingMap GetBitcastMap(absl::Span<const int64_t> input_shape,
                          const Shape& output_shape,
                          mlir::MLIRContext* mlir_context);
IndexingMap GetBitcastMap(absl::Span<const int64_t> input_shape,
                          absl::Span<const int64_t> output_shape,
                          mlir::MLIRContext* mlir_context);

// Creates an indexing map from the physical layout of the tensor to its logical
// layout.
IndexingMap GetIndexingMapFromPhysicalLayoutToLogical(
    const Shape& shape, mlir::MLIRContext* mlir_context);

// Creates an indexing map from the logical layout of the tensor to its physical
// layout.
IndexingMap GetIndexingMapFromLogicalToPhysicalLayout(
    const Shape& shape, mlir::MLIRContext* mlir_context);

// Returns the shape of the output of the instruction.
const Shape& GetOutputShape(const HloInstruction* instr, int64_t output_id);

// Computes 1D index given a shape and N-d indexing expressions.
mlir::AffineExpr LinearizeShape(
    absl::Span<const int64_t> dims,
    absl::Span<const mlir::AffineExpr> dimension_exprs,
    mlir::MLIRContext* mlir_context);

// Computes N-d indexing expressions given a linear index and a shape.
std::vector<mlir::AffineExpr> DelinearizeIndex(absl::Span<const int64_t> dims,
                                               mlir::AffineExpr linear_index,
                                               mlir::MLIRContext* mlir_context);

// Creates an identity indexing map corresponding to the parameter shape.
IndexingMap CreateIdentityMap(const Shape& shape,
                              mlir::MLIRContext* mlir_context);
IndexingMap CreateIdentityMap(absl::Span<const int64_t> dimensions,
                              mlir::MLIRContext* mlir_context);

llvm::SmallVector<mlir::AffineExpr, 4> DelinearizeInBoundsIndex(
    mlir::AffineExpr linear, absl::Span<const int64_t> sizes);

}  // namespace xla

#endif  // XLA_HLO_ANALYSIS_INDEXING_ANALYSIS_H_
