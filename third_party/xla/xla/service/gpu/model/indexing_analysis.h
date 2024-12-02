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

#ifndef XLA_SERVICE_GPU_MODEL_INDEXING_ANALYSIS_H_
#define XLA_SERVICE_GPU_MODEL_INDEXING_ANALYSIS_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
#include "llvm/ADT/SmallVector.h"
#include "mlir/IR/AffineExpr.h"
#include "mlir/IR/AffineMap.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/shape.h"

namespace xla {
namespace gpu {

using IndexingMapSet = absl::flat_hash_set<IndexingMap>;

// Contains indexing maps for all N-dimensional tensor input operands that
// correspond to a particular output.
struct HloInstructionIndexing {
  std::string ToString() const;

  // Returns true if the indexing was simplified.
  bool Simplify();

  // Creates a HloInstructionIndexing from a list of indexing maps for all
  // operands and sorted w.r.t. operand index, i.e. indexing_maps[i] corresponds
  // to operand[i] of the instruction.
  static HloInstructionIndexing FromIndexingMaps(
      absl::Span<const IndexingMap> indexing_maps);

  // Maps input operand index to the indexing map for one particular output.
  std::vector<IndexingMapSet> indexing_maps;
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

using GroupedByOpIndexingMap =
    absl::flat_hash_map<const HloInstruction*, IndexingMapSet>;

// Computes output-to-input indexing for every instruction within a fusion
// cluster starting with `target_instr` and going from def to use.
GroupedByOpIndexingMap ComputeGroupedOutputToInputIndexing(
    const HloFusionAdaptor& fusion_adaptor, HloInstructionAdaptor target_instr,
    mlir::MLIRContext* ctx);

// Groups indexing maps by instructions.
absl::flat_hash_map<const HloInstruction*, IndexingMapSet>
GroupIndexingMapsByProducers(const HloInstructionIndexing& indexing,
                             const HloInstruction* instr);

// Computes producer indexing maps and fuse/compose them with the consumer
// indexing maps.
bool FuseProducerConsumerOutputToInputIndexing(
    const HloInstruction* producer_instr,
    absl::flat_hash_map<const HloInstruction*, IndexingMapSet>*
        consumer_indexing,
    mlir::MLIRContext* mlir_context);

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

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_INDEXING_ANALYSIS_H_
