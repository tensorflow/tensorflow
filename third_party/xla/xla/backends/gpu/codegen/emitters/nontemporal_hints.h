/* Copyright 2026 The OpenXLA Authors.

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

#ifndef XLA_BACKENDS_GPU_CODEGEN_EMITTERS_NONTEMPORAL_HINTS_H_
#define XLA_BACKENDS_GPU_CODEGEN_EMITTERS_NONTEMPORAL_HINTS_H_

#include <cstdint>
#include <vector>

#include "absl/strings/string_view.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/indexing_map.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/shape.h"

namespace xla::gpu {

// How a fusion reads an operand, from its indexing maps over all roots.
struct OperandReads {
  // Every map is defined, has no runtime variables, and reads each element at
  // most once.
  bool not_broadcast = false;
  // One map, no range variables, and no other operand of the fusion reads the
  // buffer: read once per output element.
  bool streamed_once = false;
  // Another instruction, or another operand of the fusion, reads the buffer.
  bool read_elsewhere = false;
};

// OperandReads per operand id. Empty for non-fusions and scatter fusions.
// Without `dataflow`, every buffer counts as read elsewhere and no operand as
// streamed once.
std::vector<OperandReads> ComputeOperandReads(
    const HloFusionInstruction& fusion, const HloDataflowAnalysis* dataflow,
    mlir::MLIRContext* mlir_context);

// OperandReads::read_elsewhere per operand id: the only fact that depends on
// more than the fused computation.
std::vector<bool> ComputeReadElsewhere(const HloFusionInstruction& fusion,
                                       const HloDataflowAnalysis* dataflow);

// Whether `map` reads each element of `shape` at most once. False unless the
// linearized index simplifies to a sum of variables, and of variables
// floor-divided by or modulo constants, times constants.
bool ReadsEachElementAtMostOnce(const IndexingMap& map, const Shape& shape);

// Whether a wavefront load reads part of a 128-byte line of an operand.
// Checks the first two wavefronts of the first two blocks with an active
// thread. True if the scan reaches block 1024 before finding two.
bool ReadsLinesInPieces(const IndexingMap& thread_id_to_index,
                        int64_t wavefront_size, int64_t element_bytes,
                        int64_t buffer_bytes);

// Whether the kernel writes each output element once, as loop and concatenate
// emitters do.
enum class OutputContiguity { kNonContiguous, kContiguous };

// Sets the read facts (see xla/codegen/emitters/ir/xla_ops.h) on the entry's
// operand args, and xla.contiguous_output for a contiguous output.
void AnnotateNontemporalHintFacts(mlir::ModuleOp module,
                                  absl::string_view entry_function_name,
                                  const HloInstruction& hlo_instruction,
                                  const HloDataflowAnalysis* dataflow,
                                  OutputContiguity output_contiguity);

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_CODEGEN_EMITTERS_NONTEMPORAL_HINTS_H_
