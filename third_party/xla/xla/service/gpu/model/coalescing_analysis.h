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

#ifndef XLA_SERVICE_GPU_MODEL_COALESCING_ANALYSIS_H_
#define XLA_SERVICE_GPU_MODEL_COALESCING_ANALYSIS_H_

#include <optional>

#include "absl/container/flat_hash_map.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/model/indexing_analysis.h"

namespace xla {
namespace gpu {

// Returns true if all input reads are coalesced. If consumer is not nullptr,
// producer and consumer are considered as one fusion, otherwise it's only the
// producer.
bool IsReadCoalescedHeuristic(const HloFusionAnalysis& fusion_analysis,
                              const HloInstruction* producer,
                              const HloInstruction* consumer = nullptr);

// Returns true, if operand's read is coalesced.
bool IsReadCoalesced(const HloInstruction* operand, const HloInstruction* instr,
                     const absl::flat_hash_map<const HloInstruction*,
                                               IndexingMapSet>& indexing_maps,
                     mlir::MLIRContext* mlir_context);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_MODEL_COALESCING_ANALYSIS_H_
