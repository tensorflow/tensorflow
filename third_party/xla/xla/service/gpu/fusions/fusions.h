/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#ifndef XLA_SERVICE_GPU_FUSIONS_FUSIONS_H_
#define XLA_SERVICE_GPU_FUSIONS_FUSIONS_H_

#include <memory>
#include <optional>
#include <variant>

#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

struct LmhloFusionInfo {
  mlir::lmhlo::FusionOp fusion_op;
  absl::Span<const BufferAllocation* const> allocations;

  explicit LmhloFusionInfo(
      mlir::lmhlo::FusionOp fusion_op,
      absl::Span<const BufferAllocation* const> allocations)
      : fusion_op(fusion_op), allocations(allocations) {}
};

struct HloFusionInfo {
  const HloFusionInstruction* instr;
  const BufferAssignment* buffer_assignment;

  explicit HloFusionInfo(const HloFusionInstruction* instr,
                         const BufferAssignment* buffer_assignment)
      : instr(instr), buffer_assignment(buffer_assignment) {}
};

// Returns the emitter for the given fusion. Returns nullopt if the fusion
// type is not yet supported.
StatusOr<std::optional<std::unique_ptr<FusionInterface>>> GetFusionEmitter(
    HloFusionAnalysis& analysis,
    std::variant<HloFusionInfo, LmhloFusionInfo> fusion_info);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_FUSIONS_H_
