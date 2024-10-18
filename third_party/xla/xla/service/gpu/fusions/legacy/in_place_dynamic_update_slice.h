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
#ifndef XLA_SERVICE_GPU_FUSIONS_LEGACY_IN_PLACE_DYNAMIC_UPDATE_SLICE_H_
#define XLA_SERVICE_GPU_FUSIONS_LEGACY_IN_PLACE_DYNAMIC_UPDATE_SLICE_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "llvm/IR/IRBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/llvm_ir/ir_array.h"

namespace xla {
namespace gpu {

// Fusion node where the root is either:
// 1. a dynamic-update-slice op
// 2. a bitcast of a dynamic-update-slice op
// 3. a tuple op returning the result of several dynamic-update-slice ops
// 4. a tuple op returning the result of several bitcast
//    dynamic-update-slice ops
//
// Additionally, all the dynamic-update-slice ops have exactly one user. The
// fusion parameter that they update can have users (in addition to the
// dynamic-update-slice op) that read in either
// a. a dynamic-slice corresponding exactly to the slice of the parameter that
//    is updated by the dynamic-update-slice op
// b. a dynamic-slice reading in a single element anywhere in the parameter.
//    This is only allowed if the dynamic-update-slice op updates a single
//    element
//
// In both cases, the additional users must not flow into any other output
// than the dynamic-slice-update corresponding to that particular slice of the
// parameter.
//
// The assumption is that each op's input (i.e. array to update) shares the
// same slice as its output. In this case, we have a special algorithm that
// modifies the output in place without touching the un-updated elements. The
// update slice is assumed to be the exact same for all the
// dynamic-update-slice ops.
class InPlaceDynamicUpdateSliceFusion : public KernelFusionEmitterBase {
 public:
  explicit InPlaceDynamicUpdateSliceFusion(const HloFusionAnalysis& analysis)
      : analysis_(analysis),
        dus_ops_(
            GetOutputDefiningDynamicUpdateSlices(analysis.fusion_roots())) {}
  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override {
    // The mapping cannot be statically computed in general, since the offsets
    // are unknown.
    return std::nullopt;
  }

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* mlir_context) const override;

 protected:
  absl::Status EmitKernel(IrEmitterContext& ir_emitter_context,
                          const HloFusionInstruction& fusion,
                          const LaunchDimensions& launch_dims,
                          std::vector<llvm_ir::IrArray> inputs,
                          std::vector<llvm_ir::IrArray> outputs,
                          llvm::IRBuilder<>* builder) const override;

  const HloFusionAnalysis& analysis_;
  std::vector<HloInstructionAdaptor> dus_ops_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_LEGACY_IN_PLACE_DYNAMIC_UPDATE_SLICE_H_
