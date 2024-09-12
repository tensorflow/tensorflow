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
#ifndef XLA_SERVICE_GPU_FUSIONS_LEGACY_INPUT_SLICES_H_
#define XLA_SERVICE_GPU_FUSIONS_LEGACY_INPUT_SLICES_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "llvm/IR/IRBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_analysis.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/util.h"

namespace xla {
namespace gpu {

// Generates code for input-fusible slices.
//
// Prerequisite: ROOT is either a slice or a tuple of slices. The input shapes
// of all ROOT slices need to be the same while their output shapes can be
// different. On the other hand, the input ranges of slices can be
// overlapping. Further generalization/specialization when the needs are seen
// in the future.
class InputSlicesFusion : public KernelFusionEmitterBase {
 public:
  explicit InputSlicesFusion(const HloFusionAnalysis& analysis)
      : analysis_(analysis),
        unroll_factor_(CeilOfRatio(
            8, analysis.input_output_info().smallest_output_dtype_bits)) {}
  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t output_id, mlir::MLIRContext* ctx) const override;

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const override {
    // TODO(b/319081342): Implement this.
    return std::nullopt;
  }

 protected:
  absl::Status EmitKernel(IrEmitterContext& ir_emitter_context,
                          const HloFusionInstruction& fusion,
                          const LaunchDimensions& launch_dims,
                          std::vector<llvm_ir::IrArray> inputs,
                          std::vector<llvm_ir::IrArray> outputs,
                          llvm::IRBuilder<>* builder) const override;

 private:
  const HloFusionAnalysis& analysis_;
  const int unroll_factor_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_LEGACY_INPUT_SLICES_H_
