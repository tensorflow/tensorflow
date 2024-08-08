/* Copyright 2024 The OpenXLA Authors.

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
#ifndef XLA_SERVICE_GPU_FUSIONS_LEGACY_SCATTER_H_
#define XLA_SERVICE_GPU_FUSIONS_LEGACY_SCATTER_H_

#include <optional>
#include <vector>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "llvm/IR/IRBuilder.h"
#include "mlir/IR/MLIRContext.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/llvm_ir/ir_array.h"

namespace xla {
namespace gpu {

// A scatter, implemented as a loop over the updates. All scatters are in-place.
class ScatterFusion : public KernelFusionEmitterBase {
 public:
  explicit ScatterFusion(const HloFusionAnalysis& analysis);

  LaunchDimensions launch_dimensions() const override;

  std::optional<IndexingMap> ComputeThreadIdToOutputIndexing(
      int64_t root_index, mlir::MLIRContext* ctx) const override {
    // The kernel iterates over updates, whose correspondence to output
    // elements cannot be computed statically.
    return std::nullopt;
  }

  std::optional<IndexingMap> ComputeThreadIdToInputIndexing(
      int64_t root_index, int64_t hero_operand_index,
      mlir::MLIRContext* ctx) const override;

 protected:
  absl::Status EmitKernel(IrEmitterContext& ir_emitter_context,
                          const HloFusionInstruction& fusion,
                          const LaunchDimensions& launch_dims,
                          std::vector<llvm_ir::IrArray> inputs,
                          std::vector<llvm_ir::IrArray> outputs,
                          llvm::IRBuilder<>* builder) const override;

 private:
  const HloFusionAnalysis& analysis_;
  LaunchDimensionsConfig config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_LEGACY_SCATTER_H_
