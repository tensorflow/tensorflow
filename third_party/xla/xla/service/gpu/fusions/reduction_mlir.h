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
#ifndef XLA_SERVICE_GPU_FUSIONS_REDUCTION_MLIR_H_
#define XLA_SERVICE_GPU_FUSIONS_REDUCTION_MLIR_H_

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "mlir/Dialect/Bufferization/IR/BufferizableOpInterface.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/fusions/mlir/elemental_hlo_to_mlir.h"
#include "xla/service/gpu/fusions/mlir/mlir_fusion_emitter.h"
#include "xla/service/gpu/fusions/reduction_base.h"

namespace xla {
namespace gpu {

// Reduction fusion. Lowers to LLVM via MLIR. Currently not fully
// implemented: only single reduction groups, no side outputs, only row
// reductions.
class MlirReductionFusion : public ReductionFusionBase<MlirFusionEmitterBase> {
 public:
  explicit MlirReductionFusion(const HloFusionAnalysis& analysis);

  static bool IsSupported(const HloFusionAnalysis& analysis);

 protected:
  absl::Status EmitEntryFunction(
      const mlir_converter::PartitionedComputations& computations,
      const mlir_converter::CallTargetProvider& call_targets,
      mlir::func::FuncOp entry_function,
      const HloFusionInstruction& fusion) const override;

  absl::flat_hash_set<const HloInstruction*> GetInstructionsWithCustomCodegen(
      const HloFusionInstruction& fusion) const override;

 private:
  struct EmitterState;
  friend struct EmitterState;

  absl::Status EmitReduction(EmitterState& state) const;

  std::vector<const HloInstruction*> reduction_heroes_;
  // The roots for each reduction hero.
  absl::flat_hash_map<const HloInstruction*, const HloInstruction*>
      reduction_roots_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_REDUCTION_MLIR_H_
