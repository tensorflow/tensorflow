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
#ifndef XLA_SERVICE_GPU_FUSIONS_CUSTOM_H_
#define XLA_SERVICE_GPU_FUSIONS_CUSTOM_H_

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

// A wrapper for fusions implemented using the mechanism in
// xla/service/gpu/kernels. See custom_fusion.h in that folder for details.
class CustomFusionEmitter : public FusionInterface {
 public:
  absl::StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context, mlir::lmhlo::FusionOp fusion_op,
      const HloFusionInstruction& fusion) const final;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_CUSTOM_H_
