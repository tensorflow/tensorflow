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
#ifndef XLA_SERVICE_GPU_FUSIONS_COPY_H_
#define XLA_SERVICE_GPU_FUSIONS_COPY_H_

#include "xla/service/gpu/fusions/fusion_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"

namespace xla {
namespace gpu {

// Special case of a fusion consisting only of a kCopy instruction that can be
// implemented using a memcpy.
class MemcpyFusion : public FusionInterface {
 public:
  MemcpyFusion(mlir::Value src, mlir::Value dst) : src_(src), dst_(dst) {}

  StatusOr<FusionEmissionResult> Emit(IrEmitterContext& ir_emitter_context,
                                      ElementalIrEmitter& elemental_emitter,
                                      mlir::lmhlo::FusionOp fusion_op,
                                      const HloFusionInstruction& fusion,
                                      KernelReuseCache& kernel_cache,
                                      llvm::IRBuilder<>*) const final;

 private:
  mlir::Value src_;
  mlir::Value dst_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_COPY_H_
