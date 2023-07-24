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
#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSIONS_FUSION_EMITTER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSIONS_FUSION_EMITTER_H_

#include <memory>
#include <optional>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "tensorflow/compiler/xla/service/elemental_ir_emitter.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emitter_context.h"
#include "tensorflow/compiler/xla/service/gpu/kernel_reuse_cache.h"
#include "tensorflow/compiler/xla/service/gpu/launch_dimensions.h"
#include "tensorflow/compiler/xla/service/gpu/thunk.h"
#include "tensorflow/compiler/xla/service/llvm_ir/ir_array.h"

namespace xla {
namespace gpu {

struct FusionEmissionResult {
  std::vector<std::unique_ptr<Thunk>> thunks;
};

class FusionInterface {
 public:
  virtual ~FusionInterface() = default;

  virtual StatusOr<FusionEmissionResult> Emit(
      KernelReuseCache& kernel_cache, llvm::IRBuilder<>* builder) const = 0;
};

class KernelFusionEmitterBase : public FusionInterface {
 public:
  // The downstream code that is used by this emitter operates on a mix of MLIR
  // and HLO classes. Ideally this would not be the case, but it's hard to
  // change.
  KernelFusionEmitterBase(IrEmitterContext& ir_emitter_context,
                          ElementalIrEmitter& elemental_emitter,
                          mlir::lmhlo::FusionOp fusion_op,
                          const HloFusionInstruction& fusion)
      : ir_emitter_context_(ir_emitter_context),
        elemental_emitter_(elemental_emitter),
        fusion_op_(fusion_op),
        fusion_(fusion) {}

  StatusOr<FusionEmissionResult> Emit(KernelReuseCache& kernel_cache,
                                      llvm::IRBuilder<>* builder) const final;
  virtual StatusOr<LaunchDimensions> launch_dimensions() const = 0;

 protected:
  virtual Status EmitKernel(const LaunchDimensions& launch_dims,
                            std::vector<llvm_ir::IrArray> inputs,
                            std::vector<llvm_ir::IrArray> outputs,
                            llvm::IRBuilder<>* builder,
                            int kernel_index) const = 0;
  virtual int num_kernels() const { return 1; }
  const HloFusionInstruction& fusion() const { return fusion_; }
  mlir::lmhlo::FusionOp fusion_op() const { return fusion_op_; }
  IrEmitterContext& ir_emitter_context() const { return ir_emitter_context_; }
  ElementalIrEmitter& elemental_emitter() const { return elemental_emitter_; }

 private:
  IrEmitterContext& ir_emitter_context_;
  ElementalIrEmitter& elemental_emitter_;
  mlir::lmhlo::FusionOp fusion_op_;
  const HloFusionInstruction& fusion_;
};

}  // namespace gpu
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_GPU_FUSIONS_FUSION_EMITTER_H_
