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
#ifndef XLA_SERVICE_GPU_FUSIONS_FUSION_EMITTER_H_
#define XLA_SERVICE_GPU_FUSIONS_FUSION_EMITTER_H_

#include <memory>
#include <optional>
#include <vector>

#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/mlir_hlo/lhlo/IR/lhlo_ops.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/kernel_reuse_cache.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/thunk.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/status.h"
#include "xla/statusor.h"

namespace xla {
namespace gpu {

struct FusionEmissionResult {
  std::vector<std::unique_ptr<Thunk>> thunks;
};

class FusionInterface {
 public:
  virtual ~FusionInterface() = default;

  virtual StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context, mlir::lmhlo::FusionOp fusion_op,
      const HloFusionInstruction& fusion,
      KernelReuseCache& kernel_cache) const = 0;

  // Returns the fusion's launch dimensions, if applicable.
  virtual std::optional<LaunchDimensions> launch_dimensions() const {
    return std::nullopt;
  }
};

// Base class for fusions that are implemented using a single kernel, which is
// generated using LLVM.
class KernelFusionEmitterBase : public FusionInterface {
 public:
  StatusOr<FusionEmissionResult> Emit(
      IrEmitterContext& ir_emitter_context, mlir::lmhlo::FusionOp fusion_op,
      const HloFusionInstruction& fusion,
      KernelReuseCache& kernel_cache) const final;

 protected:
  virtual Status EmitKernel(IrEmitterContext& ir_emitter_context,
                            const HloFusionInstruction& fusion,
                            const LaunchDimensions& launch_dims,
                            std::vector<llvm_ir::IrArray> inputs,
                            std::vector<llvm_ir::IrArray> outputs,
                            llvm::IRBuilder<>* builder) const = 0;
};

StatusOr<std::tuple<llvm::Function*, std::vector<llvm_ir::IrArray /*inputs*/>,
                    std::vector<llvm_ir::IrArray> /*outputs*/>>
BuildKernelPrototype(IrEmitterContext& ir_emitter_context,
                     const std::string& suggested_name,
                     absl::Span<const KernelArgument> arguments,
                     size_t num_inputs,
                     const LaunchDimensions& launch_dimensions,
                     llvm::IRBuilder<>* builder);

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_FUSIONS_FUSION_EMITTER_H_
