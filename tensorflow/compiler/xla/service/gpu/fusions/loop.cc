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
#include "tensorflow/compiler/xla/service/gpu/fusions/loop.h"

#include <vector>

#include "llvm/IR/IRBuilder.h"
#include "tensorflow/compiler/xla/service/gpu/parallel_loop_emitter.h"
#include "tensorflow/compiler/xla/service/llvm_ir/fused_ir_emitter.h"

namespace xla {
namespace gpu {

Status LoopFusion::EmitKernel(const KernelArguments& args,
                              const LaunchDimensions& launch_dims,
                              std::vector<llvm_ir::IrArray> ir_arrays,
                              llvm::IRBuilder<>* builder) const {
  absl::Span<llvm_ir::IrArray> operand_arrays =
      absl::MakeSpan(ir_arrays).subspan(0,
                                        fusion_op().getInputBuffers().size());
  absl::Span<llvm_ir::IrArray> output_element_arrays =
      absl::MakeSpan(ir_arrays).subspan(fusion_op().getInputBuffers().size(),
                                        fusion_op().getOutputBuffers().size());

  FusedIrEmitter fused_emitter(elemental_emitter());
  for (int i = 0; i < fusion_op().getInputBuffers().size(); i++) {
    fused_emitter.BindGenerator(
        *fusion().fused_parameter(i), [&, i](llvm_ir::IrArray::Index index) {
          return operand_arrays[i].EmitReadArrayElement(index, builder);
        });
  }
  TF_ASSIGN_OR_RETURN(
      auto element_generator,
      fused_emitter.GetGenerator(*fusion().fused_expression_root()));

  llvm::Type* index_type =
      GetIndexTypeForKernel(fusion_op(), launch_dims.launch_bound(), builder);

  return ParallelLoopEmitter(element_generator, output_element_arrays,
                             launch_dims, builder,
                             *analysis_.GetLoopFusionConfig())
      .EmitLoop(GetIrNameFromLoc(fusion_op()->getLoc()), index_type);
}

}  // namespace gpu
}  // namespace xla
