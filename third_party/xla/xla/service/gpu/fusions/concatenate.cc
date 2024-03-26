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
#include "xla/service/gpu/fusions/concatenate.h"

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/IRBuilder.h"
#include "llvm/IR/Value.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/elemental_ir_emitter.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/gpu/parallel_loop_emitter.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/service/llvm_ir/loop_emitter.h"
#include "xla/shape.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

const Shape& GetLargestConcatOperandShape(const HloFusionAnalysis& analysis) {
  const HloInstruction* concat = analysis.fusion_heroes().front();
  int64_t dim = concat->concatenate_dimension();
  auto less = [&](const HloInstruction* lhs, const HloInstruction* rhs) {
    return lhs->shape().dimensions(dim) < rhs->shape().dimensions(dim);
  };
  HloInstruction* operand = *absl::c_max_element(concat->operands(), less);
  return operand->shape();
}

ConcatenateFusion::ConcatenateFusion(const HloFusionAnalysis& analysis)
    : analysis_(analysis) {}

std::optional<IndexingMap> ConcatenateFusion::ComputeThreadIdToOutputIndexing(
    int64_t root_index, mlir::MLIRContext* ctx) const {
  return std::nullopt;
}

std::optional<IndexingMap> ConcatenateFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* ctx) const {
  return GetDefaultThreadIdIndexingMap(launch_dimensions(), /*unroll_factor=*/1,
                                       GetLargestConcatOperandShape(analysis_),
                                       ctx);
}

absl::Status ConcatenateFusion::EmitKernel(
    IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    const LaunchDimensions& launch_dims, std::vector<llvm_ir::IrArray> inputs,
    std::vector<llvm_ir::IrArray> outputs, llvm::IRBuilder<>* builder) const {
  GpuElementalIrEmitter elemental_emitter(ir_emitter_context, builder);
  FusedIrEmitter fused_emitter(elemental_emitter);
  for (int i = 0; i < fusion.fused_parameters().size(); i++) {
    fused_emitter.BindGenerator(
        *fusion.fused_parameter(i), [&, i](llvm_ir::IrArray::Index index) {
          return inputs[i].EmitReadArrayElement(index, builder);
        });
  }

  llvm::Type* index_type =
      GetIndexTypeForKernel(&fusion, launch_dims.launch_bound(), builder);

  const HloInstruction* concat = analysis_.fusion_heroes().front();
  int64_t concat_dim = concat->concatenate_dimension();
  int64_t operand_offset = 0;

  // Emit the slices that correspond to the operands of the concat hero.
  for (const HloInstruction* operand : concat->operands()) {
    llvm_ir::BodyEmitter body_emitter =
        [&](const llvm_ir::IrArray::Index& operand_index) -> absl::Status {
      // Bind concat to generate the current operand.
      TF_ASSIGN_OR_RETURN(auto operand_generator,
                          fused_emitter.GetGenerator(*operand));
      fused_emitter.BindGenerator(*concat, [&](llvm_ir::IrArray::Index) {
        return operand_generator(operand_index);
      });

      // Create the index of the slice corresponding to the current operand.
      llvm_ir::IrArray::Index result_index = operand_index.AddOffsetToDim(
          llvm::ConstantInt::get(index_type, operand_offset), concat_dim,
          builder);
      operand_offset += operand->shape().dimensions(concat_dim);

      // Generate and write out the slice for each root.
      for (const auto& [output, root] :
           llvm::zip_equal(outputs, analysis_.fusion_roots())) {
        llvm_ir::IrArray::Index root_index = result_index.SourceIndexOfBitcast(
            concat->shape(), root->shape(), builder);
        TF_ASSIGN_OR_RETURN(auto generator, fused_emitter.GetGenerator(*root));
        TF_ASSIGN_OR_RETURN(llvm::Value * value, generator(root_index));
        output.EmitWriteArrayElement(root_index, value, builder);
      }
      return absl::OkStatus();
    };

    ParallelLoopEmitter emitter(body_emitter, operand->shape(), launch_dims,
                                builder);
    TF_RETURN_IF_ERROR(emitter.EmitLoop(fusion.name(), index_type));
  }

  return absl::OkStatus();
}

LaunchDimensions ConcatenateFusion::launch_dimensions() const {
  return CalculateLaunchDimensions(GetLargestConcatOperandShape(analysis_),
                                   analysis_.device_info());
}

}  // namespace gpu
}  // namespace xla
