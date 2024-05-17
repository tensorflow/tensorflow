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
#include "xla/service/gpu/fusions/in_place_dynamic_update_slice.h"

#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/IR/IRBuilder.h"
#include "mlir/IR/MLIRContext.h"  // from @llvm-project
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/service/gpu/elemental_ir_emitter.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/gpu/launch_dimensions.h"
#include "xla/service/gpu/model/indexing_map.h"
#include "xla/service/llvm_ir/dynamic_update_slice_util.h"
#include "xla/service/llvm_ir/fused_ir_emitter.h"
#include "xla/service/llvm_ir/ir_array.h"
#include "xla/status.h"

namespace xla {
namespace gpu {
namespace {

constexpr int kDUSUpdateIndex = 1;

}  // namespace

LaunchDimensions InPlaceDynamicUpdateSliceFusion::launch_dimensions() const {
  const auto& update_shape = dus_ops_.front()->operand(1)->shape();
  return CalculateLaunchDimensions(update_shape, analysis_.device_info());
}

std::optional<IndexingMap>
InPlaceDynamicUpdateSliceFusion::ComputeThreadIdToInputIndexing(
    int64_t root_index, int64_t hero_operand_index,
    mlir::MLIRContext* mlir_context) const {
  if (hero_operand_index != kDUSUpdateIndex) {
    return std::nullopt;
  }
  auto launch_dims = launch_dimensions();
  // It is guaranteed that all DUS ops have the same output shape at this point.
  const auto& update_shape =
      dus_ops_.front()->operand(kDUSUpdateIndex)->shape();
  return GetDefaultThreadIdIndexingMap(launch_dims, /*unroll_factor=*/1,
                                       update_shape, mlir_context);
}

absl::Status InPlaceDynamicUpdateSliceFusion::EmitKernel(
    IrEmitterContext& ir_emitter_context, const HloFusionInstruction& fusion,
    const LaunchDimensions& launch_dims, std::vector<llvm_ir::IrArray> inputs,
    std::vector<llvm_ir::IrArray> outputs, llvm::IRBuilder<>* builder) const {
  // In case a dynamic slice update's output is bitcasted, we need to ensure we
  // write to the output array using the shape and layout of the dynamic slice
  // update. This cast is known to be safe to do iff, in the case the output of
  // the dynamic slice update is bitcasted, that bitcast is either the fusion's
  // output, or has a single user and is part of the fusion's tuple output.
  // This condition should be enforced explicitly in the
  // 'CanEmitFusedDynamicUpdateSliceInPlaceForGpu' matcher.
  for (auto [op, output] : llvm::zip(dus_ops_, outputs)) {
    output = output.CastToShape(op->shape(), builder);
  }

  auto* fused_computation = fusion.fused_instructions_computation();
  GpuElementalIrEmitter elemental_emitter(ir_emitter_context, builder);
  FusedIrEmitter fused_emitter(elemental_emitter);
  for (auto [index, input] : llvm::enumerate(inputs)) {
    auto fused_operand = fused_computation->parameter_instruction(index);
    fused_emitter.BindGenerator(
        *fused_operand, [input = input, builder,
                         fused_operand](const llvm_ir::IrArray::Index& index) {
          return input.EmitReadArrayElement(index, builder,
                                            fused_operand->name());
        });
  }

  std::vector<std::pair<const HloInstruction*, const llvm_ir::IrArray>>
      dus_and_output_array;
  dus_and_output_array.reserve(dus_ops_.size());

  for (auto [op, output] : llvm::zip(dus_ops_, outputs)) {
    dus_and_output_array.push_back(std::make_pair(op, output));
  }

  return llvm_ir::EmitParallelFusedDynamicUpdateSliceInPlace(
      fused_computation, dus_and_output_array, &fused_emitter, launch_dims,
      builder);
}

}  // namespace gpu
}  // namespace xla
