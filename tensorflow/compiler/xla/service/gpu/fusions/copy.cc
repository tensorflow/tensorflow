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
#include "tensorflow/compiler/xla/service/gpu/fusions/copy.h"

#include <memory>

#include "tensorflow/compiler/xla/service/gpu/copy_thunk.h"

namespace xla {
namespace gpu {

StatusOr<FusionEmissionResult> MemcpyFusion::Emit(
    IrEmitterContext& ir_emitter_context, ElementalIrEmitter& elemental_emitter,
    mlir::lmhlo::FusionOp fusion_op, const HloFusionInstruction& fusion,
    KernelReuseCache& kernel_cache, llvm::IRBuilder<>*) const {
  auto src_buffer = *GetAllocationSlice(src_, ir_emitter_context.allocations());
  auto dst_buffer = *GetAllocationSlice(dst_, ir_emitter_context.allocations());
  FusionEmissionResult result;
  if (src_buffer != dst_buffer) {
    result.thunks.emplace_back(std::make_unique<DeviceToDeviceCopyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(fusion_op),
        /*source_buffer=*/src_buffer,
        /*destination_buffer=*/dst_buffer,
        /*mem_size=*/ShapeUtil::ByteSizeOf(GetShape(src_)),
        /*source_value=*/src_,
        /*destination_value=*/dst_));
  }
  return result;
}

}  // namespace gpu
}  // namespace xla
