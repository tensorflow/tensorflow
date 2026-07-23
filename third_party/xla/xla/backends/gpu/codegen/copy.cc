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
#include "xla/backends/gpu/codegen/copy.h"

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/runtime/device_to_device_copy_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/service/shaped_slice.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

AsyncThunkSequence MemcpyFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  std::vector<BufferAllocation::Slice> src_buffers;
  std::vector<Shape> src_shapes;
  for (const HloInstructionAdaptor& root_adaptor : analysis_.fusion_roots()) {
    const HloInstruction* root = &root_adaptor.instruction();
    const HloInstruction* src_instr =
        fusion.operand(root->operand(0)->parameter_number());
    ASSIGN_OR_RETURN(
        BufferAllocation::Slice slice,
        ir_emitter_context.buffer_assignment().GetUniqueSlice(src_instr, {}));
    src_buffers.push_back(slice);
    src_shapes.push_back(root->operand(0)->shape());
  }

  std::vector<BufferAllocation::Slice> dst_buffers;
  RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      fusion.shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                         ir_emitter_context.buffer_assignment().GetUniqueSlice(
                             &fusion, index));
        dst_buffers.push_back(slice);
        return absl::OkStatus();
      }));

  ThunkSequence thunks;
  for (int i = 0; i < src_buffers.size(); ++i) {
    if (src_buffers[i] != dst_buffers[i]) {
      thunks.emplace_back(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              &fusion, ir_emitter_context.GetNextThunkId()),
          /*source_buffer=*/ShapedSlice{src_buffers[i], src_shapes[i]},
          /*destination_buffer=*/ShapedSlice{dst_buffers[i], src_shapes[i]},
          /*mem_size=*/src_buffers[i].size()));
    }
  }
  return std::move(thunks);
}

}  // namespace gpu
}  // namespace xla
