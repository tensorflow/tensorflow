/* Copyright 2025 The OpenXLA Authors.

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
#include "xla/backends/gpu/codegen/sort.h"

#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/codegen/llvm/llvm_emitter.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/shaped_slice.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/xla_data.pb.h"
#include "xla/tsl/platform/status_macros.h"

namespace xla {
namespace gpu {

absl::StatusOr<FusionEmissionResult> SortFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  std::vector<BufferAllocation::Slice> src_buffers;
  std::vector<BufferAllocation::Slice> dst_buffers;
  std::vector<Shape> src_shapes;
  src_buffers.reserve(fusion.operand_count());
  dst_buffers.reserve(fusion.operand_count());
  src_shapes.reserve(fusion.operand_count());
  const HloSortInstruction* sort =
      Cast<HloSortInstruction>(fusion.fused_expression_root());
  Shape keys_shape = sort->operand(0)->shape();
  for (int64_t i = 0; i < sort->operand_count(); ++i) {
    // We assume that the layout of all involved operands and
    // outputs is the same.
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, sort->operand(i)->shape(),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));
    ShapeIndex shape_index =
        sort->operand_count() > 1 ? ShapeIndex({i}) : ShapeIndex({});
    TF_RET_CHECK(LayoutUtil::LayoutsInShapesEqual(
        keys_shape, ShapeUtil::GetSubshape(sort->shape(), shape_index),
        Layout::Equal().IgnoreMemorySpace().IgnoreElementSize()));
    // We expect only parameters or iotas as operand of sort.
    if (HloPredicateIsOp<HloOpcode::kParameter>(sort->operand(i))) {
      const HloInstruction* src_instr =
          fusion.operand(sort->operand(i)->parameter_number());
      ASSIGN_OR_RETURN(
          BufferAllocation::Slice slice,
          ir_emitter_context.buffer_assignment().GetUniqueSlice(src_instr, {}));
      src_buffers.push_back(slice);
      src_shapes.push_back(sort->operand(i)->shape());
      ASSIGN_OR_RETURN(slice,
                       ir_emitter_context.buffer_assignment().GetUniqueSlice(
                           &fusion, shape_index));
      dst_buffers.push_back(slice);
    } else {
      TF_RET_CHECK(HloPredicateIsOp<HloOpcode::kIota>(sort->operand(i)));
    }
  }

  FusionEmissionResult result;
  for (int i = 0; i < src_buffers.size(); ++i) {
    if (src_buffers[i] != dst_buffers[i]) {
      result.thunks.emplace_back(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(
              &fusion, ir_emitter_context.GetNextThunkId()),
          /*source_buffer=*/ShapedSlice{src_buffers[i], src_shapes[i]},
          /*destination_buffer=*/ShapedSlice{dst_buffers[i], src_shapes[i]},
          /*mem_size=*/src_buffers[i].size()));
    }
  }
  std::string op_name(sort->name());
  result.module = ir_emitter_context.CreateLLVMModule(op_name);
  ASSIGN_OR_RETURN(
      ThunkSequence sort_thunks,
      EmitBitonicSortLLVMIR(sort, result.module.get(), &ir_emitter_context));
  result.thunks.insert(result.thunks.end(),
                       std::make_move_iterator(sort_thunks.begin()),
                       std::make_move_iterator(sort_thunks.end()));
  return result;
}

}  // namespace gpu
}  // namespace xla
