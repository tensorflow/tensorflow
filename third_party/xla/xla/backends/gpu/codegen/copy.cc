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

#include <algorithm>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "xla/backends/gpu/codegen/fusion_emitter.h"
#include "xla/backends/gpu/runtime/copy_thunk.h"
#include "xla/backends/gpu/runtime/thunk.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/utils/hlo_traversal.h"
#include "xla/literal_util.h"
#include "xla/service/buffer_assignment.h"
#include "xla/service/call_graph.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/service/gpu/ir_emitter_context.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

absl::StatusOr<FusionEmissionResult> MemcpyFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  std::vector<BufferAllocation::Slice> src_buffers;
  for (const HloInstructionAdaptor& root_adaptor : analysis_.fusion_roots()) {
    const HloInstruction* root = &root_adaptor.instruction();
    const HloInstruction* src_instr =
        fusion.operand(root->operand(0)->parameter_number());
    TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                        buffer_assignment_->GetUniqueSlice(src_instr, {}));
    src_buffers.push_back(slice);
  }

  std::vector<BufferAllocation::Slice> dst_buffers;
  TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
      fusion.shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return absl::OkStatus();
        }
        TF_ASSIGN_OR_RETURN(BufferAllocation::Slice slice,
                            buffer_assignment_->GetUniqueSlice(&fusion, index));
        dst_buffers.push_back(slice);
        return absl::OkStatus();
      }));

  FusionEmissionResult result;
  for (int i = 0; i < src_buffers.size(); ++i) {
    if (src_buffers[i] != dst_buffers[i]) {
      result.thunks.emplace_back(std::make_unique<DeviceToDeviceCopyThunk>(
          Thunk::ThunkInfo::WithProfileAnnotation(&fusion),
          /*source_buffer=*/src_buffers[i],
          /*destination_buffer=*/dst_buffers[i],
          /*mem_size=*/src_buffers[i].size()));
    }
  }
  return result;
}

absl::StatusOr<FusionEmissionResult> DynamicMemcpyFusion::Emit(
    IrEmitterContext& ir_emitter_context,
    const HloFusionInstruction& fusion) const {
  CHECK_EQ(analysis_.fusion_roots().size(), 1);

  const auto* src_instr = &analysis_.fusion_root(0).GetOperand(0).instruction();
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice src_buffer,
                      buffer_assignment_->GetUniqueSlice(src_instr, {}));
  TF_ASSIGN_OR_RETURN(BufferAllocation::Slice dst_buffer,
                      buffer_assignment_->GetUniqueSlice(&fusion, {}));

  FusionEmissionResult result;
  if (src_buffer != dst_buffer) {
    result.thunks.emplace_back(std::make_unique<DynamicMemcpyThunk>(
        Thunk::ThunkInfo::WithProfileAnnotation(&fusion),
        /*source_buffer=*/src_buffer,
        /*destination_buffer=*/dst_buffer,
        /*mem_size=*/dst_buffer.size(),
        /*descriptor=*/descriptor_));
  }
  return result;
}

namespace {

bool IsZeroOffset(const HloInstruction* slice, int dim) {
  return slice->dynamic_slice_sizes()[dim] ==
         slice->operand(0)->shape().dimensions(dim);
}

std::vector<const HloInstruction*> GetCallStack(
    const HloInstruction& instruction, const CallGraph& call_graph) {
  const HloInstruction* current = &instruction;
  std::vector<const HloInstruction*> stack;
  while (current) {
    stack.push_back(current);

    auto callers = call_graph.GetComputationCallers(current->parent());
    if (callers.size() == 1) {
      current = callers[0];
    } else {
      // Failed to determine a unique caller, so we stop here. The rest of the
      // call stack (if there is one, we can also be in the entry computation)
      // will be missing.
      current = nullptr;
    }
  }
  std::reverse(stack.begin(), stack.end());
  return stack;
}

}  // namespace

bool DynamicMemcpyFusion::IsCandidateFusion(
    const HloFusionInstruction& instruction) {
  const HloInstruction* root = instruction.fused_expression_root();
  if (root->opcode() != HloOpcode::kDynamicSlice) {
    return false;
  }

  // Only contiguous slices can be represented by a memcpy.
  if (!IsContiguousSlice(*root)) {
    VLOG(5) << "Slice is not contiguous.";
    return false;
  }

  std::optional<absl::InlinedVector<int64_t, 4>> strides =
      ShapeUtil::ByteStrides(root->operand(0)->shape());
  if (!strides) {
    VLOG(5) << "Failed to get byte strides.";
    return false;
  }

  if (root->operand(0)->opcode() != HloOpcode::kParameter) {
    VLOG(5) << "Not a slice of a parameter.";
    return false;
  }

  int rank = root->operand(0)->shape().rank();
  for (int i = 0; i < rank; ++i) {
    auto* operand = root->operand(i + 1);
    if (!IsZeroOffset(root, i) && operand->opcode() != HloOpcode::kConstant &&
        operand->opcode() != HloOpcode::kParameter) {
      VLOG(5) << "Dimension " << i << " is not a constant or a parameter.";
      return false;
    }
  }

  // This might be a dynamic memcpy fusion. We need to actually analyze the data
  // dependencies of the parameters to know for sure.
  return true;
}

std::optional<DynamicMemcpyThunk::MemcpyDescriptor>
DynamicMemcpyFusion::GetMemcpyDescriptorForFusion(
    const HloFusionInstruction& fusion, const CallGraph& call_graph) {
  if (!IsCandidateFusion(fusion)) {
    return std::nullopt;
  }

  const HloInstruction* slice = fusion.fused_expression_root();
  const Shape& slice_input_shape = slice->operand(0)->shape();
  std::optional<absl::InlinedVector<int64_t, 4>> strides =
      ShapeUtil::ByteStrides(slice_input_shape);
  if (!strides) {
    return std::nullopt;
  }

  int rank = slice_input_shape.rank();
  auto stack = GetCallStack(fusion, call_graph);

  VLOG(5) << "Preconditions passed, trying to build a memcpy descriptor.";
  DynamicMemcpyThunk::MemcpyDescriptor descriptor;
  for (int i = 0; i < rank; ++i) {
    auto* operand = slice->operand(i + 1);
    // If this dimension's offset is always clamped to 0, we can skip it.
    if (IsZeroOffset(slice, i)) {
      VLOG(5) << "Offset for dimension " << i << " is clamped to 0.";
      continue;
    }

    if (operand->opcode() == HloOpcode::kConstant) {
      std::optional<int64_t> value =
          LiteralUtil::LiteralAsScalarInt64(operand->literal());
      if (!value) {
        return std::nullopt;
      }

      VLOG(5) << "Offset for dimension " << i << " is constant: " << *value
              << ".";
      descriptor.src_byte_static_offset += *value * (*strides)[i];
      continue;
    }

    auto functional_dependency =
        ResolveFunctionalDependencyOnInductionVariable(stack, operand);
    if (!functional_dependency) {
      VLOG(5) << "Offset for dimension " << i << " is not statically known.";
      return std::nullopt;
    }

    // The while loop must actually be a for loop.
    auto loop_config = functional_dependency->loop
                           ->backend_config<xla::WhileLoopBackendConfig>();
    if (!loop_config.ok() || !loop_config->has_known_init_step() ||
        !loop_config->has_known_trip_count()) {
      VLOG(5) << "Offset for dimension " << i
              << " depends on loop with unknown behavior.";
      return std::nullopt;
    }

    VLOG(5) << "Offset for dimension " << i << " is dynamic.";
    descriptor.src_dynamic_offsets.emplace_back() = {
        functional_dependency->loop, functional_dependency->induction_var,
        functional_dependency->derived_value,
        /*dimension_size=*/slice_input_shape.dimensions(i),
        /*byte_stride=*/(*strides)[i]};
  }

  return descriptor;
}

}  // namespace gpu
}  // namespace xla
