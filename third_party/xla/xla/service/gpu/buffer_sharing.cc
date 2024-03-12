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

#include "xla/service/gpu/buffer_sharing.h"

#include <cstdint>
#include <optional>
#include <queue>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"

namespace xla {
namespace gpu {

std::optional<bool> FusionCanShareBufferHint(const HloInstruction* user,
                                             const HloInstruction* operand,
                                             const ShapeIndex& user_index) {
  const HloFusionInstruction* fusion = DynCast<HloFusionInstruction>(user);
  if (fusion == nullptr) {
    return std::nullopt;
  }

  // First, do the trivial check: if the fusion operand and the fusion output
  // have a different number of elements or have a different element byte size,
  // the buffer cannot be shared.
  const Shape& user_subshape =
      ShapeUtil::GetSubshape(user->shape(), user_index);
  const Shape& operand_shape = operand->shape();
  const bool shapes_equal = ShapeUtil::Equal(operand_shape, user_subshape);
  if (!shapes_equal) {
    if (!operand_shape.IsArray() || !user_subshape.IsArray()) {
      return false;
    }
    // We cannot share the buffer if the iteration space is not the same.
    if (ShapeUtil::ElementsIn(operand_shape) !=
        ShapeUtil::ElementsIn(user_subshape)) {
      return false;
    }
    // The buffers needed for 'user_subshape' and 'operand_shape' need to have
    // the same size, otherwise they cannot be shared. We already checked that
    // the number of elements are the same, so now we check the number of bytes
    // needed for the element types.
    if (ShapeUtil::ByteSizeOfPrimitiveType(operand_shape.element_type()) !=
        ShapeUtil::ByteSizeOfPrimitiveType(user_subshape.element_type())) {
      return false;
    }
  }

  // Allow multiple output users, if they end in reductions.
  // This only works for the reduction emitter, as it calculates the reduction
  // first, i.e. before processing other outputs (that may overwrite the input).
  stream_executor::GpuDeviceInfoProto device_info;
  stream_executor::DeviceDescription device_description(device_info);
  auto analysis = HloFusionAnalysis::Create(fusion, &device_description);
  bool is_reduction_emitter = analysis.GetEmitterFusionKind() ==
                              HloFusionAnalysis::EmitterFusionKind::kReduction;
  const HloInstruction* reduction_hero =
      is_reduction_emitter ? reduction_hero = analysis.FindHeroReduction()
                           : nullptr;

  // We need to make sure that the fusion parameter is accessed in the same
  // iteration order as the fusion output. Also, there should not be any other
  // fusion output that accesses it in a different iteration order. To make sure
  // that the iteration order is the same, we only allow ops on the path from
  // fusion parameter to fusion output which are elementwise (no copy) or
  // bitcast or an elementwise dynamic update slice (i.e. with the first operand
  // being on this path).
  // In addition to that, we can also share the buffer for Scatter fusions if
  // the scatter is the single output of the fusion.
  HloInstruction* fusion_param =
      user->fused_parameter(user->operand_index(operand));
  HloInstruction* output = user->fused_expression_root();
  if (output->opcode() == HloOpcode::kTuple) {
    CHECK(!user_index.empty());
    output = output->mutable_operand(user_index[0]);
  }
  CHECK_NE(output->opcode(), HloOpcode::kTuple);
  const HloInstruction* non_bitcast_root = output;
  if (non_bitcast_root->opcode() == HloOpcode::kBitcast) {
    non_bitcast_root = non_bitcast_root->operand(0);
  }
  std::queue<HloInstruction*> q;
  absl::flat_hash_set<HloInstruction*> visited;
  q.push(fusion_param);
  visited.insert(fusion_param);
  bool found_path_to_output = false;
  while (!q.empty()) {
    HloInstruction* hlo_operand = q.front();
    q.pop();
    if (hlo_operand == output) {
      found_path_to_output = true;
      // We still need to process the users of 'hlo_operand'. There can be other
      // users in addition to the tuple user.
    }
    // Reduction emitter processes the reduction first, so the values below it
    // will not interfere with buffer sharing.
    if (hlo_operand == reduction_hero) {
      continue;
    }
    for (HloInstruction* hlo : hlo_operand->users()) {
      if (visited.insert(hlo).second) {
        q.push(hlo);
      }
      // For scatter, we can share the buffer if the path goes through one of
      // the scatter inputs.
      if (hlo == non_bitcast_root && hlo->opcode() == HloOpcode::kScatter) {
        int64_t num_scatter_inputs =
            hlo->shape().IsTuple() ? hlo->shape().tuple_shapes_size() : 1;
        if (hlo->operand_index(hlo_operand) < num_scatter_inputs &&
            absl::c_count(hlo->operands(), hlo_operand) == 1) {
          continue;
        }
      }
      if (non_bitcast_root->opcode() == HloOpcode::kDynamicUpdateSlice &&
          hlo->opcode() == HloOpcode::kDynamicSlice &&
          non_bitcast_root->operand(0) == hlo->operand(0) &&
          hlo->shape() == non_bitcast_root->operand(1)->shape()) {
        // We can still share the buffer in this case if the same slice is
        // accessed by the DUS and the DS. If they don't access the same slice,
        // the two slices might partially overlap and read/write the same index
        // at different times, and then we cannot guarantee that we read before
        // it is overwritten. However if both access only a single element,
        // there also can be no race condition.
        if (!ShapeUtil::IsEffectiveScalar(hlo->shape()) ||
            !ShapeUtil::IsEffectiveScalar(
                non_bitcast_root->operand(1)->shape())) {
          // Now compare all the slice start operands of 'hlo' and
          // 'non_bitcast_root'.
          for (int64_t i = 1; i < hlo->operand_count(); ++i) {
            if (hlo->operand(i) != non_bitcast_root->operand(i + 1)) {
              return false;
            }
          }
        }
      } else if ((!hlo->IsElementwiseOnOperand(
                      hlo->operand_index(hlo_operand)) ||
                  hlo->opcode() == HloOpcode::kCopy) &&
                 hlo->opcode() != HloOpcode::kBitcast &&
                 hlo->opcode() != HloOpcode::kTuple && hlo != reduction_hero) {
        // This check also catches the case that we reach a different fusion
        // output, as that fusion output would have a tuple op as user, which we
        // do not allow here.
        // Even if 'hlo' is not elementwise on the operand, it is ok if we are
        // coming from the second operand and 'hlo' is a DynamicUpdateSlice
        // which is the non_bitcast_root. This corresponds to the special case
        // above, where we allow a DynamicSlice if it accesses the exact same
        // slice than the DynamicUpdateSlice. When we are coming from the first
        // operand, IsElementwiseOnOperand() will return true for a
        // DynamicUpdateSlice.
        if (hlo != non_bitcast_root ||
            hlo->opcode() != HloOpcode::kDynamicUpdateSlice ||
            hlo->operand_index(hlo_operand) != 1) {
          return false;
        }
      }
    }
  }
  // Special case: multi-output fusions with Scatter or DynamicUpdateSlice. For
  // Scatter, we currently do not support multi-output fusions anyway, but still
  // handle it here. To be on the safe side, check for !IsElementwise() instead
  // of checking whether it is Scatter or DynamicUpdateSlice.
  if (user->IsMultiOutputFusion() && !non_bitcast_root->IsElementwise()) {
    // Check if any other fusion output was reached. If yes, we cannot share,
    // because the order in which the output is written might be different.
    for (HloInstruction* operand : user->fused_expression_root()->operands()) {
      if (operand != output && visited.find(operand) != visited.end()) {
        return false;
      }
    }
  }
  return found_path_to_output;
}

std::optional<bool> CanShareBufferHint(const HloInstruction* user,
                                       const HloInstruction* operand,
                                       const ShapeIndex& user_index) {
  switch (user->opcode()) {
    case HloOpcode::kAllReduce:
    case HloOpcode::kCollectiveBroadcast:
      // NCCL all-reduce and collective-broadcast can be performed in-place.
      return user->operand_count() == 1 ||
             (user_index.size() == 1 &&
              user->operand(user_index[0]) == operand);
    case HloOpcode::kCustomCall:
      // The matrix bias operand can be overwritten in-place.
      if (user->custom_call_target() == kCublasLtMatmulCallTarget) {
        GemmBackendConfig config =
            std::move(user->backend_config<GpuBackendConfig>())
                ->gemm_backend_config();
        return (config.beta() != 0.) && user->operand(2) == operand;
      }
      // The operand of cholesky can be shared with the first output.
      if (user->custom_call_target() == kCusolverCholeskyCallTarget) {
        return user_index.size() == 1 && user_index[0] == 0;
      }
      return false;
    case HloOpcode::kFusion:
      return FusionCanShareBufferHint(user, operand, user_index);
    default:
      return std::nullopt;
  }
}

}  // namespace gpu
}  // namespace xla
