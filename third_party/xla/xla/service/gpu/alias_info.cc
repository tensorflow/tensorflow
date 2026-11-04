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

#include "xla/service/gpu/alias_info.h"

#include <cstdint>
#include <optional>
#include <queue>
#include <tuple>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "llvm/ADT/STLExtras.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/primitive_util.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/stream_executor/device_description.h"
#include "xla/stream_executor/device_description.pb.h"

namespace xla::gpu {
namespace {
std::optional<bool> FusionCanShareBufferHint(
    const HloInstruction* operand, const HloFusionInstruction* user,
    const ShapeIndex& user_index,
    const se::DeviceDescription& device_description) {
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
    // the number of elements are the same, so now we check the number of bits
    // needed for the element types.
    if (primitive_util::BitWidth(operand_shape.element_type()) !=
        primitive_util::BitWidth(user_subshape.element_type())) {
      return false;
    }
  }

  // Allow multiple output users, if they end in reductions.
  // This only works for the reduction emitter, as it calculates the reduction
  // first, i.e. before processing other outputs (that may overwrite the input).
  auto analysis = HloFusionAnalysis::Create(*user, device_description);
  // Can be nullptr if the fusion is not using the reduction emitter.
  const HloInstruction* reduction_hero = analysis.FindHeroReduction();

  // We need to make sure that the fusion parameter is accessed in the same
  // iteration order as the fusion output. Also, there should not be any other
  // fusion output that accesses it in a different iteration order. To make sure
  // that the iteration order is the same, we only allow ops on the path from
  // fusion parameter to fusion output which are elementwise (no copy) or
  // bitcast or an elementwise dynamic update slice (i.e. with the first operand
  // being on this path).
  // In addition to that, we can also share the buffer for Scatter fusions if
  // the scatter is the single output of the fusion.
  // If 'operand' is passed to the fusion several times, all the corresponding
  // fusion parameters read the same buffer, so we have to check the paths from
  // all of them. This is the same as checking a single fusion parameter with
  // the users of all of them.
  HloInstruction* output = user->fused_expression_root();
  for (int64_t index : user_index) {
    if (output->opcode() != HloOpcode::kTuple) {
      break;
    }
    output = output->mutable_operand(index);
  }
  const HloInstruction* non_bitcast_root = output;
  if (non_bitcast_root->opcode() == HloOpcode::kBitcast) {
    non_bitcast_root = non_bitcast_root->operand(0);
  }
  std::queue<std::pair<HloInstruction*, bool>> q;
  absl::flat_hash_set<HloInstruction*> non_ds_visited;
  absl::flat_hash_set<HloInstruction*> ds_visited;
  auto enqueue = [&](HloInstruction* instr, bool from_ds) {
    absl::flat_hash_set<HloInstruction*>& visited_set =
        from_ds ? ds_visited : non_ds_visited;
    if (visited_set.insert(instr).second) {
      q.push({instr, from_ds});
    }
  };
  for (int64_t operand_number : user->OperandIndices(operand)) {
    enqueue(user->fused_parameter(operand_number), /*from_ds=*/false);
  }
  bool found_path_to_output = false;
  while (!q.empty()) {
    auto [hlo_operand, from_ds] = q.front();
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
      // For scatter, we can share the buffer if the path goes through one of
      // the scatter inputs.
      if (!from_ds && hlo == non_bitcast_root &&
          hlo->opcode() == HloOpcode::kScatter) {
        int64_t num_scatter_inputs =
            hlo->shape().IsTuple() ? hlo->shape().tuple_shapes().size() : 1;
        if (hlo->operand_index(hlo_operand) < num_scatter_inputs &&
            absl::c_count(hlo->operands(), hlo_operand) == 1) {
          if (hlo->shape().IsTuple() &&
              user_index != ShapeIndex{hlo->operand_index(hlo_operand)}) {
            return false;
          }
          enqueue(hlo, /*from_ds=*/false);
          continue;
        }
      }
      // For sort, we can share the buffer if the operand appears only once. We
      // can share it with that output buffer that corresponds to the operand.
      if (!from_ds && hlo == non_bitcast_root &&
          hlo->opcode() == HloOpcode::kSort &&
          absl::c_count(hlo->operands(), hlo_operand) == 1) {
        if (hlo->shape().IsTuple() &&
            user_index != ShapeIndex{hlo->operand_index(hlo_operand)}) {
          return false;
        }
        enqueue(hlo, /*from_ds=*/false);
        continue;
      }
      if (!from_ds &&
          non_bitcast_root->opcode() == HloOpcode::kDynamicUpdateSlice &&
          hlo->opcode() == HloOpcode::kDynamicSlice &&
          hlo->operand(0) == hlo_operand &&
          hlo->OperandIndices(hlo_operand).size() == 1 &&
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
        enqueue(hlo, /*from_ds=*/true);
      } else {
        for (int64_t op_idx : hlo->OperandIndices(hlo_operand)) {
          if ((!hlo->IsElementwiseOnOperand(op_idx) ||
               hlo->opcode() == HloOpcode::kCopy) &&
              hlo->opcode() != HloOpcode::kBitcast &&
              hlo->opcode() != HloOpcode::kTuple && hlo != reduction_hero) {
            // This check also catches the case that we reach a different fusion
            // output, as that fusion output would have a tuple op as user,
            // which we do not allow here.
            // Even if 'hlo' is not elementwise on the operand, it is ok if we
            // are coming from the second operand after visiting a matching
            // DynamicSlice and 'hlo' is a DynamicUpdateSlice which is the
            // non_bitcast_root. When we are coming from the first operand,
            // IsElementwiseOnOperand() will return true for a
            // DynamicUpdateSlice.
            if (!from_ds || hlo != non_bitcast_root ||
                hlo->opcode() != HloOpcode::kDynamicUpdateSlice ||
                op_idx != 1) {
              return false;
            }
          }
        }
        enqueue(hlo, hlo == non_bitcast_root ? false : from_ds);
      }
    }
  }
  if (user->IsMultiOutputFusion()) {
    // Check if any other fusion output was reached. If yes, we need to check
    // whether that fusion output has the same iteration order. If not, we
    // cannot share.
    bool other_root_was_reached = false;
    bool reached_other_non_elementwise_root = false;
    bool reached_root_has_transpose_hero = false;
    for (auto [root, hero] :
         llvm::zip(analysis.fusion_roots(), analysis.fusion_heroes())) {
      if (non_ds_visited.contains(&root.instruction()) ||
          ds_visited.contains(&root.instruction())) {
        if (&root.instruction() != output) {
          other_root_was_reached = true;
          const HloInstruction* non_bitcast_other_root = &root.instruction();
          if (non_bitcast_other_root->opcode() == HloOpcode::kBitcast) {
            non_bitcast_other_root = non_bitcast_other_root->operand(0);
          }
          if (non_bitcast_other_root != reduction_hero &&
              !non_bitcast_other_root->IsElementwise()) {
            reached_other_non_elementwise_root = true;
          }
        }
        if (hero.opcode() == HloOpcode::kTranspose ||
            hero.opcode() == HloOpcode::kCopy) {
          reached_root_has_transpose_hero = true;
        }
      }
    }
    if (other_root_was_reached) {
      // If a root was reached that has a transpose hero, that root will use the
      // iteration order of the transpose operand. The other root will have a
      // different iteration order, so we cannot share the buffer.
      // Special case: multi-output fusions with Scatter or DynamicUpdateSlice.
      // For Scatter, we currently do not support multi-output fusions anyway,
      // but still handle it here. To be on the safe side, check for
      // !IsElementwise() on both the target root and any other reached root
      // (excluding reduction_hero).
      if (!non_bitcast_root->IsElementwise() ||
          reached_other_non_elementwise_root ||
          reached_root_has_transpose_hero) {
        return false;
      }
    }
  }
  return found_path_to_output;
}
}  // namespace

bool GpuAliasInfo::IsNoOpForAliasAnalysis(
    const HloInstruction* instruction) const {
  return AliasInfo::IsNoOpForAliasAnalysis(instruction) ||
         instruction->opcode() == HloOpcode::kReshape;
}

std::optional<bool> GpuAliasInfo::MayAlias(const HloInstruction* operand,
                                           const ShapeIndex& operand_index,
                                           const HloInstruction* user,
                                           const ShapeIndex& user_index) const {
  const HloFusionInstruction* fusion = DynCast<HloFusionInstruction>(user);
  if (fusion != nullptr) {
    if (GetCustomFusionConfigName(fusion) == kDynamicSliceFusionConfigName) {
      // DynamicSliceFusionRewriterV2 creates these fusions before CopyInsertion
      // without setting output_to_operand_aliasing, and
      // DynamicSliceFusionV2Thunk::VerifyBufferAssignment requires the DUS
      // output to alias its target parameter even when that parameter has
      // additional internal uses inside the fusion.
      if (!operand_index.empty() ||
          fusion->OperandIndices(operand).size() != 1 ||
          !MustAlias(operand, operand_index, fusion, user_index)) {
        return false;
      }
      const HloInstruction* output = fusion->fused_expression_root();
      ShapeIndex idx = user_index;
      while (true) {
        std::tie(output, idx) = FollowTupleIndirection(output, idx);
        if (idx.empty() && IsNoOpForAliasAnalysis(output)) {
          output = output->operand(0);
          continue;
        }
        break;
      }
      for (const auto& [in_op, out_idx] : GetInPlaceInputOutputPairs(output)) {
        if (out_idx == idx &&
            output->OperandIndices(output->operand(in_op.operand_number))
                    .size() != 1) {
          return false;
        }
      }
      return true;
    }
    return FusionCanShareBufferHint(operand, fusion, user_index,
                                    device_description_);
  }
  return std::nullopt;
}
}  // namespace xla::gpu
