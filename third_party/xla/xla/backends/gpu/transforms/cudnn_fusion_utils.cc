/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/cudnn_fusion_utils.h"

#include <algorithm>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/shape_util.h"

namespace xla {
namespace gpu {

namespace {

bool IsEpilogueOpSupportedByCuDNN(const HloInstruction& hlo,
                                  bool can_fuse_reduce, bool is_nchw) {
  if (is_nchw) {
    return false;
  }
  const HloOpcode opcode = hlo.opcode();
  switch (opcode) {
    case HloOpcode::kAbs:
    case HloOpcode::kAdd:
    case HloOpcode::kCeil:
    case HloOpcode::kCompare:
    case HloOpcode::kConvert:
    case HloOpcode::kCos:
    case HloOpcode::kDivide:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kLog:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNegate:
    case HloOpcode::kPower:
    case HloOpcode::kRsqrt:
    case HloOpcode::kSelect:
    case HloOpcode::kSin:
    case HloOpcode::kSqrt:
    case HloOpcode::kSubtract:
    case HloOpcode::kTan:
    case HloOpcode::kTanh:
    case HloOpcode::kClamp:
      return true;
    // hlo normalization adds bitcast to group reduction dimensions, only
    // fuse such bitcast
    case HloOpcode::kBitcast:
      return hlo.user_count() == 1 &&
             hlo.users()[0]->opcode() == HloOpcode::kReduce &&
             IsEpilogueOpSupportedByCuDNN(*hlo.users()[0], can_fuse_reduce,
                                          is_nchw);
    case HloOpcode::kBroadcast:
      return ShapeUtil::IsScalar(hlo.operand(0)->shape());
    case HloOpcode::kConstant:
      return ShapeUtil::IsScalar(hlo.shape());
    case HloOpcode::kReduce:
      return can_fuse_reduce && ShapeUtil::IsScalar(hlo.shape());
    default:
      return false;
  }
}

HloInstruction* FuseTowardOperand(
    HloInstruction* hlo, HloComputation::Builder& builder,
    std::vector<HloInstruction*>& fusion_params,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& fused_hlo_map,
    bool is_nchw) {
  if (auto it = fused_hlo_map.find(hlo); it != fused_hlo_map.end()) {
    return it->second;
  }
  HloInstruction* fused_hlo;
  if (IsEpilogueOpSupportedByCuDNN(*hlo, /*can_fuse_reduce=*/false, is_nchw) &&
      hlo->user_count() == 1) {
    HloInstruction::InstructionVector new_operands;
    for (int i = 0; i < hlo->operand_count(); ++i) {
      HloInstruction* operand = hlo->mutable_operand(i);
      new_operands.push_back(FuseTowardOperand(operand, builder, fusion_params,
                                               fused_hlo_map, is_nchw));
    }
    fused_hlo = builder.AddInstruction(
        hlo->CloneWithNewOperands(hlo->shape(), new_operands));
  } else {
    fusion_params.push_back(hlo);
    fused_hlo = builder.AddInstruction(HloInstruction::CreateParameter(
        fusion_params.size() - 1, hlo->shape(), hlo->name()));
  }
  CHECK(fused_hlo_map.insert({hlo, fused_hlo}).second);
  return fused_hlo;
}

bool ShouldKeepFusingUsers(HloInstruction* hlo, bool& can_fuse_reduce,
                           bool is_nchw) {
  if (hlo->user_count() == 0 || hlo->opcode() == HloOpcode::kReduce) {
    return false;
  }
  bool cached_can_fuse_reduce = can_fuse_reduce;
  for (HloInstruction* user : hlo->users()) {
    if (!IsEpilogueOpSupportedByCuDNN(*user, can_fuse_reduce, is_nchw)) {
      can_fuse_reduce = cached_can_fuse_reduce;
      return false;
    }
    can_fuse_reduce = can_fuse_reduce && user->opcode() != HloOpcode::kReduce;
  }
  return true;
}

}  // namespace

FusionState::Snapshot FusionState::TakeSnapshot() const {
  return {static_cast<int>(fusible_users.size()),
          static_cast<int>(fusion_outputs.size()), can_fuse_reduce};
}

void FusionState::RestoreSnapshot(const Snapshot& snapshot) {
  fusible_users.resize(snapshot.fusible_users_size);
  fusion_outputs.resize(snapshot.fusion_outputs_size);
  can_fuse_reduce = snapshot.can_fuse_reduce;
}

void FuseTowardUsers(
    HloComputation::Builder& builder,
    std::vector<HloInstruction*>& fusion_params,
    std::vector<HloInstruction*>& fusible_users,
    absl::flat_hash_map<HloInstruction*, HloInstruction*>& fused_hlo_map,
    bool is_nchw) {
  for (auto user : fusible_users) {
    HloInstruction::InstructionVector new_operands;
    for (int i = 0; i < user->operand_count(); ++i) {
      HloInstruction* operand = user->mutable_operand(i);
      HloInstruction* fused_operand = FuseTowardOperand(
          operand, builder, fusion_params, fused_hlo_map, is_nchw);
      new_operands.push_back(fused_operand);
    }
    HloInstruction* fused_user = builder.AddInstruction(
        user->CloneWithNewOperands(user->shape(), new_operands));
    CHECK(fused_hlo_map.insert({user, fused_user}).second);
  }
}

bool GrowFusionDFS(HloInstruction* hlo, HloReachabilityMap* reachability,
                   FusionState& state,
                   absl::flat_hash_map<HloInstruction*, bool>& fusible_cache,
                   bool is_nchw,
                   absl::FunctionRef<bool(const std::vector<HloInstruction*>&)>
                       is_outputs_valid) {
  if (fusible_cache.contains(hlo)) {
    return fusible_cache[hlo];
  }

  const auto snapshot = state.TakeSnapshot();

  bool is_endpoint =
      !ShouldKeepFusingUsers(hlo, state.can_fuse_reduce, is_nchw);
  bool is_subgraph_valid = true;
  if (!is_endpoint) {
    for (HloInstruction* user : hlo->users()) {
      if (!GrowFusionDFS(user, reachability, state, fusible_cache, is_nchw,
                         is_outputs_valid)) {
        is_subgraph_valid = false;
        is_endpoint = true;
        break;
      }
    }
  }

  if (!is_subgraph_valid) {
    state.RestoreSnapshot(snapshot);
  }

  if (is_endpoint) {
    state.fusion_outputs.push_back(hlo);
  }

  bool is_valid = is_outputs_valid(state.fusion_outputs);
  if (is_valid && is_endpoint) {
    is_valid =
        std::none_of(state.fusible_users.begin(), state.fusible_users.end(),
                     [&reachability, hlo](HloInstruction* user) {
                       return reachability->IsReachable(hlo, user);
                     });
  }
  if (is_valid) {
    is_valid = std::none_of(state.fusion_outputs.begin(),
                            state.fusion_outputs.end() - (is_endpoint ? 1 : 0),
                            [&reachability, hlo](HloInstruction* output) {
                              return reachability->IsReachable(output, hlo);
                            });
  }

  if (is_valid) {
    state.fusible_users.push_back(hlo);
  }
  return fusible_cache[hlo] = is_valid;
}

}  // namespace gpu
}  // namespace xla
