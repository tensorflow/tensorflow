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

#include "xla/service/p2p_schedule_preparation.h"

#include <cstdint>
#include <memory>
#include <set>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {

namespace {

// Returns a boolean to indicate whether the operation is a non-host P2P
// operation. We exclude non-host P2P operations for two reasons: (1) this
// pass currently only amend control dependence for non-host P2P operations.
// (2) we need to exclude host P2P operations when looking for a nested chain
// of non-host P2P operations.
bool IsP2POp(const HloInstruction* op) {
  auto p2p = DynCastOrNull<HloSendRecvInstruction>(op);
  return p2p != nullptr && !p2p->is_host_transfer();
}

bool IsP2PDoneOp(const HloInstruction* op) {
  return IsP2POp(op) && (op->opcode() == HloOpcode::kRecvDone ||
                         op->opcode() == HloOpcode::kSendDone);
}

enum P2PGroupKind { kUnpipelined = 0, kPipelined = 1, kUnrecognized = 2 };

// A P2P group node represents the P2P instructions that are in the same
// computation and have the same channel ID. This includes one Send/SendDone
// and one Recv/RecvDone. If the P2P instructions for the given channel ID are
// pipelined, the group node for the computation containing the while-loop also
// records the while-loop instruction.
//
struct P2PGroupNode {
  bool RecordParentComputation(HloComputation* parent) {
    if (parent_computation == nullptr) {
      parent_computation = parent;
      return true;
    }
    return parent_computation == parent;
  }

  bool RecordDoneOp(HloSendRecvInstruction* p2p) {
    if (!RecordParentComputation(p2p->parent())) {
      return false;
    }

    if (p2p->opcode() == HloOpcode::kRecvDone) {
      if (recv_done == nullptr) {
        recv_done = Cast<HloRecvDoneInstruction>(p2p);
        return true;
      }
    } else if (p2p->opcode() == HloOpcode::kSendDone) {
      if (send_done == nullptr) {
        send_done = Cast<HloSendDoneInstruction>(p2p);
        return true;
      }
    }
    return false;
  }

  bool RecordWhileOp(HloInstruction* while_op) {
    if (while_loop != nullptr) {
      return false;
    }
    if (!RecordParentComputation(while_op->parent())) {
      return false;
    }
    while_loop = while_op;
    return true;
  }

  bool Incomplete() const {
    return recv_done == nullptr || send_done == nullptr;
  }

  bool IncompletePipelinedParent() const {
    return Incomplete() || while_loop == nullptr;
  }

  HloRecvDoneInstruction* recv_done = nullptr;
  HloSendDoneInstruction* send_done = nullptr;
  // The computation that contains the Send and Recv instructions.
  HloComputation* parent_computation = nullptr;
  // The while-loop instruction that calls the while-body with the pipelined
  // P2P Send and Recv instructions.
  HloInstruction* while_loop = nullptr;
};

static constexpr int kUnpipelinedNodeIdx = 0;
static constexpr int kPipelinedChildNodeIdx = 0;
static constexpr int kPipelinedParentNodeIdx = 1;

// Represent a P2P instruction group for a given channel.
//
// A kUnpipelined P2P group contains only one P2PGroupNode while a kPipelined
// P2P group contains a P2PGroupNode for the while-body and a P2PGroupNode
// for the computation with the while-loop instruction calling the while-body.
struct P2PGroup {
  Status RecordDoneOpForUnpipelinedGroup(HloSendRecvInstruction* p2p) {
    if (kind == kUnrecognized) {
      // Leave unrecognized P2P groups alone.
      return OkStatus();
    }
    if (kind != kUnpipelined) {
      return InternalError("Expected unpipelined group");
    }
    P2PGroupNode& node = nodes[kUnpipelinedNodeIdx];
    if (!node.RecordDoneOp(p2p)) {
      kind = kUnrecognized;
    }
    return OkStatus();
  }

  Status RecordDoneOpForPipelinedGroup(HloSendRecvInstruction* p2p) {
    if (kind == kUnrecognized) {
      // Leave unrecognized P2P groups alone.
      return OkStatus();
    }
    if (kind == kUnpipelined) {
      if (nodes[kPipelinedParentNodeIdx].parent_computation != nullptr) {
        return InternalError("Expected unpipelined group");
      }
      kind = kPipelined;
    }
    P2PGroupNode& node = nodes[kPipelinedParentNodeIdx];
    if (!node.RecordDoneOp(p2p)) {
      kind = kUnrecognized;
    }
    return OkStatus();
  }

  Status RecordWhileOpToPipelinedGroup(HloInstruction* while_op) {
    if (kind == kUnrecognized) {
      // Leave unrecognized P2P groups alone.
      return OkStatus();
    }
    if (kind == kUnpipelined) {
      return InternalError("Expected pipelined group");
    }
    P2PGroupNode& node = nodes[kPipelinedParentNodeIdx];
    if (!node.RecordWhileOp(while_op)) {
      kind = kUnrecognized;
    }
    return OkStatus();
  }

  P2PGroupKind kind = kUnpipelined;
  P2PGroupNode nodes[2];
};

//  Maps the channel ID to the corresponding P2P operation group.
using P2PGroupMap = absl::flat_hash_map<int64_t, P2PGroup>;

// Maps the computation to the channel IDs used by the computation for P2P
// operations. We use std::set instead of hash set for deterministic iterators.
using P2PInComputation =
    absl::flat_hash_map<const HloComputation*, std::set<int64_t>>;

// If the while-body contains a P2P chain that use the same channel as another
// P2P chain in the caller computation, assume these two P2P chain belong to
// the same pipelined P2P sequence. Adds the WhileOp to the pipelined group
// representation in this case.
Status MayAddWhileOpToPipelinedGroup(HloInstruction* while_op,
                                     P2PInComputation& p2p_in_computation,
                                     P2PGroupMap& p2p_group_map) {
  HloComputation* body = while_op->called_computations()[0];
  auto p2p = p2p_in_computation.find(body);
  if (p2p == p2p_in_computation.end()) {
    return OkStatus();
  }
  int pipelined_group = 0;
  for (auto channel : p2p->second) {
    auto p2p_group = p2p_group_map.find(channel);
    if (p2p_group == p2p_group_map.end() ||
        p2p_group->second.kind != kPipelined) {
      continue;
    }
    pipelined_group++;
    if (pipelined_group > 1) {
      return InternalError(
          "Expecting only one pipelined P2P group for each while-loop");
    }
    TF_RETURN_IF_ERROR(
        p2p_group->second.RecordWhileOpToPipelinedGroup(while_op));
  }
  return OkStatus();
}

// For an unpipelined Send-Recv chain, add control dependence to enforce this
// ordering:
//   recv => send => recv-done => send-done.
Status ChainUnpipelinedP2P(P2PGroupNode& node) {
  HloSendRecvInstruction* recv_done = node.recv_done;
  HloRecvInstruction* recv =
      DynCast<HloRecvInstruction>(recv_done->mutable_operand(0));
  HloSendRecvInstruction* send_done = node.send_done;
  HloSendInstruction* send =
      DynCast<HloSendInstruction>(send_done->mutable_operand(0));
  // We want the Recv to be scheduled before the Send.
  TF_RETURN_IF_ERROR(recv->AddControlDependencyTo(send));
  VLOG(10) << "Add control predecessor " << send->ToString();
  // We want the Send to be scheduled before RecvDone to prevent the scheduler
  // from interleaving two Send-Recv sequences.
  TF_RETURN_IF_ERROR(send->AddControlDependencyTo(recv_done));
  VLOG(10) << "Add control predecessor " << recv_done->ToString();

  // We want the RecvDone to be scheduled before the SendDone.
  TF_RETURN_IF_ERROR(recv_done->AddControlDependencyTo(send_done));
  VLOG(10) << "Add control predecessor " << send_done->ToString();
  return OkStatus();
}

// For the pipelined Send-Recv chain in a while-body, we need to make sure
// that the Send is scheduled before Recv as the Send release the unrolled
// Recv before entering the transformed loop. We let the scheduler to decide
// where to schedule send-done and recv-done. For example, if the while-body
// has other unpiplined Send-Recv chains, it may produce this ordering:
//   send => send-done => other Send-Recv chains => recv => recv-done
// If the while-body doesn't have other Send-Recv chains, it may produce this
// ordering:
//   send => recv => recv-done => send-done
Status ChainPipelinedP2PChild(P2PGroupNode& node) {
  HloSendRecvInstruction* recv_done = node.recv_done;
  HloRecvInstruction* recv =
      DynCast<HloRecvInstruction>(recv_done->mutable_operand(0));
  HloSendRecvInstruction* send_done = node.send_done;
  HloSendInstruction* send =
      DynCast<HloSendInstruction>(send_done->mutable_operand(0));
  // We want the Send to be scheduled before the Send.
  TF_RETURN_IF_ERROR(send->AddControlDependencyTo(recv));
  VLOG(10) << "Add control predecessor " << recv->ToString();
  return OkStatus();
}

// Inspects the instructions in the computation to find out whether the
// computation directly or indirectly invoke P2P operations, and records the
// finding in p2p_in_computation_cache. Also returns the boolean result.
bool FindP2PInComputation(
    absl::flat_hash_map<const HloComputation*, bool>& p2p_in_computation_cache,
    const HloComputation* computation) {
  auto it = p2p_in_computation_cache.find(computation);
  if (it != p2p_in_computation_cache.end()) {
    return it->second;
  }
  bool result = false;
  for (HloInstruction* instr : computation->instructions()) {
    if (IsP2POp(instr)) {
      result = true;
      break;
    }
    for (const HloComputation* called_computation :
         instr->called_computations()) {
      if (FindP2PInComputation(p2p_in_computation_cache, called_computation)) {
        result = true;
        break;
      }
    }
  }
  p2p_in_computation_cache[computation] = result;
  return result;
}

// Returns a boolean to indicate whether there are any operation in the range
// [start, end] that contains non-host P2P transfer that are reachable from
// the given instruction.
bool OperationChainHasP2P(
    P2PInComputation& p2p_in_computation,
    const std::vector<HloInstruction*>::const_iterator& start,
    const std::vector<HloInstruction*>::const_iterator& end,
    const HloReachabilityMap* reachability, const HloInstruction* instr) {
  for (auto it_op = start; it_op != end; ++it_op) {
    const HloInstruction* op = *it_op;
    if (!reachability->IsReachable(instr, op)) continue;

    if (IsP2POp(op)) {
      return true;
    }

    for (const HloComputation* called_comp : op->called_computations()) {
      auto p2p_in_comp = p2p_in_computation.find(called_comp);
      if (p2p_in_comp != p2p_in_computation.end()) {
        return true;
      }
    }
  }
  return false;
}

// Collects P2P send-done and recv-done instructions from the computation and
// group them by channel IDs.
Status CollectP2PGroups(const HloComputation* computation,
                        P2PInComputation& p2p_in_computation,
                        P2PGroupMap& p2p_group_map) {
  for (auto hlo : computation->MakeInstructionPostOrder()) {
    if (hlo->opcode() == HloOpcode::kWhile) {
      TF_RETURN_IF_ERROR(MayAddWhileOpToPipelinedGroup(hlo, p2p_in_computation,
                                                       p2p_group_map));
      continue;
    }
    if (!IsP2PDoneOp(hlo)) {
      continue;
    }
    HloSendRecvInstruction* p2p = Cast<HloSendRecvInstruction>(hlo);
    int64_t channel = p2p->channel_id().value();
    auto p2p_group = p2p_group_map.find(channel);
    if (p2p_group == p2p_group_map.end()) {
      // First time to see this P2P channel, assume it is for a kUnpipelined
      // P2P group and may turn it into a kPipelined group or kUnrecognized
      // group.
      P2PGroup group;
      TF_RETURN_IF_ERROR(group.RecordDoneOpForUnpipelinedGroup(p2p));
      p2p_group_map[channel] = group;
    } else {
      P2PGroup& group = p2p_group->second;
      if (group.nodes[kUnpipelinedNodeIdx].parent_computation == computation) {
        TF_RETURN_IF_ERROR(group.RecordDoneOpForUnpipelinedGroup(p2p));
      } else {
        // We are at the parent computation for a pipelined P2P group.
        TF_RETURN_IF_ERROR(group.RecordDoneOpForPipelinedGroup(p2p));
      }
    }
    // We can't rely on the operation on p2p_group_map above to find out
    // whether it is the first time to handle this channel for the current
    // computation, as we may drop information in the present of kUncognized
    // groups.
    auto p2p_in_comp = p2p_in_computation.find(computation);
    if (p2p_in_comp == p2p_in_computation.end()) {
      // First time to see a P2P channel for the computation.
      p2p_in_computation[computation] = {channel};
    } else {
      // Add the channel only if it is not recorded yet.
      p2p_in_comp->second.insert(channel);
    }
  }

  // Now finalize each group, in particular, if a kPipelined or kUnpipelined
  // group is missing some instructions, change the group to kUnrecognized.
  for (auto& [channel, p2p_group] : p2p_group_map) {
    if (p2p_group.kind == kUnpipelined) {
      if (p2p_group.nodes[kUnpipelinedNodeIdx].Incomplete()) {
        p2p_group.kind = kUnrecognized;
      }
    } else if (p2p_group.kind == kPipelined) {
      if (p2p_group.nodes[kPipelinedChildNodeIdx].Incomplete() ||
          p2p_group.nodes[kPipelinedParentNodeIdx]
              .IncompletePipelinedParent()) {
        p2p_group.kind = kUnrecognized;
      }
    }
  }
  // Erase kUnrecognized groups.
  absl::erase_if(p2p_group_map, [](const auto& p2p_group) {
    return p2p_group.second.kind == kUnrecognized;
  });

  return OkStatus();
}

}  // namespace

StatusOr<bool> P2PSchedulePreparation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  P2PGroupMap p2p_group_map;
  P2PInComputation p2p_in_computation;

  std::vector<HloComputation*> all_computations =
      module->MakeComputationPostOrder(execution_threads);

  // Collect P2P group information. We visit computations in the order of
  // callees to callers, so that when we process a while-loop instruction, we
  // already have information for the while-body.
  for (auto iter = all_computations.begin(); iter != all_computations.end();
       ++iter) {
    TF_RETURN_IF_ERROR(
        CollectP2PGroups(*iter, p2p_in_computation, p2p_group_map));
  }

  // We visit computations in the order of callers to callees, so that the
  // computation containing the while-loop is processed before the while-body.
  for (auto iter = all_computations.rbegin(); iter != all_computations.rend();
       ++iter) {
    HloComputation* computation = *iter;
    // Record the P2P chain result and the corresponding send-done
    // instruction. The unpipelined P2P chain result is the recv-done
    // instruction while the pipelined P2P chain result is the while-loop.
    absl::flat_hash_map<HloInstruction*, HloSendRecvInstruction*>
        p2p_result_to_send_done;
    auto p2p_in_comp = p2p_in_computation.find(computation);
    if (p2p_in_comp == p2p_in_computation.end()) {
      // No P2P operations in the computation, do nothing.
      continue;
    }
    std::set<int64_t>& p2p_channels = p2p_in_comp->second;
    // If the current computation is a while-body and has a pipelined P2P chain,
    // record such a P2P group.
    P2PGroup* pipelined_group = nullptr;
    for (int64_t channel : p2p_channels) {
      auto it = p2p_group_map.find(channel);
      if (it == p2p_group_map.end()) {
        // The instructions that use the channel don't form an interested P2P
        // group, do nothing.
        continue;
      }

      changed = true;

      P2PGroup& p2p_group = it->second;
      P2PGroupKind kind = p2p_group.kind;
      if (kind == P2PGroupKind::kUnpipelined) {
        TF_RETURN_IF_ERROR(
            ChainUnpipelinedP2P(p2p_group.nodes[kUnpipelinedNodeIdx]));
        p2p_result_to_send_done[p2p_group.nodes[kUnpipelinedNodeIdx]
                                    .recv_done] =
            p2p_group.nodes[kUnpipelinedNodeIdx].send_done;
        continue;
      }

      // For Pipelined group.
      if (computation !=
          p2p_group.nodes[kPipelinedParentNodeIdx].parent_computation) {
        // We are at the computation for the while-body of the pipelined group.
        TF_RETURN_IF_ERROR(
            ChainPipelinedP2PChild(p2p_group.nodes[kPipelinedChildNodeIdx]));
        p2p_result_to_send_done[p2p_group.nodes[kPipelinedChildNodeIdx]
                                    .recv_done] =
            p2p_group.nodes[kPipelinedChildNodeIdx].send_done;
        if (pipelined_group != nullptr) {
          return InternalError("Expected <=1 pipelined group in a while-body");
        }
        pipelined_group = &p2p_group;
      } else {
        // We are at the computation that contains the pipelined while-loop. No
        // need to add control dependence as the natural data dependence express
        // this ordering:
        // recv => recv-done => while-loop => send => send-done
        p2p_result_to_send_done[p2p_group.nodes[kPipelinedParentNodeIdx]
                                    .while_loop] =
            p2p_group.nodes[kPipelinedParentNodeIdx].send_done;
      }
    }

    if (pipelined_group != nullptr) {
      int64_t pipelined_channel = pipelined_group->nodes[kPipelinedChildNodeIdx]
                                      .recv_done->channel_id()
                                      .value();
      HloInstruction* pipelined_send_done =
          pipelined_group->nodes[kPipelinedChildNodeIdx].send_done;
      HloInstruction* pipelined_recv =
          pipelined_group->nodes[kPipelinedChildNodeIdx]
              .recv_done->mutable_operand(0);
      for (int64_t channel : p2p_channels) {
        if (channel == pipelined_channel) {
          continue;
        }
        auto it = p2p_group_map.find(channel);
        if (it == p2p_group_map.end()) {
          // The instructions that use the channel don't form an interested P2P
          // group, do nothing.
          continue;
        }

        P2PGroup& other_group = it->second;
        if (other_group.kind != P2PGroupKind::kUnpipelined) {
          return InternalError("Expected an unpipelined group");
        }

        // Add control dependence to make sure the unpipelined Recv is ordered
        // after the pipelined Send-done and the unpipelined Send-done is
        // ordered before the pipelined Recv.
        HloInstruction* other_recv =
            other_group.nodes[kUnpipelinedNodeIdx].recv_done->mutable_operand(
                0);
        HloInstruction* other_send_done =
            other_group.nodes[kUnpipelinedNodeIdx].send_done;
        TF_RETURN_IF_ERROR(
            pipelined_send_done->AddControlDependencyTo(other_recv));
        TF_RETURN_IF_ERROR(
            other_send_done->AddControlDependencyTo(pipelined_recv));
      }
    }

    // If an instruction has nested computation with P2P, we need to enforce
    // its order related to the P2P chains in the current computation.
    std::unique_ptr<HloReachabilityMap> reachability;
    std::vector<HloInstruction*> all_instructions =
        computation->MakeInstructionPostOrder();
    for (auto it = all_instructions.begin(); it != all_instructions.end();
         ++it) {
      HloInstruction* hlo = *it;
      auto p2p_result = p2p_result_to_send_done.find(hlo);
      if (p2p_result == p2p_result_to_send_done.end()) {
        continue;
      }
      if (reachability == nullptr) {
        reachability = HloReachabilityMap::Build(computation);
      }
      HloSendRecvInstruction* send_done = p2p_result->second;
      const HloInstruction* send_data = nullptr;
      if (p2p_result->first->opcode() == HloOpcode::kWhile) {
        // For pipelined P2P, need to exclude send_data from the while-loop
        // users to avoid creating circular dependence.
        send_data = send_done->operand(0)->operand(0);
        VLOG(10) << "Excluding send_data from while-loop users "
                 << send_data->ToString();
      }

      for (HloInstruction* user : hlo->users()) {
        if (user != send_data &&
            OperationChainHasP2P(p2p_in_computation, it, all_instructions.end(),
                                 reachability.get(), user)) {
          // We need to schedule send_done before user to avoid scheduler
          // deadlock.
          TF_RETURN_IF_ERROR(send_done->AddControlDependencyTo(user));
          VLOG(10) << "Add control predecessor from " << user->ToString()
                   << " to " << send_done->ToString();
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
