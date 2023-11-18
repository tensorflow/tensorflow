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
#include "xla/hlo/utils/hlo_query.h"
#include "xla/status.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

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

// Returns whether the instruction is a collective operation, for the purpose
// of detecting whether the computation directly invokes collective
// operations. As such, we only need to detect one of the instructions for a
// pair of asynchronous collective operation. We detect the Done op because it
// has a link to the corresponding Start op. We include Send and Recv
// operations, regardless whether they are on hosts or on devices.
bool IsCollectiveOp(const HloInstruction* op) {
  HloOpcode opcode = op->opcode();
  return hlo_query::IsAsyncCollectiveDoneOp(opcode,
                                            /*include_send_recv=*/true) ||
         (hlo_query::IsCollectiveCommunicationOp(opcode) &&
          !hlo_query::IsAsyncCollectiveStartOp(opcode,
                                               /*include_send_recv=*/true));
}

// Returns the corresponding Done op if the input is a Start op. Otherwise,
// returns the op itself.
HloInstruction* GetStartOpForDoneOp(HloInstruction* op) {
  switch (op->opcode()) {
    case HloOpcode::kAllReduceDone:
    case HloOpcode::kAllGatherDone:
    case HloOpcode::kCollectivePermuteDone:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecvDone:
      return op->mutable_operand(0);
    default:
      return op;
  }
}

enum P2PGroupKind { kUnpipelined = 0, kPipelined = 1, kUnrecognized = 2 };

// A P2P group node represents the P2P instructions that are in the same
// computation and have the same channel ID. This includes one Send/SendDone
// and one Recv/RecvDone. If the P2P instructions for the given channel ID are
// pipelined, the group node for the computation containing the while-loop
// also records the while-loop instruction.
//
struct P2PGroupNode {
  bool RecordParentComputation(HloComputation* parent) {
    if (computation == nullptr) {
      computation = parent;
      return true;
    }
    return computation == parent;
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
  HloComputation* computation = nullptr;
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
      if (nodes[kPipelinedParentNodeIdx].computation != nullptr) {
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

//  Maps a channel ID to the corresponding P2P operation group.
using P2PGroupMap = absl::flat_hash_map<int64_t, P2PGroup>;

// Maps a computation to the channel IDs used by the computation for P2P
// operations. We use std::set instead of hash set for deterministic
// iterators.
using P2PInComputation =
    absl::flat_hash_map<const HloComputation*, std::set<int64_t>>;

// Maps a computation to a boolean that indicates whether the computation
// invokes collective operations directly or indirectly.
using CollectiveInComputation =
    absl::flat_hash_map<const HloComputation*, bool>;

bool MayInvokeCollectiveOp(
    const HloInstruction* hlo,
    const CollectiveInComputation& collective_in_computation) {
  if (IsCollectiveOp(hlo)) {
    return true;
  }
  for (auto callee : hlo->called_computations()) {
    auto collective_in_comp = collective_in_computation.find(callee);
    if (collective_in_comp != collective_in_computation.end() &&
        collective_in_comp->second) {
      return true;
    }
  }
  return false;
}

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
Status ConnectUnpipelinedP2P(const P2PGroupNode& node) {
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
Status ConnectPipelinedP2PChild(const P2PGroupNode& node) {
  HloSendRecvInstruction* recv_done = node.recv_done;
  HloRecvInstruction* recv =
      DynCast<HloRecvInstruction>(recv_done->mutable_operand(0));
  HloSendRecvInstruction* send_done = node.send_done;
  HloSendInstruction* send =
      DynCast<HloSendInstruction>(send_done->mutable_operand(0));
  // We want the Send to be scheduled before the Recv.
  TF_RETURN_IF_ERROR(send->AddControlDependencyTo(recv));
  VLOG(10) << "Add control predecessor " << recv->ToString();
  return OkStatus();
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
// group them by channel IDs. Also records whether the computation invokes
// collective operation directly or indirectly.
Status GatherP2PGroupsAndCollectiveInfo(
    const HloComputation* computation, P2PInComputation& p2p_in_computation,
    P2PGroupMap& p2p_group_map,
    CollectiveInComputation& collective_in_computation) {
  collective_in_computation[computation] = false;
  for (auto hlo : computation->MakeInstructionPostOrder()) {
    // Record the use of collective operations.
    if (IsCollectiveOp(hlo)) {
      collective_in_computation[computation] = true;
    } else {
      // Propagate CollectiveInComputation from callees to callers.
      for (auto callee : hlo->called_computations()) {
        auto collective_in_comp = collective_in_computation.find(callee);
        if (collective_in_comp != collective_in_computation.end()) {
          collective_in_computation[computation] |= collective_in_comp->second;
        }
      }
    }

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
      if (group.nodes[kUnpipelinedNodeIdx].computation == computation) {
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

// For a given computation, adds control dependence to chain the recognized
// pipelined or unpipelined P2P group in the computation. Returns the total
// number of P2P chains and if the computation is a while-body with a pipelined
// P2P group, returns such a group or a nullptr.
StatusOr<int> ConnectP2PChain(HloComputation* computation,
                              const P2PGroupMap& p2p_group_map,
                              const std::set<int64_t>& p2p_channels) {
  // If the current computation is a while-body and has a pipelined P2P chain,
  // record such a P2P group.
  const P2PGroup* pipelined_group = nullptr;
  int num_p2p_chains = 0;
  for (int64_t channel : p2p_channels) {
    auto it = p2p_group_map.find(channel);
    if (it == p2p_group_map.end()) {
      // The instructions that use the channel don't form an interested P2P
      // group, do nothing.
      continue;
    }
    num_p2p_chains++;
    const P2PGroup& p2p_group = it->second;
    P2PGroupKind kind = p2p_group.kind;
    if (kind == P2PGroupKind::kUnpipelined) {
      TF_RETURN_IF_ERROR(
          ConnectUnpipelinedP2P(p2p_group.nodes[kUnpipelinedNodeIdx]));
      continue;
    }

    // For Pipelined group.
    if (computation != p2p_group.nodes[kPipelinedParentNodeIdx].computation) {
      // We are at the computation for the while-body of the pipelined group.
      TF_RETURN_IF_ERROR(
          ConnectPipelinedP2PChild(p2p_group.nodes[kPipelinedChildNodeIdx]));
      if (pipelined_group != nullptr) {
        return InternalError("Expected <=1 pipelined group in a while-body");
      }
      pipelined_group = &p2p_group;
    } else {
    }
  }
  return num_p2p_chains;
}

Status OrderBefore(HloReachabilityMap* reachability, HloInstruction* a,
                   HloInstruction* b) {
  if (!reachability->IsReachable(a, b)) {
    TF_RETURN_IF_ERROR(a->AddControlDependencyTo(b));
    VLOG(10) << "add control predecessor " << b->ToString();
    reachability->UpdateReachabilityThroughInstruction(b);
  }
  return OkStatus();
}

// Adds control dependence to linearize other collective ops with respect to
// the given unpipelined P2P chain which is ordered as follows:
//   Recv => Send => Recv-Done => Send-Done
// We intend to schedule collective ops ordered before Recv-Done before Recv
// and collective ops ordered after Recv-Done after Send-Done.
Status ChainCollectivesWithUnpipelinedP2P(
    const P2PGroupMap& p2p_group_map, const P2PGroupNode& node,
    const std::vector<HloInstruction*>::iterator& begin,
    const std::vector<HloInstruction*>::iterator& recv_done_iter,
    const std::vector<HloInstruction*>::iterator& end,
    const CollectiveInComputation& collective_in_computation,
    HloReachabilityMap* reachability) {
  HloSendRecvInstruction* send_done =
      DynCast<HloSendRecvInstruction>(node.send_done);
  HloInstruction* recv = (*recv_done_iter)->mutable_operand(0);
  auto in_current_p2p_chain = [&](const HloInstruction* hlo) {
    const HloSendRecvInstruction* p2p =
        DynCastOrNull<HloSendRecvInstruction>(hlo);
    return p2p != nullptr && p2p->channel_id() == send_done->channel_id();
  };

  for (auto it = begin; it != end; ++it) {
    HloInstruction* hlo = *it;
    if (!MayInvokeCollectiveOp(hlo, collective_in_computation) ||
        in_current_p2p_chain(hlo)) {
      continue;
    }

    // Handle a P2P chain when we see its Send-Done.
    if (hlo->opcode() == HloOpcode::kRecvDone) {
      continue;
    }
    if (hlo->opcode() == HloOpcode::kSendDone) {
      auto group_it = p2p_group_map.find(hlo->channel_id().value());
      if (group_it == p2p_group_map.end()) {
        LOG(INFO) << "Warn unhandled P2P " << hlo->ToString();
        continue;
      }
      const P2PGroup& p2p_group = group_it->second;
      P2PGroupKind kind = p2p_group.kind;
      if (kind == P2PGroupKind::kPipelined &&
          recv->parent() !=
              p2p_group.nodes[kPipelinedParentNodeIdx].computation) {
        // The pipelined P2P in the "child" is already ordered with respected to
        // other P2P chains.
        continue;
      }
      if (reachability->IsReachable(recv, hlo)) {
        HloInstruction* recv2 = p2p_group
                                    .nodes[kind == P2PGroupKind::kUnpipelined
                                               ? kUnpipelinedNodeIdx
                                               : kPipelinedParentNodeIdx]
                                    .recv_done->mutable_operand(0);
        TF_RETURN_IF_ERROR(OrderBefore(reachability, send_done, recv2));
      } else {
        TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, recv));
      }
      continue;
    }

    // The hlo is not a Send/Recv instruction.
    if (reachability->IsReachable(hlo, send_done)) {
      TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, recv));
    } else {
      TF_RETURN_IF_ERROR(
          OrderBefore(reachability, send_done, GetStartOpForDoneOp(hlo)));
    }
  }

  return OkStatus();
}

// Adds control dependence to linearize other collective ops with respect to
// the given pipelined P2P chain in the computation containing the pipelined
// while-loop, which is ordered as follows:
//   Recv => Recv-Done => While-loop => Send => SendDone
// We intend to schedule collective ops ordered before the while-loop before
// Recv and collective ops ordered after the while-loop after Send-Done.
Status ChainCollectivesWithPipelinedP2PParent(
    const P2PGroupMap& p2p_group_map, const P2PGroupNode& node,
    const std::vector<HloInstruction*>::iterator& begin,
    const std::vector<HloInstruction*>::iterator& while_loop_iter,
    const std::vector<HloInstruction*>::iterator& end,
    const CollectiveInComputation& collective_in_computation,
    HloReachabilityMap* reachability) {
  HloInstruction* recv = node.recv_done->mutable_operand(0);
  HloSendRecvInstruction* send_done =
      DynCast<HloSendRecvInstruction>(node.send_done);
  auto in_current_p2p_chain = [&](const HloInstruction* hlo) {
    if (hlo->opcode() == HloOpcode::kWhile) {
      return node.while_loop == hlo;
    }
    const HloSendRecvInstruction* p2p =
        DynCastOrNull<HloSendRecvInstruction>(hlo);
    return p2p != nullptr && p2p->channel_id() == send_done->channel_id();
  };

  for (auto it = begin; it != end; ++it) {
    HloInstruction* hlo = *it;
    if (!MayInvokeCollectiveOp(hlo, collective_in_computation) ||
        in_current_p2p_chain(hlo)) {
      continue;
    }

    // Handle a P2P chain when we see its Send-done.
    if (hlo->opcode() == HloOpcode::kRecvDone) {
      continue;
    }
    if (hlo->opcode() == HloOpcode::kSendDone) {
      auto group_it = p2p_group_map.find(hlo->channel_id().value());
      if (group_it == p2p_group_map.end()) {
        LOG(INFO) << "Warn unhandled P2P " << hlo->ToString();
        continue;
      }
      const P2PGroup& p2p_group = group_it->second;
      P2PGroupKind kind = p2p_group.kind;
      if (kind == P2PGroupKind::kPipelined &&
          recv->parent() !=
              p2p_group.nodes[kPipelinedParentNodeIdx].computation) {
        // The pipelined P2P in the "child" is already ordered with respected to
        // other P2P chains.
        continue;
      }
      if (reachability->IsReachable(recv, hlo)) {
        HloInstruction* recv2 = p2p_group
                                    .nodes[kind == P2PGroupKind::kUnpipelined
                                               ? kUnpipelinedNodeIdx
                                               : kPipelinedParentNodeIdx]
                                    .recv_done->mutable_operand(0);
        TF_RETURN_IF_ERROR(OrderBefore(reachability, send_done, recv2));
      } else {
        TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, recv));
      }
      continue;
    }

    // The hlo is not a Send/Recv instruction.
    if (reachability->IsReachable(hlo, send_done)) {
      TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, recv));
    } else {
      TF_RETURN_IF_ERROR(
          OrderBefore(reachability, send_done, GetStartOpForDoneOp(hlo)));
    }
  }

  return OkStatus();
}

// Adds control dependence to linearize other collective ops with respect to
// the given pipelined P2P chain in the computation for the pipelined
// while-loop, which is ordered as follows:
//   Send => Send-Done
//   Recv => Recv-Done
//   Send => Recv
// All collective ops should be scheduled after Send-Done and Before
Status ChainCollectivesWithPipelinedP2PChild(
    const P2PGroupMap& p2p_group_map, const P2PGroupNode& node,
    const std::vector<HloInstruction*>::iterator& begin,
    const std::vector<HloInstruction*>::iterator& end,
    const CollectiveInComputation& collective_in_computation,
    HloReachabilityMap* reachability) {
  HloInstruction* send_done = node.send_done;
  HloSendRecvInstruction* recv =
      DynCast<HloSendRecvInstruction>(node.recv_done->mutable_operand(0));
  auto in_current_p2p_chain = [&](const HloInstruction* hlo) {
    const HloSendRecvInstruction* p2p =
        DynCastOrNull<HloSendRecvInstruction>(hlo);
    return p2p != nullptr && p2p->channel_id() == recv->channel_id();
  };

  // If an hlo may invoke collective operation and is ordered before the
  // Send, checks that it is not reachable to Send-Done and adds control
  // dependence to make sure it is scheduled after Send-Done and before Recv.
  for (auto it = begin; it != end; ++it) {
    HloInstruction* hlo = *it;
    if (!MayInvokeCollectiveOp(hlo, collective_in_computation) ||
        in_current_p2p_chain(hlo)) {
      continue;
    }
    if (reachability->IsReachable(hlo, send_done) ||
        reachability->IsReachable(recv, hlo)) {
      return InternalError("Detect deadlock in input HLO");
    }

    // Handle a P2P chain when we see its Send-done.
    if (hlo->opcode() == HloOpcode::kRecvDone) {
      continue;
    }
    if (hlo->opcode() == HloOpcode::kSendDone) {
      auto group_it = p2p_group_map.find(hlo->channel_id().value());
      if (group_it == p2p_group_map.end()) {
        continue;
      }
      const P2PGroup& p2p_group = group_it->second;
      P2PGroupKind kind = p2p_group.kind;
      if (kind == P2PGroupKind::kPipelined &&
          recv->parent() !=
              p2p_group.nodes[kPipelinedParentNodeIdx].computation) {
        // The pipelined P2P in the "child" is already ordered with respected to
        // other P2P chains.
        continue;
      }

      HloInstruction* recv2 = p2p_group
                                  .nodes[kind == P2PGroupKind::kUnpipelined
                                             ? kUnpipelinedNodeIdx
                                             : kPipelinedParentNodeIdx]
                                  .recv_done->mutable_operand(0);
      TF_RETURN_IF_ERROR(OrderBefore(reachability, send_done, recv2));
      TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, recv));

      continue;
    }

    // The hlo is not a Send/Recv instruction.
    TF_RETURN_IF_ERROR(
        OrderBefore(reachability, send_done, GetStartOpForDoneOp(hlo)));
    TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, recv));
  }

  return OkStatus();
}

}  // namespace

StatusOr<bool> P2PSchedulePreparation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  P2PGroupMap p2p_group_map;
  P2PInComputation p2p_in_computation;
  CollectiveInComputation collective_in_computation;

  std::vector<HloComputation*> all_computations =
      module->MakeComputationPostOrder(execution_threads);

  // Gather P2P groups as well as whether computation invoke collective
  // operations directly or indirectly. We visit computations in the order of
  // callees to callers, so that when we process a while-loop instruction, we
  // already have information for the while-body.
  for (auto iter = all_computations.begin(); iter != all_computations.end();
       ++iter) {
    TF_RETURN_IF_ERROR(GatherP2PGroupsAndCollectiveInfo(
        *iter, p2p_in_computation, p2p_group_map, collective_in_computation));
  }

  if (p2p_group_map.empty()) {
    return false;
  }

  // Chain P2P groups and make sure that other collectives are either scheduled
  // before or after P2P groups.
  //
  // We visit computations in the order of callers to callees, so that the
  // computation containing the while-loop is processed before the while-body.
  for (auto iter = all_computations.rbegin(); iter != all_computations.rend();
       ++iter) {
    HloComputation* computation = *iter;
    auto p2p_in_comp = p2p_in_computation.find(computation);
    if (p2p_in_comp == p2p_in_computation.end()) {
      // No recognized P2P chains in the computation, do nothing.
      continue;
    }

    std::set<int64_t>& p2p_channels = p2p_in_comp->second;
    TF_ASSIGN_OR_RETURN(
        int num_p2p_chains,
        ConnectP2PChain(computation, p2p_group_map, p2p_channels));
    if (num_p2p_chains == 0) {
      continue;
    }

    VLOG(10) << "processing computation " << computation->name()
             << " num_p2p_chains " << num_p2p_chains;
    // Add control dependence to linearize collective operations with respect to
    // each P2P chain.
    std::unique_ptr<HloReachabilityMap> reachability;
    std::vector<HloInstruction*> all_instructions =
        computation->MakeInstructionPostOrder();
    std::vector<HloInstruction*>::iterator begin = all_instructions.begin();
    std::vector<HloInstruction*>::iterator end = all_instructions.end();
    for (auto instr_it = begin; instr_it != end; ++instr_it) {
      HloInstruction* hlo = *instr_it;
      if (!IsP2POp(hlo)) {
        continue;
      }
      HloSendRecvInstruction* p2p = Cast<HloSendRecvInstruction>(hlo);
      int64_t channel = p2p->channel_id().value();
      auto group_it = p2p_group_map.find(channel);
      if (group_it == p2p_group_map.end()) {
        continue;
      }

      if (reachability == nullptr) {
        reachability = HloReachabilityMap::Build(computation);
      }
      P2PGroup& p2p_group = group_it->second;
      P2PGroupKind kind = p2p_group.kind;
      VLOG(10) << "connect other collectives with channel " << channel
               << " kind " << (int)kind;

      if (kind == P2PGroupKind::kUnpipelined &&
          hlo->opcode() == HloOpcode::kRecvDone) {
        // Case 1: Unpipelined P2P chain
        //   Send => Recv => Recv-Done => Send-Done
        // We intend to schedule collective ops ordered before Recv-Done before
        // Send and collective ops ordered after Recv-Done after Send-Done.
        TF_RETURN_IF_ERROR(ChainCollectivesWithUnpipelinedP2P(
            p2p_group_map, p2p_group.nodes[kUnpipelinedNodeIdx], begin,
            instr_it, end, collective_in_computation, reachability.get()));
      } else if (kind == P2PGroupKind::kPipelined &&
                 computation ==
                     p2p_group.nodes[kPipelinedParentNodeIdx].computation &&
                 hlo->opcode() == HloOpcode::kRecvDone) {
        // Case 2: Pipelined P2P chain in the "parent", that is, the computation
        //   containing the while-loop:
        //   Recv => Recv-Done => While-loop => Send => SendDone
        // We intend to schedule collective ops ordered before the while-loop
        // before Recv and collective ops ordered after the while-loop after
        // Send-Done.
        const HloInstruction* while_loop =
            p2p_group.nodes[kPipelinedParentNodeIdx].while_loop;
        std::vector<HloInstruction*>::iterator while_loop_it = instr_it + 1;
        while ((*while_loop_it) != while_loop) while_loop_it++;
        TF_RETURN_IF_ERROR(ChainCollectivesWithPipelinedP2PParent(
            p2p_group_map, p2p_group.nodes[kPipelinedParentNodeIdx], begin,
            while_loop_it, end, collective_in_computation, reachability.get()));
      } else if (kind == P2PGroupKind::kPipelined &&
                 computation !=
                     p2p_group.nodes[kPipelinedParentNodeIdx].computation &&
                 hlo->opcode() == HloOpcode::kSend) {
        // Case 3: Pipelined P2P chain in the "child", that is, the computation
        // for the while-body:
        //   Send => Send-Done. ... Recv => Recv-Done
        // All collective ops should be scheduled after Send-Done and Before
        // Recv.
        TF_RETURN_IF_ERROR(ChainCollectivesWithPipelinedP2PChild(
            p2p_group_map, p2p_group.nodes[kPipelinedChildNodeIdx], begin, end,
            collective_in_computation, reachability.get()));
      }
      VLOG(10) << "finish connect other collectives with channel " << channel
               << " kind " << (int)kind;
    }
  }
  return true;
}

}  // namespace xla
