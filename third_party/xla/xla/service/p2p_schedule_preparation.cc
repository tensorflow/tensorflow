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

#include "xla/service/p2p_schedule_preparation.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_reachability.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
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
  auto p2p = DynCast<HloSendRecvInstruction>(op);
  return p2p != nullptr && !p2p->is_host_transfer();
}

// Returns whether the instruction is a collective operation, for the purpose
// of detecting whether the computation directly invokes collective
// operations. As such, we only need to detect one of the instructions for a
// pair of asynchronous collective operation. We detect the Done op because it
// has a link to the corresponding Start op. We include Send and Recv
// operations, regardless whether they are on hosts or on devices.
bool IsCollectiveOp(const HloInstruction* op) {
  HloOpcode opcode = op->opcode();
  // TODO(b/309639264): We temporarily make this pass to also order custom-calls
  // with respect to P2P chains, to workaround an NVIDIA bug. Remove the code
  // for custom-calls once the bug has been fixed.
  if (opcode == HloOpcode::kCustomCall) {
    return true;
  }

  return hlo_query::IsAsyncCollectiveDoneOp(op, /*include_send_recv=*/true) ||
         (hlo_query::IsCollectiveCommunicationOp(opcode) &&
          !hlo_query::IsAsyncCollectiveStartOp(op, /*include_send_recv=*/true));
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

enum P2PRuntimeStream { kUnknown = 0, kStream0 = 1, kStream1 = 2 };

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

  bool RecordP2POp(HloSendRecvInstruction* p2p) {
    if (!RecordParentComputation(p2p->parent())) {
      return false;
    }

    switch (p2p->opcode()) {
      case HloOpcode::kRecvDone:
        if (recv_done == nullptr) {
          recv_done = Cast<HloRecvDoneInstruction>(p2p);
          return true;
        }
        break;
      case HloOpcode::kSendDone:
        if (send_done == nullptr) {
          send_done = Cast<HloSendDoneInstruction>(p2p);
          return true;
        }
        break;
      case HloOpcode::kRecv:
        if (recv == nullptr) {
          recv = Cast<HloRecvInstruction>(p2p);
          return true;
        }
        break;
      case HloOpcode::kSend:
        if (send == nullptr) {
          send = Cast<HloSendInstruction>(p2p);
          return true;
        }
        break;
      default:
        break;
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
    return recv_done == nullptr || send_done == nullptr || recv == nullptr ||
           send == nullptr;
  }

  bool IncompletePipelinedParent() const {
    return Incomplete() || while_loop == nullptr;
  }

  // Returns the pipeline stream used to execute the P2P instructions in the
  // group.
  P2PRuntimeStream GetRuntimeStream(const HloInstruction* start) const {
    auto it = start->frontend_attributes().map().find(kSendRecvPipelineAttr);
    if (it != start->frontend_attributes().map().end()) {
      if (it->second == "0") {
        return kStream0;
      }
      if (it->second == "1") {
        return kStream1;
      }
    }
    return kUnknown;
  }

  // Finds the pipeline stream from the frontend attribute of the Send/Recv in
  // the pipeline group node, verifies they both have the same value and returns
  // the stream.
  P2PRuntimeStream GetRuntimeStream() const {
    P2PRuntimeStream send_stream = GetRuntimeStream(send);
    P2PRuntimeStream recv_stream = GetRuntimeStream(recv);
    if (send_stream != recv_stream) {
      return kUnknown;
    }
    return send_stream;
  }

  int64_t GetChannel() const { return recv->channel_id().value(); }

  HloRecvDoneInstruction* recv_done = nullptr;
  HloSendDoneInstruction* send_done = nullptr;
  HloRecvInstruction* recv = nullptr;
  HloSendInstruction* send = nullptr;
  // The computation that contains the Send and Recv instructions.
  HloComputation* computation = nullptr;
  // The while-loop instruction that calls the while-body with the pipelined
  // P2P Send and Recv instructions.
  HloInstruction* while_loop = nullptr;
};

//  Maps a channel ID to the corresponding P2P operation group.
struct P2PGroup;
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

// Represents the start and end of a region marked by an ordered P2P instruction
// chain.
using ChainStartEnd =
    std::pair<HloSendRecvInstruction*, HloSendRecvInstruction*>;

static constexpr int kUnpipelinedNodeIdx = 0;
static constexpr int kPipelinedChildNodeIdx = 0;
static constexpr int kPipelinedParentNodeIdx = 1;

// Represent a P2P instruction group for a given channel.
//
// A kUnpipelined P2P group contains only one P2PGroupNode while a kPipelined
// P2P group contains a P2PGroupNode for the while-body and a P2PGroupNode
// for the computation with the while-loop instruction calling the while-body.
// If a group forms a cycle with another group, records the other group as a
// complement group.
struct P2PGroup {
  absl::Status RecordP2POpForUnpipelinedGroup(HloSendRecvInstruction* p2p) {
    if (kind == kUnrecognized) {
      // Leave unrecognized P2P groups alone.
      return absl::OkStatus();
    }
    if (kind != kUnpipelined) {
      return Internal("Expected unpipelined group");
    }
    P2PGroupNode& node = nodes[kUnpipelinedNodeIdx];
    if (!node.RecordP2POp(p2p)) {
      kind = kUnrecognized;
    }
    return absl::OkStatus();
  }

  absl::Status RecordP2POpForPipelinedGroup(HloSendRecvInstruction* p2p) {
    if (kind == kUnrecognized) {
      // Leave unrecognized P2P groups alone.
      return absl::OkStatus();
    }
    if (kind == kUnpipelined) {
      if (nodes[kPipelinedParentNodeIdx].computation != nullptr) {
        return Internal("Expected unpipelined group");
      }
      kind = kPipelined;
    }
    P2PGroupNode& node = nodes[kPipelinedParentNodeIdx];
    if (!node.RecordP2POp(p2p)) {
      kind = kUnrecognized;
    }
    return absl::OkStatus();
  }

  absl::Status RecordWhileOpToPipelinedGroup(HloInstruction* while_op) {
    if (kind == kUnrecognized) {
      // Leave unrecognized P2P groups alone.
      return absl::OkStatus();
    }
    if (kind == kUnpipelined) {
      return Internal("Expected pipelined group");
    }
    P2PGroupNode& node = nodes[kPipelinedParentNodeIdx];
    if (!node.RecordWhileOp(while_op)) {
      kind = kUnrecognized;
    }
    return absl::OkStatus();
  }

  // Finds the pipeline stream from the frontend attribute of the Send/Recv in
  // the pipeline group, verifies they all have the same value and records
  // the stream.
  bool RecordRuntimeStream() {
    P2PRuntimeStream child_stream =
        nodes[kPipelinedChildNodeIdx].GetRuntimeStream();
    if (kind == kPipelined) {
      P2PRuntimeStream parent_stream =
          nodes[kPipelinedParentNodeIdx].GetRuntimeStream();
      if (child_stream != parent_stream || child_stream == kUnknown) {
        return false;
      }
    }
    // Record the stream.
    runtime_stream = child_stream;
    return true;
  }

  // Records the other group that forms a cycle with this group, assuming that
  // we handle only two groups that form a cycle.
  absl::Status RecordComplementGroup(P2PGroupMap& p2p_group_map) {
    CHECK(!complement_group_channel.has_value() && runtime_stream == kStream1);
    for (auto& [channel, p2p_group] : p2p_group_map) {
      if (&p2p_group == this ||
          p2p_group.ChildComputation() != ChildComputation()) {
        continue;
      }
      if (p2p_group.kind == kPipelined &&
          p2p_group.ParentComputation() == ParentComputation()) {
        // Found two pipelined group for the same while loop, verify that they
        // have different valid pipeline stream.
        if (p2p_group.runtime_stream != kStream0) {
          return Internal(
              "Expected different pipeline stream for complement group");
        }
        // Set the complement_group_channel for the current group.
        complement_group_channel = channel;
        // Set the complement_group_channel for the complement-group.
        p2p_group.complement_group_channel = GetChannel();
      } else if (p2p_group.kind == kUnpipelined &&
                 p2p_group.runtime_stream == kStream0) {
        complement_group_channel = channel;
        p2p_group.complement_group_channel = GetChannel();
      }
    }
    return absl::OkStatus();
  }

  // Returns the parent computation assuming this is a kPipelined group.
  HloComputation* ParentComputation() const { return GetParent().computation; }

  // Returns the child computation for the group.
  HloComputation* ChildComputation() const { return GetChild().computation; }

  int64_t GetChannel() const { return nodes[kUnpipelinedNodeIdx].GetChannel(); }
  P2PGroupNode& GetChild() { return nodes[kPipelinedChildNodeIdx]; }
  P2PGroupNode& GetParent() { return nodes[kPipelinedParentNodeIdx]; }
  const P2PGroupNode& GetChild() const { return nodes[kPipelinedChildNodeIdx]; }
  const P2PGroupNode& GetParent() const {
    return nodes[kPipelinedParentNodeIdx];
  }

  // Returns the start and end of a region marked by a pipelined chain in the
  // given computation, which is the region with the pipelined P2P instructions.
  ChainStartEnd GetChainStartEnd(const HloComputation* computation,
                                 const P2PGroupMap& p2p_group_map) const {
    if (computation == ChildComputation()) {
      if (!InCycle()) {
        return std::make_pair(GetChild().recv, GetChild().send_done);
      }
      if (runtime_stream == kStream1) {
        return std::make_pair(
            GetComplementGroup(p2p_group_map)->GetChild().recv,
            GetChild().send_done);
      }
      return std::make_pair(
          GetChild().recv,
          GetComplementGroup(p2p_group_map)->GetChild().send_done);
    }

    CHECK(kind == kPipelined && computation == ParentComputation());
    if (!InCycle()) {
      return std::make_pair(GetParent().recv, GetParent().send_done);
    }
    if (runtime_stream == kStream1) {
      return std::make_pair(GetComplementGroup(p2p_group_map)->GetParent().recv,
                            GetParent().send_done);
    }
    return std::make_pair(
        GetParent().recv,
        GetComplementGroup(p2p_group_map)->GetParent().send_done);
  }

  HloInstruction* GetWhileOp() const {
    return nodes[kPipelinedParentNodeIdx].while_loop;
  }

  bool InCycle() const { return complement_group_channel.has_value(); }
  P2PGroup* GetComplementGroup(P2PGroupMap& p2p_group_map) const {
    CHECK(InCycle());
    return &p2p_group_map.at(*complement_group_channel);
  }
  const P2PGroup* GetComplementGroup(const P2PGroupMap& p2p_group_map) const {
    CHECK(InCycle());
    return &p2p_group_map.at(*complement_group_channel);
  }
  P2PGroupKind kind = kUnpipelined;
  P2PGroupNode nodes[2];
  P2PRuntimeStream runtime_stream = kUnknown;
  // The channel id for another P2PGroup that forms a cycle with this group.
  std::optional<int64_t> complement_group_channel = std::nullopt;
};

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

// If the while-body contains a P2P group that uses the same channel as any
// Send operand of the while-op, we assume these two P2P groups belong to the
// same pipelined P2P sequence. Adds the WhileOp to the pipelined group
// representation in this case.
absl::Status MayAddWhileOpToPipelinedGroup(HloInstruction* while_op,
                                           P2PInComputation& p2p_in_computation,
                                           P2PGroupMap& p2p_group_map) {
  if (while_op->while_init()->opcode() != HloOpcode::kTuple) {
    // A while-init should contain the loop index variable. So if a while-init
    // is not a tuple, it only contains the loop index variable and shouldn't
    // contain any pipelined Send operand.
    return absl::OkStatus();
  }
  HloComputation* body = while_op->called_computations()[0];
  auto p2p_in_while = p2p_in_computation.find(body);
  if (p2p_in_while == p2p_in_computation.end()) {
    return absl::OkStatus();
  }
  int pipelined_group = 0;
  // Check whether the while-op init contains a token from a Send result.
  for (auto hlo : while_op->while_init()->operands()) {
    if (hlo->opcode() != HloOpcode::kSendDone) {
      continue;
    }
    int64_t channel_id = hlo->channel_id().value();
    if (p2p_in_while->second.find(channel_id) == p2p_in_while->second.end()) {
      continue;
    }
    auto group = p2p_group_map.find(channel_id);
    if (group == p2p_group_map.end() || group->second.kind != kPipelined) {
      continue;
    }
    pipelined_group++;
    if (pipelined_group > 2) {
      return Internal(
          "Expecting up to two pipelined P2P groups for each while-loop");
    }
    TF_RETURN_IF_ERROR(group->second.RecordWhileOpToPipelinedGroup(while_op));
  }
  return absl::OkStatus();
}

absl::Status OrderBefore(HloInstruction* i1, HloInstruction* i2) {
  TF_RETURN_IF_ERROR(i1->AddControlDependencyTo(i2));
  VLOG(10) << "Add control predecessor " << i2->ToString();
  return absl::OkStatus();
}

// Adds control dependence to enforce this ordering:
//   recv => send => recv-done => send-done.
absl::Status ConnectP2P1NodeChain(const P2PGroupNode& node) {
  HloRecvDoneInstruction* recv_done = node.recv_done;
  HloRecvInstruction* recv = node.recv;
  HloSendDoneInstruction* send_done = node.send_done;
  HloSendInstruction* send = node.send;
  TF_RETURN_IF_ERROR(OrderBefore(recv, send));
  TF_RETURN_IF_ERROR(OrderBefore(send, recv_done));
  TF_RETURN_IF_ERROR(OrderBefore(recv_done, send_done));
  return absl::OkStatus();
}

// For an unpipelined Send-Recv chain, adds control dependence to enforce this
// ordering:
//   recv => send => recv-done => send-done.
absl::Status ConnectUnpipelinedP2P(const P2PGroup& p2p_group) {
  return ConnectP2P1NodeChain(p2p_group.GetChild());
}

// For a single pipelined Send-Recv chain in a while-body, adds control
// dependence toenforce this ordering:
//   recv => send => recv-done => send-done
absl::Status ConnectPipelined1P2PChild(const P2PGroup& p2p_group) {
  return ConnectP2P1NodeChain(p2p_group.GetChild());
}

// For aSend-Recv chain involving two channels, adds control dependence to
// enforce this ordering:
//   recv.0 => send.0 => recv.1 => send.1 =>
//   recv-done.0 => recv-done.1 => send-done.0 => send-done.1
absl::Status ConnectP2P2NodeChain(const P2PGroupNode& node0,
                                  const P2PGroupNode& node1) {
  HloSendRecvInstruction* recv_done0 = node0.recv_done;
  HloRecvInstruction* recv0 = node0.recv;
  HloSendRecvInstruction* send_done0 = node0.send_done;
  HloSendInstruction* send0 = node0.send;
  HloSendRecvInstruction* recv_done1 = node1.recv_done;
  HloRecvInstruction* recv1 = node1.recv;
  HloSendRecvInstruction* send_done1 = node1.send_done;
  HloSendInstruction* send1 = node1.send;

  TF_RETURN_IF_ERROR(OrderBefore(recv_done0, recv_done1));
  TF_RETURN_IF_ERROR(OrderBefore(recv_done1, send_done0));
  TF_RETURN_IF_ERROR(OrderBefore(send_done0, send_done1));

  TF_RETURN_IF_ERROR(OrderBefore(recv0, send0));
  TF_RETURN_IF_ERROR(OrderBefore(send0, recv1));
  TF_RETURN_IF_ERROR(OrderBefore(recv1, send1));

  TF_RETURN_IF_ERROR(OrderBefore(send1, recv_done0));

  return absl::OkStatus();
}

// For a pipelined Send-Recv chain with two channel groups forming a cycle in a
// while-body computation, we enforce this ordering:
//   recv.0 => send.0 => recv.1 => send.1 =>
//   recv-done.0 => recv-done.1 => send-done.0 => send-done.1
absl::Status ConnectPipelined2P2PChild(const P2PGroup& p2p_group,
                                       const P2PGroupMap& p2p_group_map) {
  return ConnectP2P2NodeChain(
      p2p_group.GetComplementGroup(p2p_group_map)->GetChild(),
      p2p_group.GetChild());
}

// For a pipelined Send-Recv chain with one group in the while-body calling
// computation, we enforce this ordering:
//   recv => send => recv-done => send-done
absl::Status ConnectPipelined1P2PParent(const P2PGroup& p2p_group) {
  return ConnectP2P1NodeChain(p2p_group.GetParent());
}

// For a pipelined Send-Recv chain with two channel groups forming a cycle
// in the while-body calling computation, we enforce this ordering:
//   recv.0 => send.0 => recv.1 => send.1 => =>
//   recv-done.0 => recv-done.1 => send-done.0 => send-done.1
absl::Status ConnectPipelined2P2PParent(const P2PGroup& p2p_group,
                                        const P2PGroupMap& p2p_group_map) {
  return ConnectP2P2NodeChain(
      p2p_group.GetComplementGroup(p2p_group_map)->GetParent(),
      p2p_group.GetParent());
}

// For a Send-Recv chain with two channel groups forming a cycle in a while-body
// annotated for pipelining but not pipelined (due to skip pipelining pass), we
// enforece this ordering:
//   recv.0 => send.0 => recv.1 => send.1 =>
//   recv-done.0 => recv-done.1 => send-done.0 => send-done.1
absl::Status ConnectUnpipelined2P2P(const P2PGroup& p2p_group,
                                    const P2PGroupMap& p2p_group_map) {
  CHECK(p2p_group.runtime_stream == kStream1);
  return ConnectP2P2NodeChain(
      p2p_group.GetComplementGroup(p2p_group_map)->GetChild(),
      p2p_group.GetChild());
}

// Collects P2P send-done and recv-done instructions from the computation,
// groups them by channel IDs, records pipeline decision for groups and connects
// groups that form a cycle for pipelining. Also records whether the computation
// invokes collective operation directly or indirectly.
absl::Status GatherP2PGroupsAndCollectiveInfo(
    const HloComputation* computation, P2PInComputation& p2p_in_computation,
    P2PGroupMap& p2p_group_map,
    CollectiveInComputation& collective_in_computation) {
  collective_in_computation[computation] = false;
  std::vector<HloInstruction*> while_ops;
  for (auto hlo : computation->MakeInstructionPostOrder()) {
    // Record the use of collective operations.
    if (MayInvokeCollectiveOp(hlo, collective_in_computation)) {
      collective_in_computation[computation] = true;
    }

    if (hlo->opcode() == HloOpcode::kWhile) {
      // The pipelined Recv-done/Send-done appears after the while-op. As
      // such, the pipelined group hasn't been constructed at this point.
      // Keep the while-op and add to the pipelined group later.
      while_ops.push_back(hlo);
      continue;
    }
    if (!IsP2POp(hlo)) {
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
      TF_RETURN_IF_ERROR(group.RecordP2POpForUnpipelinedGroup(p2p));
      p2p_group_map[channel] = group;
    } else {
      P2PGroup& group = p2p_group->second;
      if (group.ChildComputation() == computation) {
        TF_RETURN_IF_ERROR(group.RecordP2POpForUnpipelinedGroup(p2p));
      } else {
        // We are at the parent computation for a pipelined P2P group.
        TF_RETURN_IF_ERROR(group.RecordP2POpForPipelinedGroup(p2p));
      }
    }
    // We can't rely on the operation on p2p_group_map above to find out
    // whether it is the first time to handle this channel for the current
    // computation, as we may drop information in the present of kUnrecognized
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

  for (auto hlo : while_ops) {
    TF_RETURN_IF_ERROR(
        MayAddWhileOpToPipelinedGroup(hlo, p2p_in_computation, p2p_group_map));
  }

  // Now finalize each group, in particular, if a kPipelined or kUnpipelined
  // group is missing some instructions, a kPipelined group missing a pipeline
  // stream or have inconsistent pipeline streams, change the group to
  // kUnrecognized.
  for (auto& [channel, p2p_group] : p2p_group_map) {
    if (p2p_group.kind == kUnpipelined) {
      if (p2p_group.nodes[kUnpipelinedNodeIdx].Incomplete() ||
          !p2p_group.RecordRuntimeStream()) {
        p2p_group.kind = kUnrecognized;
      }
    } else if (p2p_group.kind == kPipelined) {
      if (p2p_group.nodes[kPipelinedChildNodeIdx].Incomplete() ||
          p2p_group.nodes[kPipelinedParentNodeIdx]
              .IncompletePipelinedParent() ||
          !p2p_group.RecordRuntimeStream()) {
        p2p_group.kind = kUnrecognized;
      }
    }
  }

  // Erase kUnrecognized groups.
  absl::erase_if(p2p_group_map, [](const auto& p2p_group) {
    return p2p_group.second.kind == kUnrecognized;
  });

  // Connect two groups that form a cycle, both for pipelined and unpipelined
  // cases for the current computation. We only build such a connection when we
  // are processing the group for kStream1 stream, and for parent computation
  // for a pipelined group.
  for (auto& [channel, p2p_group] : p2p_group_map) {
    if ((p2p_group.kind == kPipelined &&
         p2p_group.ParentComputation() != computation) ||
        p2p_group.InCycle() || p2p_group.runtime_stream != kStream1) {
      continue;
    }
    TF_RETURN_IF_ERROR(p2p_group.RecordComplementGroup(p2p_group_map));
  }

  return absl::OkStatus();
}

// For a given computation, adds control dependence to chain a pipelined or
// unpipelined P2P group in the computation. Returns the total number of such
// chains. If the computation is a while-body, verifies that at most one group
// or two groups forming a cycle are pipelined and returns the pipelined group.
absl::StatusOr<std::pair<int, const P2PGroup*>> ConnectP2PChain(
    HloComputation* computation, const P2PGroupMap& p2p_group_map,
    const std::set<int64_t>& p2p_channels) {
  // If the current computation is a while-body and has a pipelined P2P chain,
  // record such a P2P group.
  const P2PGroup* pipelined_group = nullptr;
  int num_p2p_chains = 0;
  for (int64_t channel : p2p_channels) {
    auto it = p2p_group_map.find(channel);
    if (it == p2p_group_map.end()) {
      // The instructions that use the channel don't form an interesting P2P
      // group, do nothing.
      continue;
    }
    num_p2p_chains++;
    const P2PGroup& p2p_group = it->second;
    P2PGroupKind kind = p2p_group.kind;
    if (kind == P2PGroupKind::kUnpipelined) {
      if (!p2p_group.InCycle()) {
        TF_RETURN_IF_ERROR(ConnectUnpipelinedP2P(p2p_group));
      } else if (p2p_group.runtime_stream == kStream1) {
        TF_RETURN_IF_ERROR(ConnectUnpipelined2P2P(p2p_group, p2p_group_map));
      }
      continue;
    }

    if (!p2p_group.InCycle()) {
      if (computation == p2p_group.ParentComputation()) {
        TF_RETURN_IF_ERROR(ConnectPipelined1P2PParent(p2p_group));
      } else {
        // A pipeline of one group.
        if (pipelined_group != nullptr) {
          return Internal("Expected <=1 pipelined group in a while-body");
        }
        pipelined_group = &p2p_group;
        TF_RETURN_IF_ERROR(ConnectPipelined1P2PChild(p2p_group));
      }
      continue;
    }

    // A pipeline of two groups that form a cycle. We process the pipeline when
    // we see the group with kStream1.
    if (p2p_group.runtime_stream != kStream1) {
      continue;
    }

    if (computation == p2p_group.ParentComputation()) {
      TF_RETURN_IF_ERROR(ConnectPipelined2P2PParent(p2p_group, p2p_group_map));
    } else {
      if (pipelined_group != nullptr) {
        return Internal(
            "Expected only two pipelined groups forming a cycle in a "
            "while-body");
      }
      pipelined_group = &p2p_group;
      TF_RETURN_IF_ERROR(ConnectPipelined2P2PChild(p2p_group, p2p_group_map));
    }
  }
  return std::make_pair(num_p2p_chains, pipelined_group);
}

absl::Status OrderBefore(HloReachabilityMap* reachability, HloInstruction* a,
                         HloInstruction* b) {
  VLOG(10) << "OrderBefore " << a->ToString() << " " << b->ToString();
  if (!reachability->IsReachable(a, b)) {
    TF_RETURN_IF_ERROR(a->AddControlDependencyTo(b));
    VLOG(10) << "add control predecessor " << b->ToString();
    reachability->UpdateReachabilityThroughInstruction(b);
  }
  return absl::OkStatus();
}

// Adds control dependence to linearize other collective ops with respect to
// the given P2P chain, which is either an unpipelined P2P chain, or a pipelined
// P2P chain in the while-loop calling computation. The P2P chain can be one of
// the following:
//   Recv => Send => Recv-Done => Send-Done (unpipelined, or pipelined 1)
//   Recv.0 => Send.0 => Recv.1 => Send.1 => Recv-Done.0 => Send-Done.0
//      Recv-Done.1 => Send-Done.1 (pipelined 2)
// We intend to schedule collective ops ordered before the beginning of such a
// chain or after the ending of such a chain.
absl::Status LinearizeCollectivesWithOtherP2P(
    const P2PGroupMap& p2p_group_map, const P2PGroup& group,
    const CollectiveInComputation& collective_in_computation,
    const std::vector<HloInstruction*>::iterator& chain_start_iter,
    const std::vector<HloInstruction*>::iterator& begin_iter,
    const std::vector<HloInstruction*>::iterator& end_iter,
    HloReachabilityMap* reachability) {
  HloComputation* computation = (*chain_start_iter)->parent();
  ChainStartEnd start_end = group.GetChainStartEnd(computation, p2p_group_map);
  // We refer to the P2P chain represented by `group` chain A.
  for (auto it = begin_iter; it != end_iter; ++it) {
    HloInstruction* hlo = *it;
    if (IsP2POp(hlo)) {
      auto group_it = p2p_group_map.find(hlo->channel_id().value());
      if (group_it == p2p_group_map.end()) {
        continue;
      }
      const P2PGroup& cur_group = group_it->second;
      P2PGroupKind kind = cur_group.kind;
      // May linearize chain A with chain B represented by `cur_group`.
      if (kind == P2PGroupKind::kPipelined &&
          computation == cur_group.ChildComputation()) {
        // Chain B a pipelined P2P chain with `computation` as a while-body. We
        // already linearize the two chains in
        // LinearizeCollectivesWithPipelinedP2PChild.
        continue;
      }
      ChainStartEnd cur_start_end =
          cur_group.GetChainStartEnd(computation, p2p_group_map);
      if (cur_start_end.first != hlo) {
        // We will linearize the two chains when we see the first instruction in
        // chain B.
        continue;
      }
      if (it <= chain_start_iter) {
        // We already linearize the two chains when we call this routine for
        // `cur_group`.
        continue;
      }

      if (reachability->IsReachable(start_end.first, cur_start_end.second)) {
        // Order chain A before chain B.
        TF_RETURN_IF_ERROR(
            OrderBefore(reachability, start_end.second, cur_start_end.first));
      } else {
        // Order chain B before chain A.
        TF_RETURN_IF_ERROR(
            OrderBefore(reachability, cur_start_end.second, start_end.first));
      }
      continue;
    }

    if (!MayInvokeCollectiveOp(hlo, collective_in_computation)) {
      continue;
    }
    if (hlo->opcode() == HloOpcode::kWhile &&
        group.kind == P2PGroupKind::kPipelined && group.GetWhileOp() == hlo) {
      // This is the while-op for chain A. No need to add control dependence.
      continue;
    }

    if (hlo_query::IsAsyncCollectiveDoneOp(hlo, /*include_send_recv=*/false)) {
      if (reachability->IsReachable(start_end.first, hlo)) {
        // Order chain A before the async op.
        TF_RETURN_IF_ERROR(OrderBefore(reachability, start_end.second,
                                       GetStartOpForDoneOp(hlo)));
      } else {
        // Order the async op before chain A.
        TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, start_end.first));
      }
    }
    // CustomCall or other op that indirectly invoke collectives.
    if (reachability->IsReachable(start_end.first, hlo)) {
      // Order chain A before the op.
      TF_RETURN_IF_ERROR(OrderBefore(reachability, start_end.second, hlo));
    } else {
      // Order the op before chain A.
      TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, start_end.first));
    }
  }

  return absl::OkStatus();
}

// Adds control dependence to linearize other collective ops with respect to
// the given pipelined P2P chain in the computation for the pipelined
// while-loop. All Collective ops should be scheduled before the chain.
absl::Status LinearizeCollectivesWithPipelinedP2PChild(
    const P2PGroupMap& p2p_group_map, const P2PGroup& group,
    const CollectiveInComputation& collective_in_computation,
    HloComputation* computation, HloReachabilityMap* reachability) {
  ChainStartEnd start_end = group.GetChainStartEnd(computation, p2p_group_map);

  // If an hlo may invoke collective operation, we add control dependence to
  // make sure that the hlo is scheduled before the pipelined chain starts.
  for (HloInstruction* hlo : computation->MakeInstructionPostOrder()) {
    // For async collective ops, only the done version of the op passes this
    // check, to avoid handling async ops twice.
    if (!MayInvokeCollectiveOp(hlo, collective_in_computation)) {
      continue;
    }

    HloOpcode opcode = hlo->opcode();
    // Handle a P2P group when we see its Send-done.
    if (IsP2POp(hlo) && opcode != HloOpcode::kSendDone) {
      continue;
    }
    if (hlo->opcode() == HloOpcode::kSendDone) {
      auto group_it = p2p_group_map.find(hlo->channel_id().value());
      if (group_it == p2p_group_map.end()) {
        continue;
      }
      const P2PGroup& cur_group = group_it->second;
      P2PGroupKind kind = cur_group.kind;
      if (kind == P2PGroupKind::kPipelined &&
          computation == cur_group.ChildComputation()) {
        // This is a P2P group for the pipelined in the current while-body.
        // We are looking for other collective ops outside this group.
        continue;
      }

      ChainStartEnd cur_start_end =
          cur_group.GetChainStartEnd(computation, p2p_group_map);
      TF_RETURN_IF_ERROR(
          OrderBefore(reachability, cur_start_end.second, start_end.first));

      continue;
    }

    // Async done, CustomCall, or other ops that indirectly invoke collectives.
    TF_RETURN_IF_ERROR(OrderBefore(reachability, hlo, start_end.first));
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> P2PSchedulePreparation::Run(
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
    VLOG(10) << "Gathering P2P groups and collective info for computation "
             << (*iter)->name();
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
      // No recognized P2P groups in the computation, do nothing.
      continue;
    }

    std::set<int64_t>& p2p_channels = p2p_in_comp->second;
    // Connect P2P chains and return the number of chains and the P2P group
    // representation for pipelined P2P in the current computation as a
    // while-body.
    TF_ASSIGN_OR_RETURN(
        auto result, ConnectP2PChain(computation, p2p_group_map, p2p_channels));
    if (result.first == 0) {
      continue;
    }

    VLOG(10) << "Processing computation " << computation->name()
             << " num_p2p_chains " << result.first;

    std::unique_ptr<HloReachabilityMap> reachability =
        HloReachabilityMap::Build(computation);
    if (result.second != nullptr) {
      // The current computation is a while-body with pipelined P2P chain.
      // Order all other collectives in a pipelined while-body before the
      // pipelined P2P chain.
      TF_RETURN_IF_ERROR(LinearizeCollectivesWithPipelinedP2PChild(
          p2p_group_map, *result.second, collective_in_computation, computation,
          reachability.get()));
    }

    // Add control dependence to linearize collective operations with respect to
    // other P2P chains.
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
      P2PGroup& group = group_it->second;
      P2PGroupKind kind = group.kind;
      if (kind == P2PGroupKind::kPipelined &&
          computation == group.ChildComputation()) {
        // We already linearize pipelined P2P chains in while-body with respect
        // to other collectives.
        continue;
      }

      ChainStartEnd start_end =
          group.GetChainStartEnd(computation, p2p_group_map);

      // Handle the group when we see the beginning of the chain.
      if (start_end.first != hlo) {
        continue;
      }
      VLOG(10) << "linearize other collectives with respect to channel "
               << hlo->ToString();

      TF_RETURN_IF_ERROR(LinearizeCollectivesWithOtherP2P(
          p2p_group_map, group, collective_in_computation, instr_it, begin, end,
          reachability.get()));
      VLOG(10) << "finish connect other collectives with channel ";
    }
  }

  return true;
}

}  // namespace xla
