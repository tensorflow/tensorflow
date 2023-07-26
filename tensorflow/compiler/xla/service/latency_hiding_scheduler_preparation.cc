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

#include "tensorflow/compiler/xla/service/latency_hiding_scheduler_preparation.h"

#include <memory>
#include <vector>

#include "absl/log/log.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instructions.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_reachability.h"

namespace xla {

namespace {

// Returns a boolean to indicate whether the operation is a non-host P2P
// operation. We exclude non-host P2P operations for two reasons: (1) this pass
// currently can only amend control dependence for non-host P2P operations. (2)
// we need to exclude host P2P operations when looking for a nested chain
// of non-host P2P operations.
bool IsP2POp(const HloInstruction* op) {
  auto p2p = DynCastOrNull<HloSendRecvInstruction>(op);
  return p2p != nullptr && !p2p->is_host_transfer();
}

// Returns the predecessor of op with instruction type T1 or nullptr if such a
// control predecessor doesn't exist. The routine gives an error if there are
// more than one such control predecessor.
template <typename T1>
const T1* GetChainedOp(const HloInstruction* op) {
  const T1* chained_op = nullptr;
  for (const HloInstruction* predecessor : op->control_predecessors()) {
    auto tmp = DynCastOrNull<T1>(predecessor);
    if (!tmp || !IsP2POp(tmp)) {
      continue;
    }
    CHECK_EQ(chained_op, nullptr);
    chained_op = tmp;
  }
  return chained_op;
}

// Given a send_done, returns the recv_done if it is in a chain of
//   Recv => Send => RecvDone => SendDone
// Returns nullptr if such a chain doesn't exist.
const HloRecvDoneInstruction* GetChainedRecvDone(
    const HloSendDoneInstruction* send_done) {
  const HloRecvDoneInstruction* recv_done =
      GetChainedOp<HloRecvDoneInstruction>(send_done);
  if (!recv_done) {
    return nullptr;
  }

  auto send = DynCast<HloSendInstruction>(send_done->operand(0));
  CHECK_NE(send, nullptr);
  CHECK_EQ(send->is_host_transfer(), false);

  const HloRecvInstruction* recv = GetChainedOp<HloRecvInstruction>(send);
  if (!recv) {
    return nullptr;
  }
  if (recv_done->operand(0) != recv) {
    return nullptr;
  }

  return recv_done;
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
// [start, end] that contains non-host P2P transfer that are reachable from the
// given instruction.
bool OperationChainHasP2P(
    absl::flat_hash_map<const HloComputation*, bool>& p2p_in_computation_cache,
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
      if (FindP2PInComputation(p2p_in_computation_cache, called_comp)) {
        return true;
      }
    }
  }
  return false;
}

}  // namespace

StatusOr<bool> LatencyHidingSchedulerPreparation::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  absl::flat_hash_map<const HloComputation*, bool> p2p_in_computation_cache;
  for (HloComputation* computation : module->computations(execution_threads)) {
    std::unique_ptr<HloReachabilityMap> reachability;
    std::vector<HloInstruction*> all_instructions =
        computation->MakeInstructionPostOrder();
    for (auto it = all_instructions.begin(); it != all_instructions.end();
         ++it) {
      HloInstruction* hlo = *it;
      if (hlo->opcode() != HloOpcode::kSendDone) {
        continue;
      }
      auto send_done = Cast<HloSendDoneInstruction>(hlo);
      if (send_done->is_host_transfer()) {
        continue;
      }
      const HloRecvDoneInstruction* recv_done = GetChainedRecvDone(send_done);
      if (!recv_done) {
        continue;
      }
      if (reachability == nullptr) {
        reachability = HloReachabilityMap::Build(computation);
      }
      for (HloInstruction* recv_data : recv_done->users()) {
        if (OperationChainHasP2P(p2p_in_computation_cache, it,
                                 all_instructions.end(), reachability.get(),
                                 recv_data)) {
          // We need to schedule send_done before recv_data to avoid deadlock.
          TF_RETURN_IF_ERROR(send_done->AddControlDependencyTo(recv_data));
          VLOG(10) << "Add control predecessor to " << recv_data->ToString();
          changed = true;
        }
      }
    }
  }
  return changed;
}

}  // namespace xla
