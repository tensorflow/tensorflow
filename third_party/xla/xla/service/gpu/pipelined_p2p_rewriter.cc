
/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/service/gpu/pipelined_p2p_rewriter.h"

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/hlo/utils/hlo_query.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {
// Maps a computation to a boolean that indicates whether there is any
// collective operations directly or indirectly invoked in the computation.
using CollectiveInComputation =
    absl::flat_hash_map<const HloComputation*, bool>;

using InstructionVector = HloInstruction::InstructionVector;

// Records starting index and the ending index of a pipelined while-op. They
// are the indices of the while-loop operand.
struct PipelinedP2PInfo {
  int64_t opnd_start;
  int64_t opnd_end;
};

// Returns whether the instruction is a collective operation.
bool IsCollectiveOp(const HloInstruction* op) {
  HloOpcode opcode = op->opcode();
  // TODO(NVIDIA/4364298): The information is recorded in b/309639264.
  // we need to avoid custom-calls to overlap with Send/Recv to workaround the
  // bug. Remove custom-calls here when the bug is fixed.
  if (opcode == HloOpcode::kCustomCall) {
    return true;
  }

  return hlo_query::IsCollectiveCommunicationOp(opcode) ||
         opcode == HloOpcode::kSend || opcode == HloOpcode::kRecv;
}

// Returns whether the instruction may invoke collective operations directly
// or indirectly.
bool MayInvokeCollectiveOp(
    const HloInstruction* hlo,
    const CollectiveInComputation& collective_in_computation) {
  if (IsCollectiveOp(hlo)) {
    return true;
  }
  for (HloComputation* callee : hlo->called_computations()) {
    auto collective_in_comp = collective_in_computation.find(callee);
    CHECK(collective_in_comp != collective_in_computation.end());
    if (collective_in_comp->second) {
      return true;
    }
  }
  return false;
}

// Returns the unique get-tuple-element user with the given idx or nullptr if
// there isn't such a unique user.
HloInstruction* FindUniqueGTEUserWithIndex(const HloInstruction* op,
                                           int64_t idx) {
  CHECK(op->shape().IsTuple());

  HloInstruction* gte = nullptr;
  for (auto user : op->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      continue;
    }
    if (user->tuple_index() == idx) {
      if (gte == nullptr) {
        gte = user;
      } else {
        return nullptr;
      }
    }
  }
  return gte;
}

// Returns whether there is any get-tuple-element user with the given idx.
bool HasGTEUserWithIndex(const HloInstruction* op, int64_t idx) {
  CHECK(op->shape().IsTuple());

  for (auto user : op->users()) {
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      continue;
    }
    if (user->tuple_index() == idx) {
      return true;
    }
  }
  return false;
}

// Returns the instruction hidden behind a trivial tuple or `op`. This allows
// the discovery of recv-done for the following case, for which the indirection
// would have been removed by tuple-simplification.
//    gte.0 = f32[1,1024,1024] get-tuple-element(recv-done), index=0
//    gte.1 = token get-tuple-element(recv-done.p), index=1
//    op = (f32[1,1024,1024], token[]) tuple(gte.0, gte.1)
//
// TODO(bixia): investigate the possible of implementing
// m::TrivialTuple(m::RecvDone(&instr)) as suggested by code review.
HloInstruction* MaySkipTrivialTuple(HloInstruction* op) {
  if (op->opcode() != HloOpcode::kTuple) {
    return op;
  }
  HloInstruction* hidden_op = nullptr;
  for (auto opnd : op->mutable_operands()) {
    if (opnd->opcode() != HloOpcode::kGetTupleElement) {
      return op;
    }
    if (hidden_op == nullptr) {
      hidden_op = opnd->mutable_operand(0);
    } else if (opnd->mutable_operand(0) != hidden_op) {
      return op;
    }
  }
  return hidden_op;
}

// This routine is similar to the non-const version above except that the
// the given instruction is used for pattern checking only and can't be mutated.
const HloInstruction* MaySkipTrivialTuple(const HloInstruction* op) {
  // Use const_cast to avoid repeating the non-const version above to find
  // operands of the instruction through operands() instead of
  // mutable_operands().
  return MaySkipTrivialTuple(const_cast<HloInstruction*>(op));
}

// Finds a consecutive block of balanced SendDone/RecvDone in the while_init
// of a while-loop, assuming its while_init is a tuple.
std::optional<PipelinedP2PInfo>
FindConsecutiveAndBalanceBlockOfSendDoneRecvDone(
    const HloInstruction* while_init) {
  PipelinedP2PInfo pipelined_p2p_info{0, 0};
  // Return whether the first SendDone/RecvDone has been seen.
  auto has_started = [&]() {
    return pipelined_p2p_info.opnd_start != pipelined_p2p_info.opnd_end;
  };
  // Record the difference between the number of SendDone and RecvDone in a
  // consecutive block.
  int difference = 0;
  // If SendDone/RecvDone exists in a consecutive block in the while_init
  // tuple, find such block.
  for (int64_t i = 0; i < while_init->operand_count(); ++i) {
    const HloInstruction* op = while_init->operand(i);
    if ((op->opcode() == HloOpcode::kRecvDone ||
         op->opcode() == HloOpcode::kSendDone) &&
        op->frontend_attributes().map().count(kSendRecvPipelineAttr) > 0) {
      if (op->opcode() == HloOpcode::kRecvDone) {
        difference++;
      } else {
        difference--;
      }
      if (!has_started()) {
        pipelined_p2p_info.opnd_start = i;
      }
      pipelined_p2p_info.opnd_end = i + 1;
    } else {
      if (has_started()) {
        VLOG(10) << "End a consecutive block";
        break;
      }
    }
  }

  if (difference != 0) {
    VLOG(10) << "Mismatch number of SendDone and RecvDone: " << difference;
    return std::nullopt;
  }

  if (has_started()) {
    // Check for SendDone/RecvDone outside the consecutive block.
    for (int64_t i = pipelined_p2p_info.opnd_end;
         i < while_init->operand_count(); ++i) {
      const HloInstruction* op = while_init->operand(i);
      if (op->opcode() == HloOpcode::kRecvDone ||
          op->opcode() == HloOpcode::kSendDone) {
        VLOG(10) << "SendDone/RecvDone outside the consecutive block";
        return std::nullopt;
        break;
      }
    }
  }

  if (!has_started()) {
    VLOG(10) << "No SendDone/RecvDone in while-init ";
    return std::nullopt;
  }

  return pipelined_p2p_info;
}

// Checks whether the while-op, its while-body and while-condition have a
// recognized pipelined pattern. If a pipelined pattern is found, returns the
// first and last indices for the pipelined instruction in the while-init tuple.
// For pipelined Send/Recv to work, the SendDone/RecvDone doesn't have to be in
// a consecutive block, but this simplifies the implementation and is the
// pattern that the current gpu-p2p-pipeliner generated.
//
// As a summary, this is what the routine looks for:
//
// . The while-init has a tuple with a single user.
// . The while-init has a consecutive block of SendDone and RecvDone. The
//   numbers of SendDone and RecvDone are the same, and there isn't any other
//   SendDone and RecvDone outside the block.
// . The while-body has a single tuple parameter.
// . For the while-op result tuple and the while-body parameter tuple:
//     The index corresponding to the index of SendDone in while-init should not
//       correspond to any get-element-tuple user.
//     The index corresponding to the index of RecvDone in while-init should
//       correspond to a single get-element-tuple user.
// . In the while-body result tuple, the operand with an index corresponding to
//   the index in the while-init SendDone and RecvDone should also be a SendDone
//   or RecvDone.
//
// TODO(bixia): support pipelined SendDone/RecvDone not in a consecutive block
// if the gpu-p2p-pipeliner will ever generate such code in the future.
std::optional<PipelinedP2PInfo> FindPipelinedP2P(
    const HloInstruction* while_op) {
  VLOG(10) << "while_op: " << while_op->ToString();
  const HloInstruction* while_init = while_op->while_init();
  if (while_init->opcode() != HloOpcode::kTuple ||
      while_init->user_count() != 1) {
    return std::nullopt;
  }

  // The while-body and while-condition should have one parameter of a tuple
  // shape.
  const HloComputation* while_body = while_op->while_body();
  const HloComputation* while_condition = while_op->while_condition();
  if (while_body->num_parameters() != 1 ||
      while_condition->num_parameters() != 1) {
    return std::nullopt;
  }

  std::optional<PipelinedP2PInfo> pipelined_p2p_info =
      FindConsecutiveAndBalanceBlockOfSendDoneRecvDone(while_init);
  if (!pipelined_p2p_info.has_value()) {
    return std::nullopt;
  }

  VLOG(10) << "opnd_start " << pipelined_p2p_info->opnd_start << " opnd_end "
           << pipelined_p2p_info->opnd_end;

  // In the while-result or while-body parameter, the index for RecvDone should
  // correspond to one get-tuple-element user and the index for SendDone should
  // not correspond to any get-tuple-element user.
  for (int64_t i = pipelined_p2p_info->opnd_start;
       i < pipelined_p2p_info->opnd_end; ++i) {
    const HloInstruction* op = while_init->operand(i);
    if (op->opcode() == HloOpcode::kRecvDone) {
      if (!FindUniqueGTEUserWithIndex(while_op, i)) {
        VLOG(10) << "While result get-tuple-element user with index " << i
                 << " not unique";
        return std::nullopt;
      }
      if (!FindUniqueGTEUserWithIndex(while_body->parameter_instruction(0),
                                      i)) {
        VLOG(10) << "While-body parameter get-tuple-element user with index "
                 << i << " not unique";
        return std::nullopt;
      }
    } else {
      CHECK(op->opcode() == HloOpcode::kSendDone);
      if (HasGTEUserWithIndex(while_op, i) ||
          HasGTEUserWithIndex(while_body->parameter_instruction(0), i)) {
        VLOG(10) << "SendDone with index " << i << " has unexpected users";
        return std::nullopt;
      }
    }
  }

  // The element in the while-body result tuple corresponding to the pipelined
  // SendDone/RecvDone in the while-init have the same opcode.
  const HloInstruction* root = while_body->root_instruction();
  for (int64_t i = pipelined_p2p_info->opnd_start;
       i < pipelined_p2p_info->opnd_end; ++i) {
    const HloInstruction* op_init = while_init->operand(i);
    const HloInstruction* op_root = root->operand(i);
    op_root = MaySkipTrivialTuple(op_root);
    if (op_init->opcode() != op_root->opcode()) {
      VLOG(10) << "Mismatching opcode, op_init: " << op_init->ToString()
               << " op_root: " << op_root->ToString();
      return std::nullopt;
    }
  }

  return pipelined_p2p_info.value();
}

absl::Status RemoveOpFromParent(HloInstruction* op) {
  TF_RETURN_IF_ERROR(op->DropAllControlDeps());
  TF_RETURN_IF_ERROR(op->parent()->RemoveInstruction(op));
  return absl::OkStatus();
}

absl::Status ReplaceOpInSequence(HloInstruction* old_op, HloInstruction* new_op,
                                 HloInstructionSequence& instruction_sequence) {
  VLOG(10) << "old_op: " << old_op->ToString();
  VLOG(10) << "new_op: " << new_op->ToString();
  instruction_sequence.replace_instruction(old_op, new_op);
  return RemoveOpFromParent(old_op);
}

absl::Status ReplaceUsesAndUpdateSequence(
    HloInstruction* old_op, HloInstruction* new_op,
    HloInstructionSequence& instruction_sequence, bool diff_shape = false) {
  VLOG(10) << "old_op: " << old_op->ToString();
  VLOG(10) << "new_op: " << new_op->ToString();
  if (diff_shape) {
    TF_RETURN_IF_ERROR(old_op->ReplaceAllUsesWithDifferentShape(new_op));
  } else {
    TF_RETURN_IF_ERROR(old_op->ReplaceAllUsesWith(new_op));
  }
  return ReplaceOpInSequence(old_op, new_op, instruction_sequence);
}

absl::Status ReplaceUsesAndUpdateSequence(
    const InstructionVector& old_ops, const InstructionVector& new_ops,
    HloInstructionSequence& instruction_sequence) {
  CHECK(old_ops.size() == new_ops.size());
  for (int64_t i = 0; i < old_ops.size(); ++i) {
    TF_RETURN_IF_ERROR(ReplaceUsesAndUpdateSequence(old_ops[i], new_ops[i],
                                                    instruction_sequence));
  }
  return absl::OkStatus();
}

absl::Status RemoveDoneOpsAndUpdateSequence(
    const InstructionVector& ops,
    HloInstructionSequence& instruction_sequence) {
  auto remove_op = [&](HloInstruction* op) {
    VLOG(10) << "op: " << op->ToString();
    TF_RETURN_IF_ERROR(RemoveOpFromParent(op));
    instruction_sequence.remove_instruction(op);
    return absl::OkStatus();
  };
  for (auto op : ops) {
    if (op->opcode() == HloOpcode::kTuple) {
      InstructionVector to_remove;
      HloInstruction* tuple_op = op;
      op = MaySkipTrivialTuple(tuple_op);
      to_remove.push_back(tuple_op);
      for (auto opnd : tuple_op->mutable_operands()) {
        to_remove.push_back(opnd);
      }
      for (auto opnd : to_remove) {
        TF_RETURN_IF_ERROR(remove_op(opnd));
      }
    }
    TF_RETURN_IF_ERROR(remove_op(op));
  }
  return absl::OkStatus();
}

bool InsertBeforeFirstCollectiveOp(
    const InstructionVector& ops,
    const CollectiveInComputation& collective_in_computation,
    HloInstructionSequence& instruction_sequence, int64_t& idx,
    int64_t& idx_tot) {
  bool inserted = false;
  while (idx < idx_tot) {
    HloInstruction* hlo = instruction_sequence.instructions()[idx];
    if (MayInvokeCollectiveOp(hlo, collective_in_computation)) {
      for (auto op : ops) {
        instruction_sequence.insert_instruction(op, idx);
        idx++;
        idx_tot++;
      }
      inserted = true;
      break;
    }
    idx++;
  }
  return inserted;
}

void CopyInstructionInfo(const HloInstruction* old_op, HloInstruction* new_op) {
  new_op->set_metadata(old_op->metadata());
  new_op->add_frontend_attributes(old_op->frontend_attributes());
  new_op->CopyBackendConfigFrom(old_op);
}

HloInstruction* CreateRecvDoneFrom(const HloInstruction* old_recv_done,
                                   HloInstruction* recv,
                                   HloComputation* computation) {
  HloInstruction* recv_done =
      computation->AddInstruction(HloInstruction::CreateRecvDone(
          recv, old_recv_done->channel_id().value()));
  CopyInstructionInfo(old_recv_done, recv_done);
  return recv_done;
}

HloInstruction* CreateSendDoneFrom(const HloInstruction* old_send_done,
                                   HloInstruction* send,
                                   HloComputation* computation) {
  HloInstruction* send_done =
      computation->AddInstruction(HloInstruction::CreateSendDone(
          send, old_send_done->channel_id().value()));
  CopyInstructionInfo(old_send_done, send_done);
  return send_done;
}

absl::Status RewritePipelinedP2PWhileBody(
    const CollectiveInComputation& collective_in_computation,
    const std::vector<Shape>& new_parameter_shapes, HloInstruction* while_op,
    int64_t opnd_start, int64_t opnd_end) {
  HloComputation* computation = while_op->while_body();
  HloInstruction* while_init = while_op->while_init();
  HloInstruction* root = computation->root_instruction();
  HloInstructionSequence& instruction_sequence =
      computation->parent()->schedule().GetOrCreateSequence(computation);

  HloInstruction* param = computation->parameter_instruction(0);
  *param->mutable_shape() = ShapeUtil::MakeTupleShape(new_parameter_shapes);

  InstructionVector recv_dones;
  InstructionVector new_recv_dones;
  InstructionVector new_send_dones;
  for (int64_t i = opnd_start; i < opnd_end; ++i) {
    const HloInstruction* op = root->operand(i);
    op = MaySkipTrivialTuple(op);
    if (op->opcode() == HloOpcode::kRecvDone) {
      HloInstruction* gte = FindUniqueGTEUserWithIndex(param, i);
      CHECK(gte != nullptr);
      recv_dones.push_back(gte);

      // Create the new RecvDone using the new while-body parameter.
      HloInstruction* recv = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(param, i));

      HloInstruction* recv_done = CreateRecvDoneFrom(op, recv, computation);
      new_recv_dones.push_back(recv_done);
      continue;
    }
    CHECK(op->opcode() == HloOpcode::kSendDone);
    //  Create the new SendDone using the new while-op result.
    HloInstruction* send = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(param, i));
    HloInstruction* send_done = CreateSendDoneFrom(op, send, computation);
    new_send_dones.push_back(send_done);
  }
  TF_RETURN_IF_ERROR(ReplaceUsesAndUpdateSequence(recv_dones, new_recv_dones,
                                                  instruction_sequence));

  // Create a new root tuple.
  InstructionVector done_ops;
  InstructionVector new_opnds;
  for (int64_t i = 0; i < while_init->operand_count(); ++i) {
    HloInstruction* op = root->mutable_operand(i);
    if (i >= opnd_start && i < opnd_end) {
      new_opnds.push_back(MaySkipTrivialTuple(op)->mutable_operand(0));
      done_ops.push_back(op);
    } else {
      new_opnds.push_back(op);
    }
  }
  HloInstruction* new_root =
      computation->AddInstruction(HloInstruction::CreateTuple(new_opnds));
  computation->set_root_instruction(new_root,
                                    /*accept_different_shape=*/true);
  TF_RETURN_IF_ERROR(computation->RemoveInstruction(root));
  instruction_sequence.replace_instruction(root, new_root);

  TF_RETURN_IF_ERROR(
      RemoveDoneOpsAndUpdateSequence(done_ops, instruction_sequence));

  // Find a place to put the new SendDone. It will be either the first
  // may-invoke-collective ops that is not in the pipelined Send/Recv chain or
  // the first op in the pipelined Send/Recv chain.
  int64_t idx = 0;
  int64_t idx_end = instruction_sequence.size();
  bool inserted =
      InsertBeforeFirstCollectiveOp(new_send_dones, collective_in_computation,
                                    instruction_sequence, idx, idx_end);
  CHECK(inserted);  // There are Send/Recv in the while-body, expect inserted.
  CHECK(idx_end == instruction_sequence.size());

  TF_RETURN_IF_ERROR(computation->parent()->schedule().Update());
  return absl::OkStatus();
}

void RewritePipelinedP2PWhileCond(
    const std::vector<Shape>& new_parameter_shapes, HloInstruction* while_op) {
  HloComputation* computation = while_op->while_condition();
  HloInstruction* param = computation->parameter_instruction(0);
  *param->mutable_shape() = ShapeUtil::MakeTupleShape(new_parameter_shapes);
  VLOG(10) << computation->ToString();
}

// Rewrites the while-op with a recognized pipelined SendDone/RecvDone pattern
// to pipeline Send/Recv instead.
absl::Status TransformLoop(
    const PipelinedP2PInfo& pipelined_info,
    const CollectiveInComputation& collective_in_computation, int64_t& idx,
    int64_t& idx_end, HloInstructionSequence& instruction_sequence,
    HloInstruction* while_op) {
  HloComputation* computation = while_op->parent();
  int64_t opnd_start = pipelined_info.opnd_start;
  int64_t opnd_end = pipelined_info.opnd_end;
  VLOG(10) << "Transform pipelined while-op " << while_op->ToString();
  HloInstruction* while_init = while_op->while_init();
  InstructionVector new_while_init_opnds;
  std::vector<Shape> new_parameter_shapes;
  for (int64_t i = 0; i < while_init->operand_count(); ++i) {
    HloInstruction* op = while_init->mutable_operand(i);
    if (i >= opnd_start && i < opnd_end) {
      // Get Send/Recv from SendDone/RecvDone.
      new_while_init_opnds.push_back(op->mutable_operand(0));
    } else {
      new_while_init_opnds.push_back(op);
    }
    new_parameter_shapes.push_back(new_while_init_opnds.back()->shape());
  }

  RewritePipelinedP2PWhileCond(new_parameter_shapes, while_op);
  TF_RETURN_IF_ERROR(RewritePipelinedP2PWhileBody(
      collective_in_computation, new_parameter_shapes, while_op, opnd_start,
      opnd_end));
  HloInstruction* new_while_init = computation->AddInstruction(
      HloInstruction::CreateTuple(new_while_init_opnds), "while-init");
  VLOG(10) << "new_while_init: " << new_while_init->ToString();
  HloInstruction* new_while_op = computation->AddInstruction(
      HloInstruction::CreateWhile(
          while_op->while_body()->root_instruction()->shape(),
          while_op->while_condition(), while_op->while_body(), new_while_init),
      "while-result");
  CopyInstructionInfo(while_op, new_while_op);
  VLOG(10) << "new_while_op: " << new_while_op->ToString();

  InstructionVector recv_dones;
  InstructionVector new_recv_dones;
  InstructionVector new_send_dones;
  InstructionVector done_ops;
  for (int64_t i = opnd_start; i < opnd_end; ++i) {
    HloInstruction* op = while_init->mutable_operand(i);
    done_ops.push_back(op);
    if (op->opcode() == HloOpcode::kRecvDone) {
      HloInstruction* gte = FindUniqueGTEUserWithIndex(while_op, i);
      CHECK(gte != nullptr);
      recv_dones.push_back(gte);

      // Create the new RecvDone using the new while-op result.
      HloInstruction* recv = computation->AddInstruction(
          HloInstruction::CreateGetTupleElement(new_while_op, i));
      HloInstruction* recv_done = computation->AddInstruction(
          HloInstruction::CreateRecvDone(recv, op->channel_id().value()));
      new_recv_dones.push_back(recv_done);
      CopyInstructionInfo(op, recv_done);
      continue;
    }
    CHECK(op->opcode() == HloOpcode::kSendDone);
    //  Create the new SendDone using the new while-op result.
    HloInstruction* send = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(new_while_op, i));
    HloInstruction* send_done = computation->AddInstruction(
        HloInstruction::CreateSendDone(send, op->channel_id().value()));
    new_send_dones.push_back(send_done);
    CopyInstructionInfo(op, send_done);
  }

  TF_RETURN_IF_ERROR(ReplaceUsesAndUpdateSequence(
      while_op, new_while_op, instruction_sequence, /*diff_shape*/ true));
  TF_RETURN_IF_ERROR(
      ReplaceOpInSequence(while_init, new_while_init, instruction_sequence));
  TF_RETURN_IF_ERROR(ReplaceUsesAndUpdateSequence(recv_dones, new_recv_dones,
                                                  instruction_sequence));
  TF_RETURN_IF_ERROR(
      RemoveDoneOpsAndUpdateSequence(done_ops, instruction_sequence));

  int64_t opnd_tot = opnd_end - opnd_start;
  // Verify that the numbers of ops we have removed from the sequence is
  // opnd_tot and they are before the position of the new while-op.
  CHECK(idx_end == instruction_sequence.size() + opnd_tot);
  CHECK(instruction_sequence.instructions()[idx - opnd_tot] == new_while_op);

  // Update idx_end to reflect the current size of the instruction sequence.
  // Update idx to right after the new while-op.
  idx_end -= opnd_tot;
  idx = idx - opnd_tot + 1;
  bool inserted =
      InsertBeforeFirstCollectiveOp(new_send_dones, collective_in_computation,
                                    instruction_sequence, idx, idx_end);
  CHECK(idx_end == instruction_sequence.size());
  // If there isn't any may-invoke-collective ops after the while-op, add
  // the new SendDone ops before the last instruction in the sequence.
  if (!inserted) {
    CHECK(idx_end == idx);
    idx--;
    for (auto send_done : new_send_dones) {
      instruction_sequence.insert_instruction(send_done, idx++);
    }
  }
  return absl::OkStatus();
}

// Find while-loop with pipelined Send/Recv and rotates the SendDone/RecvDone
// for such while-loop.
absl::StatusOr<bool> ProcessComputation(
    HloModule* module, HloComputation* computation,
    CollectiveInComputation& collective_in_computation) {
  VLOG(10) << "Process compuation " << computation->name();
  bool changed = false;
  HloInstructionSequence& instruction_sequence =
      module->schedule().GetOrCreateSequence(computation);
  int64_t idx = 0;
  int64_t idx_end = instruction_sequence.size();
  while (idx < idx_end) {
    HloInstruction* hlo = instruction_sequence.instructions()[idx];

    if (MayInvokeCollectiveOp(hlo, collective_in_computation)) {
      collective_in_computation[computation] = true;
    }

    if (hlo->opcode() != HloOpcode::kWhile) {
      idx++;
      continue;
    }

    std::optional<PipelinedP2PInfo> pipelined_info = FindPipelinedP2P(hlo);
    if (!pipelined_info.has_value()) {
      idx++;
      continue;
    }
    TF_RETURN_IF_ERROR(TransformLoop(pipelined_info.value(),
                                     collective_in_computation, idx, idx_end,
                                     instruction_sequence, hlo));
    changed = true;
  }
  return changed;
}
}  // namespace

absl::StatusOr<bool> PipelinedP2PRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  if (!module->has_schedule()) return changed;
  CollectiveInComputation collective_in_computation;
  // Visit the computations in the order of callees to callers, so that
  // while-body is processed before while-op.
  for (auto* computation :
       module->MakeComputationPostOrder(execution_threads)) {
    if (computation->IsFusionComputation()) {
      collective_in_computation[computation] = false;
      continue;
    }

    TF_ASSIGN_OR_RETURN(
        bool cur_changed,
        ProcessComputation(module, computation, collective_in_computation));
    changed |= cur_changed;
  }

  if (changed) {
    TF_RETURN_IF_ERROR(module->schedule().Update());
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
