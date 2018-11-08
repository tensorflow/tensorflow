/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/plugin/poplar/driver/inplace_util.h"

#include "tensorflow/compiler/plugin/poplar/driver/compiler_annotations.h"
#include "tensorflow/compiler/plugin/poplar/kernels/custom_kernels_util.h"

#include "tensorflow/compiler/xla/service/hlo_module.h"

namespace xla {
namespace poplarplugin {
namespace InplaceUtil {
namespace {
bool AllPeersAreNotViewChanging(HloInstruction* inplace,
                                HloInstruction* inplace_parent,
                                const CompilerAnnotations& annotations) {
  // Need to make sure all the peers of inplace instruction are not view
  // changing on this operand (cond 3a).
  for (auto* user : inplace_parent->users()) {
    if (user == inplace) {
      continue;
    }
    auto peer_info = GetHloInstructionDescription(user, annotations);
    if (peer_info->IsViewChangingType(user)) {
      auto view_changing_peer_info =
          static_cast<ViewChangingHloInstructionDescription*>(peer_info.get());
      for (auto idx :
           view_changing_peer_info->GetViewChangingOperandIndexes()) {
        if (user->mutable_operand(idx) == inplace_parent) {
          return false;
        }
      }
    }
  }
  return true;
}

bool IsNotDependencyOfPeers(HloInstruction* inplace,
                            HloInstruction* inplace_parent,
                            HloReachabilityMap* reachability_map,
                            std::vector<HloInstruction*>& added_dependencies) {
  HloComputation* comp = inplace->parent();
  // Verify that inplace is not a dependency of any of the peers (cond 3b).
  for (auto* peer : inplace_parent->users()) {
    if (peer == inplace) {
      continue;
    }

    // If peer is a depenency of inplace, this can't be inplace
    if (reachability_map->IsReachable(inplace, peer)) {
      return false;
    } else {
      // If there already wasn't a control depdenency then insert it
      if (!reachability_map->IsReachable(peer, inplace)) {
        peer->AddControlDependencyTo(inplace);
        comp->UpdateReachabilityThroughInstruction(inplace, reachability_map);
        added_dependencies.push_back(peer);
      }
    }
  }
  return true;
}
}

HloInstructionDescription::HloInstructionDescription() {}
bool HloInstructionDescription::IsInPlaceType(const HloInstruction*) {
  return false;
};
bool HloInstructionDescription::IsViewChangingType(const HloInstruction*) {
  return false;
};

NotInplaceHloInstructionDescription::NotInplaceHloInstructionDescription() {}

InplaceHloInstructionDescription::InplaceHloInstructionDescription() {}
InplaceHloInstructionDescription::InplaceHloInstructionDescription(
    const OperandIndexes& inplace_operand_indexes)
    : inplace_operand_indexes_(std::move(inplace_operand_indexes)) {}

bool InplaceHloInstructionDescription::IsInPlaceType(const HloInstruction*) {
  return true;
}

const OperandIndexes
InplaceHloInstructionDescription::GetInplaceOperandIndexes() const {
  return inplace_operand_indexes_;
}

ViewChangingHloInstructionDescription::ViewChangingHloInstructionDescription(
    const OperandIndexes& view_operand_indexes)
    : view_operand_indexes_(std::move(view_operand_indexes)) {}

bool ViewChangingHloInstructionDescription::IsViewChangingType(
    const HloInstruction*) {
  return true;
}

const OperandIndexes
ViewChangingHloInstructionDescription::GetViewChangingOperandIndexes() const {
  return view_operand_indexes_;
}

GetTupleElementHloInstructionDescription::
    GetTupleElementHloInstructionDescription()
    : ViewChangingHloInstructionDescription(OperandIndexes({0})) {}

bool GetTupleElementHloInstructionDescription::IsViewChangingType(
    const HloInstruction* inst) {
  // Check it is a GTE on a parameter - if it is then it is not view changing.
  const HloInstruction* op = inst->operand(0);
  if (op->opcode() != HloOpcode::kParameter) {
    return true;
  }
  // Check that this is a unique GTE
  for (auto user : op->users()) {
    // The parameter can only be accessed by other GTEs
    if (user->opcode() != HloOpcode::kGetTupleElement) {
      return true;
    }
    if (user != inst && user->tuple_index() == inst->tuple_index()) {
      return true;
    }
  }
  return false;
}

std::unique_ptr<HloInstructionDescription> GetHloInstructionDescription(
    const HloInstruction* inst, const CompilerAnnotations& annotations) {
  switch (inst->opcode()) {
    // Unary Elementwise ops - inplace on operand 0.
    case HloOpcode::kAbs:
    case HloOpcode::kRoundNearestAfz:
    case HloOpcode::kCeil:
    case HloOpcode::kClz:
    case HloOpcode::kConvert:
    case HloOpcode::kBitcastConvert:
    case HloOpcode::kCopy:
    case HloOpcode::kCos:
    case HloOpcode::kExp:
    case HloOpcode::kExpm1:
    case HloOpcode::kFloor:
    case HloOpcode::kImag:
    case HloOpcode::kIsFinite:
    case HloOpcode::kLog:
    case HloOpcode::kLog1p:
    case HloOpcode::kNot:
    case HloOpcode::kNegate:
    case HloOpcode::kReal:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kSign:
    case HloOpcode::kSin:
    case HloOpcode::kTanh:
    // Binary Elementwise ops - inplace on operand 0.
    case HloOpcode::kAdd:
    case HloOpcode::kAtan2:
    case HloOpcode::kComplex:
    case HloOpcode::kDivide:
    case HloOpcode::kEq:
    case HloOpcode::kGe:
    case HloOpcode::kGt:
    case HloOpcode::kLe:
    case HloOpcode::kLt:
    case HloOpcode::kMaximum:
    case HloOpcode::kMinimum:
    case HloOpcode::kMultiply:
    case HloOpcode::kNe:
    case HloOpcode::kPower:
    case HloOpcode::kRemainder:
    case HloOpcode::kSubtract:
    case HloOpcode::kAnd:
    case HloOpcode::kOr:
    case HloOpcode::kXor:
    case HloOpcode::kShiftLeft:
    case HloOpcode::kShiftRightArithmetic:
    case HloOpcode::kShiftRightLogical:
    // These ops are implemented as inplace ops as well.
    case HloOpcode::kBitcast:
    case HloOpcode::kDynamicUpdateSlice: {
      // All of the above ops are inplace on operand 0.
      return absl::make_unique<InplaceHloInstructionDescription>(
          OperandIndexes({0}));
    }
    // Sort is inplace on all operands.
    case HloOpcode::kSort: {
      OperandIndexes indexes(inst->operand_count());
      std::iota(indexes.begin(), indexes.end(), 0);
      return absl::make_unique<InplaceHloInstructionDescription>(indexes);
    }

    // View changing ops.
    // View changing ops on operand 0.
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose:
    case HloOpcode::kBroadcast: {
      return absl::make_unique<ViewChangingHloInstructionDescription>(
          OperandIndexes({0}));
    }
    // View changing on all operands.
    case HloOpcode::kConcatenate:
    case HloOpcode::kTuple:
    case HloOpcode::kMap:
    case HloOpcode::kFusion: {
      OperandIndexes indexes(inst->operand_count());
      std::iota(indexes.begin(), indexes.end(), 0);
      return absl::make_unique<ViewChangingHloInstructionDescription>(indexes);
    }
    // Other view changing.
    case HloOpcode::kPad: {
      // view changing on the first 2 ops.
      return absl::make_unique<ViewChangingHloInstructionDescription>(
          OperandIndexes({0, 1}));
    }

    // Not inplace ops.
    case HloOpcode::kSelect:
    case HloOpcode::kBatchNormGrad:
    case HloOpcode::kBatchNormInference:
    case HloOpcode::kBatchNormTraining:
    case HloOpcode::kClamp:
    case HloOpcode::kTupleSelect:
    case HloOpcode::kRng:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kConvolution:
    case HloOpcode::kDot:
    case HloOpcode::kIota:
    case HloOpcode::kReduce:
    case HloOpcode::kParameter:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kReduceWindow: {
      return absl::make_unique<NotInplaceHloInstructionDescription>();
    }

    // Special cases.
    case HloOpcode::kGetTupleElement: {
      return absl::make_unique<GetTupleElementHloInstructionDescription>();
    }

    case HloOpcode::kCall: {
      // Check if the call is inplace.
      auto it = annotations.inplace_calls.find(inst);
      if (it != annotations.inplace_calls.end()) {
        // If the call is inplace, then get the operands at affected indexes.
        auto inplace_call_description = it->second;
        return absl::make_unique<InplaceHloInstructionDescription>(
            inplace_call_description.GetInplaceOperandIndexes());
      } else {
        // TODO T4848
        return absl::make_unique<NotInplaceHloInstructionDescription>();
      }
    }

    case HloOpcode::kWhile: {
      // TODO T4848
      return absl::make_unique<NotInplaceHloInstructionDescription>();
    }

    case HloOpcode::kCustomCall: {
      if (IPUCustomKernelsUtil::IsPoplibsOp(inst)) {
        // For custom Poplibs Ops, get num_inplace_operands attribute which
        // indicates the following:
        // If num_inplace_operands == 0 then the op is NotInplaceHloInstruction;
        // Else the op is inplace on the first num_inplace_operands operands.
        auto attribute_map = IPUCustomKernelsUtil::AttributeMap(inst);
        auto statusor =
            attribute_map.GetAttributeAsUInt64("num_inplace_operands");
        if (!statusor.ok()) {
          LOG(FATAL) << "Custom Poplibs op " << inst->name()
                     << " is missing \"num_inplace_operands\" attribute.";
        }
        uint64 num_inplace_operands = statusor.ValueOrDie();
        if (num_inplace_operands) {
          OperandIndexes indexes(num_inplace_operands);
          std::iota(indexes.begin(), indexes.end(), 0);
          return absl::make_unique<InplaceHloInstructionDescription>(indexes);
        } else {
          return absl::make_unique<NotInplaceHloInstructionDescription>();
        }
      } else {
        OperandIndexes indexes(inst->operand_count());
        std::iota(indexes.begin(), indexes.end(), 0);
        return absl::make_unique<ViewChangingHloInstructionDescription>(
            indexes);
      }
    }

    // Unimplemented ops.
    case HloOpcode::kAllToAll:
    case HloOpcode::kCollectivePermute:
    case HloOpcode::kCrossReplicaSum:
    case HloOpcode::kDomain:
    case HloOpcode::kFft:
    case HloOpcode::kGather:
    case HloOpcode::kAfterAll:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kScatter:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kTrace:
    default: {
      VLOG(1)
          << "Unrecognized op, consider classifying it as it to inplace ops";
      // For safety, mark the op as view changing on all the operands, meaning
      // that outputs can be never overwritten by an inplace ops.
      OperandIndexes indexes(inst->operand_count());
      std::iota(indexes.begin(), indexes.end(), 0);
      return absl::make_unique<ViewChangingHloInstructionDescription>(indexes);
    }
  }
}

bool IsInPlace(HloInstruction* inst, const CompilerAnnotations& annotations,
               HloReachabilityMap* reachability_map) {
  // An instruction is inplace if:
  // 1. It has an inplace type, and
  // 2. It's inplace operand instructions are not view changing type, and
  // 3. For each inplace operand instruction:
  //   a) The peers (users of the same operands) are not view changing on that
  //      operand.
  //   b) Instruction is not a dependency of peer.
  // TODO the 2nd assumption could probably be relaxed.
  auto info = GetHloInstructionDescription(inst, annotations);

  // Verify it is inplace (cond 1).
  if (!info->IsInPlaceType(inst)) {
    return false;
  }
  const auto& inplace_info =
      *static_cast<InplaceHloInstructionDescription*>(info.get());

  // Keep track of all control dependencies we add.
  std::vector<HloInstruction*> added_dependencies;
  HloComputation* comp = inst->parent();

  bool is_inplace = true;
  // Go trough all the inplace operands.
  for (auto op_idx : inplace_info.GetInplaceOperandIndexes()) {
    HloInstruction* op = inst->mutable_operand(op_idx);
    // Verify they are not view changing (cond 2).
    if (GetHloInstructionDescription(op, annotations)->IsViewChangingType(op)) {
      is_inplace = false;
      break;
    }
    // Need to make sure all the peers of inplace instruction are not view
    // changing on this operand (cond 3a).
    if (!AllPeersAreNotViewChanging(inst, op, annotations)) {
      is_inplace = false;
      break;
    }
    // Verify that inplace is not a dependency of any of the peers (cond 3b).
    if (!IsNotDependencyOfPeers(inst, op, reachability_map,
                                added_dependencies)) {
      is_inplace = false;
      break;
    }
  }

  if (!is_inplace) {
    // If we can't make this op inplace, remove all the dependencies which we
    // have added.
    for (auto* depenency : added_dependencies) {
      depenency->RemoveControlDependencyTo(inst);
    }
    comp->UpdateReachabilityThroughInstruction(inst, reachability_map);
  }
  return is_inplace;
}
}  // namespace InplaceUtil
}  // namespace poplarplugin
}  // namespace xla
