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
bool IsNotDependencyOfPeers(HloInstruction* inplace,
                            HloInstruction* inplace_parent,
                            HloReachabilityMap* reachability_map,
                            std::vector<HloInstruction*>& added_dependencies) {
  HloComputation* comp = inplace->parent();
  for (auto* peer : inplace_parent->users()) {
    if (peer == inplace) {
      continue;
    }
    if (inplace->opcode() == HloOpcode::kGetTupleElement) {
      // Special case for GTE - it's not a dependency if all other users of
      // parent are GTEs and there is no other GTE with the same GTE index.
      if (peer->opcode() != HloOpcode::kGetTupleElement) {
        return false;
      }
      if (peer->tuple_index() == inplace->tuple_index()) {
        return false;
      }
    } else {
      if (reachability_map->IsReachable(inplace, peer)) {
        return false;
      } else {
        // If there already wasn't a control dependency then insert it
        if (!reachability_map->IsReachable(peer, inplace)) {
          peer->AddControlDependencyTo(inplace);
          comp->UpdateReachabilityThroughInstruction(inplace, reachability_map);
          added_dependencies.push_back(peer);
        }
      }
    }
  }
  return true;
}
}  // namespace

HloInstructionDescription::HloInstructionDescription() {}
bool HloInstructionDescription::IsInPlaceType(const HloInstruction*) {
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
    // These ops are implemented as inplace ops on operand 0 as well.
    case HloOpcode::kBitcast:
    case HloOpcode::kBroadcast:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kReshape:
    case HloOpcode::kSlice:
    case HloOpcode::kTranspose: {
      // All of the above ops are inplace on operand 0.
      return absl::make_unique<InplaceHloInstructionDescription>(
          OperandIndexes({0}));
    }
    // Inplace ops on the first 2 ops.
    case HloOpcode::kPad: {
      return absl::make_unique<InplaceHloInstructionDescription>(
          OperandIndexes({0, 1}));
    }

    // Inplace on all operands.
    case HloOpcode::kConcatenate:
    case HloOpcode::kFusion:
    case HloOpcode::kMap:
    case HloOpcode::kTuple:
    case HloOpcode::kSort: {
      OperandIndexes indexes(inst->operand_count());
      std::iota(indexes.begin(), indexes.end(), 0);
      return absl::make_unique<InplaceHloInstructionDescription>(indexes);
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
        return absl::make_unique<InplaceHloInstructionDescription>(indexes);
      }
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
      LOG(FATAL) << "Unrecognized op " << inst->opcode()
                 << ". Classify whether it is an inplace op or not";
      return absl::make_unique<NotInplaceHloInstructionDescription>();
    }
  }
}

bool IsInPlace(HloInstruction* inst, const CompilerAnnotations& annotations,
               HloReachabilityMap* reachability_map) {
  // An instruction is inplace if:
  // 1. It has an inplace type, and
  // 2. For each inplace operand instruction, instruction is not a dependency of
  // peer (users of the same operands).
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
    // Verify that inplace is not a dependency of any of the peers (cond 2).
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
