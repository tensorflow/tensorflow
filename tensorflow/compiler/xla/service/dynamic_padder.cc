/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/xla/service/dynamic_padder.h"

#include <algorithm>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"

#include "absl/container/flat_hash_set.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dynamic_dimension_inference.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"

#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

// ChooseIdentityValue looks at the instruction and returns a identity value
// which, when padded, doesn't change the result of the instruction.
//
// nullopt is returned if padding doesn't need to be reset.
StatusOr<HloInstruction*> ChooseIdentityValue(HloInstruction* inst) {
  HloComputation* comp = inst->parent();
  // Padding on elementwise operation doesn't affect the result of the effective
  // data.
  if (inst->IsElementwise()) {
    return nullptr;
  }

  switch (inst->opcode()) {
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow: {
      // Because of the way we do reduce, we already require the `init` operand
      // of hlo reduce instruction to be identity value. Here we reuse the
      // operand.
      return inst->mutable_operand(1);
    }

    case HloOpcode::kConvolution:
    case HloOpcode::kDot: {
      // Use 0 as padding value for convolution and dot.
      PrimitiveType ptype = inst->shape().element_type();
      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(ptype)));
    }

    case HloOpcode::kPad: {
      return inst->mutable_operand(1);
    }
    case HloOpcode::kParameter:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kReshape:
    case HloOpcode::kTuple:
    case HloOpcode::kAllReduce:
    case HloOpcode::kBroadcast:
      return nullptr;
    default:
      return UnimplementedStrCat("Unimplimented padding for instruction: ",
                                 inst->ToString());
  }
}

}  // namespace

StatusOr<bool> DynamicPadder::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "Pre DynamicPadder HLO:";
  XLA_VLOG_LINES(2, module->ToString());
  TF_ASSIGN_OR_RETURN(DynamicDimensionInference dynamic_dimension_inference,
                      DynamicDimensionInference::Run(module));

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* inst : computation->instructions()) {
      for (int64 operand_num = 0; operand_num < inst->operand_count();
           ++operand_num) {
        HloInstruction* operand = inst->mutable_operand(operand_num);
        if (!operand->shape().IsArray()) {
          continue;
        }
        for (int64 dim = 0; dim < operand->shape().rank(); ++dim) {
          HloInstruction* dynamic_size =
              dynamic_dimension_inference.GetDynamicSize(operand, {}, dim);
          if (dynamic_size == nullptr) {
            continue;
          }
          VLOG(1) << "Has dynamic dimension of operand" << operand_num << " @"
                  << dim;
          TF_ASSIGN_OR_RETURN(HloInstruction * identity_value,
                              ChooseIdentityValue(inst));
          if (identity_value == nullptr) {
            continue;
          }

          // For each dimension, first generates a mask representing the
          // effective area of data and padded area of data using iota and
          // dynamic_size. For example, given a dimension of 7 elements and 5
          // effective elements:
          //
          // iota = [0, 1, 2, 3, 4, 5, 6]
          // broadcast_dynamic_size = [5, 5, 5, 5, 5, 5, 5]
          // mask = lt(iota, broadcast_dynamic_size) = [t, t, t, t, t, f, f]
          //
          // Once the mask is generated, the input data is then padded using the
          // mask and pad value.
          //
          const Shape mask_shape =
              ShapeUtil::ChangeElementType(operand->shape(), xla::U32);
          const Shape pred_shape =
              ShapeUtil::ChangeElementType(operand->shape(), xla::PRED);
          HloInstruction* iota = computation->AddInstruction(
              HloInstruction::CreateIota(mask_shape, dim));

          HloInstruction* broadcasted_effective_size =
              computation->AddInstruction(HloInstruction::CreateBroadcast(
                  mask_shape, dynamic_size, {}));
          HloInstruction* pred = computation->AddInstruction(
              HloInstruction::CreateBinary(pred_shape, HloOpcode::kLt, iota,
                                           broadcasted_effective_size));

          HloInstruction* broadcasted_identity_value =
              computation->AddInstruction(HloInstruction::CreateBroadcast(
                  operand->shape(), identity_value, {}));
          HloInstruction* padded =
              computation->AddInstruction(HloInstruction::CreateTernary(
                  operand->shape(), HloOpcode::kSelect, pred, operand,
                  broadcasted_identity_value));
          TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(operand_num, padded));
          operand = inst->mutable_operand(operand_num);
          changed = true;
        }
      }
    }
  }
  HloDCE dce;
  TF_ASSIGN_OR_RETURN(changed, dce.Run(module));
  VLOG(2) << "Post DynamicPadder HLO:";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace xla
