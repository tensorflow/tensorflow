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
#include "absl/strings/str_format.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/comparison_util.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/dynamic_dimension_inference.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_dce.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

namespace {

// ChooseIdentityValue looks at the instruction's operand, returns a
// identity value which, when padded, doesn't change the result of the
// instruction.
//
// nullopt is returned if padding doesn't need to be reset.
StatusOr<HloInstruction*> ChooseIdentityValue(HloInstruction* inst,
                                              int64 operand_number) {
  HloComputation* comp = inst->parent();
  // Padding on elementwise operation doesn't affect the result of the effective
  // data.
  if (inst->IsElementwise()) {
    return nullptr;
  }

  switch (inst->opcode()) {
    case HloOpcode::kReduce: {
      TF_RET_CHECK(operand_number < inst->operand_count() / 2)
          << "Only data operand with dynamic dimension is valid.";
      // Variadic reduce has different init value for different operand, given a
      // data operand number, find the init value index.
      int64 init_value_index = inst->operand_count() / 2 + operand_number;
      return inst->mutable_operand(init_value_index);
    }
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

    case HloOpcode::kSelectAndScatter: {
      return inst->mutable_operand(2);
    }
    case HloOpcode::kScatter: {
      if (operand_number != 1) {
        return nullptr;
      }
      PrimitiveType indices_ptype =
          inst->operand(operand_number)->shape().element_type();

      return comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::MaxValue(indices_ptype)));
    }
    case HloOpcode::kParameter:
    case HloOpcode::kGather:
    case HloOpcode::kDynamicSlice:
    case HloOpcode::kDynamicUpdateSlice:
    case HloOpcode::kGetDimensionSize:
    case HloOpcode::kSetDimensionSize:
    case HloOpcode::kConcatenate:
    case HloOpcode::kReshape:
    case HloOpcode::kReverse:
    case HloOpcode::kTuple:
    case HloOpcode::kAllReduce:
    case HloOpcode::kBroadcast:
    case HloOpcode::kTranspose:
    case HloOpcode::kSort:
    case HloOpcode::kSlice:
      return nullptr;
    // Assume that custom calls created by the client are valid with padded
    // dynamic dimensions.
    case HloOpcode::kCustomCall:
      return nullptr;
    default:
      return UnimplementedStrCat("Unimplemented padding for instruction: ",
                                 inst->ToString());
  }
}

bool ShouldSkipPadOnOperand(const HloInstruction* inst, int64 operand_num,
                            int64 dimension) {
  if ((inst->opcode() == HloOpcode::kReduceWindow ||
       inst->opcode() == HloOpcode::kSelectAndScatter) &&
      operand_num == 0 && inst->window().dimensions(dimension).size() == 1) {
    return true;
  }

  if (operand_num == 0 && inst->opcode() == HloOpcode::kConvolution &&
      inst->convolution_dimension_numbers().input_batch_dimension() ==
          dimension) {
    return true;
  }
  return false;
}

// Generates a mask representing the effective area of data and padded area of
// data using iota and dynamic_size. For example, given a dimension of 7
// elements and 5 effective elements:
//
// iota = [0, 1, 2, 3, 4, 5, 6]
// broadcast_dynamic_size = [5, 5, 5, 5, 5, 5, 5]
// mask = lt(iota, broadcast_dynamic_size) = [t, t, t, t, t, f, f]
//
// Once the mask is generated, the input data is then padded using the
// mask and pad value.
//
HloInstruction* PadWithScalar(HloInstruction* inst, int64 dim,
                              HloInstruction* dynamic_size,
                              HloInstruction* padding_scalar) {
  const Shape mask_shape =
      ShapeUtil::ChangeElementType(inst->shape(), xla::S32);
  const Shape pred_shape =
      ShapeUtil::ChangeElementType(inst->shape(), xla::PRED);
  HloComputation* computation = inst->parent();
  HloInstruction* iota =
      computation->AddInstruction(HloInstruction::CreateIota(mask_shape, dim));

  HloInstruction* broadcasted_effective_size = computation->AddInstruction(
      HloInstruction::CreateBroadcast(mask_shape, dynamic_size, {}));
  HloInstruction* pred =
      computation->AddInstruction(HloInstruction::CreateCompare(
          pred_shape, iota, broadcasted_effective_size,
          ComparisonDirection::kLt));

  HloInstruction* broadcasted_identity_value = computation->AddInstruction(
      HloInstruction::CreateBroadcast(inst->shape(), padding_scalar, {}));
  HloInstruction* padded = computation->AddInstruction(
      HloInstruction::CreateTernary(inst->shape(), HloOpcode::kSelect, pred,
                                    inst, broadcasted_identity_value));
  return padded;
}

// In a reshape if a dynamic dimension is splitted into multiple output
// dimensions, we need to rewrite the input of the reshape.
//
// The reason for this is that a continuous input may not be evenly reshaped
// into output.  Image we have [<=6] where valid data has size 4 and padding (P)
// data has size 2: [a,b,c,d,P,P]
//
// And we have a reshape that produces dynamic output dimensions.
//
// [<=6]
//  |
// Reshape
//  |
// [2, <=3]
//
// This should produce the same result as if the data has no padding:
//
// [4]     // [a, b, c, d]
//  |
// Reshape
//  |
// [2, 2]  // [[a,b], [c,d]]
//
// Without reshape rewriting, the result looks like:
//
// [[a,b,c]
//  [d,P,P]], which is incorrect.
//
// We need to rewrite the reshape such that it produces:
// [[a,b,P]
//  [c,d,P]]
//
// The way we do this is by a 6-steps double-sorting algorithm:
//
// 1.First we use the output shape to generate a binary 0-1 masking, which masks
// out the padded area of the output:
// [[0,0,1]
//  [0,0,1]]
//
// 2.Then we do an inverse reshape to reshape it from output shape back to input
// shape [2,3]->[6]:
//  [0,0,1,0,0,1]
//
// 3.We then generate an iota mask using the input shape:
//  [0,1,2,3,4,5]
//
// 4.Stable sort the iota mask using the binary mask as key:
//  key  [0,0,1,0,0,1]
//  value[0,1,2,3,4,5]
//     | Sort by key
//     v
//  key  [0,0,0,0,1,1]
//  value[0,1,3,4,2,5]
//
// 5.Sort the original input [a,b,c,d,P,P] using the sorted iota mask:
//  key  [0,1,3,4,2,5]
//  value[a,b,c,d,P,P]
//     | Sort by key
//     v
//  key  [0,1,2,3,4,5]
//  value[a,b,P,c,d,P]
//
// 6.Feed the sorted input to original reshape[6]->[2,3], we can get the correct
// reshape:
//  [[a,b,P]
//   [c,d,P]]
//
Status RewriteDynamicReshapeSplitInput(
    HloInstruction* reshape, int64 input_dim,
    absl::Span<const int64> output_dims,
    DynamicDimensionInference* dynamic_dimension_inference) {
  const Shape operand_shape = reshape->operand(0)->shape();
  TF_RET_CHECK(output_dims.size() > 1);

  HloComputation* comp = reshape->parent();
  const Shape mask_input_shape =
      ShapeUtil::ChangeElementType(operand_shape, xla::S32);
  const Shape mask_reshaped_shape =
      ShapeUtil::ChangeElementType(reshape->shape(), xla::S32);

  HloInstruction* zero = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
  HloInstruction* one = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::One(S32)));
  // Step 1 -- generate binary mask.
  // Mask starts with all zero, each dynamic dimension sets one dimension of the
  // mask to partially one.
  HloInstruction* binary_mask = comp->AddInstruction(
      HloInstruction::CreateBroadcast(mask_reshaped_shape, zero, {}));

  bool need_rewrite = false;

  // Index starts from 1 since there is no need to rewrite a major output
  // dimension.
  for (int64 i = 1; i < output_dims.size(); ++i) {
    const int64 output_dim = output_dims[i];
    HloInstruction* dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(reshape, {}, output_dim);
    if (dynamic_size == nullptr) {
      continue;
    }
    // If there is dynamic dimension in the output, need rewrite the input.
    need_rewrite = true;

    binary_mask = PadWithScalar(binary_mask, output_dim, dynamic_size, one);
  }
  if (!need_rewrite) {
    return Status::OK();
  }
  // Step 2.
  // Do a reverse reshape to flatten the binary mask (with output shape) back to
  // input shape.
  HloInstruction* input_shape_binary_mask = comp->AddInstruction(
      HloInstruction::CreateReshape(mask_input_shape, binary_mask));

  // Step 3. Generate iota mask.
  HloInstruction* iota_mask = comp->AddInstruction(
      HloInstruction::CreateIota(mask_input_shape, input_dim));

  // Step 4. Sort iota.
  // Use binary mark to sort iota mask, then use iota mask to reshape input.
  HloComputation::Builder comp_builder("compare_binary_iota");
  {
    HloInstruction* lhs_key =
        comp_builder.AddInstruction(HloInstruction::CreateParameter(
            0, ShapeUtil::MakeShape(S32, {}), "lhs_key_binary"));
    HloInstruction* rhs_key =
        comp_builder.AddInstruction(HloInstruction::CreateParameter(
            1, ShapeUtil::MakeShape(S32, {}), "rhs_key_binary"));

    // Values for lhs and rhs
    comp_builder.AddInstruction(HloInstruction::CreateParameter(
        2, ShapeUtil::MakeShape(S32, {}), "lhs_iota"));
    comp_builder.AddInstruction(HloInstruction::CreateParameter(
        3, ShapeUtil::MakeShape(S32, {}), "rhs_iota"));
    comp_builder.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), lhs_key,
                                      rhs_key, ComparisonDirection::kLt));
  }

  HloComputation* compare_binary_iota =
      comp->parent()->AddEmbeddedComputation(comp_builder.Build());

  HloInstruction* sorted_binary_iota =
      comp->AddInstruction(HloInstruction::CreateSort(
          ShapeUtil::MakeTupleShape({mask_input_shape, mask_input_shape}),
          input_dim, {input_shape_binary_mask, iota_mask}, compare_binary_iota,
          /*is_stable=*/true));
  HloInstruction* sorted_iota_mask =
      comp->AddInstruction(HloInstruction::CreateGetTupleElement(
          mask_input_shape, sorted_binary_iota, 1));

  // Step 5. Sort original input using iota mask as key.
  HloComputation::Builder comp_builder_iota("compare_binary_iota");
  {
    HloInstruction* lhs_key =
        comp_builder_iota.AddInstruction(HloInstruction::CreateParameter(
            0, ShapeUtil::MakeShape(S32, {}), "lhs_key_iota"));
    HloInstruction* rhs_key =
        comp_builder_iota.AddInstruction(HloInstruction::CreateParameter(
            1, ShapeUtil::MakeShape(S32, {}), "rhs_key_iota"));

    // Values for lhs and rhs
    comp_builder_iota.AddInstruction(HloInstruction::CreateParameter(
        2, ShapeUtil::MakeShape(operand_shape.element_type(), {}),
        "lhs_value"));
    comp_builder_iota.AddInstruction(HloInstruction::CreateParameter(
        3, ShapeUtil::MakeShape(operand_shape.element_type(), {}),
        "rhs_value"));
    comp_builder_iota.AddInstruction(
        HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), lhs_key,
                                      rhs_key, ComparisonDirection::kLt));
  }

  HloComputation* compare_iota_value =
      comp->parent()->AddEmbeddedComputation(comp_builder_iota.Build());

  // Temporarily removes dynamic dimension before entering sort -- we want the
  // sort to ignore dynamic dimension.
  HloInstruction* operand_static_dim_size =
      comp->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32>(operand_shape.dimensions(input_dim))));

  HloInstruction* operand_static =
      comp->AddInstruction(HloInstruction::CreateSetDimensionSize(
          operand_shape, reshape->mutable_operand(0), operand_static_dim_size,
          input_dim));

  HloInstruction* sorted_iota_value =
      comp->AddInstruction(HloInstruction::CreateSort(
          ShapeUtil::MakeTupleShape({mask_input_shape, operand_shape}),
          input_dim, {sorted_iota_mask, operand_static}, compare_iota_value,
          /*is_stable=*/true));
  // Step 6: Feed sorted input to original reshape.
  HloInstruction* sorted_operand =
      comp->AddInstruction(HloInstruction::CreateGetTupleElement(
          operand_shape, sorted_iota_value, 1));

  TF_RETURN_IF_ERROR(reshape->ReplaceOperandWith(0, sorted_operand));

  HloInstruction* reshape_dynamic = reshape;

  auto users = reshape->users();

  // Forward the output dynamic dimension.
  for (int64 output_dim : output_dims) {
    HloInstruction* output_dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(reshape, {}, output_dim);
    if (output_dynamic_size != nullptr) {
      reshape_dynamic =
          comp->AddInstruction(HloInstruction::CreateSetDimensionSize(
              reshape->shape(), reshape_dynamic, output_dynamic_size,
              output_dim));
    }
  }

  for (auto* user : users) {
    TF_RETURN_IF_ERROR(reshape->ReplaceUseWith(user, reshape_dynamic));
  }
  TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
      reshape, reshape_dynamic, {}));

  return Status::OK();
}

Status RewriteDynamicReshapeCombineInput(
    HloInstruction* reshape, int64 input_dim, int64 output_dim,
    HloInstruction* dynamic_size,
    DynamicDimensionInference* dynamic_dimension_inference) {
  // Rewrite dynamic reshape into reshape followed by a sort, all padded
  // data will be moved to the end.
  const HloInstruction* operand = reshape->operand(0);
  HloComputation* comp = reshape->parent();
  HloInstruction* zero = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
  HloInstruction* one = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::One(S32)));
  const Shape mask_shape =
      ShapeUtil::ChangeElementType(operand->shape(), xla::S32);
  const Shape mask_reshaped_shape =
      ShapeUtil::ChangeElementType(reshape->shape(), xla::S32);
  HloInstruction* broadcasted_zero = comp->AddInstruction(
      HloInstruction::CreateBroadcast(mask_shape, zero, {}));
  // Pad masking area with 1s, rest with 0s.
  HloInstruction* padding_mask =
      PadWithScalar(broadcasted_zero, input_dim, dynamic_size, one);
  HloInstruction* mask_reshaped = comp->AddInstruction(
      HloInstruction::CreateReshape(mask_reshaped_shape, padding_mask));

  // Build computation for reshape, key is the mask shape, value is reshape's
  // original data.
  HloComputation::Builder comp_builder("compare");
  HloInstruction* lhs_key =
      comp_builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeShape(S32, {}), "lhs_key"));
  HloInstruction* rhs_key =
      comp_builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeShape(S32, {}), "rhs_key"));

  // Values for lhs and rhs
  comp_builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeShape(operand->shape().element_type(), {}),
      "lhs_value"));
  comp_builder.AddInstruction(HloInstruction::CreateParameter(
      3, ShapeUtil::MakeShape(operand->shape().element_type(), {}),
      "rhs_value"));
  comp_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), lhs_key,
                                    rhs_key, ComparisonDirection::kLt));
  HloComputation* compare =
      comp->parent()->AddEmbeddedComputation(comp_builder.Build());

  HloInstruction* static_dim_size = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(
          reshape->shape().dimensions(output_dim))));

  // Temporarily removes dynamic dimension of the reshape before we send it to
  // the sort -- we want padded area to also participate in the sort.
  HloInstruction* reshape_static =
      comp->AddInstruction(HloInstruction::CreateSetDimensionSize(
          reshape->shape(), reshape, static_dim_size, output_dim));

  // Use mask_reshaped as key, sort reshaped data as value.
  HloInstruction* sort = comp->AddInstruction(HloInstruction::CreateSort(
      ShapeUtil::MakeTupleShape({mask_reshaped_shape, reshape->shape()}),
      output_dim, {mask_reshaped, reshape_static}, compare,
      /*is_stable=*/true));
  HloInstruction* dynamic_reshape = comp->AddInstruction(
      HloInstruction::CreateGetTupleElement(reshape->shape(), sort, 1));
  // Forward dynamic size to the newly created reshape.
  HloInstruction* output_dynamic_size =
      dynamic_dimension_inference->GetDynamicSize(reshape, {}, output_dim);
  TF_RET_CHECK(output_dynamic_size != nullptr);
  dynamic_reshape = comp->AddInstruction(HloInstruction::CreateSetDimensionSize(
      dynamic_reshape->shape(), dynamic_reshape, output_dynamic_size,
      output_dim));
  auto users = reshape->users();
  for (auto* user : users) {
    // Avoid cycles by not replacing the staic reshape and get_dimension_size.
    if (user != reshape_static && user != output_dynamic_size) {
      TF_RETURN_IF_ERROR(reshape->ReplaceUseWith(user, dynamic_reshape));
    }
  }

  if (reshape == comp->root_instruction()) {
    comp->set_root_instruction(dynamic_reshape);
  }

  TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
      reshape, dynamic_reshape, {}));

  return Status::OK();
}

Status RewriteDynamicReshapeSingleDim(
    HloInstruction* reshape, int64 input_dim, HloInstruction* dynamic_size,
    DynamicDimensionInference* dynamic_dimension_inference) {
  VLOG(2) << "Rewriting dynamic reshape " << reshape->ToString()
          << " input dim: " << input_dim;
  const Shape operand_shape = reshape->operand(0)->shape();
  const Shape output_shape = reshape->shape();

  const int64 static_input_dim_size = operand_shape.dimensions()[input_dim];

  // Don't need to rewrite size 1 input dims.
  if (static_input_dim_size == 1) {
    return Status::OK();
  }

  auto common_factors =
      CommonFactors(operand_shape.dimensions(), output_shape.dimensions());
  // If there are multiple input dims combining into one output dim,
  // input_dim_start and input_dim_end represent the input dimension range.
  int64 input_dim_start = -1;
  int64 input_dim_end = -1;
  // Similarly when one input dim is splitted into multiple outputs, we use
  // output_dim_start and output_dim_start to represent the output dimension
  // range.
  int64 output_dim_start = -1;
  int64 output_dim_end = -1;
  // Find common_factors that the input belong to.
  for (int64 i = 0; i < common_factors.size() - 1; ++i) {
    auto start = common_factors[i];
    auto end = common_factors[i + 1];
    if (input_dim >= start.first && input_dim < end.first) {
      // Found the common_factor group that the input_dim belongs to.
      input_dim_start = start.first;
      input_dim_end = end.first;
      output_dim_start = start.second;
      output_dim_end = end.second;
    }
  }

  TF_RET_CHECK(output_dim_end - output_dim_start > 0);

  std::vector<int64> output_dims;
  for (int64 i = output_dim_start; i < output_dim_end; ++i) {
    output_dims.push_back(i);
  }

  const int64 first_output_dim = output_dims[0];

  if (reshape->shape().dimensions(first_output_dim) < static_input_dim_size) {
    // One input dimension is splitted into multiple output dimensions.
    return RewriteDynamicReshapeSplitInput(reshape, input_dim, output_dims,
                                           dynamic_dimension_inference);
  }

  if (reshape->shape().dimensions(first_output_dim) == static_input_dim_size) {
    // Unchanged dynamic dimension doesn't need a rewrite.
    return Status::OK();
  }

  // Multiple dimensions got combined into one output.
  if (input_dim != input_dim_start) {
    // If 'input_dim' is not the first dimension that got combined into the
    // output. A reshape rewrite on the output is needed:
    //
    //  Need a write (d is dynamic):
    //  1, 2, d
    //   |
    //  Reshape
    //   |
    //   2d
    //
    //  Don't need rewrite:
    //  d, 2
    //   |
    //  Reshape
    //   |
    //   2d
    //
    return RewriteDynamicReshapeCombineInput(reshape, input_dim,
                                             first_output_dim, dynamic_size,
                                             dynamic_dimension_inference);
  }
  return Status::OK();
}

StatusOr<bool> RewriteDynamicConcat(
    HloInstruction* concat,
    DynamicDimensionInference* dynamic_dimension_inference) {
  const int64 concat_dim = concat->concatenate_dimension();
  HloComputation* comp = concat->parent();
  if (dynamic_dimension_inference->GetDynamicSize(concat, {}, concat_dim) ==
      nullptr) {
    // Concat dimension is not dynamic -- no rewrite needed.
    return false;
  }
  std::vector<HloInstruction*> offsets;
  for (int64 i = 0; i < concat->shape().dimensions_size(); ++i) {
    offsets.push_back(comp->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(0))));
  }
  HloInstruction* rewritten_concat = concat;
  // Keep track of previous users before rewrite so that we can update their
  // operands later.
  auto prev_users = concat->users();
  for (int64 i = 0; i < concat->operand_count(); ++i) {
    // Rewrite the concat by dynamic update slicing operand into the concat dim.
    HloInstruction* operand = concat->mutable_operand(i);
    rewritten_concat =
        comp->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            rewritten_concat->shape(), rewritten_concat, operand, offsets));
    // Update the offset of concat dimension by adding the size of the concat
    // dimension of the operand to it.
    HloInstruction* dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(operand, {}, concat_dim);
    if (dynamic_size == nullptr) {
      HloInstruction* static_size = comp->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(
              operand->shape().dimensions(concat_dim))));
      offsets[concat_dim] = comp->AddInstruction(HloInstruction::CreateBinary(
          ShapeUtil::MakeScalarShape(S32), HloOpcode::kAdd, offsets[concat_dim],
          static_size));
    } else {
      offsets[concat_dim] = comp->AddInstruction(HloInstruction::CreateBinary(
          ShapeUtil::MakeScalarShape(S32), HloOpcode::kAdd, offsets[concat_dim],
          dynamic_size));
    }
  }
  for (HloInstruction* user : prev_users) {
    TF_RETURN_IF_ERROR(concat->ReplaceUseWith(user, rewritten_concat));
  }
  TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
      concat, rewritten_concat, {}));
  return true;
}

StatusOr<bool> RewriteDynamicSort(
    HloInstruction* hlo,
    DynamicDimensionInference* dynamic_dimension_inference) {
  HloInstruction* dynamic_size = nullptr;
  HloSortInstruction* sort = Cast<HloSortInstruction>(hlo);
  HloComputation* comp = hlo->parent();
  int64 sort_dim = sort->sort_dimension();
  // Find the dynamic dimension in the operand.
  for (auto* operand : sort->operands()) {
    if (dynamic_size == nullptr) {
      dynamic_size =
          dynamic_dimension_inference->GetDynamicSize(operand, {}, sort_dim);
    }
  }

  if (dynamic_size == nullptr) {
    // Not a dynamic sort, ignore.
    return false;
  }

  Shape operand_shape =
      ShapeUtil::ChangeElementType(sort->operand(0)->shape(), S32);
  HloInstruction* iota =
      comp->AddInstruction(HloInstruction::CreateIota(operand_shape, sort_dim));
  HloInstruction* dynamic_size_broadcasted = comp->AddInstruction(
      HloInstruction::CreateBroadcast(operand_shape, dynamic_size, {}));
  HloInstruction* lt = comp->AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::ChangeElementType(operand_shape, PRED), iota,
      dynamic_size_broadcasted, ComparisonDirection::kLt));
  sort->AppendOperand(lt);

  const int64 param_number_before_rewritten =
      sort->called_computations()[0]->num_parameters();
  auto new_param_0 = HloInstruction::CreateParameter(
      param_number_before_rewritten, ShapeUtil::MakeScalarShape(PRED),
      "inbound_lhs");
  auto new_param_1 = HloInstruction::CreateParameter(
      param_number_before_rewritten + 1, ShapeUtil::MakeScalarShape(PRED),
      "inbound_rhs");
  std::vector<const HloInstruction*> extra_parameters{new_param_0.get(),
                                                      new_param_1.get()};
  HloComputation* sort_comp = sort->parent()->parent()->AddEmbeddedComputation(
      sort->called_computations()[0]->CloneWithReplacements(
          /*replacements=*/absl::flat_hash_map<
              const HloInstruction*, std::unique_ptr<HloInstruction>>(),
          extra_parameters));
  auto inbound_lhs =
      sort_comp->parameter_instruction(param_number_before_rewritten);
  auto inbound_rhs =
      sort_comp->parameter_instruction(param_number_before_rewritten + 1);
  sort->ReplaceCalledComputations(
      [&](HloComputation* comp) { return sort_comp; });

  // inbound_lhs & (sort_comp | !in_bound_rhs)
  // Select the lhs if it is in bounds and the rhs is out of bounds or the
  // sort_comp returns true.
  auto out_of_bound_rhs = sort_comp->AddInstruction(HloInstruction::CreateUnary(
      ShapeUtil::MakeScalarShape(PRED), HloOpcode::kNot, inbound_rhs));
  auto sort_comp_or_out_of_bound_rhs =
      sort_comp->AddInstruction(HloInstruction::CreateBinary(
          ShapeUtil::MakeScalarShape(PRED), HloOpcode::kOr,
          sort_comp->root_instruction(), out_of_bound_rhs));

  auto new_root = sort_comp->AddInstruction(HloInstruction::CreateBinary(
      ShapeUtil::MakeScalarShape(PRED), HloOpcode::kAnd, inbound_lhs,
      sort_comp_or_out_of_bound_rhs));
  sort_comp->set_root_instruction(new_root);
  Shape compare_shape =
      ShapeUtil::ChangeElementType(sort->operand(0)->shape(), PRED);
  if (sort->shape().IsTuple()) {
    // For sort that is already tuple, simply add another result to the tuple.
    *sort->mutable_shape()->add_tuple_shapes() =
        ShapeUtil::ChangeElementType(operand_shape, PRED);
  } else {
    auto sort_users = sort->users();
    auto sort_clone = comp->AddInstruction(sort->Clone());
    *sort_clone->mutable_shape() = ShapeUtil::MakeTupleShape(
        {sort->shape(), ShapeUtil::ChangeElementType(operand_shape, PRED)});
    auto rewritten_sort = comp->AddInstruction(
        HloInstruction::CreateGetTupleElement(sort->shape(), sort_clone, 0));
    for (HloInstruction* user : sort_users) {
      TF_RETURN_IF_ERROR(sort->ReplaceUseWith(user, rewritten_sort));
    }
    TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
        sort, rewritten_sort, {}));
    if (comp->root_instruction() == sort) {
      comp->set_root_instruction(rewritten_sort);
    }
  }

  return true;
}

StatusOr<bool> RewriteDynamicReshape(
    HloInstruction* reshape,
    DynamicDimensionInference* dynamic_dimension_inference) {
  bool changed = false;
  HloInstruction* operand = reshape->mutable_operand(0);

  // We append sort instructions after reshape if there is a dynamic input, and
  // the order of sort matters. Rewrite minor dimensions first in case multiple
  // inputs have dynamic dimensions to ensure correct order of sort.
  for (int64 input_dim = operand->shape().rank() - 1; input_dim >= 0;
       --input_dim) {
    HloInstruction* operand_dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(operand, {}, input_dim);

    if (operand_dynamic_size == nullptr) {
      continue;
    }
    TF_RETURN_IF_ERROR(RewriteDynamicReshapeSingleDim(
        reshape, input_dim, operand_dynamic_size, dynamic_dimension_inference));

    changed = true;
  }
  return changed;
}

// Insert pad-to-static after `inst` if `inst` has dynamic dimensions in it.
// Recurse into tuple instructions.
StatusOr<HloInstruction*> InsertPadToStaticOnInstruction(HloInstruction* inst) {
  if (inst->shape().is_static()) {
    return inst;
  }
  HloComputation* comp = inst->parent();
  if (!inst->shape().IsTuple()) {
    // The output shape of pad static is a tuple. The 0th element is the data
    // output, which is the same as input shape, but without dynamic dimensions;
    // i-th element is the dynamic dimension size for i-1th input dimension.
    Shape data_output_shape = inst->shape();  // 0th element.
    data_output_shape.clear_dynamic_dimensions();
    Shape output_shape = ShapeUtil::MakeTupleShape({data_output_shape});
    for (int64 i = 0; i < inst->shape().rank(); ++i) {
      ShapeUtil::AppendShapeToTuple(ShapeUtil::MakeScalarShape(S32),
                                    &output_shape);
    }
    HloInstruction* pad_to_static =
        comp->AddInstruction(HloInstruction::CreateCustomCall(
            output_shape, {inst}, "PadToStatic", ""));
    HloInstruction* data_output =
        comp->AddInstruction(HloInstruction::CreateGetTupleElement(
            data_output_shape, pad_to_static, 0));
    return data_output;
  }

  TF_RET_CHECK(inst->shape().IsTuple());
  std::vector<HloInstruction*> static_tuple_elements;
  for (int64 i = 0; i < inst->shape().tuple_shapes_size(); ++i) {
    // For each tuple element, if it is static, pass it through. If it is
    // dynamic, recursively call this function again.
    HloInstruction* gte =
        comp->AddInstruction(HloInstruction::CreateGetTupleElement(
            inst->shape().tuple_shapes(i), inst, i));

    if (gte->shape().is_static()) {
      static_tuple_elements.push_back(gte);
    } else {
      TF_ASSIGN_OR_RETURN(HloInstruction * static_gte,
                          InsertPadToStaticOnInstruction(gte));
      static_tuple_elements.push_back(static_gte);
    }
  }

  return comp->AddInstruction(
      HloInstruction::CreateTuple(static_tuple_elements));
}

Status InsertPadToStaticAfterModuleInputs(HloModule* module) {
  std::vector<HloInstruction*> params;
  HloComputation* entry = module->entry_computation();
  for (int64 i = 0; i < entry->num_parameters(); ++i) {
    HloInstruction* param =
        module->entry_computation()->parameter_instruction(i);
    auto users = param->users();
    TF_ASSIGN_OR_RETURN(HloInstruction * static_param,
                        InsertPadToStaticOnInstruction(param));
    for (auto* user : users) {
      TF_RETURN_IF_ERROR(param->ReplaceUseWith(user, static_param));
    }
    if (param == entry->root_instruction()) {
      module->entry_computation()->set_root_instruction(static_param);
    }
  }
  return Status::OK();
}

// For all dynamic outputs that live out of the computation, add
// slice-to-dynamic operations.
Status InsertSliceToDynamicBeforeModuleOutputs(
    const DynamicDimensionInference& dynamic_dimension_inference,
    HloModule* module) {
  auto root = module->entry_computation()->root_instruction();
  absl::flat_hash_set<ShapeIndex> dynamic_outputs;
  ShapeUtil::ForEachSubshape(
      root->shape(), [&](const Shape& subshape, const ShapeIndex& index) {
        if (subshape.IsArray()) {
          bool has_dynamic_output = false;
          for (int64 dim = 0; dim < subshape.rank(); ++dim) {
            if (dynamic_dimension_inference.GetDynamicSize(root, index, dim) !=
                nullptr) {
              CHECK_LE(index.size(), 1) << "XLA doesn't support nested output "
                                           "dimension that has dynamic size";
              has_dynamic_output = true;
            }
          }
          if (has_dynamic_output) {
            dynamic_outputs.insert(index);
          }
        }
      });
  int64 dynamic_index = 0;
  if (!dynamic_outputs.empty()) {
    if (root->shape().IsTuple()) {
      std::vector<HloInstruction*> new_root_operands;
      ShapeUtil::ForEachSubshape(root->shape(), [&](const Shape& subshape,
                                                    const ShapeIndex& index) {
        if (!subshape.IsArray()) {
          return;
        }

        auto gte = module->entry_computation()->AddInstruction(
            HloInstruction::CreateGetTupleElement(
                ShapeUtil::MakeShapeWithStaticDimensions(subshape), root,
                index[0]));

        if (dynamic_outputs.contains(index)) {
          CHECK_EQ(index.size(), 1)
              << "XLA only support 1 layer nested output tuple";
          // For dynamic outputs, creates an slice operation.
          std::vector<HloInstruction*> slice_operands;
          // First operand is the original input. Rest are dimension values.
          slice_operands.push_back(gte);
          // Keep a dynamic version of the subshape as we are removing the
          // dynamic dimension in the original root and gte.
          Shape dynamic_subshape = subshape;
          for (int64 dim = 0; dim < subshape.rank(); ++dim) {
            HloInstruction* dynamic_size =
                dynamic_dimension_inference.GetDynamicSize(root, index, dim);
            if (dynamic_size != nullptr) {
              slice_operands.push_back(dynamic_size);
            } else {
              auto const_size = HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<int32>(subshape.dimensions(dim)));
              slice_operands.push_back(
                  module->entry_computation()->AddInstruction(
                      std::move(const_size)));
            }
          }
          // This is a dynamic output, add slice operation.
          //
          // Write the backend config in the format of
          // 'dynamic_index'-'output_index'.
          //
          // dynamic_index indicates the position of this output in all dynamic
          // outputs.
          //
          // output_index indicates the position of this output in all outputs
          // (including static inputs).
          auto slice = HloInstruction::CreateCustomCall(
              dynamic_subshape, slice_operands, "SliceToDynamic",
              absl::StrFormat("%d-%d", dynamic_index++, index[0]));
          new_root_operands.push_back(
              module->entry_computation()->AddInstruction(std::move(slice)));
        } else {
          new_root_operands.push_back(gte);
        }
      });

      auto new_root = module->entry_computation()->AddInstruction(
          HloInstruction::CreateTuple(new_root_operands));
      module->entry_computation()->set_root_instruction(new_root);
    } else {
      std::vector<HloInstruction*> slice_operands;
      // First operand is the original input. Rest are dimension values.
      slice_operands.push_back(root);
      for (int64 dim = 0; dim < root->shape().rank(); ++dim) {
        HloInstruction* dynamic_size =
            dynamic_dimension_inference.GetDynamicSize(root, {}, dim);
        if (dynamic_size != nullptr) {
          slice_operands.push_back(dynamic_size);
        } else {
          auto const_size = HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32>(root->shape().dimensions(dim)));
          slice_operands.push_back(module->entry_computation()->AddInstruction(
              std::move(const_size)));
        }
        // This is a dynamic output, add slice operation.
        auto slice = module->entry_computation()->AddInstruction(
            HloInstruction::CreateCustomCall(root->shape(), slice_operands,
                                             "SliceToDynamic", "0-0"));
        module->entry_computation()->set_root_instruction(slice);
      }
    }
  }
  return Status::OK();
}

// Remove all dynamic shapes between pad-to-static and slice-to-dynamic.
//
// After this visitor the entry computation then looks like:
//  Param(dynamic)
//    |
//   GTE (dynamic)
//    |
//  PadToStatic(static)
//    |
//   .... regular computation with static shapes.
//    |
//  SliceToDynamic(dynamic)
//    |
// ROOT tuple (dynamic)
class DynamicShapeRemovingVisitor : public DfsHloVisitorWithDefault {
 public:
  Status DefaultAction(HloInstruction* hlo) override;

  Status HandleCustomCall(HloInstruction* hlo) override;

  Status HandleParameter(HloInstruction* hlo) override;

  static Status Run(HloComputation* computation) {
    DynamicShapeRemovingVisitor visitor;
    return computation->Accept(&visitor);
  }
};

Status DynamicShapeRemovingVisitor::DefaultAction(HloInstruction* hlo) {
  // Default rule: If input to an op is static, remove dynamism in output.
  bool input_is_dynamic = false;
  // Default rule:
  for (int64 i = 0; i < hlo->operand_count(); ++i) {
    if (!hlo->operand(i)->shape().is_static()) {
      input_is_dynamic = true;
    }
  }

  if (!input_is_dynamic) {
    hlo->mutable_shape()->clear_dynamic_dimensions();
  }
  return Status::OK();
}

Status DynamicShapeRemovingVisitor::HandleCustomCall(HloInstruction* hlo) {
  if (hlo->custom_call_target() == "SliceToDynamic") {
    // Don't remove slice-to-dynamic instruction.
    return Status::OK();
  }
  return DefaultAction(hlo);
}

Status DynamicShapeRemovingVisitor::HandleParameter(HloInstruction* hlo) {
  return Status::OK();
}

}  // namespace

StatusOr<bool> DynamicPadder::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "Pre DynamicPadder HLO:";

  // Removes dynamic dimensions on parameters if there is already a binding for
  // it. We do this because we have two different APIs to express a dynamic
  // dimension:
  //
  // 1. Dynamic dimension as specificed directly in the shape -- Needed for
  // Pytorch.
  //
  // 2. Dynamic dimension using dynamic parameter binding object. This
  // is needed for tensorflow.
  //
  // For case 1, we will insert "pad-to-static" instruction in the
  // beginning of xla execution, to make it into a static layout.
  //
  // For case 2, since it already has a static layout, we remove the
  // dynamic dimension.
  //
  // TODO(b/145140571): Convert all API invocations to case 1.
  //
  TF_RETURN_IF_ERROR(module->dynamic_parameter_binding().ForEachBinding(
      [&](const DynamicParameterBinding::DynamicParameter& dynamic_parameter,
          const DynamicParameterBinding::DynamicDimension& dynamic_dimension)
          -> Status {
        HloInstruction* parameter =
            module->entry_computation()->parameter_instruction(
                dynamic_dimension.parameter_num);
        ShapeUtil::UpdateDynamicDimension(parameter->mutable_shape(),
                                          dynamic_dimension.parameter_index,
                                          dynamic_dimension.dimension, false);
        return Status::OK();
      }));

  TF_RETURN_IF_ERROR(InsertPadToStaticAfterModuleInputs(module));
  TF_ASSIGN_OR_RETURN(DynamicDimensionInference dynamic_dimension_inference,
                      DynamicDimensionInference::Run(module));

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      if (inst->opcode() == HloOpcode::kConcatenate) {
        TF_ASSIGN_OR_RETURN(
            changed, RewriteDynamicConcat(inst, &dynamic_dimension_inference));
        continue;
      }
      if (inst->opcode() == HloOpcode::kSort) {
        TF_ASSIGN_OR_RETURN(
            changed, RewriteDynamicSort(inst, &dynamic_dimension_inference));
        continue;
      }
      for (int64 operand_num = 0; operand_num < inst->operand_count();
           ++operand_num) {
        HloInstruction* original_operand = inst->mutable_operand(operand_num);
        HloInstruction* operand = original_operand;
        if (!operand->shape().IsArray()) {
          continue;
        }

        if (inst->opcode() == HloOpcode::kReshape) {
          TF_ASSIGN_OR_RETURN(changed, RewriteDynamicReshape(
                                           inst, &dynamic_dimension_inference));
          continue;
        }
        for (int64 input_dim = 0; input_dim < operand->shape().rank();
             ++input_dim) {
          HloInstruction* operand_dynamic_size =
              dynamic_dimension_inference.GetDynamicSize(original_operand, {},
                                                         input_dim);
          if (operand_dynamic_size == nullptr) {
            continue;
          }
          VLOG(2) << "Has dynamic dimension of operand" << operand_num << " @"
                  << input_dim;

          if (ShouldSkipPadOnOperand(inst, operand_num, input_dim)) {
            continue;
          }

          TF_ASSIGN_OR_RETURN(HloInstruction * identity_value,
                              ChooseIdentityValue(inst, operand_num));
          if (identity_value == nullptr) {
            continue;
          }

          HloInstruction* padded = PadWithScalar(
              operand, input_dim, operand_dynamic_size, identity_value);
          TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(operand_num, padded));
          operand = inst->mutable_operand(operand_num);
          changed = true;
        }
      }
    }
  }

  TF_RETURN_IF_ERROR(InsertSliceToDynamicBeforeModuleOutputs(
      dynamic_dimension_inference, module));

  // Remove all dynamic dimensions after entry parameter and root instruction --
  // Dynamic padder will produce an equivalent static shaped graph.
  for (HloComputation* computation : module->computations()) {
    if (computation == module->entry_computation()) {
      TF_RETURN_IF_ERROR(DynamicShapeRemovingVisitor::Run(computation));
    } else {
      for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
        bool operand_is_dynamic = false;
        for (auto* operand : inst->operands()) {
          if (!operand->shape().is_static()) {
            operand_is_dynamic = true;
          }
        }
        if (!operand_is_dynamic) {
          inst->mutable_shape()->clear_dynamic_dimensions();
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
