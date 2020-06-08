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
#include "tensorflow/core/platform/errors.h"

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
    case HloOpcode::kDomain:
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
// The way we do this is by a 5-steps cumsum-gather algorithm:
//
// 1.First we use the output shape to generate a binary 0-1 masking, which masks
// out the padded area of the output:
// [[1,1,0]
//  [1,1,0]]
//
// 2.Then we do an inverse reshape to reshape it from output shape back to input
// shape [2,3]->[6]:
//  [1,1,0,1,1,0]
//
// 3.We then do a cumsum with the mask:
//  [1,2,2,3,4,4] and subtract it with 1:
//  [0,1,1,2,3,3]
//
// 4.Use the the result of cumsum as gather indicies to rearrange the original
// data. Feed the original input [a,b,c,d,P,P] and indices into gather.
//
//  operand [a,b,c,d,P,P], indices [0,1,1,2,3,3]
//     |                    |
//   Gather-----------------+
//     |
//     v
//  value[a,b,b,c,d,d], which is equivalent to [a,b,P,c,d,P] as padding value
//  doesn't matter.
//
//
// 5.Feed the sorted input to original reshape[6]->[2,3], we can now get the
// correct result:
//  [[a,b,P]
//   [c,d,P]]
//
Status RewriteDynamicReshapeSplitInput(
    HloInstruction* reshape, int64 input_dim,
    absl::Span<const int64> output_dims,
    absl::Span<HloInstruction*> output_dynamic_dims,
    DynamicDimensionInference* dynamic_dimension_inference) {
  VLOG(2) << "Reshaping input dim " << input_dim << "to "
          << VectorString(output_dims);
  const Shape operand_shape = reshape->operand(0)->shape();
  TF_RET_CHECK(output_dims.size() > 1);

  HloComputation* comp = reshape->parent();
  const Shape mask_input_shape =
      ShapeUtil::MakeShape(xla::S32, {operand_shape.dimensions(input_dim)});

  std::vector<int64> reshaped_dims;
  for (int64 output_dim : output_dims) {
    reshaped_dims.push_back(reshape->shape().dimensions(output_dim));
  }

  const Shape mask_reshaped_shape =
      ShapeUtil::MakeShape(xla::S32, reshaped_dims);

  HloInstruction* zero = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
  HloInstruction* one = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::One(S32)));
  // Step 1 -- generate binary mask.
  // Mask starts with all one, each dynamic dimension sets that dimension of the
  // mask to partially zero in the end.
  HloInstruction* binary_mask = comp->AddInstruction(
      HloInstruction::CreateBroadcast(mask_reshaped_shape, one, {}));

  bool need_rewrite = false;

  // Pad the effective dimension with 1.
  //
  // Index starts from 1 since there is no need to rewrite a major output
  // dimension.
  for (int64 i = 1; i < output_dims.size(); ++i) {
    const int64 output_dim = output_dims[i];
    HloInstruction* dynamic_size = output_dynamic_dims[output_dim];
    if (dynamic_size == nullptr) {
      continue;
    }
    // If there is dynamic dimension in the output, need to rewrite the input.
    need_rewrite = true;

    binary_mask = PadWithScalar(binary_mask, i, dynamic_size, zero);
  }
  if (!need_rewrite) {
    return Status::OK();
  }
  // Step 2.
  // Do a reverse reshape to flatten the binary mask (with output shape) back to
  // input shape.
  HloInstruction* input_shape_binary_mask = comp->AddInstruction(
      HloInstruction::CreateReshape(mask_input_shape, binary_mask));

  // Step 3. Do a cumsum on the binary mask.
  auto embedded_builder = HloComputation::Builder("add");
  {
    auto lhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(S32, {}), "lhs"));
    auto rhs = embedded_builder.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(S32, {}), "rhs"));
    embedded_builder.AddInstruction(
        HloInstruction::CreateBinary(lhs->shape(), HloOpcode::kAdd, lhs, rhs));
  }

  HloComputation* add =
      reshape->GetModule()->AddEmbeddedComputation(embedded_builder.Build());
  Window cumsum_window;
  // First dimension is unchanged.
  WindowDimension* dim = cumsum_window.add_dimensions();
  dim->set_size(operand_shape.dimensions(input_dim));
  dim->set_stride(1);
  dim->set_padding_low(operand_shape.dimensions(input_dim) - 1);
  dim->set_padding_high(0);
  dim->set_window_dilation(1);
  dim->set_base_dilation(1);
  HloInstruction* cumsum =
      comp->AddInstruction(HloInstruction::CreateReduceWindow(
          mask_input_shape, input_shape_binary_mask, zero, cumsum_window, add));

  HloInstruction* broadcast_ones = comp->AddInstruction(
      HloInstruction::CreateBroadcast(mask_input_shape, one, {}));
  cumsum = comp->AddInstruction(HloInstruction::CreateBinary(
      mask_input_shape, HloOpcode::kSubtract, cumsum, broadcast_ones));

  GatherDimensionNumbers gather_dim_numbers;
  // Use gather to rearrange the input dim dimension.
  for (int64 i = 0; i < operand_shape.dimensions_size(); ++i) {
    // Offset dim is every dimension including newly added size 1 dim, except
    // for input_dim, which acts as a batch_dim.
    if (i != input_dim) {
      gather_dim_numbers.add_offset_dims(i);
    }
  }
  // The dimension to rewrite is the index dim.
  gather_dim_numbers.add_start_index_map(input_dim);
  gather_dim_numbers.set_index_vector_dim(1);
  gather_dim_numbers.add_collapsed_slice_dims(input_dim);

  // Step 4. Gather.

  // Temporarily removes dynamic dimension before entering gather -- we want the
  // gather to ignore dynamic dimension.
  HloInstruction* operand_static_dim_size =
      comp->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32>(operand_shape.dimensions(input_dim))));
  HloInstruction* operand_static =
      comp->AddInstruction(HloInstruction::CreateSetDimensionSize(
          operand_shape, reshape->mutable_operand(0), operand_static_dim_size,
          input_dim));

  std::vector<int64> slice_sizes(operand_shape.dimensions().begin(),
                                 operand_shape.dimensions().end());
  slice_sizes[input_dim] = 1;
  HloInstruction* gather = comp->AddInstruction(HloInstruction::CreateGather(
      ShapeUtil::MakeShape(operand_shape.element_type(),
                           operand_shape.dimensions()),
      operand_static, cumsum, gather_dim_numbers, slice_sizes, true));

  // Step 6: Feed gather input to original reshape.

  TF_RETURN_IF_ERROR(reshape->ReplaceOperandWith(0, gather));

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

// RewriteDynamicReshapeCombineInput is similar to
// RewriteDynamicReshapeSplitInput, in a reshape if multiple dimensions are
// combined into one dimension, we need to rewrite the output.
//
// The reason for this is that a continuous input may not be evenly reshaped
// into output.  Image we have [2, <=3] where second dimension has size 2 and
// padding(P) data has size 1:
// [[a,b,P]
//  [c,d,P]]
//
// And we have a reshape that combines this two input dimensions.
//
// [2, <=3]
//  |
// Reshape
//  |
// [6]
//
// This should produce the same result as if the data has no padding:
//
// [2, 2]     // [[a, b], [c, d]]
//  |
// Reshape
//  |
// [4]  // [a,b,c,d]
//
// Without rewriting, the result would be:
//
// [a,b,P,c,d,P], which is incorrect.
//
// We need to rewrite the reshape such that it produces:
// [a,b,c,d,P,P]
//
// The way we do this is by a 5-steps sort-gather algorithm:
//
// 1.First we use the input shape to generate a binary 0-1 masking, which masks
// out the padded area of the output:
// [[0,0,1]
//  [0,0,1]]
//
// 2.Then we do an reshape to reshape the mask from input shape to output
// shape [2,3]->[6]:
//  [0,0,1,0,0,1]
//
// 3.We then generate an iota mask using the output shape:
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
// 5.Gather the original output [a,b,P,c,d,P] using the sorted iota mask:
//      original output       gather indices
//       [a,b,P,c,d,P]         [0,1,3,4,2,5]
//            |                    |
//          Gather ----------------+
//            |
//       [a,b,c,d,P,P]
//
Status RewriteDynamicReshapeCombineInput(
    HloInstruction* reshape, absl::Span<const int64> input_dims,
    int64 output_dim, absl::Span<HloInstruction*> input_dynamic_dims,
    DynamicDimensionInference* dynamic_dimension_inference) {
  // Rewrite dynamic reshape into reshape followed by a sort, all padded
  // data will be moved to the end.
  HloComputation* comp = reshape->parent();
  HloInstruction* zero = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
  HloInstruction* one = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::One(S32)));
  const Shape output_shape = reshape->shape();
  const Shape input_shape = reshape->operand(0)->shape();
  const Shape mask_output_shape =
      ShapeUtil::MakeShape(xla::S32, {output_shape.dimensions(output_dim)});
  std::vector<int64> input_dim_sizes;
  for (int64 input_dim : input_dims) {
    input_dim_sizes.push_back(input_shape.dimensions(input_dim));
  }

  const Shape mask_input_shape =
      ShapeUtil::MakeShape(xla::S32, input_dim_sizes);

  // Step 1 -- generate binary mask.
  // Mask starts with all zero, each dynamic dimension sets that dimension of
  // the mask to partially ones in the end.
  HloInstruction* binary_mask = comp->AddInstruction(
      HloInstruction::CreateBroadcast(mask_input_shape, zero, {}));

  bool need_rewrite = false;

  // Pad the effective dimension with 1.
  //
  // Index starts from 1 since there is no need to rewrite a major output
  // dimension.
  for (int64 i = 1; i < input_dims.size(); ++i) {
    const int64 input_dim = input_dims[i];
    HloInstruction* dynamic_size = input_dynamic_dims[input_dim];
    if (dynamic_size == nullptr) {
      continue;
    }
    // If there is a dynamic dimension in the input, need to rewrite the output.
    need_rewrite = true;

    binary_mask = PadWithScalar(binary_mask, i, dynamic_size, one);
  }
  if (!need_rewrite) {
    VLOG(2) << "No need to rewrite";
    return Status::OK();
  }

  // Step 2.
  // Do a reshape to flatten the binary mask into output_shape
  HloInstruction* output_shape_binary_mask = comp->AddInstruction(
      HloInstruction::CreateReshape(mask_output_shape, binary_mask));

  // Step 3.
  // Generate an iota with output shape.
  HloInstruction* iota =
      comp->AddInstruction(HloInstruction::CreateIota(mask_output_shape, 0));

  // Step 4.
  // Stable sort the iota mask using the binary mask as key and iota as value:

  // Build computation for sort, key is the mask, value is the iota.
  HloComputation::Builder comp_builder("compare");
  HloInstruction* lhs_key =
      comp_builder.AddInstruction(HloInstruction::CreateParameter(
          0, ShapeUtil::MakeScalarShape(S32), "lhs_key"));
  HloInstruction* rhs_key =
      comp_builder.AddInstruction(HloInstruction::CreateParameter(
          1, ShapeUtil::MakeScalarShape(S32), "rhs_key"));

  // Values for lhs and rhs
  comp_builder.AddInstruction(HloInstruction::CreateParameter(
      2, ShapeUtil::MakeScalarShape(S32), "lhs_value"));
  comp_builder.AddInstruction(HloInstruction::CreateParameter(
      3, ShapeUtil::MakeScalarShape(S32), "rhs_value"));
  comp_builder.AddInstruction(
      HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), lhs_key,
                                    rhs_key, ComparisonDirection::kLt));
  HloComputation* compare =
      comp->parent()->AddEmbeddedComputation(comp_builder.Build());

  // Use mask_reshaped as key, sort reshaped data as value.
  HloInstruction* sort = comp->AddInstruction(HloInstruction::CreateSort(
      ShapeUtil::MakeTupleShape({mask_output_shape, mask_output_shape}), 0,
      {output_shape_binary_mask, iota}, compare,
      /*is_stable=*/true));

  HloInstruction* gather_indices = comp->AddInstruction(
      HloInstruction::CreateGetTupleElement(mask_output_shape, sort, 1));

  // Step 5.Gather the original output using the sorted iota mask:

  GatherDimensionNumbers gather_dim_numbers;
  // Use gather to rearrange the output dim dimension.
  for (int64 i = 0; i < output_shape.dimensions_size(); ++i) {
    // Offset dim is every dimension including newly added size 1 dim, except
    // for input_dim, which acts as a batch_dim.
    if (i != output_dim) {
      gather_dim_numbers.add_offset_dims(i);
    }
  }
  // The dimension to rewrite is the index dim.
  gather_dim_numbers.add_start_index_map(output_dim);
  gather_dim_numbers.set_index_vector_dim(1);
  gather_dim_numbers.add_collapsed_slice_dims(output_dim);

  HloInstruction* static_dim_size = comp->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32>(
          reshape->shape().dimensions(output_dim))));

  // Temporarily removes dynamic dimension of the reshape before we send it to
  // the sort -- we want padded area to also participate in the gather.
  HloInstruction* reshape_static =
      comp->AddInstruction(HloInstruction::CreateSetDimensionSize(
          reshape->shape(), reshape, static_dim_size, output_dim));
  std::vector<int64> gather_slice_sizes(output_shape.dimensions().begin(),
                                        output_shape.dimensions().end());
  gather_slice_sizes[output_dim] = 1;
  HloInstruction* gather = comp->AddInstruction(HloInstruction::CreateGather(
      output_shape, reshape_static, gather_indices, gather_dim_numbers,
      gather_slice_sizes, true));

  // Forward dynamic size to the newly created gather.
  HloInstruction* output_dynamic_size =
      dynamic_dimension_inference->GetDynamicSize(reshape, {}, output_dim);
  TF_RET_CHECK(output_dynamic_size != nullptr);
  gather = comp->AddInstruction(HloInstruction::CreateSetDimensionSize(
      gather->shape(), gather, output_dynamic_size, output_dim));
  auto users = reshape->users();
  for (auto* user : users) {
    // Avoid cycles by not replacing the staic reshape and get_dimension_size.
    if (user != reshape_static && user != output_dynamic_size) {
      TF_RETURN_IF_ERROR(reshape->ReplaceUseWith(user, gather));
    }
  }

  if (reshape == comp->root_instruction()) {
    comp->set_root_instruction(gather);
  }

  TF_RETURN_IF_ERROR(
      dynamic_dimension_inference->ForwardDynamicSize(reshape, gather, {}));

  return Status::OK();
}

Status RewriteDynamicReshapeSingleGroup(
    HloInstruction* reshape, absl::Span<const int64> input_dims,
    absl::Span<const int64> output_dims,
    absl::Span<HloInstruction*> input_dynamic_dims,
    absl::Span<HloInstruction*> output_dynamic_dims,
    DynamicDimensionInference* dynamic_dimension_inference) {
  VLOG(2) << "Rewriting dynamic reshape " << reshape->ToString()
          << " input dims: " << VectorString(input_dims)
          << " output dims: " << VectorString(output_dims);

  const Shape operand_shape = reshape->operand(0)->shape();
  const Shape output_shape = reshape->shape();

  if (input_dims.size() == 1) {
    int64 input_dim = input_dims[0];
    // Size 1 dimension doesn't need a rewrite.
    if (operand_shape.dimensions()[input_dim] == 1) {
      return Status::OK();
    }
    // One input dimension is splitted into multiple output dimensions.
    return RewriteDynamicReshapeSplitInput(reshape, input_dim, output_dims,
                                           output_dynamic_dims,
                                           dynamic_dimension_inference);
  }

  if (output_dims.size() == 1) {
    int64 output_dim = output_dims[0];
    if (output_shape.dimensions()[output_dim] == 1) {
      return Status::OK();
    }
    // One input dimension is splitted into multiple output dimensions.
    return RewriteDynamicReshapeCombineInput(reshape, input_dims, output_dim,
                                             input_dynamic_dims,
                                             dynamic_dimension_inference);
  }
  // Shouldn't get here;
  TF_RET_CHECK(false);
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
  std::vector<HloInstruction*> input_dynamic_dims;
  for (int64 dim = 0; dim < operand->shape().dimensions_size(); ++dim) {
    input_dynamic_dims.push_back(
        dynamic_dimension_inference->GetDynamicSize(operand, {}, dim));
  }

  std::vector<HloInstruction*> output_dynamic_dims;
  for (int64 dim = 0; dim < reshape->shape().dimensions_size(); ++dim) {
    output_dynamic_dims.push_back(
        dynamic_dimension_inference->GetDynamicSize(reshape, {}, dim));
  }

  auto common_factors = CommonFactors(operand->shape().dimensions(),
                                      reshape->shape().dimensions());
  // Find common_factors that the input belongs to.
  for (int64 i = 0; i < common_factors.size() - 1; ++i) {
    auto start = common_factors[i];
    auto end = common_factors[i + 1];
    std::vector<int64> input_dims;
    std::vector<int64> output_dims;
    for (int64 dim = start.first; dim < end.first; ++dim) {
      input_dims.push_back(dim);
    }
    for (int64 dim = start.second; dim < end.second; ++dim) {
      output_dims.push_back(dim);
    }

    VLOG(2) << "input_dims: " << VectorString(input_dims);
    VLOG(2) << "output_dims: " << VectorString(output_dims);

    if (input_dims.empty() || output_dims.empty()) {
      continue;
    }
    bool has_dynamic_dimension = absl::c_any_of(output_dims, [&](int64 dim) {
      HloInstruction* operand_dynamic_size =
          dynamic_dimension_inference->GetDynamicSize(reshape, {}, dim);

      return operand_dynamic_size != nullptr ||
             reshape->shape().is_dynamic_dimension(dim);
    });

    if (!has_dynamic_dimension) {
      // Don't need to rewrite any group without dynamic dimensions.
      VLOG(2) << "All dimensions are static in this common factor group";
      continue;
    }

    if (input_dims.size() == 1 && output_dims.size() == 1) {
      // The dimension is unchanged. No rewrite needed.
      continue;
    }
    if (input_dims.size() > 1 && output_dims.size() > 1) {
      // We don't support the case when a dynamic dimension is both combined
      // with and splitted into other dimensions:
      //
      //  [x, yz]
      //     | Reshape
      //  [xy, z]
      //
      // TODO(yunxing): This can be supported by canonicalizing
      // the offending reshape into two reshapes:
      //
      //  [x,yz]
      //     | Reshape
      //  [x, y, z]
      //     | Reshape
      //  [xy, z]
      //
      return Unimplemented(
          "Dynamic input dimension to reshape that is both splitted and "
          "combined is not supported %s",
          reshape->ToString());
    }

    TF_RETURN_IF_ERROR(RewriteDynamicReshapeSingleGroup(
        reshape, input_dims, output_dims, absl::MakeSpan(input_dynamic_dims),
        absl::MakeSpan(output_dynamic_dims), dynamic_dimension_inference));
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
  explicit DynamicShapeRemovingVisitor(
      const DynamicPadder::OpSupportsDynamismHandler&
          op_supports_dynamism_handler,
      const DynamicDimensionInference& dynamic_dimension_inference)
      : op_supports_dynamism_handler_(op_supports_dynamism_handler),
        dynamic_dimension_inference_(dynamic_dimension_inference) {}

  Status DefaultAction(HloInstruction* hlo) override;

  Status HandleCustomCall(HloInstruction* hlo) override;

  Status HandleTuple(HloInstruction* hlo) override;
  Status HandleGetTupleElement(HloInstruction* hlo) override;

  Status HandleParameter(HloInstruction* hlo) override;

  static Status Run(HloComputation* computation,
                    const DynamicPadder::OpSupportsDynamismHandler&
                        op_supports_dynamism_handler,
                    const DynamicDimensionInference& dynamic_shape_inference,
                    bool require_dynamic_output) {
    DynamicShapeRemovingVisitor visitor(op_supports_dynamism_handler,
                                        dynamic_shape_inference);
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    // If the outputs is required to be dynamic form, insert static to dynamic
    // conversion as root.
    if (require_dynamic_output) {
      HloInstruction* root = computation->root_instruction();
      if (dynamic_shape_inference.HasDynamicDimension(root)) {
        HloInstruction* new_root = visitor.ConvertToDynamic(root);
        computation->set_root_instruction(new_root);
      }
    }
    return Status::OK();
  }

 private:
  // If a tensor produced by `inst` is in dynamic form, convert it to static and
  // returns the new instruction.
  HloInstruction* ConvertToStatic(HloInstruction* inst);

  // If a tensor produced by `inst` is in static form, convert it to dynamic and
  // returns the new instruction.
  HloInstruction* ConvertToDynamic(HloInstruction* inst);

  const DynamicPadder::OpSupportsDynamismHandler& op_supports_dynamism_handler_;

  const DynamicDimensionInference& dynamic_dimension_inference_;
};

HloInstruction* DynamicShapeRemovingVisitor::ConvertToDynamic(
    HloInstruction* inst) {
  auto* comp = inst->parent();
  const Shape& shape = inst->shape();
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> dynamic_operands;
    for (int64 i = 0; i < shape.tuple_shapes_size(); ++i) {
      auto operand = inst->mutable_operand(i);
      if (dynamic_dimension_inference_.HasDynamicDimension(operand)) {
        // Recurse.
        dynamic_operands.push_back(ConvertToDynamic(operand));
      } else {
        dynamic_operands.push_back(operand);
      }
    }
    return comp->AddInstruction(HloInstruction::CreateTuple(dynamic_operands));
  } else {
    // Collect the data input, as well as dimension sizes, and feed them to
    // slice to dynamic to create a dynamic tensor.
    Shape output_shape = shape;  // 0th element.
    CHECK(output_shape.is_static());
    std::vector<HloInstruction*> slice_operand;
    slice_operand.push_back(inst);
    for (int64 i = 0; i < output_shape.dimensions_size(); ++i) {
      auto dimension_size =
          dynamic_dimension_inference_.GetDynamicSize(inst, {}, i);
      if (dimension_size == nullptr) {
        dimension_size = comp->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32>(output_shape.dimensions(i))));
      } else {
        output_shape.set_dynamic_dimension(i, true);
      }
      slice_operand.push_back(dimension_size);
    }
    return comp->AddInstruction(HloInstruction::CreateCustomCall(
        output_shape, slice_operand, "SliceToDynamic"));
  }
}

HloInstruction* DynamicShapeRemovingVisitor::ConvertToStatic(
    HloInstruction* inst) {
  auto* comp = inst->parent();
  const Shape& shape = inst->shape();
  CHECK(shape.is_dynamic());
  if (shape.IsTuple()) {
    std::vector<HloInstruction*> static_operands;
    for (int64 i = 0; i < shape.tuple_shapes_size(); ++i) {
      auto operand = inst->mutable_operand(i);
      if (shape.tuple_shapes(i).is_dynamic()) {
        static_operands.push_back(ConvertToStatic(operand));
      } else {
        static_operands.push_back(operand);
      }
    }
    return comp->AddInstruction(HloInstruction::CreateTuple(static_operands));
  } else {
    // The output shape of pad static is a tuple. The 0th element is the data
    // output, which is the same as input shape, but without dynamic dimensions.
    // i-th element is the dynamic dimension size for i-1th input dimension.
    Shape data_output_shape = shape;  // 0th element.
    data_output_shape.clear_dynamic_dimensions();
    Shape output_shape = ShapeUtil::MakeTupleShape({data_output_shape});
    for (int64 i = 0; i < shape.rank(); ++i) {
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
}

Status DynamicShapeRemovingVisitor::DefaultAction(HloInstruction* hlo) {
  const bool input_is_dynamic = absl::c_any_of(
      hlo->operands(),
      [](const HloInstruction* hlo) { return hlo->shape().is_dynamic(); });

  // By default, ops don't support dynamic lowering.
  OpDynamismSupport op_support = OpDynamismSupport::kNoSupport;
  if (op_supports_dynamism_handler_) {
    op_support = op_supports_dynamism_handler_(hlo);
  }
  if (op_support == OpDynamismSupport::kNoSupport) {
    for (auto* sub_computation : hlo->called_computations()) {
      for (auto* param : sub_computation->parameter_instructions()) {
        param->mutable_shape()->clear_dynamic_dimensions();
      }
    }
  }
  // If the input to an op is static and the op doesn't support
  // dynamic output, remove dynamism in output -- dynamic_padder should have
  // rewritten it to support static shapes.
  if (!input_is_dynamic && op_support == OpDynamismSupport::kNoSupport) {
    hlo->mutable_shape()->clear_dynamic_dimensions();
    return Status::OK();
  }

  // Op doesn't support dynamic tensor: For each operand rewrite dynamic input
  // into static input using pad_to_static.
  if (input_is_dynamic && op_support == OpDynamismSupport::kNoSupport) {
    VLOG(1) << "op doesn't support dynamic tensor: " << hlo->ToString();
    for (int64 i = 0; i < hlo->operand_count(); ++i) {
      if (hlo->operand(i)->shape().is_dynamic()) {
        auto static_operand = ConvertToStatic(hlo->mutable_operand(i));
        TF_RETURN_IF_ERROR(hlo->ReplaceOperandWith(i, static_operand));
      }
    }
    // This op doesn't support dynamic lowering so the op has to be static.
    hlo->mutable_shape()->clear_dynamic_dimensions();
    return Status::OK();
  }

  // If the op requires dynamic tensor and input is static -- construct a
  // dynamic tensor from the static tensor to feed it.
  if (!input_is_dynamic && op_support == OpDynamismSupport::kRequired) {
    VLOG(1) << "op doesn't support static tensor: " << hlo->ToString();
    for (int64 i = 0; i < hlo->operand_count(); ++i) {
      auto operand = hlo->mutable_operand(i);
      if (dynamic_dimension_inference_.HasDynamicDimension(operand)) {
        auto dynamic_operand = ConvertToDynamic(hlo->mutable_operand(i));
        TF_RETURN_IF_ERROR(hlo->ReplaceOperandWith(i, dynamic_operand));
      }
    }
    return Status::OK();
  }

  return Status::OK();
}

Status DynamicShapeRemovingVisitor::HandleGetTupleElement(HloInstruction* hlo) {
  *hlo->mutable_shape() =
      hlo->operand(0)->shape().tuple_shapes(hlo->tuple_index());
  return Status::OK();
}

Status DynamicShapeRemovingVisitor::HandleTuple(HloInstruction* hlo) {
  for (int64 i = 0; i < hlo->operand_count(); ++i) {
    *hlo->mutable_shape()->mutable_tuple_shapes(i) = hlo->operand(i)->shape();
  }
  return Status::OK();
}

Status DynamicShapeRemovingVisitor::HandleParameter(HloInstruction* hlo) {
  return Status::OK();
}

Status DynamicShapeRemovingVisitor::HandleCustomCall(HloInstruction* hlo) {
  if (hlo->custom_call_target() == "SliceToDynamic" ||
      hlo->custom_call_target() == "PadToStatic") {
    // Those ops support are created to handle dynamic tensors so by their
    // nature they support dynamic lowering.
    return Status::OK();
  }

  return DefaultAction(hlo);
}

}  // namespace

StatusOr<bool> DynamicPadder::Run(HloModule* module) {
  bool changed = false;
  VLOG(2) << "Pre DynamicPadder HLO:";
  XLA_VLOG_LINES(2, module->ToString());
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
  TF_ASSIGN_OR_RETURN(
      DynamicDimensionInference dynamic_dimension_inference,
      DynamicDimensionInference::Run(module, custom_call_handler_));

  for (HloComputation* computation : module->computations()) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      OpDynamismSupport has_dynamism_support = OpDynamismSupport::kNoSupport;
      if (op_supports_dynamism_handler_ != nullptr) {
        has_dynamism_support = op_supports_dynamism_handler_(inst);
      }
      // This op support dynamic lowering, no padding is required.
      if (has_dynamism_support != OpDynamismSupport::kNoSupport) {
        continue;
      }
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
      if (inst->opcode() == HloOpcode::kReshape) {
        TF_ASSIGN_OR_RETURN(
            changed, RewriteDynamicReshape(inst, &dynamic_dimension_inference));
        continue;
      }
      for (int64 operand_num = 0; operand_num < inst->operand_count();
           ++operand_num) {
        HloInstruction* original_operand = inst->mutable_operand(operand_num);
        HloInstruction* operand = original_operand;
        if (!operand->shape().IsArray()) {
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

  // There are ops that only support dynamic lowering and ops that only support
  // static lowering, add dynamic<->static tensor conversion around the boundary
  // between those ops, as well as the root instruction.
  auto computations = module->MakeComputationPostOrder();
  // Reverse postorder so that if caller doesn't support dynamic tensor (while,
  // etc), change their called computation to only take static tensors.
  for (auto it = computations.rbegin(); it != computations.rend(); ++it) {
    HloComputation* computation = *it;
    // if slice_dynamic_output_ is set and this is entry computation, we need
    // the output tensor to be in dynamic form.
    bool require_dynamic_output =
        slice_dynamic_output_ && computation == module->entry_computation();
    TF_RETURN_IF_ERROR(DynamicShapeRemovingVisitor::Run(
        computation, op_supports_dynamism_handler_, dynamic_dimension_inference,
        /*require_dynamic_output=*/require_dynamic_output));
  }

  HloDCE dce;
  TF_ASSIGN_OR_RETURN(changed, dce.Run(module));
  VLOG(2) << "Post DynamicPadder HLO:";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace xla
