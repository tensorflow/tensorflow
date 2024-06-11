/* Copyright 2019 The OpenXLA Authors.

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
#include "xla/service/dynamic_padder.h"

#include <cstdint>
#include <functional>
#include <iterator>
#include <set>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/client/xla_builder.h"
#include "xla/comparison_util.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/dynamic_parameter_binding.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/literal_util.h"
#include "xla/service/call_graph.h"
#include "xla/service/dynamic_dimension_inference.h"
#include "xla/service/dynamic_window_utils.h"
#include "xla/service/hlo_creation_utils.h"
#include "xla/service/hlo_dce.h"
#include "xla/service/pattern_matcher.h"
#include "xla/service/shape_inference.h"
#include "xla/service/tuple_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/statusor.h"
#include "xla/util.h"
#include "xla/window_util.h"
#include "xla/xla_data.pb.h"
#include "tsl/lib/monitoring/gauge.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

auto* dynamic_padding_gauge = tsl::monitoring::Gauge<bool, 0>::New(
    "/tensorflow/core/use_dynamic_padding_gauge",
    "Tracks if dynamic padder is used.");

// ChooseIdentityValue looks at the instruction's operand, returns a
// identity value which, when padded, doesn't change the result of the
// instruction.
//
// nullopt is returned if padding doesn't need to be reset.
absl::StatusOr<HloInstruction*> ChooseIdentityValue(HloInstruction* inst,
                                                    int64_t operand_number) {
  // Padding on elementwise operation doesn't affect the result of the effective
  // data.
  if (inst->IsElementwise()) {
    return nullptr;
  }
  if (inst->opcode() == HloOpcode::kSelectAndScatter ||
      inst->IsCustomCall("DynamicSelectAndScatterSamePadding")) {
    if (operand_number == 1) {
      return inst->mutable_operand(2);
    }
    TF_RET_CHECK(operand_number == 0);
    HloComputation* select = inst->called_computations()[0];

    if (Match(select->root_instruction(),
              match::Compare(match::Parameter(), match::Parameter())
                  .WithComparisonDirection(ComparisonDirection::kGe))) {
      return inst->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::MinValue(inst->operand(0)->shape().element_type())));
    } else {
      return Unimplemented(
          "Only select and scatter with `max` as select function is "
          "supported, got %s",
          select->ToString());
    }
  }
  switch (inst->opcode()) {
    case HloOpcode::kReduce: {
      auto* reduce = Cast<HloReduceInstruction>(inst);
      TF_RET_CHECK(operand_number < reduce->input_count())
          << "Only data operand with dynamic dimension is valid.";
      // Variadic reduce has different init value for different operand, given
      // a data operand number, find the init value index.
      int64_t init_value_index = reduce->input_count() + operand_number;
      return inst->mutable_operand(init_value_index);
    }
    case HloOpcode::kReduceWindow: {
      auto* reduce_window = Cast<HloReduceWindowInstruction>(inst);
      TF_RET_CHECK(operand_number < reduce_window->input_count())
          << "Only data operand with dynamic dimension is valid.";
      // Variadic reduce has different init value for different operand, given
      // a data operand number, find the init value index.
      int64_t init_value_index = reduce_window->input_count() + operand_number;
      return inst->mutable_operand(init_value_index);
    }

    case HloOpcode::kConvolution:
    case HloOpcode::kDot: {
      // Use 0 as padding value for convolution and dot.
      //
      // Note that the output type (inst->shape().element_type()) isn't
      // necessarily the same as the input type (element type of operands).  For
      // example, a dot can take s8 as input and output s32.
      PrimitiveType ptype = inst->operand(0)->shape().element_type();
      return inst->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(ptype)));
    }

    case HloOpcode::kPad:
      return inst->mutable_operand(1);
    case HloOpcode::kScatter: {
      if (operand_number != 1) {
        return nullptr;
      }
      PrimitiveType indices_ptype =
          inst->operand(operand_number)->shape().element_type();

      return inst->AddInstruction(
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
    case HloOpcode::kReduceScatter:
    case HloOpcode::kBroadcast:
    case HloOpcode::kTranspose:
    case HloOpcode::kSort:
    case HloOpcode::kSlice:
    case HloOpcode::kDomain:
      return nullptr;
    case HloOpcode::kCustomCall:
      // Assume that custom calls created by the client are valid with padded
      // dynamic dimensions.
      return nullptr;
    default:
      return UnimplementedStrCat("Unimplemented padding for instruction: ",
                                 inst->ToString());
  }
}

absl::StatusOr<bool> ReplaceGetSize(
    HloInstruction* instr,
    DynamicDimensionInference* dynamic_dimension_inference) {
  if (instr->opcode() != HloOpcode::kGetDimensionSize) {
    return false;
  }
  HloComputation* computation = instr->parent();

  TF_ASSIGN_OR_RETURN(auto legal_shape,
                      ShapeInference::InferGetDimensionSizeShape(
                          instr->operand(0)->shape(), instr->dimension()));
  TF_RET_CHECK(ShapeUtil::Equal(instr->shape(), legal_shape))
      << "instr->shape() " << instr->shape().ToString() << " , "
      << "legal_shape " << legal_shape.ToString();
  TF_RET_CHECK(ShapeUtil::HasPrimitiveType(instr->shape(), S32));
  HloInstruction* operand = instr->mutable_operand(0);
  int64_t dim = instr->dimension();
  HloInstruction* dynamic_size =
      dynamic_dimension_inference->GetDynamicSize(operand, {}, dim);
  if (dynamic_size != nullptr) {
    TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(dynamic_size));
    // The dependency between a instruction and its dynamic dimensions is not
    // modeled in the IR. As instr is being replaced by dynamic_size, also tell
    // dynamic dimension inference that the instruction is being replaced.
    dynamic_dimension_inference->ReplaceAllDynamicDimensionUsesWith(
        instr, dynamic_size);
  } else {
    int32_t size = instr->operand(0)->shape().dimensions(dim);
    HloInstruction* new_instr = computation->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(size)));
    TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(new_instr));
    dynamic_dimension_inference->ReplaceAllDynamicDimensionUsesWith(instr,
                                                                    new_instr);
  }
  return true;
}

absl::StatusOr<bool> ReplaceSetSize(HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kSetDimensionSize) {
    return false;
  }

  TF_RET_CHECK(Shape::Equal().IgnoreDynamicDimension()(
      instr->shape(), instr->operand(0)->shape()))
      << "instr->shape() " << instr->shape().ToString() << " , "
      << "instruction operand shape " << instr->operand(0)->shape();
  HloInstruction* operand = instr->mutable_operand(0);

  TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(operand));
  return true;
}

absl::StatusOr<bool> ReplaceSetBound(HloInstruction* instr) {
  if (instr->opcode() != HloOpcode::kCustomCall ||
      instr->custom_call_target() != "SetBound") {
    return false;
  }

  TF_RET_CHECK(Shape::Equal().IgnoreDynamicDimension()(
      instr->shape(), instr->operand(0)->shape()))
      << "instr->shape() " << instr->shape().ToString() << " , "
      << "instruction operand shape " << instr->operand(0)->shape();
  HloInstruction* operand = instr->mutable_operand(0);

  TF_RETURN_IF_ERROR(instr->ReplaceAllUsesWith(operand));
  return true;
}

bool ShouldSkipPadOnOperand(
    const HloInstruction* inst, int64_t operand_num, int64_t dimension,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  switch (inst->opcode()) {
    case HloOpcode::kConvolution: {
      if (operand_num == 0) {
        if (dimension ==
            inst->convolution_dimension_numbers().input_batch_dimension()) {
          return true;
        }
        const auto& spatial_dims =
            inst->convolution_dimension_numbers().input_spatial_dimensions();
        for (int64_t spatial_dim = 0; spatial_dim < spatial_dims.size();
             ++spatial_dim) {
          // A spatial dimemnsion with a window of size 1 does not need
          // padding.
          if (spatial_dims[spatial_dim] == dimension &&
              inst->window().dimensions(spatial_dim).size() == 1) {
            return true;
          }
        }
      }
      return operand_num == 1 &&
             (dimension == inst->convolution_dimension_numbers()
                               .kernel_output_feature_dimension());
    }
    case HloOpcode::kDot: {
      if (operand_num == 0) {
        return !absl::c_linear_search(
            inst->dot_dimension_numbers().lhs_contracting_dimensions(),
            dimension);
      }
      return !absl::c_linear_search(
          inst->dot_dimension_numbers().rhs_contracting_dimensions(),
          dimension);
    }
    case HloOpcode::kReduce:
      return !absl::c_linear_search(inst->dimensions(), dimension);
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kReduceWindow:
      return inst->window().dimensions(dimension).size() == 1;
    case HloOpcode::kAsyncStart:
      if (!HloInstruction::IsThreadIncluded(inst->async_execution_thread(),
                                            execution_threads)) {
        // Async-start not included in specificed execution thread set will use
        // metadata-prefix version of dynamic shapes (result of
        // slice-to-dynamic) so there is no need to do pad on operand.
        return true;
      }
      return false;
    default:
      return false;
  }
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
HloInstruction* PadWithScalar(HloInstruction* inst, int64_t dim,
                              HloInstruction* dynamic_size,
                              HloInstruction* padding_scalar) {
  CHECK(inst != nullptr && dynamic_size != nullptr &&
        padding_scalar != nullptr);
  const Shape mask_shape =
      ShapeUtil::MakeShape(xla::S32, inst->shape().dimensions());
  const Shape pred_shape =
      ShapeUtil::MakeShape(xla::PRED, inst->shape().dimensions());
  HloInstruction* iota =
      inst->AddInstruction(HloInstruction::CreateIota(mask_shape, dim));

  HloInstruction* broadcasted_effective_size = inst->AddInstruction(
      HloInstruction::CreateBroadcast(mask_shape, dynamic_size, {}));
  HloInstruction* pred = inst->AddInstruction(HloInstruction::CreateCompare(
      pred_shape, iota, broadcasted_effective_size, ComparisonDirection::kLt));

  HloInstruction* broadcasted_identity_value =
      inst->AddInstruction(HloInstruction::CreateBroadcast(
          ShapeUtil::MakeStaticShape(inst->shape()), padding_scalar, {}));
  HloInstruction* padded = inst->AddInstruction(HloInstruction::CreateTernary(
      ShapeUtil::MakeStaticShape(inst->shape()), HloOpcode::kSelect, pred, inst,
      broadcasted_identity_value));
  return padded;
}

// Generate a 1-0 mask for input_dim where 1 means data in dynamic shape.
HloInstruction* GenerateBinaryMask(
    HloInstruction* reshape, int64_t input_dim,
    absl::Span<const int64_t> output_dims,
    absl::Span<HloInstruction*> output_dynamic_dims, HloInstruction* one,
    HloInstruction* zero, bool split_input) {
  Shape input_shape =
      split_input ? reshape->operand(0)->shape() : reshape->shape();
  Shape output_shape =
      split_input ? reshape->shape() : reshape->operand(0)->shape();
  const Shape mask_input_shape =
      ShapeUtil::MakeShape(xla::S32, {input_shape.dimensions(input_dim)});
  const Shape pred_input_shape =
      ShapeUtil::MakeShape(xla::PRED, {input_shape.dimensions(input_dim)});
  HloInstruction* pred_true = reshape->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<bool>(true)));
  HloInstruction* input_shape_pred_mask = reshape->AddInstruction(
      HloInstruction::CreateBroadcast(pred_input_shape, pred_true, {}));
  bool need_rewrite = false;
  // Iota contains a linear index for each element in input shape.
  HloInstruction* iota =
      reshape->AddInstruction(HloInstruction::CreateIota(mask_input_shape, 0));

  // Compute the multi-dimensional indices from a linear index and
  // compare to dynamic dimension size to generate the mask.
  // For a 2x3x3 shape, iota is first set to:
  //   [0, 1, 2, 3, 4, 5, 6, 7, 8, 9,10,11,12,13,14,15,16,17]
  // iota % 3 gives the index for the last dimension.
  //   [0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]
  // Then iota goes to:
  //   [0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5] (after div 3)
  // iota % 3 gives the index of the second last dimension.
  //   [0, 0, 0, 1, 1, 1, 2, 2, 2, 0, 0, 0, 1, 1, 1, 2, 2, 2]
  // Then iota goes to:
  //   [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1] (after div 3)
  // It gives the index of the major dimension.
  // For example, element 16 in the original iota will in the end get index
  // (1, 2, 1). Each index is used for generating the mask (if necessary) by
  // comparing to the dynamic size value for that dimension.
  //
  // Skip index 0 since there is no need to rewrite a major output dimension.
  for (int64_t i = 1; i < output_dims.size(); ++i) {
    if (output_dynamic_dims[output_dims[i]] != nullptr) {
      // If there is dynamic dimension in the output, need to rewrite the input.
      need_rewrite = true;
      break;
    }
  }
  if (!need_rewrite) {
    return nullptr;
  }

  for (int64_t i = output_dims.size() - 1; i > 0; i--) {
    const int64_t output_dim = output_dims[i];
    HloInstruction* dynamic_size = output_dynamic_dims[output_dim];
    HloInstruction* static_output_dim_size = reshape->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
            output_shape.dimensions(output_dim))));
    HloInstruction* broadcasted_static_output_dim_size =
        reshape->AddInstruction(HloInstruction::CreateBroadcast(
            mask_input_shape, static_output_dim_size, {}));
    if (dynamic_size != nullptr) {
      // Generate the mask for output_dim.
      HloInstruction* dim_index =
          reshape->AddInstruction(HloInstruction::CreateBinary(
              mask_input_shape, HloOpcode::kRemainder, iota,
              broadcasted_static_output_dim_size));
      HloInstruction* broadcasted_effective_size = reshape->AddInstruction(
          HloInstruction::CreateBroadcast(mask_input_shape, dynamic_size, {}));
      HloInstruction* selected =
          reshape->AddInstruction(HloInstruction::CreateCompare(
              pred_input_shape, dim_index, broadcasted_effective_size,
              ComparisonDirection::kLt));

      // Merge the mask.
      input_shape_pred_mask = reshape->AddInstruction(
          HloInstruction::CreateBinary(pred_input_shape, HloOpcode::kAnd,
                                       input_shape_pred_mask, selected));
    }

    // Update iota values by "shifting out" dimension i.
    iota = reshape->AddInstruction(
        HloInstruction::CreateBinary(mask_input_shape, HloOpcode::kDivide, iota,
                                     broadcasted_static_output_dim_size));
  }

  HloInstruction* broadcasted_one = reshape->AddInstruction(
      HloInstruction::CreateBroadcast(mask_input_shape, one, {}));
  HloInstruction* broadcasted_zero = reshape->AddInstruction(
      HloInstruction::CreateBroadcast(mask_input_shape, zero, {}));
  return reshape->AddInstruction(HloInstruction::CreateTernary(
      mask_input_shape, HloOpcode::kSelect, input_shape_pred_mask,
      broadcasted_one, broadcasted_zero));
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
// The way we do this is by a 4-steps cumsum-gather algorithm:
//
// 1.First we use the output shape to generate a binary 0-1 masking, which masks
// out the padded area of the flattened output shape:
// [1,1,0,1,1,0]
//
// 2.We then do a cumsum with the mask:
//  [1,2,2,3,4,4] and subtract it with 1:
//  [0,1,1,2,3,3]
//
// 3.Use the result of cumsum as gather indices to rearrange the original
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
// 4.Feed the sorted input to original reshape[6]->[2,3], we can now get the
// correct result:
//  [[a,b,P]
//   [c,d,P]]
//
absl::StatusOr<bool> RewriteDynamicReshapeSplitInput(
    HloInstruction* reshape, int64_t input_dim,
    absl::Span<const int64_t> output_dims,
    absl::Span<HloInstruction*> output_dynamic_dims,
    DynamicDimensionInference* dynamic_dimension_inference) {
  VLOG(2) << "Reshaping input dim " << input_dim << " to "
          << VectorString(output_dims);
  const Shape operand_shape = reshape->operand(0)->shape();
  TF_RET_CHECK(output_dims.size() > 1);

  const Shape mask_input_shape =
      ShapeUtil::MakeShape(xla::S32, {operand_shape.dimensions(input_dim)});
  const Shape pred_input_shape =
      ShapeUtil::MakeShape(xla::PRED, {operand_shape.dimensions(input_dim)});

  HloInstruction* zero = reshape->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
  HloInstruction* one = reshape->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::One(S32)));

  // Step 1 -- generate binary mask.
  HloInstruction* input_shape_binary_mask =
      GenerateBinaryMask(reshape, input_dim, output_dims, output_dynamic_dims,
                         one, zero, /*split_input=*/true);
  if (input_shape_binary_mask == nullptr) {
    // No need to rewrite.
    VLOG(2) << "No need to rewrite";
    return false;
  }

  // Step 2. Do a cumsum on the binary mask.

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
      reshape->AddInstruction(HloInstruction::CreateReduceWindow(
          mask_input_shape, input_shape_binary_mask, zero, cumsum_window, add));

  HloInstruction* broadcast_ones = reshape->AddInstruction(
      HloInstruction::CreateBroadcast(mask_input_shape, one, {}));
  cumsum = reshape->AddInstruction(HloInstruction::CreateBinary(
      mask_input_shape, HloOpcode::kSubtract, cumsum, broadcast_ones));

  GatherDimensionNumbers gather_dim_numbers;
  // Use gather to rearrange the input dim dimension.
  for (int64_t i = 0; i < operand_shape.dimensions_size(); ++i) {
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

  // Step 3. Gather.

  // Temporarily removes dynamic dimension before entering gather -- we want the
  // gather to ignore dynamic dimension.
  HloInstruction* operand_static_dim_size =
      reshape->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0<int32_t>(operand_shape.dimensions(input_dim))));
  HloInstruction* operand_static =
      reshape->AddInstruction(HloInstruction::CreateSetDimensionSize(
          operand_shape, reshape->mutable_operand(0), operand_static_dim_size,
          input_dim));

  std::vector<int64_t> slice_sizes(operand_shape.dimensions().begin(),
                                   operand_shape.dimensions().end());
  slice_sizes[input_dim] = 1;
  HloInstruction* gather = reshape->AddInstruction(HloInstruction::CreateGather(
      ShapeUtil::MakeShape(operand_shape.element_type(),
                           operand_shape.dimensions()),
      operand_static, cumsum, gather_dim_numbers, slice_sizes, true));

  // Step 4: Feed gather input to original reshape.

  TF_RETURN_IF_ERROR(reshape->ReplaceOperandWith(0, gather));

  HloInstruction* reshape_dynamic = reshape;

  auto users = reshape->users();

  // Forward the output dynamic dimension.
  for (int64_t output_dim : output_dims) {
    HloInstruction* output_dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(reshape, {}, output_dim);
    if (output_dynamic_size != nullptr) {
      reshape_dynamic =
          reshape->AddInstruction(HloInstruction::CreateSetDimensionSize(
              reshape->shape(), reshape_dynamic, output_dynamic_size,
              output_dim));
    }
  }

  for (auto* user : users) {
    TF_RETURN_IF_ERROR(reshape->ReplaceUseWith(user, reshape_dynamic));
  }
  TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
      reshape, reshape_dynamic, {}));

  return true;
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
// The way we do this is by a 4-steps sort-gather algorithm:
//
// 1.First we use the input shape to generate a binary 0-1 masking, which masks
// out the padded area of the output:
//  [1,1,0,1,1,0]
//
// 2.We then generate an iota mask using the output shape:
//  [0,1,2,3,4,5]
//
// 3.Stable sort the iota mask using the binary mask as key:
//  key  [1,1,0,1,1,0]
//  value[0,1,2,3,4,5]
//     | Sort by key
//     v
//  key  [1,1,1,1,0,0]
//  value[0,1,3,4,2,5]
//
// 4.Gather the original output [a,b,P,c,d,P] using the sorted iota mask:
//      original output       gather indices
//       [a,b,P,c,d,P]         [0,1,3,4,2,5]
//            |                    |
//          Gather ----------------+
//            |
//       [a,b,c,d,P,P]
//
absl::StatusOr<bool> RewriteDynamicReshapeCombineInput(
    HloInstruction* reshape, absl::Span<const int64_t> input_dims,
    int64_t output_dim, absl::Span<HloInstruction*> input_dynamic_dims,
    DynamicDimensionInference* dynamic_dimension_inference) {
  // Rewrite dynamic reshape into reshape followed by a sort, all padded
  // data will be moved to the end.
  HloInstruction* zero = reshape->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
  HloInstruction* one = reshape->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::One(S32)));
  const Shape output_shape = reshape->shape();
  const Shape input_shape = reshape->operand(0)->shape();
  const Shape mask_output_shape =
      ShapeUtil::MakeShape(xla::S32, {output_shape.dimensions(output_dim)});

  // Step 1.
  // Generate binary mask.
  HloInstruction* output_shape_binary_mask =
      GenerateBinaryMask(reshape, output_dim, input_dims, input_dynamic_dims,
                         one, zero, /*split_input=*/false);
  if (output_shape_binary_mask == nullptr) {
    VLOG(2) << "No need to rewrite";
    return false;
  }

  // Step 2.
  // Generate an iota with output shape.
  HloInstruction* iota =
      reshape->AddInstruction(HloInstruction::CreateIota(mask_output_shape, 0));

  // Step 3.
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
                                    rhs_key, ComparisonDirection::kGt));
  HloComputation* compare =
      reshape->GetModule()->AddEmbeddedComputation(comp_builder.Build());

  // Use mask_reshaped as key, sort reshaped data as value.
  HloInstruction* sort = reshape->AddInstruction(HloInstruction::CreateSort(
      ShapeUtil::MakeTupleShape({mask_output_shape, mask_output_shape}), 0,
      {output_shape_binary_mask, iota}, compare,
      /*is_stable=*/true));

  HloInstruction* gather_indices = reshape->AddInstruction(
      HloInstruction::CreateGetTupleElement(mask_output_shape, sort, 1));

  // Step 4.Gather the original output using the sorted iota mask:

  GatherDimensionNumbers gather_dim_numbers;
  // Use gather to rearrange the output dim dimension.
  for (int64_t i = 0; i < output_shape.dimensions_size(); ++i) {
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

  HloInstruction* static_dim_size = reshape->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
          reshape->shape().dimensions(output_dim))));

  // Temporarily removes dynamic dimension of the reshape before we send it to
  // the sort -- we want padded area to also participate in the gather.
  Shape reshape_static_shape = reshape->shape();
  reshape_static_shape.set_dynamic_dimension(output_dim, false);
  HloInstruction* reshape_static =
      reshape->AddInstruction(HloInstruction::CreateSetDimensionSize(
          reshape_static_shape, reshape, static_dim_size, output_dim));
  std::vector<int64_t> gather_slice_sizes(output_shape.dimensions().begin(),
                                          output_shape.dimensions().end());
  gather_slice_sizes[output_dim] = 1;
  HloInstruction* gather = reshape->AddInstruction(HloInstruction::CreateGather(
      output_shape, reshape_static, gather_indices, gather_dim_numbers,
      gather_slice_sizes, true));

  // Forward dynamic size to the newly created gather.
  HloInstruction* output_dynamic_size =
      dynamic_dimension_inference->GetDynamicSize(reshape, {}, output_dim);
  TF_RET_CHECK(output_dynamic_size != nullptr);
  gather = reshape->AddInstruction(HloInstruction::CreateSetDimensionSize(
      gather->shape(), gather, output_dynamic_size, output_dim));
  auto users = reshape->users();
  for (auto* user : users) {
    // Avoid cycles by not replacing the static reshape and get_dimension_size.
    if (user != reshape_static && user != output_dynamic_size) {
      TF_RETURN_IF_ERROR(reshape->ReplaceUseWith(user, gather));
    }
  }

  if (reshape == reshape->parent()->root_instruction()) {
    reshape->parent()->set_root_instruction(gather);
  }

  TF_RETURN_IF_ERROR(
      dynamic_dimension_inference->ForwardDynamicSize(reshape, gather, {}));

  return true;
}

absl::StatusOr<bool> RewriteDynamicReshapeSingleGroup(
    HloInstruction* reshape, absl::Span<const int64_t> input_dims,
    absl::Span<const int64_t> output_dims,
    absl::Span<HloInstruction*> input_dynamic_dims,
    absl::Span<HloInstruction*> output_dynamic_dims,
    DynamicDimensionInference* dynamic_dimension_inference) {
  VLOG(2) << "Rewriting dynamic reshape " << reshape->ToString()
          << " input dims: " << VectorString(input_dims)
          << " output dims: " << VectorString(output_dims);

  const Shape operand_shape = reshape->operand(0)->shape();
  const Shape output_shape = reshape->shape();

  if (input_dims.size() == 1) {
    int64_t input_dim = input_dims[0];
    // Size 1 dimension doesn't need a rewrite.
    if (operand_shape.dimensions()[input_dim] == 1) {
      return false;
    }
    // One input dimension is split into multiple output dimensions.
    return RewriteDynamicReshapeSplitInput(reshape, input_dim, output_dims,
                                           output_dynamic_dims,
                                           dynamic_dimension_inference);
  }

  if (output_dims.size() == 1) {
    int64_t output_dim = output_dims[0];
    if (output_shape.dimensions()[output_dim] == 1) {
      return false;
    }
    // One input dimension is split into multiple output dimensions.
    return RewriteDynamicReshapeCombineInput(reshape, input_dims, output_dim,
                                             input_dynamic_dims,
                                             dynamic_dimension_inference);
  }

  // Shouldn't get here.
  TF_RET_CHECK(false);
  return false;
}

absl::StatusOr<bool> RewriteReverse(
    HloInstruction* reverse,
    DynamicDimensionInference* dynamic_dimension_inference) {
  // When we have [A, B, C, D, E] and reverse them, we get [E, D, C, B, A].
  // However, if the dynamic size is 2, we expect B, A to be in front:
  // [B, A, P, P, P].
  //
  // We do this by running a pad and dynamic slice on the result:
  // [A, B, C, D, E]
  //      |
  //    reverse
  //      |
  // [E, D, C, B, A]
  //      |
  //     pad # Use pad to double the size of the dimension to avoid OOB.
  //      |
  // [E, D, C, B, A, P, P, P, P, P]
  //      |
  //  dynamic slice
  //      |
  // [B, A, P, P, P]
  auto reverse_dims = reverse->dimensions();
  const Shape& reverse_shape = reverse->shape();
  std::set<int64_t> dynamic_reverse_dims;
  for (int64_t reverse_dim : reverse_dims) {
    HloInstruction* dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(reverse, {}, reverse_dim);
    if (dynamic_size == nullptr) {
      // Reverse dimension is not dynamic -- no rewrite needed.
      continue;
    }
    dynamic_reverse_dims.insert(reverse_dim);
  }

  if (dynamic_reverse_dims.empty()) {
    // We only need to rewrite dynamic dimensions that are also reverse
    // dimensions.
    return false;
  }

  PaddingConfig padding;
  // Doubles dynamic dimension size using a pad.
  Shape pad_shape = reverse_shape;
  for (int i = 0; i < reverse_shape.rank(); ++i) {
    auto dimension = padding.add_dimensions();
    if (dynamic_reverse_dims.count(i) > 0) {
      dimension->set_edge_padding_low(0);
      dimension->set_edge_padding_high(reverse_shape.dimensions(i));
      dimension->set_interior_padding(0);
      pad_shape.set_dimensions(i, 2 * pad_shape.dimensions(i));
    }
  }
  HloInstruction* cloned_reverse = reverse->AddInstruction(reverse->Clone());
  HloInstruction* zero = reverse->AddInstruction(HloInstruction::CreateConstant(
      LiteralUtil::Zero(pad_shape.element_type())));
  HloInstruction* pad = reverse->AddInstruction(
      HloInstruction::CreatePad(pad_shape, cloned_reverse, zero, padding));
  std::vector<HloInstruction*> start_indices;
  start_indices.reserve(reverse_shape.rank());
  for (int i = 0; i < reverse_shape.rank(); ++i) {
    if (dynamic_reverse_dims.count(i) > 0) {
      // Start at bound_size - dynamic_size.
      HloInstruction* bound_size =
          reverse->AddInstruction(HloInstruction::CreateConstant(
              LiteralUtil::CreateR0<int32_t>(reverse_shape.dimensions(i))));
      HloInstruction* dynamic_size =
          dynamic_dimension_inference->GetDynamicSize(reverse, {}, i);
      HloInstruction* start_offset =
          reverse->AddInstruction(HloInstruction::CreateBinary(
              ShapeUtil::MakeScalarShape(S32), HloOpcode::kSubtract, bound_size,
              dynamic_size));
      start_indices.push_back(start_offset);
    } else {
      HloInstruction* zero = reverse->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
      start_indices.push_back(zero);
    }
  }
  HloInstruction* dynamic_reverse =
      reverse->AddInstruction(HloInstruction::CreateDynamicSlice(
          reverse_shape, pad, start_indices, reverse_shape.dimensions()));
  TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
      reverse, dynamic_reverse, {}));
  TF_RETURN_IF_ERROR(reverse->ReplaceAllUsesWith(dynamic_reverse));
  return true;
}

HloInstruction* RewriteInputWithDynamicPadding(
    HloInstruction* conv, HloInstruction* input, HloInstruction* padding_value,
    absl::Span<HloInstruction*> padding_before, Window* input_window,
    absl::FunctionRef<int64_t(int64_t)> window_dim_to_shape_dim) {
  HloInstruction* zero_s32 = conv->AddInstruction(
      HloInstruction::CreateConstant(LiteralUtil::Zero(S32)));
  // Padded shape represents the bounded shape after dynamic padding.
  Shape padded_shape = input->shape();
  PaddingConfig padding_configs;
  for (int64_t i = 0; i < input->shape().rank(); ++i) {
    PaddingConfig::PaddingConfigDimension padding_dim;
    *padding_configs.add_dimensions() = padding_dim;
  }
  std::vector<HloInstruction*> start_indices(input->shape().rank(), zero_s32);
  for (int64_t dim_index = 0; dim_index < input_window->dimensions_size();
       ++dim_index) {
    if (padding_before[dim_index] == nullptr) {
      continue;
    }
    int64_t shape_dim = window_dim_to_shape_dim(dim_index);

    WindowDimension* window_dim = input_window->mutable_dimensions(dim_index);
    auto* padding_dim = padding_configs.mutable_dimensions(shape_dim);
    const int64_t dilated_window_size = window_util::DilatedBound(
        window_dim->size(), window_dim->window_dilation());
    // Use dilated window size as low padding and static padding_high +
    // padding_low as high padding to make sure the following dynamic slice is
    // valid and doesn't go out of bound.
    //
    // See go/xla-dynamic-spatial-dim for more details.
    padding_dim->set_edge_padding_low(dilated_window_size);
    padding_dim->set_edge_padding_high(window_dim->padding_high() +
                                       window_dim->padding_low());
    padding_dim->set_interior_padding(window_dim->base_dilation() - 1);
    HloInstruction* slicing_start =
        conv->AddInstruction(HloInstruction::CreateBinary(
            ShapeUtil::MakeScalarShape(S32), HloOpcode::kSubtract,
            conv->AddInstruction(
                HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
                    padding_dim->edge_padding_low()))),
            padding_before[dim_index]));
    start_indices[shape_dim] = slicing_start;

    padded_shape.mutable_dimensions()[shape_dim] =
        window_dim->padding_low() +
        window_util::DilatedBound(padded_shape.dimensions(shape_dim),
                                  window_dim->base_dilation()) +
        window_dim->padding_high();
    window_dim->clear_padding_high();
    window_dim->clear_padding_low();
    window_dim->set_base_dilation(1);
    input->mutable_shape()->set_dynamic_dimension(shape_dim, false);
  }
  // Reconstruct dynamic padding using pad and dynamic slice.

  HloInstruction* pad =
      MakePadHlo(input, padding_value, padding_configs).value();
  input = conv->AddInstruction(HloInstruction::CreateDynamicSlice(
      padded_shape, pad, start_indices, padded_shape.dimensions()));
  return input;
}

absl::StatusOr<bool> RewriteDynamicConvolutionInputGrad(
    HloInstruction* custom_call_conv,
    DynamicDimensionInference* dynamic_dimension_inference) {
  HloInstruction* grad = custom_call_conv->mutable_operand(1);
  HloInstruction* kernel = custom_call_conv->mutable_operand(2);
  TF_RET_CHECK(kernel->shape().is_static());
  auto dnums = custom_call_conv->convolution_dimension_numbers();
  Window window = custom_call_conv->window();
  HloInstruction* zero =
      custom_call_conv->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(custom_call_conv->shape().element_type())));
  std::vector<HloInstruction*> padding_before(
      dnums.input_spatial_dimensions_size(), nullptr);
  for (int64_t spatial_dim_index = 0;
       spatial_dim_index < dnums.input_spatial_dimensions_size();
       ++spatial_dim_index) {
    int64_t input_spatial_dim =
        dnums.input_spatial_dimensions(spatial_dim_index);
    HloInstruction* operand_dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(
            custom_call_conv->mutable_operand(1), {}, input_spatial_dim);
    if (operand_dynamic_size == nullptr) {
      continue;
    }
    grad = PadWithScalar(grad, input_spatial_dim, operand_dynamic_size, zero);
    HloInstruction* slice =
        custom_call_conv->AddInstruction(HloInstruction::CreateSlice(
            ShapeUtil::MakeShape(S32, {1}),
            custom_call_conv->mutable_operand(0), {input_spatial_dim},
            {input_spatial_dim + 1}, {1}));
    HloInstruction* dynamic_input_size = custom_call_conv->AddInstruction(
        HloInstruction::CreateReshape(ShapeUtil::MakeScalarShape(S32), slice));
    const WindowDimension& window_dim = window.dimensions(spatial_dim_index);
    // Window stride of forward prop is same as base dilation of backward prop.
    DynamicWindowDims dynamic_window_dims = GetWindowedInputGradSize(
        dynamic_input_size, /*window_size=*/window_dim.size(),
        /*window_dilation=*/window_dim.window_dilation(),
        /*window_stride=*/window_dim.base_dilation(),
        custom_call_conv->padding_type());
    padding_before[spatial_dim_index] = dynamic_window_dims.padding_before;
  }

  if (custom_call_conv->padding_type() == PaddingType::PADDING_SAME) {
    grad = RewriteInputWithDynamicPadding(
        custom_call_conv, grad, zero, absl::MakeSpan(padding_before), &window,
        [&](int64_t dim) { return dnums.input_spatial_dimensions(dim); });
  }

  PrecisionConfig precision_config;
  if (custom_call_conv->precision_config().operand_precision_size() == 3) {
    // We are not interested in the precision config of the first operand, which
    // is the input_sizes.
    *precision_config.mutable_operand_precision() = {
        custom_call_conv->precision_config().operand_precision().begin() + 1,
        custom_call_conv->precision_config().operand_precision().end()};
  }
  HloInstruction* static_conv =
      custom_call_conv->AddInstruction(HloInstruction::CreateConvolve(
          custom_call_conv->shape(), grad, kernel,
          custom_call_conv->feature_group_count(),
          custom_call_conv->batch_group_count(), window,
          custom_call_conv->convolution_dimension_numbers(),
          custom_call_conv->precision_config()));
  TF_RETURN_IF_ERROR(custom_call_conv->ReplaceAllUsesWith(static_conv));
  TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
      custom_call_conv, static_conv, {}));
  return true;
}

absl::StatusOr<bool> RewriteDynamicConvolutionForward(
    HloInstruction* custom_call_conv,
    DynamicDimensionInference* dynamic_dimension_inference) {
  HloInstruction* input = custom_call_conv->mutable_operand(0);
  HloInstruction* kernel = custom_call_conv->mutable_operand(1);
  Window window = custom_call_conv->window();
  auto dnums = custom_call_conv->convolution_dimension_numbers();
  HloInstruction* zero =
      custom_call_conv->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(custom_call_conv->shape().element_type())));
  std::vector<HloInstruction*> padding_before(
      dnums.input_spatial_dimensions_size(), nullptr);
  for (int64_t spatial_dim_index = 0;
       spatial_dim_index < dnums.input_spatial_dimensions_size();
       ++spatial_dim_index) {
    int64_t input_spatial_dim =
        dnums.input_spatial_dimensions(spatial_dim_index);
    HloInstruction* operand_dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(
            custom_call_conv->mutable_operand(0), {}, input_spatial_dim);
    if (operand_dynamic_size == nullptr) {
      continue;
    }

    input = PadWithScalar(input, input_spatial_dim, operand_dynamic_size, zero);
    const WindowDimension& window_dim = window.dimensions(spatial_dim_index);
    DynamicWindowDims dynamic_window_dims = GetWindowedOutputSize(
        operand_dynamic_size, window_dim.size(), window_dim.window_dilation(),
        window_dim.stride(), custom_call_conv->padding_type());
    padding_before[spatial_dim_index] = dynamic_window_dims.padding_before;
  }
  // Input feature dim can be dynamic too, reset it to zero.
  const int64_t input_feature_dim = dnums.input_feature_dimension();
  if (HloInstruction* input_feature_dynamic_size =
          dynamic_dimension_inference->GetDynamicSize(
              custom_call_conv->mutable_operand(0), {}, input_feature_dim)) {
    input = PadWithScalar(input, input_feature_dim, input_feature_dynamic_size,
                          zero);
  }

  if (custom_call_conv->padding_type() == PaddingType::PADDING_SAME) {
    input = RewriteInputWithDynamicPadding(
        custom_call_conv, input, zero, absl::MakeSpan(padding_before), &window,
        [&](int64_t dim) { return dnums.input_spatial_dimensions(dim); });
  }

  HloInstruction* static_conv =
      custom_call_conv->AddInstruction(HloInstruction::CreateConvolve(
          custom_call_conv->shape(), input, kernel,
          custom_call_conv->feature_group_count(),
          custom_call_conv->batch_group_count(), window,
          custom_call_conv->convolution_dimension_numbers(),
          custom_call_conv->precision_config()));
  TF_RETURN_IF_ERROR(custom_call_conv->ReplaceAllUsesWith(static_conv));
  TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
      custom_call_conv, static_conv, {}));
  return true;
}

absl::StatusOr<bool> RewriteDynamicConvolutionKernelGrad(
    HloInstruction* custom_call_conv,
    DynamicDimensionInference* dynamic_dimension_inference) {
  HloInstruction* activations = custom_call_conv->mutable_operand(0);
  HloInstruction* gradients = custom_call_conv->mutable_operand(1);
  TF_RET_CHECK(dynamic_dimension_inference->HasDynamicDimension(activations));
  TF_RET_CHECK(dynamic_dimension_inference->HasDynamicDimension(gradients));
  Window window = custom_call_conv->window();
  auto dnums = custom_call_conv->convolution_dimension_numbers();
  HloInstruction* zero =
      custom_call_conv->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::Zero(custom_call_conv->shape().element_type())));
  std::vector<HloInstruction*> padding_before(
      dnums.input_spatial_dimensions_size(), nullptr);
  for (int64_t spatial_dim_index = 0;
       spatial_dim_index < dnums.input_spatial_dimensions_size();
       ++spatial_dim_index) {
    int64_t input_spatial_dim =
        dnums.input_spatial_dimensions(spatial_dim_index);
    int64_t kernel_spatial_dim =
        dnums.kernel_spatial_dimensions(spatial_dim_index);
    HloInstruction* activations_dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(
            custom_call_conv->mutable_operand(0), {}, input_spatial_dim);
    if (activations_dynamic_size != nullptr) {
      activations = PadWithScalar(activations, input_spatial_dim,
                                  activations_dynamic_size, zero);
    }

    HloInstruction* gradients_dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(
            custom_call_conv->mutable_operand(1), {}, kernel_spatial_dim);
    if (gradients_dynamic_size != nullptr) {
      gradients = PadWithScalar(gradients, kernel_spatial_dim,
                                gradients_dynamic_size, zero);
    }
    if (activations_dynamic_size == nullptr ||
        gradients_dynamic_size == nullptr) {
      TF_RET_CHECK(activations_dynamic_size == nullptr &&
                   gradients_dynamic_size == nullptr);
      continue;
    }
    int64_t output_spatial_dim =
        dnums.output_spatial_dimensions(spatial_dim_index);
    const WindowDimension& window_dim = window.dimensions(spatial_dim_index);
    DynamicWindowDims dynamic_window_dims = GetWindowedOutputSize(
        activations_dynamic_size, /*window_size=*/
        custom_call_conv->shape().dimensions(output_spatial_dim),
        /*window_dilation=*/window_dim.stride(),
        /*window_stride=*/window_dim.window_dilation(),
        custom_call_conv->padding_type());
    padding_before[spatial_dim_index] = dynamic_window_dims.padding_before;
  }

  // We only need to pad input feature on lhs to 0 -- it's mathematically
  // equivalent to padding both lhs and rhs to 0.
  const int64_t input_feature_dim = dnums.input_feature_dimension();
  if (HloInstruction* input_feature_dynamic_size =
          dynamic_dimension_inference->GetDynamicSize(
              custom_call_conv->mutable_operand(0), {}, input_feature_dim)) {
    activations = PadWithScalar(activations, input_feature_dim,
                                input_feature_dynamic_size, zero);
  }

  if (custom_call_conv->padding_type() == PaddingType::PADDING_SAME) {
    activations = RewriteInputWithDynamicPadding(
        custom_call_conv, activations, zero, absl::MakeSpan(padding_before),
        &window,
        [&](int64_t dim) { return dnums.input_spatial_dimensions(dim); });
  }

  HloInstruction* static_conv =
      custom_call_conv->AddInstruction(HloInstruction::CreateConvolve(
          custom_call_conv->shape(), activations, gradients,
          custom_call_conv->feature_group_count(),
          custom_call_conv->batch_group_count(), window,
          custom_call_conv->convolution_dimension_numbers(),
          custom_call_conv->precision_config()));
  TF_RETURN_IF_ERROR(custom_call_conv->ReplaceAllUsesWith(static_conv));
  TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
      custom_call_conv, static_conv, {}));
  return true;
}

absl::StatusOr<bool> RewriteDynamicReduceWindowSamePadding(
    HloInstruction* hlo,
    DynamicDimensionInference* dynamic_dimension_inference) {
  if (hlo->shape().IsTuple()) {
    // TODO (b/73062247) variadic reduce window is not yet supported here.
    return Unimplemented("DynamicReduceWindowSamePadding not yet supported.");
  }
  HloInstruction* input = hlo->mutable_operand(0);
  HloInstruction* init = hlo->mutable_operand(1);
  int64_t rank = hlo->shape().rank();
  Window window = hlo->window();
  std::vector<HloInstruction*> padding_before(hlo->shape().rank(), nullptr);
  for (int64_t dim_index = 0; dim_index < rank; ++dim_index) {
    HloInstruction* operand_dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(hlo->mutable_operand(0), {},
                                                    dim_index);
    if (operand_dynamic_size == nullptr) {
      continue;
    }
    const WindowDimension& window_dim = window.dimensions(dim_index);
    if (window_util::IsTrivialWindowDimension(window_dim)) {
      continue;
    }
    input = PadWithScalar(input, dim_index, operand_dynamic_size, init);

    DynamicWindowDims dynamic_window_dims = GetWindowedOutputSize(
        operand_dynamic_size, window_dim.size(), window_dim.window_dilation(),
        window_dim.stride(), PaddingType::PADDING_SAME);
    padding_before[dim_index] = dynamic_window_dims.padding_before;
  }

  input = RewriteInputWithDynamicPadding(
      hlo, input, init, absl::MakeSpan(padding_before), &window,
      [](int64_t dim) { return dim; });

  HloInstruction* rewritten =
      hlo->AddInstruction(HloInstruction::CreateReduceWindow(
          hlo->shape(), input, init, window, hlo->called_computations()[0]));
  TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(rewritten));
  TF_RETURN_IF_ERROR(
      dynamic_dimension_inference->ForwardDynamicSize(hlo, rewritten, {}));
  return true;
}

absl::StatusOr<bool> RewriteDynamicSelectAndScatterSamePadding(
    HloInstruction* hlo,
    DynamicDimensionInference* dynamic_dimension_inference) {
  HloInstruction* input = hlo->mutable_operand(0);
  HloInstruction* source = hlo->mutable_operand(1);
  HloInstruction* init = hlo->mutable_operand(2);
  TF_ASSIGN_OR_RETURN(HloInstruction * input_padding_value,
                      ChooseIdentityValue(hlo, /*operand_number=*/0));
  int64_t rank = hlo->shape().rank();
  Window window = hlo->window();
  std::vector<HloInstruction*> padding_before(hlo->shape().rank(), nullptr);
  for (int64_t dim_index = 0; dim_index < rank; ++dim_index) {
    const WindowDimension& window_dim = window.dimensions(dim_index);
    if (window_util::IsTrivialWindowDimension(window_dim)) {
      continue;
    }
    HloInstruction* operand_dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(hlo->mutable_operand(0), {},
                                                    dim_index);
    if (operand_dynamic_size == nullptr) {
      continue;
    }

    input = PadWithScalar(input, dim_index, operand_dynamic_size,
                          input_padding_value);

    HloInstruction* source_dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(hlo->mutable_operand(1), {},
                                                    dim_index);
    if (source_dynamic_size == nullptr) {
      continue;
    }
    source = PadWithScalar(source, dim_index, source_dynamic_size, init);

    DynamicWindowDims dynamic_window_dims = GetWindowedOutputSize(
        operand_dynamic_size, window_dim.size(), window_dim.window_dilation(),
        window_dim.stride(), PaddingType::PADDING_SAME);
    padding_before[dim_index] = dynamic_window_dims.padding_before;
  }

  input = RewriteInputWithDynamicPadding(
      hlo, input, input_padding_value, absl::MakeSpan(padding_before), &window,
      [](int64_t dim) { return dim; });

  // RewriteInputWithDynamicPadding adds padding to the input. However those
  // inputs should not be materialized in select and scatter's output and we
  // need to slice them out using dynamic slice. To prevent dynamic slicegoing
  // OOB, we first add some high-pad to the output to leave enough space.
  HloInstruction* rewritten =
      hlo->AddInstruction(HloInstruction::CreateSelectAndScatter(
          input->shape(), input, hlo->called_computations()[0], window, source,
          init, hlo->called_computations()[1]));
  std::vector<HloInstruction*> start_indices(
      input->shape().rank(), hlo->AddInstruction(HloInstruction::CreateConstant(
                                 LiteralUtil::Zero(S32))));
  PaddingConfig padding_configs;
  for (int64_t dim_index = 0; dim_index < rank; ++dim_index) {
    PaddingConfig::PaddingConfigDimension padding_dim;
    if (padding_before[dim_index] != nullptr) {
      const WindowDimension& window_dim = window.dimensions(dim_index);
      const int64_t dilated_window_size = window_util::DilatedBound(
          window_dim.size(), window_dim.window_dilation());
      padding_dim.set_edge_padding_high(dilated_window_size);
      start_indices[dim_index] = padding_before[dim_index];
    }
    *padding_configs.add_dimensions() = padding_dim;
  }
  HloInstruction* padded = MakePadHlo(rewritten, init, padding_configs).value();
  rewritten = hlo->AddInstruction(HloInstruction::CreateDynamicSlice(
      hlo->shape(), padded, start_indices, hlo->shape().dimensions()));
  TF_RETURN_IF_ERROR(hlo->ReplaceAllUsesWith(rewritten));
  TF_RETURN_IF_ERROR(
      dynamic_dimension_inference->ForwardDynamicSize(hlo, rewritten, {}));
  return true;
}

absl::StatusOr<bool> RewriteDynamicConcat(
    HloInstruction* concat,
    DynamicDimensionInference* dynamic_dimension_inference) {
  const int64_t concat_dim = concat->concatenate_dimension();
  if (dynamic_dimension_inference->GetDynamicSize(concat, {}, concat_dim) ==
      nullptr) {
    // Concat dimension is not dynamic -- no rewrite needed.
    return false;
  }
  std::vector<HloInstruction*> offsets;
  for (int64_t i = 0; i < concat->shape().dimensions_size(); ++i) {
    offsets.push_back(concat->AddInstruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(0))));
  }
  HloInstruction* rewritten_concat = concat;
  // Keep track of previous users before rewrite so that we can update their
  // operands later.
  auto prev_users = concat->users();
  for (int64_t i = 0; i < concat->operand_count(); ++i) {
    // Rewrite the concat by dynamic update slicing operand into the concat dim.
    HloInstruction* operand = concat->mutable_operand(i);
    rewritten_concat =
        concat->AddInstruction(HloInstruction::CreateDynamicUpdateSlice(
            rewritten_concat->shape(), rewritten_concat, operand, offsets));
    // Update the offset of concat dimension by adding the size of the concat
    // dimension of the operand to it.
    HloInstruction* dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(operand, {}, concat_dim);
    if (dynamic_size == nullptr) {
      HloInstruction* static_size = concat->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
              operand->shape().dimensions(concat_dim))));
      offsets[concat_dim] = concat->AddInstruction(HloInstruction::CreateBinary(
          ShapeUtil::MakeScalarShape(S32), HloOpcode::kAdd, offsets[concat_dim],
          static_size));
    } else {
      offsets[concat_dim] = concat->AddInstruction(HloInstruction::CreateBinary(
          ShapeUtil::MakeScalarShape(S32), HloOpcode::kAdd, offsets[concat_dim],
          dynamic_size));
    }
  }
  TF_RETURN_IF_ERROR(concat->ReplaceUsesWith(prev_users, rewritten_concat));
  TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
      concat, rewritten_concat, {}));
  return true;
}

absl::StatusOr<bool> RewriteDynamicSort(
    HloInstruction* hlo,
    DynamicDimensionInference* dynamic_dimension_inference) {
  HloInstruction* dynamic_size = nullptr;
  HloSortInstruction* sort = Cast<HloSortInstruction>(hlo);
  int64_t sort_dim = sort->sort_dimension();
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
  Shape broadcast_shape = ShapeUtil::MakeStaticShape(operand_shape);
  HloInstruction* iota = hlo->AddInstruction(
      HloInstruction::CreateIota(broadcast_shape, sort_dim));
  HloInstruction* dynamic_size_broadcasted = hlo->AddInstruction(
      HloInstruction::CreateBroadcast(broadcast_shape, dynamic_size, {}));
  HloInstruction* lt = hlo->AddInstruction(HloInstruction::CreateCompare(
      ShapeUtil::ChangeElementType(broadcast_shape, PRED), iota,
      dynamic_size_broadcasted, ComparisonDirection::kLt));
  sort->AppendOperand(lt);

  const int64_t param_number_before_rewritten =
      sort->called_computations()[0]->num_parameters();
  auto new_param_0 = HloInstruction::CreateParameter(
      param_number_before_rewritten, ShapeUtil::MakeScalarShape(PRED),
      "inbound_lhs");
  auto new_param_1 = HloInstruction::CreateParameter(
      param_number_before_rewritten + 1, ShapeUtil::MakeScalarShape(PRED),
      "inbound_rhs");
  std::vector<const HloInstruction*> extra_parameters{new_param_0.get(),
                                                      new_param_1.get()};
  HloComputation* sort_comp = sort->GetModule()->AddEmbeddedComputation(
      sort->called_computations()[0]->CloneWithReplacements(
          /*replacements=*/nullptr, extra_parameters));
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
  if (sort->shape().IsTuple()) {
    // For sort that is already tuple, simply add another result to the tuple.
    *sort->mutable_shape()->add_tuple_shapes() =
        ShapeUtil::ChangeElementType(operand_shape, PRED);
  } else {
    auto sort_users = sort->users();
    auto sort_clone = hlo->AddInstruction(sort->Clone());
    *sort_clone->mutable_shape() = ShapeUtil::MakeTupleShape(
        {sort->shape(), ShapeUtil::ChangeElementType(operand_shape, PRED)});
    auto rewritten_sort = hlo->AddInstruction(
        HloInstruction::CreateGetTupleElement(sort->shape(), sort_clone, 0));
    for (HloInstruction* user : sort_users) {
      TF_RETURN_IF_ERROR(sort->ReplaceUseWith(user, rewritten_sort));
    }
    TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
        sort, rewritten_sort, {}));
    if (hlo->parent()->root_instruction() == sort) {
      hlo->parent()->set_root_instruction(rewritten_sort);
    }
  }

  return true;
}

absl::StatusOr<bool> RewriteDynamicBinaryOp(
    HloInstruction* binary,
    DynamicDimensionInference* dynamic_dimension_inference) {
  HloInstruction* operand_0 = binary->mutable_operand(0);
  HloInstruction* operand_1 = binary->mutable_operand(1);

  TF_RET_CHECK(operand_0->shape().rank() == operand_1->shape().rank());
  auto dims_0 = dynamic_dimension_inference->GetDynamicSizes(operand_0, {});
  auto dims_1 = dynamic_dimension_inference->GetDynamicSizes(operand_1, {});
  bool changed = false;
  for (int64_t i = 0; i < dims_0.size(); ++i) {
    HloInstruction* dim_0 = dims_0[i];
    HloInstruction* dim_1 = dims_1[i];

    if (dims_0[i] != dims_1[i] && dims_0[i] != nullptr &&
        dims_1[i] != nullptr) {
      changed = true;
      // It is possible that a dynamic dimension of one operand is size 1 while
      // the other is greater than one. According to implicit broadcast
      // semantics, we need to insert broadcast in this case to make the dynamic
      // shape match.

      // An implicit broadcast is inserted by slicing the small shape into a
      // size 1 slice, reshape out the size 1 dimension then broadcast to the
      // full shape:
      //
      // Input [2, <=5, 3]
      //   |
      // Slice [2, 1, 3]
      //   |
      // Reshape [2, 3]
      //   |
      // Broadcast [2, 5, 3]
      auto rewrite_operand = [&](HloInstruction* pred,
                                 HloInstruction* operand) -> HloInstruction* {
        Shape static_shape = ShapeUtil::MakeStaticShape(operand->shape());
        pred = binary->AddInstruction(HloInstruction::CreateBroadcast(
            ShapeUtil::ChangeElementType(static_shape, PRED), pred, {}));
        Shape slice_shape = static_shape;
        slice_shape.set_dimensions(i, 1);
        std::vector<int64_t> start_indices(slice_shape.rank(), 0);
        std::vector<int64_t> strides(slice_shape.rank(), 1);
        HloInstruction* slice = binary->AddInstruction(
            HloInstruction::CreateSlice(slice_shape, operand, start_indices,
                                        slice_shape.dimensions(), strides));
        Shape reshape_shape = ShapeUtil::DeleteDimension(i, slice_shape);
        HloInstruction* reshape = binary->AddInstruction(
            HloInstruction::CreateReshape(reshape_shape, slice));
        std::vector<int64_t> broadcast_dims;
        broadcast_dims.reserve(static_shape.rank() - 1);
        // Broadcast to all dims execpt for i.
        for (int64_t j = 0; j < static_shape.rank(); ++j) {
          if (j != i) {
            broadcast_dims.push_back(j);
          }
        }

        HloInstruction* broadcast = binary->parent()->AddInstruction(
            HloInstruction::CreateBroadcast(static_shape, reshape,
                                            broadcast_dims),
            "implicit_broadcast");

        // Use a select instead of conditional as elementwise operations promote
        // more fusion.
        HloInstruction* select =
            binary->AddInstruction(HloInstruction::CreateTernary(
                static_shape, HloOpcode::kSelect, pred, broadcast, operand));
        return select;
      };

      HloInstruction* one = binary->AddInstruction(
          HloInstruction::CreateConstant(LiteralUtil::One(S32)));

      auto operand_0_needs_broadcast = binary->parent()->AddInstruction(
          HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), dim_0,
                                        dim_1, ComparisonDirection::kLt),
          "lhs_less_than_rhs");
      auto is_one = binary->parent()->AddInstruction(
          HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), dim_0,
                                        one, ComparisonDirection::kEq),
          "lhs_is_one");
      operand_0_needs_broadcast = binary->parent()->AddInstruction(
          HloInstruction::CreateBinary(ShapeUtil::MakeShape(PRED, {}),
                                       HloOpcode::kAnd, is_one,
                                       operand_0_needs_broadcast),
          "lhs_needs_implicit_broadcast");
      operand_0 = rewrite_operand(operand_0_needs_broadcast, operand_0);

      auto operand_1_needs_broadcast = binary->parent()->AddInstruction(
          HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), dim_1,
                                        dim_0, ComparisonDirection::kLt),
          "rhs_less_than_lhs");
      is_one = binary->parent()->AddInstruction(
          HloInstruction::CreateCompare(ShapeUtil::MakeShape(PRED, {}), dim_1,
                                        one, ComparisonDirection::kEq),
          "rhs_is_one");
      operand_1_needs_broadcast = binary->parent()->AddInstruction(
          HloInstruction::CreateBinary(ShapeUtil::MakeShape(PRED, {}),
                                       HloOpcode::kAnd, is_one,
                                       operand_1_needs_broadcast),
          "lhs_needs_implicit_broadcast");
      operand_1 = rewrite_operand(operand_1_needs_broadcast, operand_1);
    }
  }
  if (changed) {
    TF_RETURN_IF_ERROR(binary->ReplaceOperandWith(0, operand_0));
    TF_RETURN_IF_ERROR(binary->ReplaceOperandWith(1, operand_1));
  }
  return changed;
}

absl::StatusOr<bool> RewriteDynamicUpdateSlice(
    HloInstruction* hlo,
    DynamicDimensionInference* dynamic_dimension_inference) {
  HloDynamicUpdateSliceInstruction* dus =
      Cast<HloDynamicUpdateSliceInstruction>(hlo);
  // Suppose we have a base area that we want to update:
  // +------------------------+
  // |                        |
  // |                  base  |
  // |                        |
  // +------------------------+
  //
  // A partial update with dynamic padding looks like this:
  //
  //           +------+-------+
  //           |update|padding|
  //           +------+-------+
  //
  // We don't want the padding to overwrite the base area:
  //
  // +------------------------+
  // |         +------+-------+
  // |<-begin->|update|padding| (what we want to avoid)
  // |         +------+-------+
  // +------------------------+
  //
  // Instead we want to keep the base area untouched except for the update
  // region:
  //
  // +------------------------+
  // |         +------+       |
  // |<-begin->|update|  base | (what we want)
  // |         +------+       |
  // +------------------------+
  //
  // We do this by dynamic slicing the base area out first with the same begin
  // index:
  //
  //           +--------------+
  // <-begin-> |         base |
  //           +--------------+
  //
  // Then replace the update's padding part with base:
  //
  //           +------+-------+
  //           |update|  base |
  //           +------+-------+
  //
  // Then do the DUS.

  HloInstruction* update = dus->mutable_operand(1);
  HloInstruction* base = dus->mutable_operand(0);
  std::vector<HloInstruction*> dynamic_dims_in_partial_update(
      update->shape().rank(), nullptr);
  bool needs_rewrite = false;
  for (int64_t i = 0; i < update->shape().rank(); ++i) {
    if (update->shape().dimensions(i) < base->shape().dimensions(i)) {
      HloInstruction* dynamic_dim =
          dynamic_dimension_inference->GetDynamicSize(update, {}, i);

      if (dynamic_dim != nullptr) {
        dynamic_dims_in_partial_update[i] = dynamic_dim;
        needs_rewrite = true;
      }
    }
  }

  if (!needs_rewrite) {
    return false;
  }
  std::vector<HloInstruction*> indices;
  indices.reserve(dus->operand_count() - 2);
  for (int64_t i = 2; i < dus->operand_count(); ++i) {
    indices.push_back(dus->mutable_operand(i));
  }
  HloInstruction* base_slice =
      dus->AddInstruction(HloInstruction::CreateDynamicSlice(
          update->shape(), base, indices, update->shape().dimensions()));

  for (int64_t i = 0; i < dynamic_dims_in_partial_update.size(); ++i) {
    HloInstruction* dynamic_dim = dynamic_dims_in_partial_update[i];
    if (dynamic_dim != nullptr) {
      Shape mask_shape_int = ShapeUtil::ChangeElementType(update->shape(), S32);
      Shape mask_shape_pred =
          ShapeUtil::ChangeElementType(update->shape(), PRED);
      // Generate mask using iota and dynamic_dim.
      HloInstruction* iota =
          dus->AddInstruction(HloInstruction::CreateIota(mask_shape_int, i));
      HloInstruction* broadcast_dim = dus->AddInstruction(
          HloInstruction::CreateBroadcast(mask_shape_int, dynamic_dim, {}));
      HloInstruction* pred = dus->AddInstruction(HloInstruction::CreateCompare(
          mask_shape_pred, iota, broadcast_dim, ComparisonDirection::kLt));
      // Update `update` to include base.
      update = dus->AddInstruction(HloInstruction::CreateTernary(
          update->shape(), HloOpcode::kSelect, pred, update, base_slice));
    }
  }
  TF_RETURN_IF_ERROR(dus->ReplaceOperandWith(1, update));

  return true;
}

absl::StatusOr<bool> RewriteDynamicReshape(
    HloInstruction* reshape,
    DynamicDimensionInference* dynamic_dimension_inference) {
  bool changed = false;
  HloInstruction* operand = reshape->mutable_operand(0);
  std::vector<HloInstruction*> input_dynamic_dims;
  for (int64_t dim = 0; dim < operand->shape().dimensions_size(); ++dim) {
    input_dynamic_dims.push_back(
        dynamic_dimension_inference->GetDynamicSize(operand, {}, dim));
  }

  std::vector<HloInstruction*> output_dynamic_dims;
  for (int64_t dim = 0; dim < reshape->shape().dimensions_size(); ++dim) {
    output_dynamic_dims.push_back(
        dynamic_dimension_inference->GetDynamicSize(reshape, {}, dim));
  }

  auto common_factors = CommonFactors(operand->shape().dimensions(),
                                      reshape->shape().dimensions());

  // Scan first to see if we need to decompose the reshape to a
  // flatten-unflatten pair.
  bool need_flatten_unflatten = false;
  auto is_dynamic_dimension = [&](int64_t dim) {
    HloInstruction* operand_dynamic_size =
        dynamic_dimension_inference->GetDynamicSize(reshape, {}, dim);
    return operand_dynamic_size != nullptr ||
           reshape->shape().is_dynamic_dimension(dim);
  };

  auto should_skip_common_factor_group = [&](DimensionVector input_dims,
                                             DimensionVector output_dims) {
    if (input_dims.empty() || output_dims.empty()) {
      return true;
    }
    if (absl::c_none_of(output_dims, is_dynamic_dimension)) {
      // Don't need to rewrite any group without dynamic dimensions.
      VLOG(2) << "All dimensions are static in this common factor group";
      return true;
    }
    if (input_dims.size() == 1 && output_dims.size() == 1) {
      // The dimension is unchanged. No rewrite needed.
      return true;
    }
    return false;
  };

  for (int64_t i = 0; i < common_factors.size() - 1; ++i) {
    auto start = common_factors[i];
    auto end = common_factors[i + 1];
    DimensionVector input_dims;
    DimensionVector output_dims;
    for (int64_t dim = start.first; dim < end.first; ++dim) {
      input_dims.push_back(dim);
    }
    for (int64_t dim = start.second; dim < end.second; ++dim) {
      output_dims.push_back(dim);
    }
    if (should_skip_common_factor_group(input_dims, output_dims)) {
      continue;
    }
    if (input_dims.size() > 1 && output_dims.size() > 1) {
      need_flatten_unflatten = true;
      break;
    }
  }

  if (need_flatten_unflatten) {
    VLOG(2) << "Rewrite dynamic reshape to flatten-unflatten pair. "
            << reshape->ToString();
    int64_t num_elements = ShapeUtil::ElementsIn(operand->shape());
    Shape flattened_shape =
        ShapeUtil::MakeShape(operand->shape().element_type(), {num_elements});
    HloInstruction* flatten = operand->parent()->AddInstruction(
        HloInstruction::CreateReshape(flattened_shape, operand),
        absl::StrCat(reshape->name(), ".flatten"));

    HloInstruction* dynamic_size =
        operand->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(num_elements)));
    for (int64_t i = 0; i < operand->shape().rank(); i++) {
      HloInstruction* dynamic_dim_size =
          dynamic_dimension_inference->GetDynamicSize(operand, {}, i);
      if (dynamic_dim_size != nullptr) {
        HloInstruction* static_dim_size = operand->AddInstruction(
            HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(
                operand->shape().dimensions(i))));
        dynamic_size = operand->AddInstruction(HloInstruction::CreateBinary(
            dynamic_size->shape(), HloOpcode::kDivide, dynamic_size,
            static_dim_size));
        dynamic_size = operand->AddInstruction(HloInstruction::CreateBinary(
            dynamic_size->shape(), HloOpcode::kMultiply, dynamic_size,
            dynamic_dim_size));
      }
    }
    dynamic_dimension_inference->SetDynamicSize(flatten, {}, 0, dynamic_size);

    Shape unflattened_shape = ShapeUtil::MakeStaticShape(reshape->shape());
    HloInstruction* unflatten = reshape->parent()->AddInstruction(
        HloInstruction::CreateReshape(unflattened_shape, flatten),
        absl::StrCat(reshape->name(), ".unflatten"));
    TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
        reshape, unflatten, {}));

    TF_ASSIGN_OR_RETURN(
        bool changed_unused,
        RewriteDynamicReshape(flatten, dynamic_dimension_inference));
    TF_ASSIGN_OR_RETURN(
        changed_unused,
        RewriteDynamicReshape(unflatten, dynamic_dimension_inference));

    TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
        reshape, unflatten, {}));
    TF_RETURN_IF_ERROR(reshape->ReplaceAllUsesWith(unflatten));

    return true;
  }

  // Find common_factors that the input belongs to.
  for (int64_t i = 0; i < common_factors.size() - 1; ++i) {
    auto start = common_factors[i];
    auto end = common_factors[i + 1];
    DimensionVector input_dims;
    DimensionVector output_dims;
    for (int64_t dim = start.first; dim < end.first; ++dim) {
      input_dims.push_back(dim);
    }
    for (int64_t dim = start.second; dim < end.second; ++dim) {
      output_dims.push_back(dim);
    }

    VLOG(2) << "input_dims: " << VectorString(input_dims);
    VLOG(2) << "output_dims: " << VectorString(output_dims);

    if (should_skip_common_factor_group(input_dims, output_dims)) {
      continue;
    }
    if (input_dims.size() > 1 && output_dims.size() > 1) {
      return Internal(
          "Should be handled by decomposing reshape into "
          "flatten-unflatten pair. %s",
          reshape->ToString());
    }

    TF_ASSIGN_OR_RETURN(bool c, RewriteDynamicReshapeSingleGroup(
                                    reshape, input_dims, output_dims,
                                    absl::MakeSpan(input_dynamic_dims),
                                    absl::MakeSpan(output_dynamic_dims),
                                    dynamic_dimension_inference));
    changed |= c;
  }

  if (reshape->opcode() == HloOpcode::kDynamicReshape) {
    auto* static_reshape =
        reshape->AddInstruction(HloInstruction::CreateReshape(
            reshape->shape(), reshape->mutable_operand(0)));
    TF_RETURN_IF_ERROR(reshape->ReplaceAllUsesWith(static_reshape));
    TF_RETURN_IF_ERROR(dynamic_dimension_inference->ForwardDynamicSize(
        reshape, static_reshape, {}));
    changed = true;
  }

  return changed;
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
class DynamicShapeRemovingVisitor : public DfsHloRewriteVisitor {
 public:
  explicit DynamicShapeRemovingVisitor(
      const OpSupportsDynamismHandler& op_supports_dynamism_handler,
      DynamicDimensionInference* dynamic_dimension_inference,
      const absl::flat_hash_set<absl::string_view>& execution_threads)
      : op_supports_dynamism_handler_(op_supports_dynamism_handler),
        dynamic_dimension_inference_(dynamic_dimension_inference),
        execution_threads_(execution_threads) {}

  absl::Status DefaultAction(HloInstruction* hlo) override;

  absl::Status HandleCustomCall(HloInstruction* hlo) override;

  absl::Status HandleTuple(HloInstruction* hlo) override;
  absl::Status HandleGetTupleElement(HloInstruction* hlo) override;

  absl::Status HandleParameter(HloInstruction* hlo) override;
  absl::Status HandleInfeed(HloInstruction* hlo) override;

  absl::Status HandleAsyncStart(HloInstruction* hlo) override;
  absl::Status HandleAsyncUpdate(HloInstruction* hlo) override;
  absl::Status HandleAsyncDone(HloInstruction* hlo) override;

  absl::Status HandleWhile(HloInstruction* hlo) override;
  absl::Status HandleConditional(HloInstruction* hlo) override;

  absl::Status HandleGetDimensionSize(HloInstruction* hlo) override;
  absl::Status HandleSetDimensionSize(HloInstruction* hlo) override;

  static absl::StatusOr<bool> Run(
      HloComputation* computation,
      const OpSupportsDynamismHandler& op_supports_dynamism_handler,
      DynamicDimensionInference* dynamic_shape_inference,
      const absl::flat_hash_set<absl::string_view>& execution_threads,
      bool require_dynamic_output) {
    DynamicShapeRemovingVisitor visitor(op_supports_dynamism_handler,
                                        dynamic_shape_inference,
                                        execution_threads);
    TF_RETURN_IF_ERROR(computation->Accept(&visitor));
    // If the outputs is required to be dynamic form, insert static to dynamic
    // conversion as root.
    if (require_dynamic_output) {
      HloInstruction* root = computation->root_instruction();
      if (dynamic_shape_inference->HasDynamicDimension(root)) {
        TF_ASSIGN_OR_RETURN(HloInstruction * new_root,
                            visitor.ConvertToDynamic(root));
        computation->set_root_instruction(new_root);
      }
    }
    return visitor.changed();
  }

 private:
  // If a tensor produced by `inst` is in static form, convert it to dynamic and
  // returns the new instruction.
  absl::StatusOr<HloInstruction*> ConvertToDynamic(HloInstruction* inst);

  // Same as above, but for all of the instructions operands.  The operands will
  // be replaced by dynamic operands as needed.
  absl::Status ConvertOperandsToDynamic(HloInstruction* inst);

  const OpSupportsDynamismHandler& op_supports_dynamism_handler_;

  DynamicDimensionInference* dynamic_dimension_inference_;

  absl::flat_hash_set<absl::string_view> execution_threads_;
};

absl::StatusOr<HloInstruction*> DynamicShapeRemovingVisitor::ConvertToDynamic(
    HloInstruction* inst) {
  if (!dynamic_dimension_inference_->HasDynamicDimension(inst)) {
    return absl::OkStatus();
  }
  MarkAsChanged();
  Shape shape = dynamic_dimension_inference_->GetDynamicShape(inst);
  auto gtes = TupleUtil::DisassembleTupleInstruction(inst);

  gtes.ForEachMutableElement([&](const ShapeIndex& index,
                                 HloInstruction** element) {
    const Shape& subshape = ShapeUtil::GetSubshape(shape, index);
    if (!subshape.IsArray()) {
      return;
    }
    if (!dynamic_dimension_inference_->HasDynamicDimension(inst, index)) {
      return;
    }
    // Collect the data input, as well as dimension sizes, and feed them to
    // slice to dynamic to create a dynamic tensor.
    std::vector<HloInstruction*> slice_operand;
    slice_operand.push_back(*element);
    for (int64_t i = 0; i < subshape.dimensions_size(); ++i) {
      auto dimension_size =
          dynamic_dimension_inference_->GetDynamicSize(inst, index, i);
      if (dimension_size == nullptr) {
        dimension_size = inst->AddInstruction(HloInstruction::CreateConstant(
            LiteralUtil::CreateR0<int32_t>(subshape.dimensions(i))));
      }
      slice_operand.push_back(dimension_size);
    }
    *element = inst->AddInstruction(HloInstruction::CreateCustomCall(
        subshape, slice_operand, "SliceToDynamic"));
  });

  return TupleUtil::AssembleTupleInstruction(inst->parent(), std::move(gtes));
}

absl::Status DynamicShapeRemovingVisitor::ConvertOperandsToDynamic(
    HloInstruction* inst) {
  for (int64_t i = 0; i < inst->operand_count(); ++i) {
    auto operand = inst->mutable_operand(i);
    if (dynamic_dimension_inference_->HasDynamicDimension(operand)) {
      TF_ASSIGN_OR_RETURN(auto dynamic_operand,
                          ConvertToDynamic(inst->mutable_operand(i)));
      TF_RETURN_IF_ERROR(inst->ReplaceOperandWith(i, dynamic_operand));
      MarkAsChanged();
    }
  }
  return absl::OkStatus();
}

absl::Status DynamicShapeRemovingVisitor::DefaultAction(HloInstruction* hlo) {
  // By default, ops don't support dynamic lowering.
  OpDynamismSupport op_support = OpDynamismSupport::kNoSupport;
  if (op_supports_dynamism_handler_) {
    op_support = op_supports_dynamism_handler_(hlo);
  }

  // If the op requires dynamic tensor and input is static -- construct a
  // dynamic tensor from the static tensor to feed it.
  if (op_support == OpDynamismSupport::kRequired) {
    VLOG(1) << "op doesn't support static tensor: " << hlo->ToString();
    return ConvertOperandsToDynamic(hlo);
  }

  const bool input_is_dynamic = absl::c_any_of(
      hlo->operands(),
      [](const HloInstruction* hlo) { return hlo->shape().is_dynamic(); });

  // If the input to an op is static, we are done.
  if (!input_is_dynamic) {
    return absl::OkStatus();
  }

  // Op doesn't support dynamic tensor, but by now we should have already
  // removed the dynamic dimensions for such ops.
  TF_RET_CHECK(op_support != OpDynamismSupport::kNoSupport)
      << "Dynamic input unexpectedly found for unsupported instruction: "
      << hlo->ToString();

  return absl::OkStatus();
}

absl::Status DynamicShapeRemovingVisitor::HandleGetTupleElement(
    HloInstruction* hlo) {
  return absl::OkStatus();
}

absl::Status DynamicShapeRemovingVisitor::HandleTuple(HloInstruction* hlo) {
  return absl::OkStatus();
}

absl::Status DynamicShapeRemovingVisitor::HandleInfeed(HloInstruction* hlo) {
  return absl::OkStatus();
}

absl::Status DynamicShapeRemovingVisitor::HandleParameter(HloInstruction* hlo) {
  return absl::OkStatus();
}

absl::Status DynamicShapeRemovingVisitor::HandleCustomCall(
    HloInstruction* hlo) {
  if (hlo->custom_call_target() == "SliceToDynamic" ||
      hlo->custom_call_target() == "PadToStatic") {
    // Those ops support are created to handle dynamic tensors so by their
    // nature they support dynamic lowering.
    return absl::OkStatus();
  }

  return DefaultAction(hlo);
}

absl::Status DynamicShapeRemovingVisitor::HandleAsyncStart(
    HloInstruction* hlo) {
  if (HloInstruction::IsThreadIncluded(hlo->async_execution_thread(),
                                       execution_threads_)) {
    return absl::OkStatus();
  }
  return ConvertOperandsToDynamic(hlo);
}

absl::Status DynamicShapeRemovingVisitor::HandleAsyncUpdate(
    HloInstruction* hlo) {
  return absl::OkStatus();
}

absl::Status DynamicShapeRemovingVisitor::HandleAsyncDone(HloInstruction* hlo) {
  return absl::OkStatus();
}

absl::Status DynamicShapeRemovingVisitor::HandleWhile(HloInstruction* hlo) {
  return absl::OkStatus();
}

absl::Status DynamicShapeRemovingVisitor::HandleConditional(
    HloInstruction* hlo) {
  return absl::OkStatus();
}

absl::Status DynamicShapeRemovingVisitor::HandleGetDimensionSize(
    HloInstruction* hlo) {
  return absl::OkStatus();
}

absl::Status DynamicShapeRemovingVisitor::HandleSetDimensionSize(
    HloInstruction* hlo) {
  *hlo->mutable_shape() = hlo->operand(0)->shape();
  hlo->mutable_shape()->set_dynamic_dimension(hlo->dimension(), false);
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<bool> DynamicPadder::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  VLOG(2) << "Pre DynamicPadder HLO:";
  XLA_VLOG_LINES(2, module->ToString());

  // Run DCE before inference, in case earlier passes left dead instructions
  // that could cause us to insert PadToStatic when it isn't desired.
  HloDCE dce;
  TF_ASSIGN_OR_RETURN(bool changed, dce.Run(module, execution_threads));

  TF_ASSIGN_OR_RETURN(
      DynamicDimensionInference dynamic_dimension_inference,
      DynamicDimensionInference::Run(
          module, options_.op_supports_dynamism_handler,
          options_.custom_call_handler, options_.shape_check_mode,
          options_.assertion_generator, execution_threads));

  changed |= dynamic_dimension_inference.changed();
  std::vector<HloComputation*> computations =
      module->MakeComputationPostOrder(execution_threads);

  for (HloComputation* computation : computations) {
    for (HloInstruction* inst : computation->MakeInstructionPostOrder()) {
      OpDynamismSupport has_dynamism_support = OpDynamismSupport::kNoSupport;
      if (options_.op_supports_dynamism_handler != nullptr) {
        has_dynamism_support = options_.op_supports_dynamism_handler(inst);
      }
      // This op support dynamic lowering, no padding is required.
      if (has_dynamism_support != OpDynamismSupport::kNoSupport) {
        continue;
      }
      if (inst->opcode() == HloOpcode::kConcatenate) {
        TF_ASSIGN_OR_RETURN(
            bool c, RewriteDynamicConcat(inst, &dynamic_dimension_inference));
        changed |= c;
        continue;
      }
      if (inst->opcode() == HloOpcode::kReverse) {
        TF_ASSIGN_OR_RETURN(bool c,
                            RewriteReverse(inst, &dynamic_dimension_inference));
        changed |= c;
        continue;
      }
      if (inst->opcode() == HloOpcode::kSort) {
        TF_ASSIGN_OR_RETURN(
            bool c, RewriteDynamicSort(inst, &dynamic_dimension_inference));
        changed |= c;
        continue;
      }
      if (inst->opcode() == HloOpcode::kReshape ||
          inst->opcode() == HloOpcode::kDynamicReshape) {
        TF_ASSIGN_OR_RETURN(
            bool c, RewriteDynamicReshape(inst, &dynamic_dimension_inference));
        changed |= c;
        continue;
      }

      // Elementwise binary with dynamic shapes have implicit broadcast
      // semantics.
      if (inst->IsElementwiseBinary()) {
        TF_ASSIGN_OR_RETURN(
            bool c, RewriteDynamicBinaryOp(inst, &dynamic_dimension_inference));
        changed |= c;
        continue;
      }

      if (inst->opcode() == HloOpcode::kDynamicUpdateSlice) {
        TF_ASSIGN_OR_RETURN(bool c, RewriteDynamicUpdateSlice(
                                        inst, &dynamic_dimension_inference));
        changed |= c;
        continue;
      }

      if (inst->IsCustomCall("DynamicConvolutionInputGrad")) {
        TF_ASSIGN_OR_RETURN(bool c, RewriteDynamicConvolutionInputGrad(
                                        inst, &dynamic_dimension_inference));
        changed |= c;
        continue;
      }

      if (inst->IsCustomCall("DynamicConvolutionForward")) {
        TF_ASSIGN_OR_RETURN(bool c, RewriteDynamicConvolutionForward(
                                        inst, &dynamic_dimension_inference));
        changed |= c;
        continue;
      }

      if (inst->IsCustomCall("DynamicConvolutionKernelGrad")) {
        TF_ASSIGN_OR_RETURN(bool c, RewriteDynamicConvolutionKernelGrad(
                                        inst, &dynamic_dimension_inference));
        changed |= c;
        continue;
      }

      if (inst->IsCustomCall("DynamicReduceWindowSamePadding")) {
        TF_ASSIGN_OR_RETURN(bool c, RewriteDynamicReduceWindowSamePadding(
                                        inst, &dynamic_dimension_inference));
        changed |= c;
        continue;
      }

      if (inst->IsCustomCall("DynamicSelectAndScatterSamePadding")) {
        TF_ASSIGN_OR_RETURN(bool c, RewriteDynamicSelectAndScatterSamePadding(
                                        inst, &dynamic_dimension_inference));
        changed |= c;
        continue;
      }

      for (int64_t operand_num = 0; operand_num < inst->operand_count();
           ++operand_num) {
        HloInstruction* original_operand = inst->mutable_operand(operand_num);
        HloInstruction* operand = original_operand;
        if (!operand->shape().IsArray()) {
          continue;
        }

        for (int64_t input_dim = 0; input_dim < operand->shape().rank();
             ++input_dim) {
          HloInstruction* operand_dynamic_size =
              dynamic_dimension_inference.GetDynamicSize(original_operand, {},
                                                         input_dim);
          if (operand_dynamic_size == nullptr) {
            continue;
          }
          VLOG(2) << "Has dynamic dimension of operand" << operand_num << " @"
                  << input_dim;

          if (ShouldSkipPadOnOperand(inst, operand_num, input_dim,
                                     execution_threads)) {
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
  // DynamicDimensionInference can leave behind dead, partially inferred
  // computations, but we want to ensure that ops that do not support dynamic
  // shapes do not remain once the DynamicPadder is done.  So we filter out
  // those computations using a CallGraph.
  auto call_graph = CallGraph::Build(module, execution_threads);
  computations = module->MakeComputationPostOrder(execution_threads);
  // Reverse postorder so that if caller doesn't support dynamic tensor, change
  // their called computation to only take static tensors.
  for (auto it = computations.rbegin(); it != computations.rend(); ++it) {
    HloComputation* computation = *it;
    if (!call_graph->CanReach(module->entry_computation(), computation)) {
      continue;
    }
    // if slice_dynamic_output_ is set and this is entry computation, we need
    // the output tensor to be in dynamic form.
    bool require_dynamic_output = options_.slice_dynamic_output &&
                                  computation == module->entry_computation();
    changed |= require_dynamic_output;
    TF_ASSIGN_OR_RETURN(bool c,
                        DynamicShapeRemovingVisitor::Run(
                            computation, options_.op_supports_dynamism_handler,
                            &dynamic_dimension_inference, execution_threads,
                            /*require_dynamic_output=*/require_dynamic_output));
    changed |= c;
  }

  if (changed) {
    dynamic_padding_gauge->GetCell()->Set(changed);
    module->set_is_dynamic(true);
  }

  for (auto* computation : module->computations(execution_threads)) {
    if (!call_graph->CanReach(module->entry_computation(), computation)) {
      continue;
    }
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(
          bool c, ReplaceGetSize(instruction, &dynamic_dimension_inference));
      changed |= c;
    }
  }

  for (auto* computation : module->computations(execution_threads)) {
    if (!call_graph->CanReach(module->entry_computation(), computation)) {
      continue;
    }
    for (auto instruction : computation->MakeInstructionPostOrder()) {
      TF_ASSIGN_OR_RETURN(bool c, ReplaceSetSize(instruction));
      changed |= c;

      TF_ASSIGN_OR_RETURN(c, ReplaceSetBound(instruction));
      changed |= c;
    }
  }

  if (changed) {
    HloDCE dce;
    TF_ASSIGN_OR_RETURN(bool c, dce.Run(module, execution_threads));
    changed |= c;
  }

  VLOG(2) << "Post DynamicPadder HLO:";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace xla
