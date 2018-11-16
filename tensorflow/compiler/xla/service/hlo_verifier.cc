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

#include <set>

#include "absl/container/flat_hash_map.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {

Status VerifyNotSparse(const Shape& shape) {
  return ShapeUtil::ForEachSubshapeWithStatus(
      shape, [](const Shape& subshape, const ShapeIndex&) -> Status {
        if (LayoutUtil::IsSparseArray(subshape)) {
          return InternalError("Sparse arrays are not yet fully supported: %s",
                               ShapeUtil::HumanStringWithLayout(subshape));
        }
        return Status::OK();
      });
}

bool IsCallerInstruction(HloInstruction* hlo) {
  switch (hlo->opcode()) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kWhile:
    case HloOpcode::kCrossReplicaSum:
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kFusion:
      return true;
    default:
      return false;
  }
}

Status ShapeVerifier::Preprocess(HloInstruction* hlo) {
  if (!hlo->called_computations().empty() && !IsCallerInstruction(hlo)) {
    return InternalError(
        "Called computations specified for non-caller instruction  %s",
        hlo->ToString());
  }
  return VerifyNotSparse(hlo->shape());
}

namespace {

Status CheckOperandCount(const HloInstruction* hlo, int expected) {
  if (hlo->operand_count() != expected) {
    return InternalError("Expected %d operands for %s instruction: %s",
                         expected, HloOpcodeString(hlo->opcode()),
                         hlo->ToString());
  }
  return Status::OK();
}

Status CheckParameterCount(const HloInstruction* calling_instruction,
                           const HloComputation* computation, int expected) {
  if (computation->num_parameters() != expected) {
    return InternalError(
        "Expected computation %s called from %s to have %d parameters, has %d",
        computation->name(), calling_instruction->name(), expected,
        computation->num_parameters());
  }
  return Status::OK();
}

}  // namespace

Status ShapeVerifier::HandleElementwiseUnary(HloInstruction* hlo) {
  return CheckUnaryShape(hlo);
}

Status ShapeVerifier::HandleElementwiseBinary(HloInstruction* hlo) {
  return CheckBinaryShape(hlo);
}

Status ShapeVerifier::HandleClamp(HloInstruction* clamp) {
  return CheckTernaryShape(clamp);
}

Status ShapeVerifier::HandleSelect(HloInstruction* select) {
  return CheckTernaryShape(select);
}

Status ShapeVerifier::HandleTupleSelect(HloInstruction* tuple_select) {
  return CheckTernaryShape(tuple_select);
}

Status ShapeVerifier::HandleConcatenate(HloInstruction* concatenate) {
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : concatenate->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(concatenate,
                    ShapeInference::InferConcatOpShape(
                        operand_shapes, concatenate->concatenate_dimension()));
}

Status ShapeVerifier::HandleConvert(HloInstruction* convert) {
  TF_RETURN_IF_ERROR(CheckOperandCount(convert, 1));
  return CheckShape(convert, ShapeInference::InferConvertShape(
                                 convert->operand(0)->shape(),
                                 convert->shape().element_type()));
}

Status ShapeVerifier::HandleBitcastConvert(HloInstruction* convert) {
  TF_RETURN_IF_ERROR(CheckOperandCount(convert, 1));
  return CheckShape(convert, ShapeInference::InferBitcastConvertShape(
                                 convert->operand(0)->shape(),
                                 convert->shape().element_type()));
}

Status ShapeVerifier::HandleCopy(HloInstruction* copy) {
  return CheckUnaryShape(copy);
}

Status ShapeVerifier::HandleDot(HloInstruction* dot) {
  TF_RETURN_IF_ERROR(CheckOperandCount(dot, 2));
  TF_ASSIGN_OR_RETURN(const Shape expected,
                      ShapeInference::InferDotOpShape(
                          dot->operand(0)->shape(), dot->operand(1)->shape(),
                          dot->dot_dimension_numbers()));
  return CheckShape(dot, expected);
}

Status ShapeVerifier::HandleConvolution(HloInstruction* convolution) {
  TF_RETURN_IF_ERROR(CheckOperandCount(convolution, 2));
  TF_ASSIGN_OR_RETURN(
      const Shape expected,
      ShapeInference::InferConvolveShape(
          convolution->operand(0)->shape(), convolution->operand(1)->shape(),
          convolution->feature_group_count(), convolution->window(),
          convolution->convolution_dimension_numbers()));
  return CheckShape(convolution, expected);
}

Status ShapeVerifier::HandleFft(HloInstruction* fft) {
  TF_RETURN_IF_ERROR(CheckOperandCount(fft, 1));
  TF_ASSIGN_OR_RETURN(
      const Shape expected,
      ShapeInference::InferFftShape(fft->operand(0)->shape(), fft->fft_type(),
                                    fft->fft_length()));
  return CheckShape(fft, expected);
}

Status ShapeVerifier::HandleCrossReplicaSum(HloInstruction* crs) {
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : crs->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(crs,
                    ShapeInference::InferCrossReplicaSumShape(operand_shapes));
}

Status ShapeVerifier::HandleAllToAll(HloInstruction* hlo) {
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : hlo->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(hlo,
                    ShapeInference::InferAllToAllTupleShape(operand_shapes));
}

Status ShapeVerifier::HandleCollectivePermute(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(CheckOperandCount(hlo, 1));
  return CheckShape(hlo, ShapeInference::InferCollectivePermuteShape(
                             hlo->operand(0)->shape()));
}

Status ShapeVerifier::HandleReducePrecision(HloInstruction* reduce_precision) {
  TF_RETURN_IF_ERROR(CheckOperandCount(reduce_precision, 1));
  return CheckShape(reduce_precision, ShapeInference::InferReducePrecisionShape(
                                          reduce_precision->operand(0)->shape(),
                                          reduce_precision->exponent_bits(),
                                          reduce_precision->mantissa_bits()));
}

Status ShapeVerifier::CheckIsTokenOperand(const HloInstruction* instruction,
                                          int64 operand_no) {
  const HloInstruction* token = instruction->operand(operand_no);
  if (!ShapeUtil::Equal(token->shape(), ShapeUtil::MakeTokenShape())) {
    return InternalError(
        "Expected operand %d to be token-shaped, actual shape is "
        "%s:\n%s",
        operand_no, StringifyShape(token->shape()), instruction->ToString());
  }
  return Status::OK();
}

Status ShapeVerifier::CheckOperandAndParameter(
    const HloInstruction* instruction, int64 operand_number,
    const HloComputation* computation, int64 parameter_number) {
  const HloInstruction* operand = instruction->operand(operand_number);
  const HloInstruction* parameter =
      computation->parameter_instruction(parameter_number);
  if (!ShapesSame(operand->shape(), parameter->shape())) {
    return InternalError("Operand %s shape does not match parameter's %s in %s",
                         operand->ToString(), parameter->ToString(),
                         instruction->ToString());
  }
  return Status::OK();
}

Status ShapeVerifier::HandleInfeed(HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 1));
  HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(instruction);
  TF_RETURN_IF_ERROR(CheckIsTokenOperand(instruction, 0));

  // The output of infeed is a tuple containing the data value and a token.
  return CheckShape(infeed,
                    ShapeUtil::MakeTupleShape(
                        {infeed->infeed_shape(), ShapeUtil::MakeTokenShape()}));
}

Status ShapeVerifier::HandleOutfeed(HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 2));
  HloOutfeedInstruction* outfeed = Cast<HloOutfeedInstruction>(instruction);
  TF_RETURN_IF_ERROR(CheckIsTokenOperand(instruction, 1));

  // Outfeed has a separate shape field for the value which is outfed to the
  // host. The shape of the instruction itself is always a token.
  if (!ShapesSame(outfeed->outfeed_shape(), outfeed->operand(0)->shape())) {
    return InternalError(
        "Expected outfeed shape to be equal to operand's shape %s, "
        "actual shape is %s:\n%s",
        StringifyShape(outfeed->operand(0)->shape()),
        StringifyShape(outfeed->outfeed_shape()), outfeed->ToString());
  }
  return CheckShape(outfeed, ShapeUtil::MakeTokenShape());
}

bool ShapeVerifier::HasCompatibleElementTypes(const Shape& shape_0,
                                              const Shape& shape_1,
                                              const Shape& result_shape) {
  return ShapeUtil::SameElementType(shape_0, shape_1) &&
         (ShapeUtil::SameElementType(shape_0, result_shape) ||
          (allow_mixed_precision_ &&
           ShapeUtil::SameElementTypeIgnoringFpPrecision(shape_0,
                                                         result_shape)));
}

Status ShapeVerifier::HandleRng(HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 2));

  const Shape& shape_0 = instruction->operand(0)->shape();
  const Shape& shape_1 = instruction->operand(1)->shape();
  if (!ShapeUtil::IsScalar(shape_0) || !ShapeUtil::IsScalar(shape_1)) {
    return InternalError(
        "Expected scalar types for the two operands of Rng instruction: %s",
        instruction->ToString());
  }

  if (!HasCompatibleElementTypes(shape_0, shape_1, instruction->shape())) {
    return InternalError(
        "Expected compatible element types for the result and the two operands"
        " of Rng instruction: %s",
        instruction->ToString());
  }

  PrimitiveType element_type = shape_0.element_type();
  switch (instruction->random_distribution()) {
    case RNG_UNIFORM:
      if (!primitive_util::IsFloatingPointType(element_type) &&
          !primitive_util::IsIntegralType(element_type) &&
          element_type != PRED) {
        return InternalError(
            "Element type not supported."
            " Expected element to be of floating point type, integral type or"
            " predicate type for RngUniform: %s",
            instruction->ToString());
      }
      break;

    case RNG_NORMAL:
      if (!primitive_util::IsFloatingPointType(element_type)) {
        return InternalError(
            "Element type not supported."
            " Expected element to be FloatingPointType for RngNormal: %s",
            instruction->ToString());
      }
      break;
    default:
      return InternalError(
          "Invalid Rng distribution %s",
          RandomDistribution_Name(instruction->random_distribution()));
  }

  return Status::OK();
}

Status ShapeVerifier::HandleReverse(HloInstruction* reverse) {
  TF_RETURN_IF_ERROR(CheckOperandCount(reverse, 1));
  return CheckShape(
      reverse, ShapeInference::InferReverseShape(reverse->operand(0)->shape(),
                                                 reverse->dimensions()));
}

Status ShapeVerifier::HandleSort(HloInstruction* sort) {
  if (sort->operand_count() < 1) {
    return InternalError("Expected at least 1 operand for %s instruction: %s",
                         HloOpcodeString(sort->opcode()), sort->ToString());
  }
  for (int64 operand = 1; operand < sort->operand_count(); ++operand) {
    if (!ShapeUtil::SameDimensions(sort->operand(0)->shape(),
                                   sort->operand(operand)->shape())) {
      return InternalError(
          "Expected sort to have to have the same dimensions for the keys "
          "and the values. Keys shape is: %s\n, Values shape (operand index "
          "%lld) is: %s",
          StringifyShape(sort->operand(0)->shape()), operand,
          StringifyShape(sort->operand(operand)->shape()));
    }
  }
  return CheckVariadicShape(sort);
}

Status ShapeVerifier::HandleConstant(HloInstruction* constant) {
  TF_RETURN_IF_ERROR(CheckOperandCount(constant, 0));
  if (!Cast<HloConstantInstruction>(constant)->HasLiteral()) {
    return InternalError("Constant is required to have a valid literal: %s",
                         constant->ToString());
  }
  return CheckShape(constant, constant->literal().shape());
}

Status ShapeVerifier::HandleIota(HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 0));
  auto* iota = Cast<HloIotaInstruction>(instruction);
  const int64 rank = ShapeUtil::Rank(iota->shape());
  if (rank == 0) {
    return InternalError("Iota does not support scalars.");
  }
  int64 iota_dimension = iota->iota_dimension();
  if (iota_dimension >= rank) {
    return InternalError(
        "The iota dimension cannot go beyond the operation rank.");
  }
  return Status::OK();
}

Status ShapeVerifier::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  TF_RETURN_IF_ERROR(CheckOperandCount(get_tuple_element, 1));
  return CheckShape(get_tuple_element,
                    ShapeInference::InferGetTupleElementShape(
                        get_tuple_element->operand(0)->shape(),
                        get_tuple_element->tuple_index()));
}

Status ShapeVerifier::HandleReduce(HloInstruction* reduce) {
  if (reduce->operand_count() % 2 != 0) {
    return InternalError(
        "Expected an even number of operands for %s instruction: %s",
        HloOpcodeString(reduce->opcode()), reduce->ToString());
  }

  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : reduce->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(reduce, ShapeInference::InferReduceShape(
                                operand_shapes, reduce->dimensions(),
                                reduce->to_apply()->ComputeProgramShape()));
}

Status ShapeVerifier::HandleBitcast(HloInstruction* bitcast) {
  TF_RETURN_IF_ERROR(CheckOperandCount(bitcast, 1));
  return Status::OK();
}

Status ShapeVerifier::HandleBroadcast(HloInstruction* broadcast) {
  TF_RETURN_IF_ERROR(CheckOperandCount(broadcast, 1));
  // HLO broadcast has no exact analog at the proto level so there is no
  // ShapeInference method. Check the output shape explicitly.
  const Shape& operand_shape = broadcast->operand(0)->shape();
  // Check for mixed precision.
  TF_RET_CHECK(SameElementType(broadcast->shape(), operand_shape));
  TF_RET_CHECK(ShapeUtil::Rank(operand_shape) ==
               broadcast->dimensions().size());
  for (int64 operand_dimension = 0;
       operand_dimension < ShapeUtil::Rank(operand_shape);
       ++operand_dimension) {
    int64 output_dimension = broadcast->dimensions()[operand_dimension];
    TF_RET_CHECK((output_dimension < ShapeUtil::Rank(broadcast->shape())) &&
                 output_dimension >= 0 &&
                 (broadcast->shape().dimensions(output_dimension) ==
                  operand_shape.dimensions(operand_dimension)))
        << broadcast->ToString() << " operand shape " << operand_shape;
  }
  return Status::OK();
}

Status ShapeVerifier::HandleReshape(HloInstruction* reshape) {
  TF_RETURN_IF_ERROR(CheckOperandCount(reshape, 1));
  // Check for mixed precision.
  const Shape& operand_shape = reshape->operand(0)->shape();
  TF_RET_CHECK(SameElementType(reshape->shape(), operand_shape));
  TF_RET_CHECK(ShapeUtil::ElementsIn(reshape->shape()) ==
               ShapeUtil::ElementsIn(operand_shape));
  return Status::OK();
}

Status ShapeVerifier::HandleTranspose(HloInstruction* transpose) {
  TF_RETURN_IF_ERROR(CheckOperandCount(transpose, 1));
  return CheckShape(
      transpose, ShapeInference::InferTransposeShape(
                     transpose->operand(0)->shape(), transpose->dimensions()));
}

Status ShapeVerifier::HandleParameter(HloInstruction* hlo) {
  TF_RETURN_IF_ERROR(CheckOperandCount(hlo, 0));
  return Status::OK();
}

Status ShapeVerifier::HandleFusion(HloInstruction* fusion) {
  auto& fused_parameters = fusion->fused_parameters();
  if (fused_parameters.size() != fusion->operand_count()) {
    return InternalError(
        "Fused parameter count (%d) does not match the number of operands (%d)"
        " passed to the fusion instruction in: %s.",
        fused_parameters.size(), fusion->operand_count(),
        fusion->ToString().c_str());
  }
  for (HloInstruction* fused_param : fused_parameters) {
    int64 param_no = fused_param->parameter_number();
    if (!ShapesSame(fused_param->shape(), fusion->operand(param_no)->shape())) {
      return InternalError(
          "Shape mismatch between parameter number %d and its operand in "
          "%s.",
          param_no, fusion->ToString().c_str());
    }
  }
  return Status::OK();
}

Status ShapeVerifier::HandleCall(HloInstruction* call) {
  TF_RETURN_IF_ERROR(
      CheckParameterCount(call, call->to_apply(), call->operand_count()));
  for (int64 i = 0; i < call->to_apply()->num_parameters(); ++i) {
    TF_RETURN_IF_ERROR(CheckOperandAndParameter(call, i, call->to_apply(), i));
  }
  // The shape of kCall should match the shape of the computation it calls.
  return CheckShape(call, call->to_apply()->root_instruction()->shape());
}

Status ShapeVerifier::HandleCustomCall(HloInstruction* instruction) {
  const HloCustomCallInstruction* custom_call =
      DynCast<const HloCustomCallInstruction>(instruction);
  TF_RET_CHECK(custom_call != nullptr);
  if (custom_call->layout_constrained()) {
    // If the layout is constrained, verify all the respective shapes have
    // layouts and that the constrained operand shapes match the shapes of the
    // operands.
    TF_RET_CHECK(LayoutUtil::HasLayout(custom_call->shape()));
    TF_RET_CHECK(custom_call->operand_count() ==
                 custom_call->operand_shapes_with_layout().size());
    for (int64 i = 0; i < custom_call->operand_count(); ++i) {
      const Shape& operand_shape_with_layout =
          custom_call->operand_shapes_with_layout()[i];
      TF_RET_CHECK(ShapeUtil::Compatible(custom_call->operand(i)->shape(),
                                         operand_shape_with_layout));
      TF_RET_CHECK(LayoutUtil::HasLayout(operand_shape_with_layout));
    }
  }
  return Status::OK();
}

Status ShapeVerifier::HandleSlice(HloInstruction* slice) {
  TF_RETURN_IF_ERROR(CheckOperandCount(slice, 1));
  return CheckShape(slice,
                    ShapeInference::InferSliceShape(
                        slice->operand(0)->shape(), slice->slice_starts(),
                        slice->slice_limits(), slice->slice_strides()));
}

Status ShapeVerifier::HandleDynamicSlice(HloInstruction* dynamic_slice) {
  TF_RETURN_IF_ERROR(CheckOperandCount(dynamic_slice, 2));
  return CheckShape(dynamic_slice, ShapeInference::InferDynamicSliceShape(
                                       dynamic_slice->operand(0)->shape(),
                                       dynamic_slice->operand(1)->shape(),
                                       dynamic_slice->dynamic_slice_sizes()));
}

Status ShapeVerifier::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
  TF_RETURN_IF_ERROR(CheckOperandCount(dynamic_update_slice, 3));
  return CheckShape(dynamic_update_slice,
                    ShapeInference::InferDynamicUpdateSliceShape(
                        dynamic_update_slice->operand(0)->shape(),
                        dynamic_update_slice->operand(1)->shape(),
                        dynamic_update_slice->operand(2)->shape()));
}

Status ShapeVerifier::HandleTuple(HloInstruction* tuple) {
  return CheckVariadicShape(tuple);
}

Status ShapeVerifier::HandleMap(HloInstruction* map) {
  std::vector<const Shape*> operand_shapes;
  int64 max_operand_rank = 0;
  for (const HloInstruction* operand : map->operands()) {
    operand_shapes.push_back(&operand->shape());
    max_operand_rank =
        std::max(max_operand_rank, ShapeUtil::Rank(operand->shape()));
  }
  // TODO(b/65689298) Remove code below once Map is generalized to accept
  // arbitrary map dimensions.
  std::vector<int64> map_dims(max_operand_rank);
  std::iota(map_dims.begin(), map_dims.end(), 0);
  return CheckShape(map, ShapeInference::InferMapShape(
                             operand_shapes,
                             map->to_apply()->ComputeProgramShape(), map_dims));
}

Status ShapeVerifier::HandleReduceWindow(HloInstruction* reduce_window) {
  TF_RETURN_IF_ERROR(CheckOperandCount(reduce_window, 2));
  return CheckShape(
      reduce_window,
      ShapeInference::InferReduceWindowShape(
          reduce_window->operand(0)->shape(),
          reduce_window->operand(1)->shape(), reduce_window->window(),
          reduce_window->to_apply()->ComputeProgramShape()));
}

Status ShapeVerifier::HandleSelectAndScatter(HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 3));
  return CheckShape(
      instruction,
      ShapeInference::InferSelectAndScatterShape(
          instruction->operand(0)->shape(),
          instruction->select()->ComputeProgramShape(), instruction->window(),
          instruction->operand(1)->shape(), instruction->operand(2)->shape(),
          instruction->scatter()->ComputeProgramShape()));
}

Status ShapeVerifier::HandleWhile(HloInstruction* xla_while) {
  TF_RETURN_IF_ERROR(CheckOperandCount(xla_while, 1));
  TF_RETURN_IF_ERROR(
      CheckParameterCount(xla_while, xla_while->while_body(), 1));
  TF_RETURN_IF_ERROR(
      CheckParameterCount(xla_while, xla_while->while_condition(), 1));
  TF_RETURN_IF_ERROR(
      CheckOperandAndParameter(xla_while, 0, xla_while->while_body(), 0));
  TF_RETURN_IF_ERROR(
      CheckOperandAndParameter(xla_while, 0, xla_while->while_condition(), 0));
  const Shape& conditional_shape =
      xla_while->while_condition()->root_instruction()->shape();
  if (!ShapesSame(conditional_shape, ShapeUtil::MakeShape(PRED, {}))) {
    return InternalError(
        "Conditional computation shape does not lead to a scalar predicate "
        "shape: %s",
        StringifyShape(conditional_shape));
  }
  // The shape of kWhile should match the shape of the body computation it
  // calls.
  return CheckShape(xla_while,
                    xla_while->while_body()->root_instruction()->shape());
}

Status ShapeVerifier::HandleConditional(HloInstruction* conditional) {
  TF_RETURN_IF_ERROR(CheckOperandCount(conditional, 3));
  TF_RETURN_IF_ERROR(
      CheckParameterCount(conditional, conditional->true_computation(), 1));
  TF_RETURN_IF_ERROR(
      CheckParameterCount(conditional, conditional->false_computation(), 1));
  TF_RETURN_IF_ERROR(CheckOperandAndParameter(
      conditional, 1, conditional->true_computation(), 0));
  TF_RETURN_IF_ERROR(CheckOperandAndParameter(
      conditional, 2, conditional->false_computation(), 0));
  TF_RETURN_IF_ERROR(
      CheckShape(conditional,
                 conditional->true_computation()->root_instruction()->shape()));
  TF_RETURN_IF_ERROR(CheckShape(
      conditional,
      conditional->false_computation()->root_instruction()->shape()));
  return Status::OK();
}

Status ShapeVerifier::HandlePad(HloInstruction* pad) {
  TF_RETURN_IF_ERROR(CheckOperandCount(pad, 2));
  return CheckShape(pad, ShapeInference::InferPadShape(pad->operand(0)->shape(),
                                                       pad->operand(1)->shape(),
                                                       pad->padding_config()));
}

Status ShapeVerifier::HandleSend(HloInstruction* send) {
  TF_RETURN_IF_ERROR(CheckOperandCount(send, 2));
  return CheckShape(send,
                    ShapeUtil::MakeTupleShape({send->operand(0)->shape(),
                                               ShapeUtil::MakeShape(U32, {}),
                                               ShapeUtil::MakeTokenShape()}));
}

Status ShapeVerifier::HandleSendDone(HloInstruction* send_done) {
  TF_RETURN_IF_ERROR(CheckOperandCount(send_done, 1));
  return CheckShape(send_done, ShapeUtil::MakeTokenShape());
}

Status ShapeVerifier::HandleRecv(HloInstruction* recv) {
  TF_RETURN_IF_ERROR(CheckOperandCount(recv, 1));
  return CheckShape(
      recv, ShapeUtil::MakeTupleShape(
                {ShapeUtil::GetTupleElementShape(recv->shape(), 0),
                 ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeTokenShape()}));
}

Status ShapeVerifier::HandleRecvDone(HloInstruction* recv_done) {
  TF_RETURN_IF_ERROR(CheckOperandCount(recv_done, 1));
  return CheckShape(
      recv_done,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::GetTupleElementShape(recv_done->operand(0)->shape(), 0),
           ShapeUtil::MakeTokenShape()}));
}

Status ShapeVerifier::HandleBatchNormTraining(
    HloInstruction* batch_norm_training) {
  TF_RETURN_IF_ERROR(CheckOperandCount(batch_norm_training, 3));
  return CheckShape(batch_norm_training,
                    ShapeInference::InferBatchNormTrainingShape(
                        batch_norm_training->operand(0)->shape(),
                        batch_norm_training->operand(1)->shape(),
                        batch_norm_training->operand(2)->shape(),
                        batch_norm_training->feature_index()));
}

Status ShapeVerifier::HandleBatchNormInference(
    HloInstruction* batch_norm_inference) {
  TF_RETURN_IF_ERROR(CheckOperandCount(batch_norm_inference, 5));
  return CheckShape(batch_norm_inference,
                    ShapeInference::InferBatchNormInferenceShape(
                        batch_norm_inference->operand(0)->shape(),
                        batch_norm_inference->operand(1)->shape(),
                        batch_norm_inference->operand(2)->shape(),
                        batch_norm_inference->operand(3)->shape(),
                        batch_norm_inference->operand(4)->shape(),
                        batch_norm_inference->feature_index()));
}

Status ShapeVerifier::HandleBatchNormGrad(HloInstruction* batch_norm_grad) {
  TF_RETURN_IF_ERROR(CheckOperandCount(batch_norm_grad, 5));
  return CheckShape(batch_norm_grad, ShapeInference::InferBatchNormGradShape(
                                         batch_norm_grad->operand(0)->shape(),
                                         batch_norm_grad->operand(1)->shape(),
                                         batch_norm_grad->operand(2)->shape(),
                                         batch_norm_grad->operand(3)->shape(),
                                         batch_norm_grad->operand(4)->shape(),
                                         batch_norm_grad->feature_index()));
}

namespace {

// Checks that the instruction does not have mixed precision floating point
// inputs.
Status CheckMixedPrecisionOperands(const HloInstruction* instruction) {
  switch (instruction->opcode()) {
    // White list the following opcodes for mixed-precision check, because
    // they involve data pass through or grouping via tuples, where the
    // precisions of buffers can be different.
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kConstant:
    case HloOpcode::kCrossReplicaSum:
    case HloOpcode::kCustomCall:
    case HloOpcode::kDomain:
    case HloOpcode::kFusion:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kInfeed:
    case HloOpcode::kOutfeed:
    case HloOpcode::kParameter:
    case HloOpcode::kRecv:
    case HloOpcode::kRecvDone:
    case HloOpcode::kReducePrecision:
    case HloOpcode::kSelect:
    case HloOpcode::kTupleSelect:
    case HloOpcode::kSend:
    case HloOpcode::kSendDone:
    case HloOpcode::kSort:
    case HloOpcode::kTuple:
    case HloOpcode::kWhile:
      break;
    default: {
      PrimitiveType fp_type = PRIMITIVE_TYPE_INVALID;
      for (auto operand : instruction->operands()) {
        TF_RETURN_IF_ERROR(ShapeUtil::ForEachSubshapeWithStatus(
            operand->shape(),
            [&](const Shape& subshape, const ShapeIndex& index) {
              if (!ShapeUtil::ElementIsFloating(subshape)) {
                return Status::OK();
              }
              if (fp_type == PRIMITIVE_TYPE_INVALID) {
                fp_type = subshape.element_type();
              } else if (fp_type != subshape.element_type()) {
                return InternalError(
                    "Seen floating point types of different precisions in "
                    "%s, but mixed precision is disallowed.",
                    instruction->ToString());
              }
              return Status::OK();
            }));
      }
    }
  }
  return Status::OK();
}

}  // namespace

Status ShapeVerifier::HandleGather(HloInstruction* gather) {
  TF_RETURN_IF_ERROR(CheckOperandCount(gather, 2));
  return CheckShape(
      gather,
      ShapeInference::InferGatherShape(
          gather->operand(0)->shape(), gather->operand(1)->shape(),
          gather->gather_dimension_numbers(), gather->gather_slice_sizes()));
}

Status ShapeVerifier::HandleScatter(HloInstruction* scatter) {
  TF_RETURN_IF_ERROR(CheckOperandCount(scatter, 3));
  return CheckShape(
      scatter, ShapeInference::InferScatterShape(
                   scatter->operand(0)->shape(), scatter->operand(1)->shape(),
                   scatter->operand(2)->shape(),
                   scatter->to_apply()->ComputeProgramShape(),
                   scatter->scatter_dimension_numbers()));
}

Status ShapeVerifier::HandleAfterAll(HloInstruction* token) {
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : token->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(token, ShapeInference::InferAfterAllShape(operand_shapes));
}

Status ShapeVerifier::HandleGetDimensionSize(HloInstruction* get_size) {
  return CheckShape(get_size,
                    ShapeInference::InferGetDimensionSizeShape(
                        get_size->operand(0)->shape(), get_size->dimension()));
}

Status ShapeVerifier::CheckShape(const HloInstruction* instruction,
                                 const Shape& inferred_shape) {
  // If allow_mixed_precision_ is false, check if there are operands with
  // different precisions. We need this check because ShapeInference allows
  // mixed precision inputs.
  if (!allow_mixed_precision_) {
    TF_RETURN_IF_ERROR(CheckMixedPrecisionOperands(instruction));
  }

  // Check if the output shape matches the expected shape.
  //
  // We treat BF16 and F32 as compatible types if mixed precision is allowed,
  // but only when the instruction defines the BF16/F32 buffer.
  bool equal = [&] {
    switch (instruction->opcode()) {
      // The opcodes below can't have implicit layout conversions, nor can they
      // implicitly transform f32 -> bf16.  Fundamentally these are either
      // reinterpreting existing data (e.g. kBitcast) or shuffling data around
      // without modifying it (e.g. kGetTupleElement, kTupleSelect).
      case HloOpcode::kBitcast:
      case HloOpcode::kCall:
      case HloOpcode::kConditional:
      case HloOpcode::kConstant:
      case HloOpcode::kCustomCall:
      case HloOpcode::kGetTupleElement:
      case HloOpcode::kInfeed:
      case HloOpcode::kOutfeed:
      case HloOpcode::kParameter:
      case HloOpcode::kRecv:
      case HloOpcode::kRecvDone:
      case HloOpcode::kSend:
      case HloOpcode::kSendDone:
      case HloOpcode::kTuple:
      case HloOpcode::kTupleSelect:
      case HloOpcode::kWhile:
        return ShapesSame(instruction->shape(), inferred_shape);

      // We allow arbitrary layout and f32->bf16 transformations on all other
      // instructions, although this may be made more strict pending discussion
      // in b/112709536.
      default:
        if (allow_mixed_precision_) {
          return ShapeUtil::CompatibleIgnoringFpPrecision(instruction->shape(),
                                                          inferred_shape);
        } else {
          return ShapeUtil::Compatible(instruction->shape(), inferred_shape);
        }
    }
  }();
  if (!equal) {
    return InternalError(
        "Expected instruction to have shape equal to %s, actual "
        "shape is %s:\n%s",
        StringifyShape(inferred_shape), StringifyShape(instruction->shape()),
        instruction->ToString());
  }
  return Status::OK();
}

Status ShapeVerifier::CheckShape(const HloInstruction* instruction,
                                 const StatusOr<Shape>& inferred_shape_status) {
  if (!inferred_shape_status.ok()) {
    Status s = inferred_shape_status.status();
    tensorflow::errors::AppendToMessage(&s, ", for instruction ",
                                        instruction->ToString());
    return s;
  }
  return CheckShape(instruction, inferred_shape_status.ValueOrDie());
}

Status ShapeVerifier::CheckUnaryShape(const HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 1));
  return CheckShape(instruction,
                    ShapeInference::InferUnaryOpShape(instruction->opcode(),
                                                      instruction->operand(0)));
}

Status ShapeVerifier::CheckBinaryShape(const HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 2));
  return CheckShape(
      instruction, ShapeInference::InferBinaryOpShape(instruction->opcode(),
                                                      instruction->operand(0),
                                                      instruction->operand(1)));
}

Status ShapeVerifier::CheckTernaryShape(const HloInstruction* instruction) {
  TF_RETURN_IF_ERROR(CheckOperandCount(instruction, 3));
  return CheckShape(instruction,
                    ShapeInference::InferTernaryOpShape(
                        instruction->opcode(), instruction->operand(0),
                        instruction->operand(1), instruction->operand(2)));
}

Status ShapeVerifier::CheckVariadicShape(const HloInstruction* instruction) {
  return CheckShape(instruction,
                    ShapeInference::InferVariadicOpShape(
                        instruction->opcode(), instruction->operands()));
}

Status ShapeVerifier::VerifyEntryComputationLayout(const HloModule& module) {
  const HloComputation* computation = module.entry_computation();
  const auto& layout = module.entry_computation_layout();
  const ShapeLayout& result_layout = layout.result_layout();

  TF_RETURN_IF_ERROR(
      ShapeUtil::ValidateShapeWithOptionalLayout(result_layout.shape()));

  TF_RETURN_IF_ERROR(VerifyNotSparse(result_layout.shape()));

  if (!ShapeUtil::Compatible(computation->root_instruction()->shape(),
                             result_layout.shape())) {
    return InternalError(
        "Shape of the root instruction of entry computation (%s) should be "
        "compatible to one specified in module's entry computation layout (%s)",
        ShapeUtil::HumanString(computation->root_instruction()->shape()),
        ShapeUtil::HumanString(result_layout.shape()));
  }

  if (computation->num_parameters() != layout.parameter_count()) {
    return InternalError(
        "Number of parameters in entry computation layout (%d) must be same "
        "as number of parameters of entry computation computation (%d)",
        layout.parameter_count(), computation->num_parameters());
  }

  for (int i = 0; i < computation->num_parameters(); ++i) {
    const HloInstruction* parameter = computation->parameter_instruction(i);
    TF_RETURN_IF_ERROR(
        ShapeUtil::ValidateShapeWithOptionalLayout(layout.parameter_shape(i)));
    TF_RETURN_IF_ERROR(VerifyNotSparse(layout.parameter_shape(i)));
    if (!ShapeUtil::Compatible(parameter->shape(), layout.parameter_shape(i))) {
      return InternalError(
          "Shape of the entry computation parameter %d is %s should be "
          "compatible to the one specified in module's entry computation "
          "layout %s",
          i, ShapeUtil::HumanString(parameter->shape()),
          ShapeUtil::HumanString(layout.parameter_shape(i)));
    }
  }

  return Status::OK();
}

string ComputationsToString(absl::Span<HloComputation* const> computations) {
  return absl::StrJoin(computations, ",",
                       [](string* s, const HloComputation* computation) {
                         s->append(computation->name());
                       });
}

// Verifies various invariants about the structure of the HLO:
//
// (1) each instruction has a non-null parent() set to the HloComputation
// which
//     contains it.
//
// (2) each computation has a non-null parent() set to the HloModule which
//     contains it.
//
// (3) the operands of each instruction are in the same computation as the
//     instruction.
Status VerifyHloStructure(HloModule* module) {
  for (const HloComputation* computation : module->computations()) {
    if (computation->parent() == nullptr) {
      return InternalError("Computation %s has a null parent pointer",
                           computation->name());
    }
    if (computation->parent() != module) {
      return InternalError(
          "Computation %s parent() does not point to parent module",
          computation->name());
    }

    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->parent() == nullptr) {
        return InternalError("Instruction %s has a null parent pointer",
                             instruction->name());
      }
      if (instruction->parent() != computation) {
        return InternalError(
            "Instruction %s parent() does not point to parent computation",
            instruction->name());
      }
    }
  }

  // Check that operands are in the same computation separately from verifying
  // parent() correctness so conditions like a null HloInstruction::parent()
  // are identified and reported explicitly above rather than reporting a
  // mismatched operand.
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      for (int i = 0; i < instruction->operand_count(); ++i) {
        const HloInstruction* operand = instruction->operand(i);
        if (operand->parent() != instruction->parent()) {
          return InternalError(
              "Operand %d (%s) of instruction %s is in a different "
              "computation: %s vs %s",
              i, operand->name(), instruction->name(),
              operand->parent()->name(), instruction->parent()->name());
        }
      }
    }
  }
  return Status::OK();
}

namespace {

// Returns true if the given Shape has a TOKEN shape as any subshape.
bool ShapeContainsToken(const Shape& shape) {
  bool contains_token = false;
  ShapeUtil::ForEachSubshape(
      shape, [&contains_token](const Shape& subshape, const ShapeIndex&) {
        if (ShapeUtil::IsToken(subshape)) {
          contains_token = true;
        }
      });
  return contains_token;
}

// Verifies that all types entering and exiting the entry computation are
// legal.
Status VerifyEntryAndExitShapes(const HloModule& module) {
  // Tokens cannot be passed as entry parameters.
  // TODO(b/80000000): Remove this constraint.
  for (int i = 0; i < module.entry_computation()->num_parameters(); ++i) {
    HloInstruction* param =
        module.entry_computation()->parameter_instruction(i);
    if (ShapeContainsToken(param->shape())) {
      return InternalError(
          "Entry parameter %d is or contains a token shape: %s", i,
          ShapeUtil::HumanString(param->shape()));
    }
  }
  return Status::OK();
}

// Checks if the given two instructions share the same channel id.
Status CheckSameChannel(const HloInstruction* instr1,
                        const HloInstruction* instr2) {
  if (instr1->channel_id() != instr2->channel_id()) {
    return InternalError(
        "Expected to have the same channel id, actual channel ids are: %s "
        "(%d), %s (%d)",
        instr1->ToString(), instr1->channel_id(), instr2->ToString(),
        instr2->channel_id());
  }
  return Status::OK();
}

// Checks if the given two instructions have the same is_host_transfer
// attribute value. Intsructions must be send/recv instructions or their
// 'done' variant.
Status CheckSameIsHostTransfer(const HloInstruction* instr1,
                               const HloInstruction* instr2) {
  const HloSendRecvInstruction* send_recv1 =
      DynCast<const HloSendRecvInstruction>(instr1);
  const HloSendRecvInstruction* send_recv2 =
      DynCast<const HloSendRecvInstruction>(instr2);
  TF_RET_CHECK(send_recv1 != nullptr);
  TF_RET_CHECK(send_recv2 != nullptr);
  if (send_recv1->is_host_transfer() != send_recv2->is_host_transfer()) {
    return InternalError(
        "Expected instructions to have the same is-host-transfer property: "
        "%s, "
        "%s ",
        instr1->ToString(), instr2->ToString());
  }
  return Status::OK();
}

// Checks various invariants of send and recv instructions.
Status VerifySendsAndRecvs(const HloModule& module) {
  absl::flat_hash_map<int64, const HloInstruction*> host_channels;
  // Host send/recv instructions must have their own unique channel.
  auto check_unique_host_channel = [&](const HloInstruction* instruction) {
    const HloSendRecvInstruction* sendrecv =
        DynCast<const HloSendRecvInstruction>(instruction);
    if (sendrecv->is_host_transfer()) {
      auto it_inserted =
          host_channels.insert({sendrecv->channel_id(), sendrecv});
      if (!it_inserted.second) {
        return FailedPrecondition(
            "Channel %d is used for multiple host send/recv instructions: "
            "%s "
            "and "
            "%s",
            sendrecv->channel_id(), sendrecv->ToString(),
            it_inserted.first->second->ToString());
      }
    }

    return Status::OK();
  };

  // Send/Recv instruction must have a single user: the corresponding
  // SendDone/RecvDone. with matching channel.
  for (const HloComputation* computation : module.computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      switch (instruction->opcode()) {
        case HloOpcode::kSend: {
          TF_RETURN_IF_ERROR(check_unique_host_channel(instruction));
          TF_RET_CHECK(instruction->users().size() == 1);
          const HloInstruction* send_done = instruction->users().front();
          TF_RET_CHECK(send_done->opcode() == HloOpcode::kSendDone);
          TF_RETURN_IF_ERROR(CheckSameChannel(instruction, send_done));
          TF_RETURN_IF_ERROR(CheckSameIsHostTransfer(instruction, send_done));
          break;
        }
        case HloOpcode::kRecv: {
          TF_RETURN_IF_ERROR(check_unique_host_channel(instruction));
          TF_RET_CHECK(instruction->users().size() == 1);
          const HloInstruction* recv_done = instruction->users().front();
          TF_RET_CHECK(recv_done->opcode() == HloOpcode::kRecvDone);
          TF_RETURN_IF_ERROR(CheckSameChannel(instruction, recv_done));
          TF_RETURN_IF_ERROR(CheckSameIsHostTransfer(instruction, recv_done));
          break;
        }
        case HloOpcode::kSendDone:
          TF_RET_CHECK(instruction->operands().size() == 1);
          TF_RET_CHECK(instruction->operand(0)->opcode() == HloOpcode::kSend);
          break;
        case HloOpcode::kRecvDone:
          TF_RET_CHECK(instruction->operands().size() == 1);
          TF_RET_CHECK(instruction->operand(0)->opcode() == HloOpcode::kRecv);
          break;
        default:
          break;
      }
    }
  }
  return Status::OK();
}

// CHECKs various invariants of a fusion instruction.
Status CheckFusionInstruction(HloInstruction* fusion) {
  // The parent fusion instruction of the fusion computation must be 'fusion'.
  HloComputation* fused_computation = fusion->fused_instructions_computation();
  if (fusion != fused_computation->FusionInstruction()) {
    return InternalError(
        "Instruction of fused computation does not match expected "
        "instruction "
        "%s.",
        fusion->ToString());
  }

  // Fused root instruction and fused parameters must all be owned by the
  // fusion computation.
  bool root_owned = false;
  const std::vector<HloInstruction*>& fused_parameters =
      fusion->fused_parameters();
  const HloInstruction* fused_root = fusion->fused_expression_root();
  std::vector<bool> parameter_owned(fused_parameters.size(), false);
  for (auto* instruction : fused_computation->instructions()) {
    if (fused_root == instruction) {
      if (root_owned) {
        return InternalError("Root appears more than once in %s.",
                             fusion->ToString());
      }
      root_owned = true;
    }
    for (int i = 0; i < fused_parameters.size(); ++i) {
      if (fused_parameters[i] == instruction) {
        if (parameter_owned[i]) {
          return InternalError("Parameter appears more than once in %s.",
                               fusion->ToString());
        }
        parameter_owned[i] = true;
      }
    }
  }
  if (!root_owned) {
    return InternalError("Root not found in computation of %s.",
                         fusion->ToString());
  }
  // Make sure all the parameter_owned entries are set
  for (int i = 0; i < parameter_owned.size(); i++) {
    if (!parameter_owned[i]) {
      return InternalError("Parameter %d not found in computation of %s.", i,
                           fusion->ToString());
    }
  }

  // Fused root must have no users.
  if (fused_root->user_count() != 0) {
    return InternalError("Root of %s may not have users.", fusion->ToString());
  }

  // All uses of fused instructions must be in the fusion computation, and
  // every non-root instruction must have at least one use.
  for (auto* instruction :
       fusion->fused_instructions_computation()->instructions()) {
    if (instruction != fused_root) {
      if (instruction->user_count() == 0) {
        return InternalError("Non-root instruction %s in %s must have users.",
                             instruction->ToString(), fusion->ToString());
      }
      for (auto& user : instruction->users()) {
        if (fused_computation != user->parent()) {
          return InternalError(
              "Non-root instruction %s in %s may not have external users.",
              instruction->ToString(), fusion->ToString());
        }
      }
    }
  }

  // Fused parameter instructions must be numbered contiguously and match up
  // (shapes equal) with their respective operand.
  CHECK_EQ(fusion->operands().size(), fused_parameters.size());
  std::vector<bool> parameter_numbers(fused_parameters.size(), false);
  for (auto fused_param : fused_parameters) {
    int64 param_no = fused_param->parameter_number();
    if (param_no < 0) {
      return InternalError("Unexpected negative parameter number %d in %s.",
                           param_no, fusion->ToString());
    }
    if (param_no >= fused_parameters.size()) {
      return InternalError(
          "Unexpected parameter number %d in %s: higher then number of "
          "parameters %lu.",
          param_no, fusion->ToString(), fused_parameters.size());
    }
    if (parameter_numbers[param_no]) {
      return InternalError(
          "Did not expect parameter number %d more than once in %s.", param_no,
          fusion->ToString());
    }
    parameter_numbers[param_no] = true;
  }
  // Make sure all the parameter_numbers entries were seen.
  for (int i = 0; i < parameter_numbers.size(); i++) {
    if (!parameter_numbers[i]) {
      return InternalError("Did not see parameter number %d in %s.", i,
                           fusion->ToString());
    }
  }

  TF_RET_CHECK(fusion->called_computations() ==
               absl::Span<HloComputation* const>(
                   {fusion->fused_instructions_computation()}))
      << "Fusion HLO calls computations other than the "
         "fused_instructions_computation: "
      << fusion->ToString() << " fusion->fused_instructions_computation(): "
      << fusion->fused_instructions_computation()->ToString()
      << " fusion->called_computations(): "
      << ComputationsToString(fusion->called_computations());

  for (const auto& fused : fusion->fused_instructions()) {
    TF_RET_CHECK(fused->parent() == fusion->fused_instructions_computation())
        << "Fused HLO was missing a parent: " << fused->ToString()
        << " parent: " << fused->parent()
        << " computation: " << fusion->parent();
  }

  // TODO(b/65423525): We'd like to check that all operands are distinct.
  // This is currently disabled due to the invariant being violated by
  // multi-output fusion.
  return Status::OK();
}

// Checks that the non-scalar operand shapes are compatible to the output
// shape, i.e., that there are no implicit broadcasts of size-one dimensions.
Status CheckElementwiseInstruction(HloInstruction* instruction) {
  const Shape& out_shape = instruction->shape();
  for (HloInstruction* operand : instruction->operands()) {
    const Shape& operand_shape = operand->shape();
    if (!ShapeUtil::CompatibleIgnoringElementType(operand_shape, out_shape)) {
      return FailedPrecondition(
          "Implicit broadcast is not allowed in HLO."
          "Found different shapes for instruction %s.\n"
          "output: %s\noperand: %s\n",
          HloOpcodeString(instruction->opcode()),
          ShapeUtil::HumanString(out_shape),
          ShapeUtil::HumanString(operand_shape));
    }
  }
  return Status::OK();
}

// Visitor which verifies various fields on the HLO instruction. This class does
// not check result shape as that is checked in the ShapeVerifier.
class InstructionVerifier : public DfsHloVisitorWithDefault {
 public:
  explicit InstructionVerifier(std::function<bool(const HloInstruction*)>
                                   instruction_can_change_layout_func)
      : instruction_can_change_layout_func_(
            instruction_can_change_layout_func) {}

  Status DefaultAction(HloInstruction*) override { return Status::OK(); }

  Status HandleFusion(HloInstruction* fusion) override {
    return CheckFusionInstruction(fusion);
  }

  Status HandleBroadcast(HloInstruction* broadcast) override {
    // If you see this failure then someone has confused the difference
    // between the HLO broadcast op, and the UserComputation broadcast
    // op. See https://groups.google.com/forum/#!topic/xla-dev/9LqijHmTt_I
    // or ComputationLowerer::Visit()
    TF_RET_CHECK(broadcast->dimensions().size() ==
                 ShapeUtil::Rank(broadcast->operand(0)->shape()))
        << "Broadcast HLO (" << broadcast->ToShortString()
        << ") has invalid number of dimensions: "
        << broadcast->dimensions().size()
        << " != " << ShapeUtil::Rank(broadcast->operand(0)->shape());
    return Status::OK();
  }

  Status HandleWhile(HloInstruction* xla_while) override {
    auto* while_cond = xla_while->while_condition();
    auto* while_body = xla_while->while_body();
    if (while_cond->num_parameters() != 1) {
      return FailedPrecondition(
          "While condition must have exactly 1 parameter; had %d : %s",
          while_cond->num_parameters(), while_cond->ToString());
    }
    if (while_body->num_parameters() != 1) {
      return FailedPrecondition(
          "While body must have exactly 1 parameter; had %d : %s",
          while_body->num_parameters(), while_body->ToString());
    }
    if (xla_while->operand_count() != 1) {
      return FailedPrecondition(
          "While loop must have exactly one operand; had %d : %s",
          xla_while->operand_count(), xla_while->ToString());
    }
    return Status::OK();
  }

  Status HandleConditional(HloInstruction* conditional) override {
    if (conditional->true_computation()->num_parameters() != 1) {
      return FailedPrecondition(
          "True computation %s of %s must have 1 parameter insted of %d",
          conditional->true_computation()->name(), conditional->ToString(),
          conditional->true_computation()->num_parameters());
    }
    if (conditional->false_computation()->num_parameters() != 1) {
      return FailedPrecondition(
          "False computation %s of %s must have 1 parameter insted of %d",
          conditional->false_computation()->name(), conditional->ToString(),
          conditional->false_computation()->num_parameters());
    }
    return Status::OK();
  }

  Status HandleElementwiseUnary(HloInstruction* instruction) override {
    return CheckElementwiseInstruction(instruction);
  }

  Status HandleElementwiseBinary(HloInstruction* instruction) override {
    return CheckElementwiseInstruction(instruction);
  }

  Status HandleGetTupleElement(HloInstruction* gte) override {
    TF_RET_CHECK(ShapeUtil::IsTuple(gte->operand(0)->shape()));
    return Status::OK();
  }

  Status HandleTranspose(HloInstruction* transpose) override {
    const Shape& shape = transpose->shape();
    const HloInstruction* operand = transpose->operand(0);
    TF_RET_CHECK(shape.dimensions().size() == transpose->dimensions().size());
    TF_RET_CHECK(shape.dimensions().size() ==
                 transpose->operand(0)->shape().dimensions().size());
    TF_RET_CHECK(std::equal(
        operand->shape().dimensions().begin(),
        operand->shape().dimensions().end(),
        Permute(transpose->dimensions(), shape.dimensions()).begin()))
        << "shape: " << shape << ", operand->shape(): " << shape
        << ", dimensions: {" << absl::StrJoin(transpose->dimensions(), ", ")
        << "}";
    return Status::OK();
  }

  Status HandleCrossReplicaSum(HloInstruction* crs) override {
    if (crs->all_reduce_id().has_value()) {
      TF_RET_CHECK(crs->all_reduce_id().value() > 0)
          << "All reduce id must be greater than 0 for "
          << crs->ToShortString();
    }
    return Status::OK();
  }

  Status Preprocess(HloInstruction* instruction) override {
    auto previous = instructions_by_name_.find(instruction->name());
    TF_RET_CHECK(previous == instructions_by_name_.end())
        << "HLO has name that is not unique within module:\n"
        << instruction->ToString()
        << " in computation: " << instruction->parent()->name()
        << "\nPrevious HLO with same name:\n"
        << previous->second->ToString()
        << " in computation: " << previous->second->parent()->name();
    instructions_by_name_[instruction->name()] = instruction;
    return Status::OK();
  }

  Status Postprocess(HloInstruction* instruction) override {
    if (instruction_can_change_layout_func_ &&
        LayoutUtil::IsDenseArray(instruction->shape()) &&
        !instruction_can_change_layout_func_(instruction)) {
      const Shape& result_shape = instruction->shape();
      const Layout& result_layout = result_shape.layout();
      for (HloInstruction* operand : instruction->operands()) {
        const Shape& operand_shape = operand->shape();
        if (LayoutUtil::IsDenseArray(operand_shape) &&
            ShapeUtil::Rank(operand_shape) == ShapeUtil::Rank(result_shape)) {
          const Layout& operand_layout = operand_shape.layout();
          TF_RET_CHECK(LayoutUtil::Equal(result_layout, operand_layout))
              << "Instruction shouldn't change layouts "
              << instruction->ToString() << " From "
              << ShapeUtil::HumanString(result_shape) << " To "
              << ShapeUtil::HumanString(operand_shape);
        }
      }
    }

    return Status::OK();
  }

 private:
  absl::flat_hash_map<string, const HloInstruction*> instructions_by_name_;
  // Determines whether an instruction can change layouts.
  std::function<bool(const HloInstruction*)>
      instruction_can_change_layout_func_;
};

}  // namespace

StatusOr<bool> HloVerifier::Run(HloModule* module) {
  TF_RET_CHECK(!module->name().empty());

  if (module->entry_computation()->IsFusionComputation()) {
    return InvalidArgument(
        "Module entry computation cannot be a fusion computation");
  }

  TF_RETURN_IF_ERROR(VerifyHloStructure(module));
  TF_RETURN_IF_ERROR(VerifySendsAndRecvs(*module));

  std::unique_ptr<ShapeVerifier> shape_verifier =
      target_metadata_->GetVerifier();
  for (auto* computation : module->computations()) {
    TF_RETURN_IF_ERROR(computation->Accept(shape_verifier.get()));

    InstructionVerifier instruction_verifier(
        instruction_can_change_layout_func_);
    TF_RETURN_IF_ERROR(computation->Accept(&instruction_verifier));
  }

  TF_RETURN_IF_ERROR(shape_verifier->VerifyEntryComputationLayout(*module));
  TF_RETURN_IF_ERROR(VerifyEntryAndExitShapes(*module));

  // If the module has a schedule, it must be valid.
  if (module->has_schedule()) {
    TF_RETURN_IF_ERROR(module->schedule().Verify());
  }

  TF_RETURN_IF_ERROR(module->input_output_alias_config().Verify(
      *module, [this](const Shape& shape) {
        return target_metadata_->ShapeSize(shape);
      }));

  TF_RETURN_IF_ERROR(module->dynamic_parameter_binding().Verify(*module));

  return false;
}

}  // namespace xla
