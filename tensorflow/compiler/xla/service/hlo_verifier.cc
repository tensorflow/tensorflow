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

#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/service/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_instructions.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/flatmap.h"

namespace xla {

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
  return CheckShape(convert, ShapeInference::InferConvertShape(
                                 convert->operand(0)->shape(),
                                 convert->shape().element_type()));
}

Status ShapeVerifier::HandleBitcastConvert(HloInstruction* convert) {
  return CheckShape(convert, ShapeInference::InferBitcastConvertShape(
                                 convert->operand(0)->shape(),
                                 convert->shape().element_type()));
}

Status ShapeVerifier::HandleCopy(HloInstruction* copy) {
  return CheckUnaryShape(copy);
}

Status ShapeVerifier::HandleDot(HloInstruction* dot) {
  TF_ASSIGN_OR_RETURN(const Shape expected,
                      ShapeInference::InferDotOpShape(
                          dot->operand(0)->shape(), dot->operand(1)->shape(),
                          dot->dot_dimension_numbers()));
  return CheckShape(dot, expected);
}

Status ShapeVerifier::HandleConvolution(HloInstruction* convolution) {
  TF_ASSIGN_OR_RETURN(
      const Shape expected,
      ShapeInference::InferConvolveShape(
          convolution->operand(0)->shape(), convolution->operand(1)->shape(),
          convolution->feature_group_count(), convolution->window(),
          convolution->convolution_dimension_numbers()));
  return CheckShape(convolution, expected);
}

Status ShapeVerifier::HandleFft(HloInstruction* fft) {
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
  return CheckShape(hlo, ShapeInference::InferCollectivePermuteShape(
                             hlo->operand(0)->shape()));
}

Status ShapeVerifier::HandleReducePrecision(HloInstruction* reduce_precision) {
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
  HloInfeedInstruction* infeed = Cast<HloInfeedInstruction>(instruction);
  TF_RETURN_IF_ERROR(CheckIsTokenOperand(instruction, 0));

  // The output of infeed is a tuple containing the data value and a token.
  return CheckShape(infeed,
                    ShapeUtil::MakeTupleShape(
                        {infeed->infeed_shape(), ShapeUtil::MakeTokenShape()}));
}

Status ShapeVerifier::HandleOutfeed(HloInstruction* instruction) {
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
  if (instruction->operand_count() != 2) {
    return InternalError("Expected two operands for Rng instruction: %s",
                         instruction->ToString());
  }

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
  return CheckShape(
      reverse, ShapeInference::InferReverseShape(reverse->operand(0)->shape(),
                                                 reverse->dimensions()));
}

Status ShapeVerifier::HandleSort(HloInstruction* sort) {
  if (sort->operand_count() == 2 &&
      !ShapeUtil::SameDimensions(sort->operand(0)->shape(),
                                 sort->operand(1)->shape())) {
    return InternalError(
        "Expected sort to have to have the same dimensions for the keys and "
        "the values. Keys shape is: %s\n, Values shape is: %s",
        StringifyShape(sort->operand(0)->shape()),
        StringifyShape(sort->operand(1)->shape()));
  }
  return CheckVariadicShape(sort);
}

Status ShapeVerifier::HandleConstant(HloInstruction* constant) {
  return CheckShape(constant, constant->literal().shape());
}

Status ShapeVerifier::HandleIota(HloInstruction* instruction) {
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
  return CheckShape(get_tuple_element,
                    ShapeInference::InferGetTupleElementShape(
                        get_tuple_element->operand(0)->shape(),
                        get_tuple_element->tuple_index()));
}

Status ShapeVerifier::HandleReduce(HloInstruction* reduce) {
  std::vector<const Shape*> operand_shapes;
  for (const HloInstruction* operand : reduce->operands()) {
    operand_shapes.push_back(&operand->shape());
  }
  return CheckShape(reduce, ShapeInference::InferReduceShape(
                                operand_shapes, reduce->dimensions(),
                                reduce->to_apply()->ComputeProgramShape()));
}

Status ShapeVerifier::HandleBitcast(HloInstruction* bitcast) {
  return Status::OK();
}

Status ShapeVerifier::HandleBroadcast(HloInstruction* broadcast) {
  // HLO broadcast has no exact analog at the proto level so there is no
  // ShapeInference method. Check the output shape explicitly.
  const Shape& operand_shape = broadcast->operand(0)->shape();
  // Check for mixed precision.
  TF_RETURN_IF_ERROR(CheckShape(broadcast, broadcast->shape()));
  TF_RET_CHECK(ShapeUtil::Rank(operand_shape) ==
               broadcast->dimensions().size());
  for (int64 operand_dimension = 0;
       operand_dimension < ShapeUtil::Rank(operand_shape);
       ++operand_dimension) {
    int64 output_dimension = broadcast->dimensions()[operand_dimension];
    TF_RET_CHECK(broadcast->shape().dimensions(output_dimension) ==
                 operand_shape.dimensions(operand_dimension))
        << broadcast->ToString() << " operand shape " << operand_shape;
  }
  return Status::OK();
}

Status ShapeVerifier::HandleReshape(HloInstruction* reshape) {
  // Check for mixed precision.
  TF_RETURN_IF_ERROR(CheckShape(reshape, reshape->shape()));
  TF_RET_CHECK(ShapeUtil::ElementsIn(reshape->shape()) ==
               ShapeUtil::ElementsIn(reshape->operand(0)->shape()));
  return Status::OK();
}

Status ShapeVerifier::HandleTranspose(HloInstruction* transpose) {
  return CheckShape(
      transpose, ShapeInference::InferTransposeShape(
                     transpose->operand(0)->shape(), transpose->dimensions()));
}

Status ShapeVerifier::HandleParameter(HloInstruction* hlo) {
  return Status::OK();
}

Status ShapeVerifier::HandleFusion(HloInstruction* fusion) {
  for (HloInstruction* fused_param : fusion->fused_parameters()) {
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
  for (int64 i = 0; i < call->to_apply()->num_parameters(); ++i) {
    TF_RETURN_IF_ERROR(CheckOperandAndParameter(call, i, call->to_apply(), i));
  }
  // The shape of kCall should match the shape of the computation it calls.
  return CheckShape(call, call->to_apply()->root_instruction()->shape());
}

Status ShapeVerifier::HandleCustomCall(HloInstruction*) { return Status::OK(); }

Status ShapeVerifier::HandleSlice(HloInstruction* slice) {
  return CheckShape(slice,
                    ShapeInference::InferSliceShape(
                        slice->operand(0)->shape(), slice->slice_starts(),
                        slice->slice_limits(), slice->slice_strides()));
}

Status ShapeVerifier::HandleDynamicSlice(HloInstruction* dynamic_slice) {
  return CheckShape(dynamic_slice, ShapeInference::InferDynamicSliceShape(
                                       dynamic_slice->operand(0)->shape(),
                                       dynamic_slice->operand(1)->shape(),
                                       dynamic_slice->dynamic_slice_sizes()));
}

Status ShapeVerifier::HandleDynamicUpdateSlice(
    HloInstruction* dynamic_update_slice) {
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
  return CheckShape(
      reduce_window,
      ShapeInference::InferReduceWindowShape(
          reduce_window->operand(0)->shape(),
          reduce_window->operand(1)->shape(), reduce_window->window(),
          reduce_window->to_apply()->ComputeProgramShape()));
}

Status ShapeVerifier::HandleSelectAndScatter(HloInstruction* instruction) {
  return CheckShape(
      instruction,
      ShapeInference::InferSelectAndScatterShape(
          instruction->operand(0)->shape(),
          instruction->select()->ComputeProgramShape(), instruction->window(),
          instruction->operand(1)->shape(), instruction->operand(2)->shape(),
          instruction->scatter()->ComputeProgramShape()));
}

Status ShapeVerifier::HandleWhile(HloInstruction* xla_while) {
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
  return CheckShape(pad, ShapeInference::InferPadShape(pad->operand(0)->shape(),
                                                       pad->operand(1)->shape(),
                                                       pad->padding_config()));
}

Status ShapeVerifier::HandleSend(HloInstruction* send) {
  return CheckShape(send,
                    ShapeUtil::MakeTupleShape({send->operand(0)->shape(),
                                               ShapeUtil::MakeShape(U32, {}),
                                               ShapeUtil::MakeTokenShape()}));
}

Status ShapeVerifier::HandleSendDone(HloInstruction* send_done) {
  return CheckShape(send_done, ShapeUtil::MakeTokenShape());
}

Status ShapeVerifier::HandleRecv(HloInstruction* recv) {
  return CheckShape(
      recv, ShapeUtil::MakeTupleShape(
                {ShapeUtil::GetTupleElementShape(recv->shape(), 0),
                 ShapeUtil::MakeShape(U32, {}), ShapeUtil::MakeTokenShape()}));
}

Status ShapeVerifier::HandleRecvDone(HloInstruction* recv_done) {
  return CheckShape(
      recv_done,
      ShapeUtil::MakeTupleShape(
          {ShapeUtil::GetTupleElementShape(recv_done->operand(0)->shape(), 0),
           ShapeUtil::MakeTokenShape()}));
}

Status ShapeVerifier::HandleBatchNormTraining(
    HloInstruction* batch_norm_training) {
  return CheckShape(batch_norm_training,
                    ShapeInference::InferBatchNormTrainingShape(
                        batch_norm_training->operand(0)->shape(),
                        batch_norm_training->operand(1)->shape(),
                        batch_norm_training->operand(2)->shape(),
                        batch_norm_training->feature_index()));
}

Status ShapeVerifier::HandleBatchNormInference(
    HloInstruction* batch_norm_inference) {
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
  return CheckShape(
      gather,
      ShapeInference::InferGatherShape(
          gather->operand(0)->shape(), gather->operand(1)->shape(),
          gather->gather_dimension_numbers(), gather->gather_slice_sizes()));
}

Status ShapeVerifier::HandleScatter(HloInstruction* scatter) {
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
  return CheckShape(instruction,
                    ShapeInference::InferUnaryOpShape(instruction->opcode(),
                                                      instruction->operand(0)));
}

Status ShapeVerifier::CheckBinaryShape(const HloInstruction* instruction) {
  return CheckShape(
      instruction, ShapeInference::InferBinaryOpShape(instruction->opcode(),
                                                      instruction->operand(0),
                                                      instruction->operand(1)));
}

Status ShapeVerifier::CheckTernaryShape(const HloInstruction* instruction) {
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

Status HloVerifier::CheckFusionInstruction(HloInstruction* fusion) const {
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

  // TODO(b/65423525): We'd like to check that all operands are distinct.
  // This is currently disabled due to the invariant being violated by
  // multi-output fusion.
  return Status::OK();
}

Status HloVerifier::CheckWhileInstruction(HloInstruction* instruction) {
  auto* while_cond = instruction->while_condition();
  auto* while_body = instruction->while_body();
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
  if (instruction->operand_count() != 1) {
    return FailedPrecondition(
        "While loop must have exactly one operand; had %d : %s",
        instruction->operand_count(), instruction->ToString());
  }
  return Status::OK();
}

Status HloVerifier::CheckConditionalInstruction(HloInstruction* instruction) {
  if (instruction->true_computation()->num_parameters() != 1) {
    return FailedPrecondition(
        "True computation %s of %s must have 1 parameter insted of %d",
        instruction->true_computation()->name(), instruction->ToString(),
        instruction->true_computation()->num_parameters());
  }
  if (instruction->false_computation()->num_parameters() != 1) {
    return FailedPrecondition(
        "False computation %s of %s must have 1 parameter insted of %d",
        instruction->false_computation()->name(), instruction->ToString(),
        instruction->false_computation()->num_parameters());
  }
  return Status::OK();
}

Status HloVerifier::CheckElementwiseInstruction(HloInstruction* instruction) {
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
  tensorflow::gtl::FlatMap<int64, const HloInstruction*> host_channels;
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

}  // namespace

StatusOr<bool> HloVerifier::Run(HloModule* module) {
  TF_RET_CHECK(!module->name().empty());
  TF_RETURN_IF_ERROR(VerifyHloStructure(module));
  TF_RETURN_IF_ERROR(VerifySendsAndRecvs(*module));

  tensorflow::gtl::FlatMap<string, const HloInstruction*> instructions;

  for (auto* computation : module->computations()) {
    for (const auto& instruction : computation->instructions()) {
      TF_RET_CHECK(instruction->parent() == computation);
      if (instruction->opcode() == HloOpcode::kFusion) {
        TF_RETURN_IF_ERROR(CheckFusionInstruction(instruction));
        TF_RET_CHECK(instruction->called_computations() ==
                     absl::Span<HloComputation* const>(
                         {instruction->fused_instructions_computation()}))
            << "Fusion HLO calls computations other than the "
               "fused_instructions_computation: "
            << instruction->ToString()
            << " instruction->fused_instructions_computation(): "
            << instruction->fused_instructions_computation()->ToString()
            << " instruction->called_computations(): "
            << ComputationsToString(instruction->called_computations());

        for (const auto& fused : instruction->fused_instructions()) {
          TF_RET_CHECK(fused->parent() ==
                       instruction->fused_instructions_computation())
              << "Fused HLO was missing a parent: " << fused->ToString()
              << " parent: " << fused->parent()
              << " computation: " << computation;
        }
      } else if (instruction->opcode() == HloOpcode::kBroadcast) {
        // If you see this failure then someone has confused the difference
        // between the HLO broadcast op, and the UserComputation broadcast
        // op. See https://groups.google.com/forum/#!topic/xla-dev/9LqijHmTt_I
        // or ComputationLowerer::Visit()
        TF_RET_CHECK(instruction->dimensions().size() ==
                     ShapeUtil::Rank(instruction->operand(0)->shape()))
            << "Broadcast HLO (" << instruction->ToShortString()
            << ") has invalid number of dimensions: "
            << instruction->dimensions().size()
            << " != " << ShapeUtil::Rank(instruction->operand(0)->shape());
      } else if (instruction->opcode() == HloOpcode::kWhile) {
        TF_RETURN_IF_ERROR(CheckWhileInstruction(instruction));
      } else if (instruction->opcode() == HloOpcode::kConditional) {
        TF_RETURN_IF_ERROR(CheckConditionalInstruction(instruction));
      } else if (instruction->opcode() !=
                     HloOpcode::kRng /* Rng operands are always scalar. */
                 && instruction->IsElementwise()) {
        TF_RETURN_IF_ERROR(CheckElementwiseInstruction(instruction));
      }

      auto previous = instructions.find(instruction->name());
      TF_RET_CHECK(previous == instructions.end())
          << "HLO has name that is not unique within module:\n"
          << instruction->ToString()
          << " in computation: " << computation->name()
          << "\nPrevious HLO with same name:\n"
          << previous->second->ToString()
          << " in computation: " << previous->second->parent()->name();
      instructions[instruction->name()] = instruction;
    }

    std::unique_ptr<ShapeVerifier> shape_verifier = shape_verifier_factory_();
    TF_RETURN_IF_ERROR(computation->Accept(shape_verifier.get()));
  }

  TF_RETURN_IF_ERROR(VerifyEntryAndExitShapes(*module));

  // If the module has a schedule, it must be valid.
  if (module->has_schedule()) {
    TF_RETURN_IF_ERROR(module->schedule().Verify());
  }

  return false;
}

}  // namespace xla
