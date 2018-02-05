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

#include "tensorflow/compiler/xla/service/hlo_verifier.h"
#include "tensorflow/compiler/xla/status_macros.h"
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
          convolution->window(), convolution->convolution_dimension_numbers()));
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

Status ShapeVerifier::HandleReducePrecision(HloInstruction* reduce_precision) {
  return CheckShape(reduce_precision, ShapeInference::InferReducePrecisionShape(
                                          reduce_precision->operand(0)->shape(),
                                          reduce_precision->exponent_bits(),
                                          reduce_precision->mantissa_bits()));
}

Status ShapeVerifier::HandleInfeed(HloInstruction*) {
  return tensorflow::Status::OK();
}

Status ShapeVerifier::HandleOutfeed(HloInstruction* outfeed) {
  // Outfeed has a separate shape field for the value which is outfed to the
  // host. The shape of the instruction itself is always nil because the outfeed
  // produces no HLO value in the graph.
  if (!ShapeUtil::Compatible(outfeed->outfeed_shape(),
                             outfeed->operand(0)->shape())) {
    return InvalidArgument(
        "Expected outfeed to have shape compatible with operand's shape %s, "
        "actual shape is %s:\n%s",
        ShapeUtil::HumanString(outfeed->operand(0)->shape()).c_str(),
        ShapeUtil::HumanString(outfeed->outfeed_shape()).c_str(),
        outfeed->ToString().c_str());
  }
  return CheckShape(outfeed, ShapeUtil::MakeNil());
}

Status ShapeVerifier::HandleRng(HloInstruction*) {
  return tensorflow::Status::OK();
}

Status ShapeVerifier::HandleReverse(HloInstruction* reverse) {
  return CheckShape(
      reverse, ShapeInference::InferReverseShape(reverse->operand(0)->shape(),
                                                 reverse->dimensions()));
}

Status ShapeVerifier::HandleSort(HloInstruction* sort) {
  return CheckUnaryShape(sort);
}

Status ShapeVerifier::HandleConstant(HloInstruction* constant) {
  return CheckShape(constant, constant->literal().shape());
}

Status ShapeVerifier::HandleGetTupleElement(HloInstruction* get_tuple_element) {
  return CheckShape(get_tuple_element,
                    ShapeInference::InferGetTupleElementShape(
                        get_tuple_element->operand(0)->shape(),
                        get_tuple_element->tuple_index()));
}

Status ShapeVerifier::HandleReduce(HloInstruction* reduce) {
  return CheckShape(
      reduce,
      ShapeInference::InferReduceShape(
          reduce->operand(0)->shape(), reduce->operand(1)->shape(),
          reduce->dimensions(), reduce->to_apply()->ComputeProgramShape()));
}

Status ShapeVerifier::HandleBitcast(HloInstruction* bitcast) {
  return tensorflow::Status::OK();
}

Status ShapeVerifier::HandleBroadcast(HloInstruction* broadcast) {
  // HLO broadcast has no exact analog at the proto level so there is no
  // ShapeInference method. Check the output shape explicitly.
  const Shape& operand_shape = broadcast->operand(0)->shape();
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
  return tensorflow::Status::OK();
}

Status ShapeVerifier::HandleReshape(HloInstruction* reshape) {
  TF_RET_CHECK(ShapeUtil::ElementsIn(reshape->shape()) ==
               ShapeUtil::ElementsIn(reshape->operand(0)->shape()));
  return tensorflow::Status::OK();
}

Status ShapeVerifier::HandleTranspose(HloInstruction* transpose) {
  return CheckShape(
      transpose, ShapeInference::InferTransposeShape(
                     transpose->operand(0)->shape(), transpose->dimensions()));
}

Status ShapeVerifier::HandleParameter(HloInstruction*) {
  return tensorflow::Status::OK();
}

Status ShapeVerifier::HandleFusion(HloInstruction*) {
  return tensorflow::Status::OK();
}

Status ShapeVerifier::HandleCall(HloInstruction* call) {
  // The shape of kCall should match the shape of the computation it calls.
  return CheckShape(call, call->to_apply()->ComputeProgramShape().result());
}

Status ShapeVerifier::HandleCustomCall(HloInstruction*) {
  return tensorflow::Status::OK();
}

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
  // The shape of kWhile should match the shape of the body computation it
  // calls.
  return CheckShape(xla_while,
                    xla_while->while_body()->ComputeProgramShape().result());
}

Status ShapeVerifier::HandleConditional(HloInstruction* conditional) {
  TF_RETURN_IF_ERROR(CheckShape(
      conditional,
      conditional->true_computation()->ComputeProgramShape().result()));
  return CheckShape(
      conditional,
      conditional->false_computation()->ComputeProgramShape().result());
}

Status ShapeVerifier::HandlePad(HloInstruction* pad) {
  return CheckShape(pad, ShapeInference::InferPadShape(pad->operand(0)->shape(),
                                                       pad->operand(1)->shape(),
                                                       pad->padding_config()));
}

Status ShapeVerifier::HandleSend(HloInstruction* send) {
  TF_RET_CHECK(send->users().size() == 1);
  const HloInstruction* send_done = send->users().front();
  TF_RET_CHECK(send_done->opcode() == HloOpcode::kSendDone);
  TF_RETURN_IF_ERROR(CheckSameChannel(send, send_done));
  return CheckShape(
      send, ShapeUtil::MakeTupleShape(
                {send->operand(0)->shape(), ShapeUtil::MakeShape(U32, {})}));
}

Status ShapeVerifier::HandleSendDone(HloInstruction* send_done) {
  TF_RET_CHECK(send_done->operands().size() == 1);
  const HloInstruction* send = send_done->operand(0);
  TF_RET_CHECK(send->opcode() == HloOpcode::kSend);
  TF_RETURN_IF_ERROR(CheckSameChannel(send, send_done));
  return CheckShape(send_done, ShapeUtil::MakeNil());
}

Status ShapeVerifier::HandleRecv(HloInstruction* recv) {
  TF_RET_CHECK(recv->users().size() == 1);
  const HloInstruction* recv_done = recv->users().front();
  TF_RET_CHECK(recv_done->opcode() == HloOpcode::kRecvDone);
  TF_RETURN_IF_ERROR(CheckSameChannel(recv, recv_done));
  return CheckShape(recv,
                    ShapeUtil::MakeTupleShape(
                        {recv_done->shape(), ShapeUtil::MakeShape(U32, {})}));
}

Status ShapeVerifier::HandleRecvDone(HloInstruction* recv_done) {
  TF_RET_CHECK(recv_done->operands().size() == 1);
  const HloInstruction* recv = recv_done->operand(0);
  TF_RET_CHECK(recv->opcode() == HloOpcode::kRecv);
  TF_RETURN_IF_ERROR(CheckSameChannel(recv, recv_done));
  return CheckShape(recv_done, recv->shape().tuple_shapes(0));
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

Status ShapeVerifier::CheckShape(const HloInstruction* instruction,
                                 const Shape& expected_shape) {
  if (!ShapeUtil::Compatible(instruction->shape(), expected_shape)) {
    return InvalidArgument(
        "Expected instruction to have shape compatible with %s, actual "
        "shape is %s:\n%s",
        ShapeUtil::HumanString(expected_shape).c_str(),
        ShapeUtil::HumanString(instruction->shape()).c_str(),
        instruction->ToString().c_str());
  }
  return tensorflow::Status::OK();
}

Status ShapeVerifier::CheckShape(const HloInstruction* instruction,
                                 const StatusOr<Shape>& expected_shape_status) {
  if (!expected_shape_status.ok()) {
    Status s = expected_shape_status.status();
    tensorflow::errors::AppendToMessage(&s, ", for instruction ",
                                        instruction->ToString());
    return s;
  }
  return CheckShape(instruction, expected_shape_status.ValueOrDie());
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

// Checks if the given two instructions shares the same channel id.
Status ShapeVerifier::CheckSameChannel(const HloInstruction* instr1,
                                       const HloInstruction* instr2) {
  if (instr1->channel_id() != instr2->channel_id()) {
    return FailedPrecondition(
        "Expected to have the same channel id, actual channel ids are: %s "
        "(%lld), %s (%lld)",
        instr1->ToString().c_str(), instr1->channel_id(),
        instr2->ToString().c_str(), instr2->channel_id());
  }
  return tensorflow::Status::OK();
}

string ComputationsToString(
    tensorflow::gtl::ArraySlice<HloComputation*> computations) {
  return tensorflow::str_util::Join(
      computations, ",", [](string* s, const HloComputation* computation) {
        s->append(computation->name());
      });
}

// Verifies various invariants about the structure of the HLO:
//
// (1) each instruction has a non-null parent() set to the HloComputation which
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
      return FailedPrecondition("Computation %s has a null parent pointer",
                                computation->name().c_str());
    }
    if (computation->parent() != module) {
      return FailedPrecondition(
          "Computation %s parent() does not point to parent module",
          computation->name().c_str());
    }

    for (const HloInstruction* instruction : computation->instructions()) {
      if (instruction->parent() == nullptr) {
        return FailedPrecondition("Instruction %s has a null parent pointer",
                                  instruction->name().c_str());
      }
      if (instruction->parent() != computation) {
        return FailedPrecondition(
            "Instruction %s parent() does not point to parent computation",
            instruction->name().c_str());
      }
    }
  }

  // Check that operands are in the same computation separately from verifying
  // parent() correctness so conditions like a null HloInstruction::parent() are
  // identified and reported explicitly above rather than reporting a mismatched
  // operand.
  for (const HloComputation* computation : module->computations()) {
    for (const HloInstruction* instruction : computation->instructions()) {
      for (int i = 0; i < instruction->operand_count(); ++i) {
        const HloInstruction* operand = instruction->operand(i);
        if (operand->parent() != instruction->parent()) {
          return FailedPrecondition(
              "Operand %d (%s) of instruction %s is in a different "
              "computation: %s vs %s",
              i, operand->name().c_str(), instruction->name().c_str(),
              operand->parent()->name().c_str(),
              instruction->parent()->name().c_str());
        }
      }
    }
  }
  return tensorflow::Status::OK();
}

Status HloVerifier::CheckFusionInstruction(HloInstruction* fusion) const {
  // The parent fusion instruction of the fusion computation must be 'fusion'.
  HloComputation* fused_computation = fusion->fused_instructions_computation();
  if (fusion != fused_computation->FusionInstruction()) {
    return FailedPrecondition(
        "Instruction of fused computation does not match expected instruction "
        "%s.",
        fusion->ToString().c_str());
  }

  // Fused root instruction and fused parameters must all be owned by the fusion
  // computation.
  bool root_owned = false;
  const std::vector<HloInstruction*>& fused_parameters =
      fusion->fused_parameters();
  const HloInstruction* fused_root = fusion->fused_expression_root();
  std::vector<bool> parameter_owned(fused_parameters.size(), false);
  for (auto* instruction : fused_computation->instructions()) {
    if (fused_root == instruction) {
      if (root_owned) {
        return FailedPrecondition("Root appears more than once in %s.",
                                  fusion->ToString().c_str());
      }
      root_owned = true;
    }
    for (int i = 0; i < fused_parameters.size(); ++i) {
      if (fused_parameters[i] == instruction) {
        if (parameter_owned[i]) {
          return FailedPrecondition("Parameter appears more than once in %s.",
                                    fusion->ToString().c_str());
        }
        parameter_owned[i] = true;
      }
    }
  }
  if (!root_owned) {
    return FailedPrecondition("Root not found in computation of %s.",
                              fusion->ToString().c_str());
  }
  // Make sure all the parameter_owned entries are set
  for (int i = 0; i < parameter_owned.size(); i++) {
    if (!parameter_owned[i]) {
      return FailedPrecondition("Parameter %d not found in computation of %s.",
                                i, fusion->ToString().c_str());
    }
  }

  // Fused root must have no users.
  if (fused_root->user_count() != 0) {
    return FailedPrecondition("Root of %s may not have users.",
                              fusion->ToString().c_str());
  }

  // All uses of fused instructions must be in the fusion computation, and every
  // non-root instruction must have at least one use.
  for (auto* instruction :
       fusion->fused_instructions_computation()->instructions()) {
    if (instruction != fused_root) {
      if (instruction->user_count() == 0) {
        return FailedPrecondition(
            "Non-root instruction %s in %s must have users.",
            instruction->ToString().c_str(), fusion->ToString().c_str());
      }
      for (auto& user : instruction->users()) {
        if (fused_computation != user->parent()) {
          return FailedPrecondition(
              "Non-root instruction %s in %s may not have external users.",
              instruction->ToString().c_str(), fusion->ToString().c_str());
        }
      }
    }
  }

  // Fused parameter instructions must be numbered contiguously and match up
  // (shapes compatible) with their respective operand.
  CHECK_EQ(fusion->operands().size(), fused_parameters.size());
  std::vector<bool> parameter_numbers(fused_parameters.size(), false);
  for (auto fused_param : fused_parameters) {
    int64 param_no = fused_param->parameter_number();
    if (param_no < 0) {
      return FailedPrecondition(
          "Unexpected negative parameter number %lld in %s.", param_no,
          fusion->ToString().c_str());
    }
    if (param_no >= fused_parameters.size()) {
      return FailedPrecondition(
          "Unexpected parameter number %lld in %s: higher then number of "
          "parameters %lu.",
          param_no, fusion->ToString().c_str(), fused_parameters.size());
    }
    if (parameter_numbers[param_no]) {
      return FailedPrecondition(
          "Did not expect parameter number %lld more than once in %s.",
          param_no, fusion->ToString().c_str());
    }
    parameter_numbers[param_no] = true;
    if (!ShapeUtil::Compatible(fused_param->shape(),
                               fusion->operand(param_no)->shape())) {
      return FailedPrecondition(
          "Shape mismatch between parameter number %lld and its operand in %s.",
          param_no, fusion->ToString().c_str());
    }
  }
  // Make sure all the parameter_numbers entries were seen
  for (int i = 0; i < parameter_numbers.size(); i++) {
    if (!parameter_numbers[i]) {
      return FailedPrecondition("Did not see parameter number %d in %s.", i,
                                fusion->ToString().c_str());
    }
  }

  // TODO(b/65423525): We'd like to check that all operands are distinct.
  // This is currently disabled due to the invariant being violated by
  // multi-output fusion.
  return tensorflow::Status::OK();
}

StatusOr<bool> HloVerifier::Run(HloModule* module) {
  TF_RETURN_IF_ERROR(VerifyHloStructure(module));

  tensorflow::gtl::FlatMap<string, const HloInstruction*> instructions;

  for (auto* computation : module->computations()) {
    for (const auto& instruction : computation->instructions()) {
      TF_RET_CHECK(instruction->parent() == computation);
      if (instruction->opcode() == HloOpcode::kFusion) {
        TF_RETURN_IF_ERROR(CheckFusionInstruction(instruction));
        TF_RET_CHECK(
            ContainersEqual(instruction->called_computations(),
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
        // op.  See https://groups.google.com/forum/#!topic/xla-dev/9LqijHmTt_I
        // or ComputationLowerer::Visit()
        TF_RET_CHECK(instruction->dimensions().size() ==
                     ShapeUtil::Rank(instruction->operand(0)->shape()))
            << "Broadcast HLO has invalid number of dimensions.";
      } else if (instruction->opcode() == HloOpcode::kWhile) {
        auto* while_cond = instruction->while_condition();
        auto* while_body = instruction->while_body();
        TF_RET_CHECK(while_cond->num_parameters() == 1)
            << "While condition must have exactly 1 parameter; had "
            << while_cond->num_parameters() << ": " << while_cond->ToString();
        TF_RET_CHECK(while_body->num_parameters() == 1)
            << "While body must have exactly 1 parameter; had "
            << while_body->num_parameters() << ": " << while_body->ToString();
        TF_RET_CHECK(instruction->operand_count() == 1)
            << "While loop must have exactly one operand; had "
            << instruction->operand_count() << ": " << instruction->ToString();

        auto* init = instruction->operand(0);
        auto* cond_param = while_cond->parameter_instruction(0);
        TF_RET_CHECK(ShapeUtil::Compatible(init->shape(), cond_param->shape()))
            << "While condition's parameter must have the same shape as the "
               "loop's 'init'. init: "
            << init->ToString() << ", param: " << cond_param->ToString();
        auto* cond_root = while_cond->root_instruction();
        TF_RET_CHECK(ShapeUtil::Compatible(cond_root->shape(),
                                           ShapeUtil::MakeShape(PRED, {})))
            << "While condition should have shape PRED: "
            << cond_root->ToString();

        auto* body_param = while_body->parameter_instruction(0);
        TF_RET_CHECK(ShapeUtil::Compatible(init->shape(), body_param->shape()))
            << "While body's parameter must have the same shape as the loop's "
               "'init'. init: "
            << init->ToString() << ", param: " << body_param->ToString();
        auto* body_root = while_body->root_instruction();
        TF_RET_CHECK(ShapeUtil::Compatible(init->shape(), body_root->shape()))
            << "While body should have same shape as the loop's 'init'. init: "
            << init->ToString() << ", body: " << body_root->ToString();
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

  return false;
}

}  // namespace xla
