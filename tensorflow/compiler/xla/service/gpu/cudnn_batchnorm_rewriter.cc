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

#include "tensorflow/compiler/xla/service/gpu/cudnn_batchnorm_rewriter.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

namespace xla {
namespace gpu {
namespace {

class Visitor : public DfsHloVisitorWithDefault {
 public:
  explicit Visitor(HloComputation* computation) : computation_(computation) {}

  static bool Run(HloComputation* computation) {
    Visitor visitor(computation);
    TF_CHECK_OK(computation->Accept(&visitor));
    return visitor.changed_;
  }

  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return Status::OK();
  }

  Status HandleBatchNormInference(HloInstruction* batch_norm) override;
  Status HandleBatchNormTraining(HloInstruction* batch_norm) override;
  Status HandleBatchNormGrad(HloInstruction* batch_norm) override;

 private:
  bool changed_ = false;
  HloComputation* computation_;
};

// cudnn defines CUDNN_BN_MIN_EPSILON = 1e-5 as the minimum acceptable epsilon
// for calls to its batchnorm ops.
bool EpsilonInRange(HloInstruction* batch_norm) {
  return batch_norm->epsilon() >= 1e-5;
}

Status Visitor::HandleBatchNormInference(HloInstruction* batch_norm) {
  if (batch_norm->operand(0)->shape().element_type() != F32) {
    VLOG(1) << "Not rewriting op with non-F32 element type: "
            << batch_norm->ToString();
    return Status::OK();
  }

  // cudnn errors out on zero-sized inputs.
  if (ShapeUtil::ElementsIn(batch_norm->operand(0)->shape()) == 0) {
    return Status::OK();
  }

  if (!EpsilonInRange(batch_norm)) {
    return Status::OK();
  }

  HloInstruction* epsilon =
      computation_->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0(batch_norm->epsilon())));
  HloInstruction* feature_index =
      computation_->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0(batch_norm->feature_index())));

  std::vector<HloInstruction*> operands(batch_norm->operands().begin(),
                                        batch_norm->operands().end());
  operands.push_back(epsilon);
  operands.push_back(feature_index);

  std::unique_ptr<HloInstruction> libcall = HloInstruction::CreateCustomCall(
      batch_norm->shape(), operands, kCudnnBatchNormForwardInferenceCallTarget);
  TF_RETURN_IF_ERROR(
      computation_->ReplaceWithNewInstruction(batch_norm, std::move(libcall)));
  changed_ = true;
  return Status::OK();
}

Status Visitor::HandleBatchNormTraining(HloInstruction* batch_norm) {
  if (batch_norm->operand(0)->shape().element_type() != F32) {
    VLOG(1) << "Not rewriting op with non-F32 element type: "
            << batch_norm->ToString();
    return Status::OK();
  }

  // cudnn errors out on zero-sized inputs.
  if (ShapeUtil::ElementsIn(batch_norm->operand(0)->shape()) == 0) {
    return Status::OK();
  }

  if (!EpsilonInRange(batch_norm)) {
    return Status::OK();
  }

  HloInstruction* epsilon =
      computation_->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0(batch_norm->epsilon())));
  HloInstruction* feature_index =
      computation_->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0(batch_norm->feature_index())));

  std::vector<HloInstruction*> operands(batch_norm->operands().begin(),
                                        batch_norm->operands().end());
  operands.push_back(epsilon);
  operands.push_back(feature_index);

  HloInstruction* libcall =
      computation_->AddInstruction(HloInstruction::CreateCustomCall(
          batch_norm->shape(), operands,
          kCudnnBatchNormForwardTrainingCallTarget));

  // The cudnn libcall returns a tuple
  //   {output, mean, rsqrt(variance + epsilon)},
  // but the batchnorm HLO returns {output, mean, variance}.  Fix it up.
  HloInstruction* inverse_stddev =
      computation_->AddInstruction(HloInstruction::CreateGetTupleElement(
          libcall->shape().tuple_shapes(2), libcall, 2));
  HloInstruction* variance_plus_epsilon =
      computation_->AddInstruction(HloInstruction::CreateBinary(
          inverse_stddev->shape(), HloOpcode::kPower, inverse_stddev,
          computation_->AddInstruction(HloInstruction::CreateBroadcast(
              inverse_stddev->shape(),
              computation_->AddInstruction(HloInstruction::CreateConstant(
                  LiteralUtil::CreateR0<float>(-2))),
              {}))));
  HloInstruction* variance =
      computation_->AddInstruction(HloInstruction::CreateBinary(
          variance_plus_epsilon->shape(), HloOpcode::kSubtract,
          variance_plus_epsilon,
          computation_->AddInstruction(HloInstruction::CreateBroadcast(
              variance_plus_epsilon->shape(), epsilon, {}))));

  // Repackage the results.
  std::unique_ptr<HloInstruction> new_tuple = HloInstruction::CreateTuple({
      computation_->AddInstruction(HloInstruction::CreateGetTupleElement(
          libcall->shape().tuple_shapes(0), libcall, 0)),
      computation_->AddInstruction(HloInstruction::CreateGetTupleElement(
          libcall->shape().tuple_shapes(1), libcall, 1)),
      variance,
  });

  TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
      batch_norm, std::move(new_tuple)));
  changed_ = true;
  return Status::OK();
}

Status Visitor::HandleBatchNormGrad(HloInstruction* batch_norm) {
  if (batch_norm->operand(0)->shape().element_type() != F32) {
    VLOG(1) << "Not rewriting op with non-F32 element type: "
            << batch_norm->ToString();
    return Status::OK();
  }

  // cudnn errors out on zero-sized inputs.
  if (ShapeUtil::ElementsIn(batch_norm->operand(0)->shape()) == 0) {
    return Status::OK();
  }

  if (!EpsilonInRange(batch_norm)) {
    return Status::OK();
  }

  HloInstruction* epsilon =
      computation_->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0(batch_norm->epsilon())));
  HloInstruction* feature_index =
      computation_->AddInstruction(HloInstruction::CreateConstant(
          LiteralUtil::CreateR0(batch_norm->feature_index())));

  // The cudnn libcall expects its input to be rsqrt(variance + epsilon), but
  // the batchnorm HLO takes plain variance as input.  Fix it up.
  HloInstruction* var_plus_epsilon =
      computation_->AddInstruction(HloInstruction::CreateBinary(
          batch_norm->operand(3)->shape(), HloOpcode::kAdd,
          batch_norm->mutable_operand(3),
          computation_->AddInstruction(HloInstruction::CreateBroadcast(
              batch_norm->operand(3)->shape(), epsilon, {}))));
  HloInstruction* inverse_stddev =
      computation_->AddInstruction(HloInstruction::CreateUnary(
          var_plus_epsilon->shape(), HloOpcode::kRsqrt, var_plus_epsilon));

  std::vector<HloInstruction*> operands(batch_norm->operands().begin(),
                                        batch_norm->operands().end());
  operands[3] = inverse_stddev;
  operands.push_back(epsilon);
  operands.push_back(feature_index);

  std::unique_ptr<HloInstruction> libcall = HloInstruction::CreateCustomCall(
      batch_norm->shape(), operands, kCudnnBatchNormBackwardCallTarget);

  TF_RETURN_IF_ERROR(
      computation_->ReplaceWithNewInstruction(batch_norm, std::move(libcall)));
  changed_ = true;
  return Status::OK();
}

}  // anonymous namespace

StatusOr<bool> CudnnBatchNormRewriter::Run(HloModule* module) {
  VLOG(2) << "CudnnBatchNormRewriter::Run(), before:";
  XLA_VLOG_LINES(2, module->ToString());

  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (Visitor::Run(comp)) {
      changed = true;
    }
  }

  VLOG(2) << "CudnnBatchNormRewriter::Run(), after:";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
