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
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/gpu/stream_executor_util.h"

namespace xla {
namespace gpu {
namespace {

class Visitor : public DfsHloVisitorWithDefault {
 public:
  explicit Visitor(HloComputation* computation, se::Stream* stream)
      : computation_(computation), stream_(stream) {}

  static bool Run(HloComputation* computation, se::Stream* stream) {
    Visitor visitor(computation, stream);
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
  se::Stream* stream_;
  HloInstruction* AddConvert(HloInstruction* hlo, PrimitiveType elem_type) {
    Shape shape = ShapeUtil::ChangeElementType(hlo->shape(), elem_type);
    return this->computation_->AddInstruction(
        HloInstruction::CreateConvert(shape, hlo));
  }
};

// cudnn defines CUDNN_BN_MIN_EPSILON = 1e-5 as the minimum acceptable epsilon
// for calls to its batchnorm ops.
bool EpsilonInRange(HloInstruction* batch_norm) {
  return batch_norm->epsilon() >= 1e-5;
}

bool IsF32BatchNormWithFP16Inputs(HloInstruction* batch_norm) {
  auto convert = batch_norm->operand(0);
  if (convert->opcode() != HloOpcode::kConvert) {
    return false;
  }
  return convert->operand(0)->shape().element_type() == F16;
}

size_t GetWorkspaceSize(se::Stream* stream, HloInstruction* batch_norm,
                        bool is_batchnorm_with_fp16_inputs, bool apply_relu,
                        bool apply_side_inputs) {
  DnnBatchDescriptors batch_descs = MakeBatchNormDescriptors(
      batch_norm->operand(0)->shape(), batch_norm->feature_index());
  se::dnn::BatchDescriptor input_desc = batch_descs.input_desc;
  se::dnn::BatchDescriptor scale_offset_desc = batch_descs.scale_offset_desc;
  se::BatchNormalizationKind kind;
  if (batch_norm->opcode() == HloOpcode::kBatchNormTraining) {
    kind = se::BatchNormalizationKind::kBatchnormForward;
  } else if (batch_norm->opcode() == HloOpcode::kBatchNormGrad) {
    kind = se::BatchNormalizationKind::kBatchnormBackward;
  } else {
    LOG(ERROR) << "Workspace can only be queried for Batchnorm training "
                  "forward and gradient.";
  }
  size_t workspace_size = 0;
  if (stream) {
    if (is_batchnorm_with_fp16_inputs) {
      stream->ThenFindBatchNormWorkspaceSize<Eigen::half, float>(
          input_desc, scale_offset_desc, &workspace_size, kind, apply_relu,
          apply_side_inputs);
    } else {
      DCHECK_EQ(apply_relu, false)
          << "cudnnGetBatchNormalizationForwardTrainingWorkspaceSize is only "
             "supported for CUDNN_DATA_HALF data type for "
             "CUDNN_BATCHNORM_OPS_BN_ACTIVATION mode. ";
      DCHECK_EQ(apply_side_inputs, false)
          << "cudnnGetBatchNormalizationForwardTrainingWorkspaceSize is only "
             "supported for CUDNN_DATA_HALF data type for "
             "CUDNN_BATCHNORM_OPS_BN_ADD_ACTIVATION mode. ";
      stream->ThenFindBatchNormWorkspaceSize<float, float>(
          input_desc, scale_offset_desc, &workspace_size, kind, apply_relu,
          apply_side_inputs);
    }
  } else {
    VLOG(1) << "Stream is nullptr. Hence workspace not queried. ";
  }
  return workspace_size;
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

  bool is_batchnorm_with_fp16_inputs = IsF32BatchNormWithFP16Inputs(batch_norm);
  if (is_batchnorm_with_fp16_inputs) {
    operands[0] = AddConvert(batch_norm->mutable_operand(0), F16);
  }
  operands.push_back(epsilon);
  operands.push_back(feature_index);

  auto batch_norm_shape = ShapeUtil::MakeShape(
      operands[0]->shape().element_type(), batch_norm->shape().dimensions());

  auto batchnorm_inference_result = HloInstruction::CreateCustomCall(
      batch_norm_shape, operands, kCudnnBatchNormForwardInferenceCallTarget);
  if (is_batchnorm_with_fp16_inputs) {
    HloInstruction* libcall =
        computation_->AddInstruction(std::move(batchnorm_inference_result));
    Shape shape_f32 = ShapeUtil::ChangeElementType(libcall->shape(), F32);
    batchnorm_inference_result =
        HloInstruction::CreateConvert(shape_f32, libcall);
  }

  TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
      batch_norm, std::move(batchnorm_inference_result)));
  changed_ = true;
  return Status::OK();
}

Status Visitor::HandleBatchNormTraining(HloInstruction* batch_norm) {
  VLOG(1) << batch_norm->ToString();
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
  bool is_batchnorm_with_fp16_inputs = IsF32BatchNormWithFP16Inputs(batch_norm);
  if (is_batchnorm_with_fp16_inputs) {
    operands[0] = AddConvert(batch_norm->mutable_operand(0), F16);
    if (batch_norm->operands().size() == 4) {
      operands[3] = AddConvert(batch_norm->mutable_operand(3), F16);
    }
  }
  operands.push_back(epsilon);
  operands.push_back(feature_index);

  std::vector<Shape> batch_norm_tuple_shape;
  batch_norm_tuple_shape.push_back(
      ShapeUtil::MakeShape(operands[0]->shape().element_type(),
                           batch_norm->shape().tuple_shapes(0).dimensions()));
  for (int i = 1; i < batch_norm->shape().tuple_shapes_size(); i++) {
    batch_norm_tuple_shape.push_back(batch_norm->shape().tuple_shapes(i));
  }

  auto num_outputs = batch_norm_tuple_shape.size();
  bool use_reserve_space = num_outputs == 4;
  // When batch-norm-training hlo has been lowered from
  // either FusedBatchnormV3 or FusedBatchNormEx, the hlo has
  // an extra reserve space output i.e, 4 outputs. This implies that an extra
  // temp workspace also needs to be allocated.
  if (use_reserve_space) {
    bool apply_relu = static_cast<HloBatchNormTrainingInstruction*>(batch_norm)
                          ->is_activation_relu();
    bool apply_side_input = batch_norm->operands().size() == 4;
    size_t workspace_size =
        GetWorkspaceSize(stream_, batch_norm, is_batchnorm_with_fp16_inputs,
                         apply_relu, apply_side_input);
    VLOG(1) << "Forward workspace required: " << workspace_size << " bytes";
    batch_norm_tuple_shape.push_back(
        ShapeUtil::MakeShape(U8, {workspace_size}));
  }

  const Shape& batch_norm_shape =
      ShapeUtil::MakeTupleShape(batch_norm_tuple_shape);

  HloInstruction* libcall =
      computation_->AddInstruction(HloInstruction::CreateCustomCall(
          batch_norm_shape, operands,
          kCudnnBatchNormForwardTrainingCallTarget));
  TF_ASSIGN_OR_RETURN(
      CudnnBatchNormBackendConfig config,
      batch_norm->backend_config<CudnnBatchNormBackendConfig>());
  se::dnn::ActivationMode activation_mode =
      static_cast<HloBatchNormTrainingInstruction*>(batch_norm)
              ->is_activation_relu()
          ? se::dnn::ActivationMode::kRelu
          : se::dnn::ActivationMode::kNone;
  config.set_activation_mode(static_cast<int64>(activation_mode));
  TF_RETURN_IF_ERROR(libcall->set_backend_config(config));
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

  HloInstruction* new_gte =
      computation_->AddInstruction(HloInstruction::CreateGetTupleElement(
          libcall->shape().tuple_shapes(0), libcall, 0));

  if (is_batchnorm_with_fp16_inputs) {
    new_gte = AddConvert(new_gte, F32);
  }
  // Repackage the results. Although this tuple is redundant when convert is not
  // inserted, TupleSimplifier eliminates the Tuple eventually
  HloInstruction::InstructionVector replacing_tuple_elements = {
      new_gte,
      computation_->AddInstruction(HloInstruction::CreateGetTupleElement(
          libcall->shape().tuple_shapes(1), libcall, 1)),
      variance};

  if (use_reserve_space) {
    replacing_tuple_elements.push_back(
        computation_->AddInstruction(HloInstruction::CreateGetTupleElement(
            libcall->shape().tuple_shapes(3), libcall, 3)));
  }
  std::unique_ptr<HloInstruction> replacing_tuple =
      HloInstruction::CreateTuple(replacing_tuple_elements);
  HloInstruction* replacing_tuple_ptr = replacing_tuple.get();
  CHECK_NE(replacing_tuple_ptr, nullptr);
  TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
      batch_norm, std::move(replacing_tuple)));
  changed_ = true;
  return Status::OK();
}

Status Visitor::HandleBatchNormGrad(HloInstruction* batch_norm) {
  VLOG(1) << batch_norm->ToString();
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
  bool is_batchnorm_with_fp16_inputs = IsF32BatchNormWithFP16Inputs(batch_norm);
  if (is_batchnorm_with_fp16_inputs) {
    HloInstruction* operand_0_convert = batch_norm->mutable_operand(0);
    operands[0] = AddConvert(operand_0_convert, F16);
    for (auto index = 1; index < operands.size(); index++) {
      if (batch_norm->operand(index)->opcode() == HloOpcode::kConvert &&
          batch_norm->operand(index)->operand(0)->shape().element_type() ==
              F16 &&
          ShapeUtil::Compatible(operand_0_convert->shape(),
                                batch_norm->mutable_operand(index)->shape())) {
        operands[index] = AddConvert(batch_norm->mutable_operand(index), F16);
      }
    }
  }
  operands[3] = inverse_stddev;
  operands.push_back(epsilon);
  operands.push_back(feature_index);

  std::vector<Shape> batch_norm_tuple_shape;
  batch_norm_tuple_shape.push_back(
      ShapeUtil::MakeShape(operands[0]->shape().element_type(),
                           batch_norm->shape().tuple_shapes(0).dimensions()));
  for (int i = 1; i < batch_norm->shape().tuple_shapes_size(); i++) {
    batch_norm_tuple_shape.push_back(batch_norm->shape().tuple_shapes(i));
  }

  auto num_inputs = batch_norm->operand_count();
  bool use_reserve_space = num_inputs == 6;
  // When batch-norm-grad hlo has been lowered from
  // either FusedBatchnormGradV3 or FusedBatchNormGradEx, the hlo has
  // an extra reserve space input i.e, 6 inputs {operand, scale, mean,
  // variance, out_grad, reserve_space}. This implies that an extra
  // temp workspace also needs to be allocated.
  if (use_reserve_space) {
    size_t workspace_size = GetWorkspaceSize(
        stream_, batch_norm, is_batchnorm_with_fp16_inputs, false, false);
    VLOG(1) << "Backward workspace required: " << workspace_size << " bytes";
    batch_norm_tuple_shape.push_back(
        ShapeUtil::MakeShape(U8, {workspace_size}));
  }

  const Shape& batch_norm_shape =
      ShapeUtil::MakeTupleShape(batch_norm_tuple_shape);
  HloInstruction* libcall =
      computation_->AddInstruction(HloInstruction::CreateCustomCall(
          batch_norm_shape, operands, kCudnnBatchNormBackwardCallTarget));

  HloInstruction* new_gte =
      computation_->AddInstruction(HloInstruction::CreateGetTupleElement(
          libcall->shape().tuple_shapes(0), libcall, 0));

  if (is_batchnorm_with_fp16_inputs) {
    new_gte = AddConvert(new_gte, F32);
  }
  // Repackage the results. Although this tuple is redundant when convert is not
  // inserted, TupleSimplifier eliminates the Tuple eventually
  std::unique_ptr<HloInstruction> replacing_tuple = HloInstruction::CreateTuple(
      {new_gte,
       computation_->AddInstruction(HloInstruction::CreateGetTupleElement(
           libcall->shape().tuple_shapes(1), libcall, 1)),
       computation_->AddInstruction(HloInstruction::CreateGetTupleElement(
           libcall->shape().tuple_shapes(2), libcall, 2))});

  TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
      batch_norm, std::move(replacing_tuple)));
  changed_ = true;
  return Status::OK();
}

}  // anonymous namespace

StatusOr<bool> CudnnBatchNormRewriter::Run(HloModule* module) {
  VLOG(2) << "CudnnBatchNormRewriter::Run(), before:";
  XLA_VLOG_LINES(2, module->ToString());

  if (allocator_ == nullptr) {
    allocator_ = stream_exec_->GetAllocator();
  }

  TF_ASSIGN_OR_RETURN(se::Stream* const stream,
                      allocator_->GetStream(stream_exec_->device_ordinal()));

  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (Visitor::Run(comp, stream)) {
      changed = true;
    }
  }

  VLOG(2) << "CudnnBatchNormRewriter::Run(), after:";
  XLA_VLOG_LINES(2, module->ToString());
  return changed;
}

}  // namespace gpu
}  // namespace xla
