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

#include "tensorflow/compiler/xla/service/batchnorm_expander.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

namespace {

using absl::optional;

// BatchNormExpanderVisitor traverses the HLO computation and rewrites BatchNorm
// operations into smaller operations.
class BatchNormExpanderVisitor : public DfsHloVisitorWithDefault {
 public:
  // Default visitor action is to do nothing and return OK.
  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return Status::OK();
  }

  Status HandleBatchNormTraining(HloInstruction* batch_norm) override;

  Status HandleBatchNormInference(HloInstruction* batch_norm) override;

  Status HandleBatchNormGrad(HloInstruction* batch_norm) override;

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation, bool rewrite_training_op,
                  bool rewrite_inference_op, bool rewrite_grad_op);

  // Returns whether any batch norm ops were rewritten.
  const bool changed() const { return changed_; }

  ~BatchNormExpanderVisitor() override = default;

 private:
  explicit BatchNormExpanderVisitor(HloComputation* computation,
                                    bool rewrite_training_op,
                                    bool rewrite_inference_op,
                                    bool rewrite_grad_op)
      : computation_(computation),
        rewrite_training_op_(rewrite_training_op),
        rewrite_inference_op_(rewrite_inference_op),
        rewrite_grad_op_(rewrite_grad_op) {}

  HloComputation* GetOrCreateScalarAddComputation(
      PrimitiveType primitive_type) {
    HloComputation::Builder b("scalar_add_computation");
    Shape shape = ShapeUtil::MakeShape(primitive_type, {});
    auto scalar_lhs = b.AddInstruction(
        HloInstruction::CreateParameter(0, shape, "scalar_lhs"));
    auto scalar_rhs = b.AddInstruction(
        HloInstruction::CreateParameter(1, shape, "scalar_rhs"));
    auto scalar_op = b.AddInstruction(HloInstruction::CreateBinary(
        shape, HloOpcode::kAdd, scalar_lhs, scalar_rhs));
    return computation_->parent()->AddEmbeddedComputation(b.Build(scalar_op));
  }

  std::unique_ptr<HloInstruction> Rsqrt(
      HloInstruction* operand,
      const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
          add_instruction) {
    HloInstruction* exponent = add_instruction(HloInstruction::CreateBroadcast(
        operand->shape(),
        add_instruction(HloInstruction::CreateConvert(
            ShapeUtil::MakeShape(operand->shape().element_type(), {}),
            add_instruction(HloInstruction::CreateConstant(
                LiteralUtil::CreateR0<float>(-0.5f))))),
        {}));
    return HloInstruction::CreateBinary(operand->shape(), HloOpcode::kPower,
                                        operand, exponent);
  }

  std::unique_ptr<HloInstruction> Mean(
      int64 element_count, HloInstruction* operand,
      const std::function<HloInstruction*(std::unique_ptr<HloInstruction>)>&
          add_instruction) {
    HloInstruction* elem_count_recip =
        add_instruction(HloInstruction::CreateBroadcast(
            operand->shape(),
            add_instruction(HloInstruction::CreateConvert(
                ShapeUtil::MakeShape(operand->shape().element_type(), {}),
                add_instruction(HloInstruction::CreateConstant(
                    LiteralUtil::CreateR0<float>(1.0 / element_count))))),
            {}));
    return HloInstruction::CreateBinary(operand->shape(), HloOpcode::kMultiply,
                                        operand, elem_count_recip);
  }

  // Replaces the existing HLO instruction old_instruction, with
  // new_instruction, and marks the optimizer status as changed.
  // Returns the Status representing the result of the replace operation.
  Status ReplaceWithNewInstruction(
      HloInstruction* old_instruction,
      std::unique_ptr<HloInstruction> new_instruction) {
    TF_RETURN_IF_ERROR(computation_->ReplaceWithNewInstruction(
        old_instruction, std::move(new_instruction)));
    changed_ = true;
    return Status::OK();
  }

  // Replaces the existing HLO instruction old_instruction, with
  // new_instruction, and marks the optimizer status as changed.
  // Returns the Status representing the result of the replace operation.
  Status ReplaceInstruction(HloInstruction* old_instruction,
                            HloInstruction* new_instruction) {
    TF_RETURN_IF_ERROR(
        computation_->ReplaceInstruction(old_instruction, new_instruction));
    changed_ = true;
    return Status::OK();
  }
  // Current HloComputation instance the BatchNormExpander is
  // traversing.
  HloComputation* computation_;

  bool rewrite_training_op_;
  bool rewrite_inference_op_;
  bool rewrite_grad_op_;

  // Whether rewrite has occurred.
  bool changed_ = false;
};

}  // namespace

bool BatchNormExpanderVisitor::Run(HloComputation* computation,
                                   bool rewrite_training_op,
                                   bool rewrite_inference_op,
                                   bool rewrite_grad_op) {
  BatchNormExpanderVisitor visitor(
      computation,
      /*rewrite_training_op=*/rewrite_training_op,
      /*rewrite_inference_op=*/rewrite_inference_op,
      /*rewrite_grad_op=*/rewrite_grad_op);
  TF_CHECK_OK(computation->Accept(&visitor));
  return visitor.changed_;
}

Status BatchNormExpanderVisitor::HandleBatchNormTraining(
    HloInstruction* batch_norm) {
  if (!rewrite_training_op_) {
    return Status::OK();
  }

  std::vector<HloInstruction*> added_instructions;
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
    HloInstruction* added_inst = computation_->AddInstruction(std::move(inst));
    added_inst->set_metadata(batch_norm->metadata());
    added_instructions.push_back(added_inst);
    return added_inst;
  };
  auto add_binary = [&](const Shape& shape, const HloOpcode opcode,
                        HloInstruction* a, HloInstruction* b) {
    return add(HloInstruction::CreateBinary(shape, opcode, a, b));
  };
  int64 instruction_count_before = computation_->instruction_count();

  // Expand batch norm training into smaller HLO ops.
  HloInstruction* operand = batch_norm->mutable_operand(0);
  const Shape operand_shape = operand->shape();
  PrimitiveType ptype = operand_shape.element_type();
  int64 feature_index = batch_norm->feature_index();
  const int64 feature_count = operand_shape.dimensions(feature_index);
  const int64 size_in_elements = ShapeUtil::ElementsIn(operand_shape);
  int64 elements_per_feature_int64 = size_in_elements / feature_count;

  HloInstruction* scale = batch_norm->mutable_operand(1);
  HloInstruction* offset = batch_norm->mutable_operand(2);
  const Shape feature_shape = scale->shape();

  auto zero_literal = LiteralUtil::CreateR0(0.0f);
  TF_ASSIGN_OR_RETURN(zero_literal, zero_literal->Convert(ptype));
  auto zero = add(HloInstruction::CreateConstant(std::move(zero_literal)));

  auto epsilon_literal = LiteralUtil::CreateR0(batch_norm->epsilon());
  TF_ASSIGN_OR_RETURN(epsilon_literal, epsilon_literal->Convert(ptype));
  auto epsilon = add(HloInstruction::CreateBroadcast(
      operand_shape,
      add(HloInstruction::CreateConstant(std::move(epsilon_literal))), {}));
  std::vector<int64> dimensions_without_feature;

  for (int64 i = 0; i < ShapeUtil::Rank(operand_shape); ++i) {
    if (i != feature_index) {
      dimensions_without_feature.push_back(i);
    }
  }

  auto scale_broadcasted = add(
      HloInstruction::CreateBroadcast(operand_shape, scale, {feature_index}));

  auto offset_broadcasted = add(
      HloInstruction::CreateBroadcast(operand_shape, offset, {feature_index}));

  HloComputation* add_reduce_computation =
      GetOrCreateScalarAddComputation(ptype);

  // X^2.
  auto operand_squared =
      add_binary(operand_shape, HloOpcode::kMultiply, operand, operand);
  // Sum[X].
  auto sum = add(HloInstruction::CreateReduce(feature_shape, operand, zero,
                                              dimensions_without_feature,
                                              add_reduce_computation));

  // Sum[X^2].
  auto squared_sum = add(HloInstruction::CreateReduce(
      feature_shape, operand_squared, zero, dimensions_without_feature,
      add_reduce_computation));

  // E[X].
  auto mean = add(Mean(elements_per_feature_int64, sum, add));

  auto mean_broadcasted = add(
      HloInstruction::CreateBroadcast(operand_shape, mean, {feature_index}));

  // E[X^2].
  auto square_mean = add(Mean(elements_per_feature_int64, squared_sum, add));

  // E^2[X].
  auto mean_square =
      add_binary(feature_shape, HloOpcode::kMultiply, mean, mean);

  // Var[X].
  auto var =
      add_binary(feature_shape, HloOpcode::kSubtract, square_mean, mean_square);

  auto var_broadcasted =
      add(HloInstruction::CreateBroadcast(operand_shape, var, {feature_index}));

  // Var[X] + epsilon.
  auto var_add_epsilon =
      add_binary(operand_shape, HloOpcode::kAdd, var_broadcasted, epsilon);

  // 1 / Sqrt[Var[X] + epsilon].
  auto rsqrt_var_add_epsilon = add(Rsqrt(var_add_epsilon, add));

  // X - E[X].
  auto operand_minus_mean = add_binary(operand_shape, HloOpcode::kSubtract,
                                       operand, mean_broadcasted);

  // (X - E[X]) / Sqrt[Var[X] + epsilon].
  auto normalized = add_binary(operand_shape, HloOpcode::kMultiply,
                               operand_minus_mean, rsqrt_var_add_epsilon);

  // (X - E[X]) / Sqrt[Var[X] + epsilon] * scale.
  auto scaled_normalized = add_binary(operand_shape, HloOpcode::kMultiply,
                                      normalized, scale_broadcasted);

  // (X - E[X]) / Sqrt[Var[X] + epsilon] * scale + offset.
  auto shifted_normalized = add_binary(operand_shape, HloOpcode::kAdd,
                                       scaled_normalized, offset_broadcasted);

  auto tuple = HloInstruction::CreateTuple({shifted_normalized, mean, var});

  if (batch_norm->has_sharding()) {
    int64 instruction_count_after = computation_->instruction_count();
    CHECK_EQ(instruction_count_after,
             instruction_count_before + added_instructions.size());
    const HloSharding& sharding = batch_norm->sharding();
    HloSharding operand_sharding =
        sharding.GetAsShapeTree(batch_norm->shape()).element({0});
    optional<int64> unique_device = batch_norm->sharding_unique_device();
    HloSharding default_sharding =
        unique_device.has_value()
            ? HloSharding::AssignDevice(unique_device.value())
            : HloSharding::Replicate();
    for (HloInstruction* inst : added_instructions) {
      if (ShapeUtil::Equal(inst->shape(), operand_shape)) {
        inst->set_sharding(operand_sharding);
      } else {
        inst->set_sharding(default_sharding);
      }
    }
    tuple->set_sharding(sharding);
  }
  TF_CHECK_OK(ReplaceWithNewInstruction(batch_norm, std::move(tuple)));
  return Status::OK();
}

Status BatchNormExpanderVisitor::HandleBatchNormInference(
    HloInstruction* batch_norm) {
  if (!rewrite_inference_op_) {
    return Status::OK();
  }
  // Expand batch norm inference into smaller HLO ops.
  HloInstruction* operand = batch_norm->mutable_operand(0);
  const Shape operand_shape = operand->shape();
  int64 feature_index = batch_norm->feature_index();
  PrimitiveType ptype = operand_shape.element_type();

  HloInstruction* scale = batch_norm->mutable_operand(1);
  HloInstruction* offset = batch_norm->mutable_operand(2);
  HloInstruction* mean = batch_norm->mutable_operand(3);
  HloInstruction* var = batch_norm->mutable_operand(4);
  const Shape feature_shape = scale->shape();

  auto epsilon_literal = LiteralUtil::CreateR0(batch_norm->epsilon());
  TF_ASSIGN_OR_RETURN(epsilon_literal, epsilon_literal->Convert(ptype));
  auto epsilon = computation_->AddInstruction(HloInstruction::CreateBroadcast(
      operand_shape,
      computation_->AddInstruction(
          HloInstruction::CreateConstant(std::move(epsilon_literal))),
      {}));

  std::vector<int64> dimensions_without_feature;

  for (int64 i = 0; i < ShapeUtil::Rank(operand_shape); ++i) {
    if (i != feature_index) {
      dimensions_without_feature.push_back(i);
    }
  }

  std::vector<HloInstruction*> added_instructions;
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
    HloInstruction* added_inst = computation_->AddInstruction(std::move(inst));
    added_inst->set_metadata(batch_norm->metadata());
    added_instructions.push_back(added_inst);
    return added_inst;
  };
  auto add_binary = [&](const Shape& shape, const HloOpcode opcode,
                        HloInstruction* a, HloInstruction* b) {
    return add(HloInstruction::CreateBinary(shape, opcode, a, b));
  };
  int64 instruction_count_before = computation_->instruction_count();

  auto scale_broadcasted = add(
      HloInstruction::CreateBroadcast(operand_shape, scale, {feature_index}));

  auto offset_broadcasted = add(
      HloInstruction::CreateBroadcast(operand_shape, offset, {feature_index}));

  auto mean_broadcasted = add(
      HloInstruction::CreateBroadcast(operand_shape, mean, {feature_index}));

  auto var_broadcasted =
      add(HloInstruction::CreateBroadcast(operand_shape, var, {feature_index}));

  // Var[X] + epsilon.
  auto var_add_epsilon =
      add_binary(operand_shape, HloOpcode::kAdd, var_broadcasted, epsilon);

  // 1 / Sqrt[Var[X] + epsilon].
  auto rsqrt_var_add_epsilon = add(Rsqrt(var_add_epsilon, add));

  // X - E[X].
  auto operand_minus_mean = add_binary(operand_shape, HloOpcode::kSubtract,
                                       operand, mean_broadcasted);

  // (X - E[X]) / Sqrt[Var[X] + epsilon].
  auto normalized = add_binary(operand_shape, HloOpcode::kMultiply,
                               operand_minus_mean, rsqrt_var_add_epsilon);

  // (X - E[X]) / Sqrt[Var[X] + epsilon] * scale.
  auto scaled_normalized = add_binary(operand_shape, HloOpcode::kMultiply,
                                      normalized, scale_broadcasted);

  // (X - E[X]) / Sqrt[Var[X] + epsilon] * scale + offset.
  auto shifted_normalized = HloInstruction::CreateBinary(
      operand_shape, HloOpcode::kAdd, scaled_normalized, offset_broadcasted);

  int64 instruction_count_after = computation_->instruction_count();
  CHECK_EQ(instruction_count_after,
           instruction_count_before + added_instructions.size());
  if (batch_norm->has_sharding()) {
    const HloSharding& sharding = batch_norm->sharding();
    optional<int64> unique_device = batch_norm->sharding_unique_device();
    HloSharding default_sharding =
        unique_device.has_value()
            ? HloSharding::AssignDevice(unique_device.value())
            : HloSharding::Replicate();
    for (HloInstruction* inst : added_instructions) {
      if (ShapeUtil::Equal(inst->shape(), operand_shape)) {
        inst->set_sharding(sharding);
      } else {
        inst->set_sharding(default_sharding);
      }
    }
    shifted_normalized->set_sharding(sharding);
  }
  TF_CHECK_OK(
      ReplaceWithNewInstruction(batch_norm, std::move(shifted_normalized)));
  return Status::OK();
}

Status BatchNormExpanderVisitor::HandleBatchNormGrad(
    HloInstruction* batch_norm) {
  // Use the following formulas to calculate gradients:
  // scale_grad =
  //   sum(output_grad * (activation - mean(activation))) * rsqrt(var + epsilon)
  //
  // offset_grad =
  //   sum(output_grad)
  //
  // activation_grad =
  //   1/N * scale * rsqrt(var + epsilon) *
  //   (N * output_grad - sum(output_grad) - (activation - mean(activation)) *
  //   sum(output_grad * (activation - mean(activation))) / (variance +
  //   epsilon))
  if (!rewrite_grad_op_) {
    return Status::OK();
  }
  std::vector<HloInstruction*> added_instructions;
  auto add = [&](std::unique_ptr<HloInstruction> inst) {
    HloInstruction* added_inst = computation_->AddInstruction(std::move(inst));
    added_inst->set_metadata(batch_norm->metadata());
    added_instructions.push_back(added_inst);
    return added_inst;
  };
  auto add_binary = [&](const Shape& shape, const HloOpcode opcode,
                        HloInstruction* a, HloInstruction* b) {
    return add(HloInstruction::CreateBinary(shape, opcode, a, b));
  };
  int64 instruction_count_before = computation_->instruction_count();

  HloInstruction* activation = batch_norm->mutable_operand(0);
  const Shape activation_shape = activation->shape();
  PrimitiveType ptype = activation_shape.element_type();
  HloInstruction* scale = batch_norm->mutable_operand(1);
  const Shape feature_shape = scale->shape();
  HloInstruction* mean = batch_norm->mutable_operand(2);
  HloInstruction* variance = batch_norm->mutable_operand(3);
  HloInstruction* grad_output = batch_norm->mutable_operand(4);

  int64 feature_index = batch_norm->feature_index();

  const int64 size_in_elements = ShapeUtil::ElementsIn(activation_shape);
  const int64 feature_count = activation_shape.dimensions(feature_index);
  const int64 elements_per_feature_int64 = size_in_elements / feature_count;

  auto zero_literal = LiteralUtil::CreateR0(0.0f);
  TF_ASSIGN_OR_RETURN(zero_literal, zero_literal->Convert(ptype));
  auto zero = add(HloInstruction::CreateConstant(std::move(zero_literal)));

  auto epsilon_literal = LiteralUtil::CreateR0(batch_norm->epsilon());
  TF_ASSIGN_OR_RETURN(epsilon_literal, epsilon_literal->Convert(ptype));
  auto epsilon_scalar =
      add(HloInstruction::CreateConstant(std::move(epsilon_literal)));
  auto epsilon_activation = add(
      HloInstruction::CreateBroadcast(activation_shape, epsilon_scalar, {}));
  auto epsilon_feature =
      add(HloInstruction::CreateBroadcast(feature_shape, epsilon_scalar, {}));

  std::vector<int64> dimensions_without_feature;

  for (int64 i = 0; i < ShapeUtil::Rank(activation_shape); ++i) {
    if (i != feature_index) {
      dimensions_without_feature.push_back(i);
    }
  }

  auto scale_broadcasted = add(HloInstruction::CreateBroadcast(
      activation_shape, scale, {feature_index}));
  auto variance_broadcasted = add(HloInstruction::CreateBroadcast(
      activation_shape, variance, {feature_index}));

  // E[X].
  auto mean_broadcasted = add(
      HloInstruction::CreateBroadcast(activation_shape, mean, {feature_index}));

  // rsqrt[Var[X] + epsilon].
  auto rsqrt_var_add_epsilon_broadcasted =
      add(Rsqrt(add_binary(activation_shape, HloOpcode::kAdd,
                           variance_broadcasted, epsilon_activation),
                add));

  auto rsqrt_var_add_epsilon = add(Rsqrt(
      add_binary(feature_shape, HloOpcode::kAdd, variance, epsilon_feature),
      add));

  // X - E[X].
  auto activation_minus_mean = add_binary(
      activation_shape, HloOpcode::kSubtract, activation, mean_broadcasted);

  // Grad[Y] * (X - E[X]).
  auto grad_output_times_activiation_minus_mean =
      add_binary(activation_shape, HloOpcode::kMultiply, grad_output,
                 activation_minus_mean);

  HloComputation* add_reduce_computation =
      GetOrCreateScalarAddComputation(ptype);

  // sum(Grad[Y] * (X - E[X])).
  auto sum_grad_output_times_activiation_minus_mean =
      add(HloInstruction::CreateReduce(
          feature_shape, grad_output_times_activiation_minus_mean, zero,
          dimensions_without_feature, add_reduce_computation));

  // Grad[beta] = Sum(Grad[Y]).
  auto grad_beta = add(HloInstruction::CreateReduce(
      feature_shape, grad_output, zero, dimensions_without_feature,
      add_reduce_computation));

  // Grad[scale] = Sum(Grad[Y] * (X - E[X]) * rsqrt[Var[X] + epsilon]).
  auto grad_scale = add_binary(feature_shape, HloOpcode::kMultiply,
                               sum_grad_output_times_activiation_minus_mean,
                               rsqrt_var_add_epsilon);

  // I2 = Sum(Grad[Y])
  auto i2 = add(HloInstruction::CreateBroadcast(activation_shape, grad_beta,
                                                {feature_index}));

  // I3 = Sum(Grad[Y] * (X - E[X]))
  auto i3 = add(HloInstruction::CreateBroadcast(
      activation_shape, sum_grad_output_times_activiation_minus_mean,
      {feature_index}));

  // I4 = (X - E[X]) * I3
  auto i4 = add_binary(activation_shape, HloOpcode::kMultiply, i3,
                       activation_minus_mean);

  // I5 = I4 / (Var[X] + epsilon)
  auto i5 = add_binary(activation_shape, HloOpcode::kDivide, i4,
                       add_binary(activation_shape, HloOpcode::kAdd,
                                  variance_broadcasted, epsilon_activation));

  // scale * rsqrt[Var[X] + epsilon] * 1/N
  auto scale_times_rsqrt_var_add_epsilon =
      add_binary(activation_shape, HloOpcode::kMultiply, scale_broadcasted,
                 rsqrt_var_add_epsilon_broadcasted);

  scale_times_rsqrt_var_add_epsilon = add(
      Mean(elements_per_feature_int64, scale_times_rsqrt_var_add_epsilon, add));

  auto elements_per_feature_literal =
      LiteralUtil::CreateR0<float>(elements_per_feature_int64);
  TF_ASSIGN_OR_RETURN(elements_per_feature_literal,
                      elements_per_feature_literal->Convert(ptype));
  auto elements_per_feature = add(
      HloInstruction::CreateConstant(std::move(elements_per_feature_literal)));
  auto i1 = add_binary(activation_shape, HloOpcode::kMultiply, grad_output,
                       add(HloInstruction::CreateBroadcast(
                           activation_shape, elements_per_feature, {})));

  // I6 = I1 - I2 - I5
  auto i6 = add_binary(
      activation_shape, HloOpcode::kSubtract,
      add_binary(activation_shape, HloOpcode::kSubtract, i1, i2), i5);

  // Grad[X] = scale * rsqrt[Var[X] + epsilon] * 1/N * I6.
  auto grad_activation = add_binary(activation_shape, HloOpcode::kMultiply,
                                    scale_times_rsqrt_var_add_epsilon, i6);
  auto tuple =
      HloInstruction::CreateTuple({grad_activation, grad_scale, grad_beta});
  if (batch_norm->has_sharding()) {
    const HloSharding& sharding = batch_norm->sharding();
    int64 instruction_count_after = computation_->instruction_count();
    CHECK_EQ(instruction_count_after,
             instruction_count_before + added_instructions.size());
    HloSharding activation_sharding =
        sharding.GetAsShapeTree(batch_norm->shape()).element({0});
    auto unique_device = batch_norm->sharding_unique_device();
    HloSharding default_sharding =
        unique_device.has_value()
            ? HloSharding::AssignDevice(unique_device.value())
            : HloSharding::Replicate();
    for (HloInstruction* inst : added_instructions) {
      if (ShapeUtil::Equal(inst->shape(), activation_shape)) {
        inst->set_sharding(activation_sharding);
      } else {
        inst->set_sharding(default_sharding);
      }
    }
    tuple->set_sharding(sharding);
  }

  TF_CHECK_OK(ReplaceWithNewInstruction(batch_norm, std::move(tuple)));

  return Status::OK();
}

StatusOr<bool> BatchNormExpander::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "BatchNormExpander::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (auto* comp : module->MakeNonfusionComputations()) {
    if (BatchNormExpanderVisitor::Run(comp, rewrite_training_op_,
                                      rewrite_inference_op_,
                                      rewrite_grad_op_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "BatchNormExpander::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
