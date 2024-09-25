/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/batchnorm_expander.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/literal.h"
#include "xla/literal_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/status.h"
#include "tsl/platform/statusor.h"

namespace xla {

namespace {

using std::optional;

// BatchNormExpanderVisitor traverses the HLO computation and rewrites BatchNorm
// operations into smaller operations.
class BatchNormExpanderVisitor : public DfsHloRewriteVisitor {
 public:
  absl::Status HandleBatchNormTraining(HloInstruction* batch_norm) override;

  absl::Status HandleBatchNormInference(HloInstruction* batch_norm) override;

  absl::Status HandleBatchNormGrad(HloInstruction* batch_norm) override;

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation, bool rewrite_training_op,
                  bool rewrite_inference_op, bool rewrite_grad_op);

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

  std::unique_ptr<HloInstruction> Rsqrt(HloInstruction* operand) {
    return HloInstruction::CreateUnary(operand->shape(), HloOpcode::kRsqrt,
                                       operand);
  }

  std::unique_ptr<HloInstruction> Mean(
      HloInstruction* element_count, HloInstruction* operand,
      absl::FunctionRef<HloInstruction*(std::unique_ptr<HloInstruction>)>
          add_instruction) {
    auto broadcast = add_instruction(HloInstruction::CreateBroadcast(
        ShapeUtil::MakeStaticShape(operand->shape()), element_count, {}));
    return HloInstruction::CreateBinary(operand->shape(), HloOpcode::kDivide,
                                        operand, broadcast);
  }

  std::unique_ptr<HloInstruction> DynamicElementCountPerFeature(
      HloInstruction* operand, int64_t feature_index,
      absl::FunctionRef<HloInstruction*(std::unique_ptr<HloInstruction>)>
          add_instruction) {
    auto elements_per_feature_s32 = add_instruction(
        HloInstruction::CreateConstant(LiteralUtil::CreateR0<int32_t>(1)));

    for (int64_t i = 0; i < operand->shape().rank(); ++i) {
      if (i == feature_index) {
        continue;
      }
      auto dynamic_dimension_size =
          add_instruction(HloInstruction::CreateGetDimensionSize(
              ShapeUtil::MakeShape(S32, {}), operand, i));
      elements_per_feature_s32 = add_instruction(HloInstruction::CreateBinary(
          ShapeUtil::MakeShape(S32, {}), HloOpcode::kMultiply,
          dynamic_dimension_size, elements_per_feature_s32));
    }

    return HloInstruction::CreateConvert(
        ShapeUtil::MakeShape(operand->shape().element_type(), {}),
        elements_per_feature_s32);
  }

  // Current HloComputation instance the BatchNormExpander is
  // traversing.
  HloComputation* computation_;

  bool rewrite_training_op_;
  bool rewrite_inference_op_;
  bool rewrite_grad_op_;
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
  return visitor.changed();
}

absl::Status BatchNormExpanderVisitor::HandleBatchNormTraining(
    HloInstruction* batch_norm) {
  if (!rewrite_training_op_) {
    return absl::OkStatus();
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
  int64_t instruction_count_before = computation_->instruction_count();

  // Expand batch norm training into smaller HLO ops.
  HloInstruction* operand = batch_norm->mutable_operand(0);
  const Shape operand_shape = operand->shape();
  PrimitiveType ptype = operand_shape.element_type();
  int64_t feature_index = batch_norm->feature_index();

  HloInstruction* scale = batch_norm->mutable_operand(1);
  HloInstruction* offset = batch_norm->mutable_operand(2);
  const Shape feature_shape = scale->shape();

  auto zero_literal = LiteralUtil::CreateR0(0.0f);
  TF_ASSIGN_OR_RETURN(zero_literal, zero_literal.Convert(ptype));
  auto zero = add(HloInstruction::CreateConstant(std::move(zero_literal)));

  auto epsilon_literal = LiteralUtil::CreateR0(batch_norm->epsilon());
  TF_ASSIGN_OR_RETURN(epsilon_literal, epsilon_literal.Convert(ptype));
  Shape scalar_broadcast_shape = ShapeUtil::MakeStaticShape(operand_shape);
  auto epsilon = add(HloInstruction::CreateBroadcast(
      scalar_broadcast_shape,
      add(HloInstruction::CreateConstant(std::move(epsilon_literal))), {}));
  std::vector<int64_t> dimensions_without_feature;
  const int64_t rank = operand_shape.rank();
  dimensions_without_feature.reserve(rank - 1);

  for (int64_t i = 0; i < rank; ++i) {
    if (i != feature_index) {
      dimensions_without_feature.push_back(i);
    }
  }

  auto elements_per_feature =
      add(DynamicElementCountPerFeature(operand, feature_index, add));

  auto feature_broadcast = [&](HloInstruction* inst) -> HloInstruction* {
    Shape feature_broadcast_shape = scalar_broadcast_shape;
    feature_broadcast_shape.set_dynamic_dimension(
        feature_index, inst->shape().is_dynamic_dimension(0));
    return add(HloInstruction::CreateBroadcast(feature_broadcast_shape, inst,
                                               {feature_index}));
  };

  auto scale_broadcasted = feature_broadcast(scale);

  auto offset_broadcasted = feature_broadcast(offset);

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
  auto mean = add(Mean(elements_per_feature, sum, add));

  auto mean_broadcasted = feature_broadcast(mean);

  // E[X^2].
  auto square_mean = add(Mean(elements_per_feature, squared_sum, add));

  // E^2[X].
  auto mean_square =
      add_binary(feature_shape, HloOpcode::kMultiply, mean, mean);

  // Var[X].
  auto var =
      add_binary(feature_shape, HloOpcode::kSubtract, square_mean, mean_square);

  auto var_broadcasted = feature_broadcast(var);

  // Var[X] + epsilon.
  auto var_add_epsilon = add_binary(var_broadcasted->shape(), HloOpcode::kAdd,
                                    var_broadcasted, epsilon);

  // 1 / Sqrt[Var[X] + epsilon].
  auto rsqrt_var_add_epsilon = add(Rsqrt(var_add_epsilon));

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
    int64_t instruction_count_after = computation_->instruction_count();
    CHECK_EQ(instruction_count_after,
             instruction_count_before + added_instructions.size());
    const HloSharding& sharding = batch_norm->sharding();
    HloSharding operand_sharding =
        sharding.GetAsShapeTree(batch_norm->shape()).element({0});
    optional<int64_t> unique_device = batch_norm->sharding_unique_device();
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
  return absl::OkStatus();
}

absl::Status BatchNormExpanderVisitor::HandleBatchNormInference(
    HloInstruction* batch_norm) {
  if (!rewrite_inference_op_) {
    return absl::OkStatus();
  }
  // Expand batch norm inference into smaller HLO ops.
  HloInstruction* operand = batch_norm->mutable_operand(0);
  const Shape operand_shape = operand->shape();
  int64_t feature_index = batch_norm->feature_index();
  PrimitiveType ptype = operand_shape.element_type();

  HloInstruction* scale = batch_norm->mutable_operand(1);
  HloInstruction* offset = batch_norm->mutable_operand(2);
  HloInstruction* mean = batch_norm->mutable_operand(3);
  HloInstruction* var = batch_norm->mutable_operand(4);
  const Shape feature_shape = scale->shape();
  Shape scalar_broadcast_shape = ShapeUtil::MakeStaticShape(feature_shape);

  auto epsilon_literal = LiteralUtil::CreateR0(batch_norm->epsilon());
  TF_ASSIGN_OR_RETURN(epsilon_literal, epsilon_literal.Convert(ptype));
  auto epsilon = computation_->AddInstruction(HloInstruction::CreateBroadcast(
      scalar_broadcast_shape,
      computation_->AddInstruction(
          HloInstruction::CreateConstant(std::move(epsilon_literal))),
      {}));

  std::vector<int64_t> dimensions_without_feature;
  const int64_t rank = operand_shape.rank();
  dimensions_without_feature.reserve(rank - 1);

  for (int64_t i = 0; i < rank; ++i) {
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
  auto feature_broadcast = [&](HloInstruction* a) {
    Shape broadcast_shape = ShapeUtil::MakeStaticShape(operand_shape);
    broadcast_shape.set_dynamic_dimension(feature_index,
                                          a->shape().is_dynamic_dimension(0));
    return add(
        HloInstruction::CreateBroadcast(broadcast_shape, a, {feature_index}));
  };

  int64_t instruction_count_before = computation_->instruction_count();
  auto true_scale = add_binary(
      feature_shape, HloOpcode::kMultiply, scale,
      add(Rsqrt(add_binary(feature_shape, HloOpcode::kAdd, var, epsilon))));
  auto true_shift = add_binary(
      feature_shape, HloOpcode::kSubtract, offset,
      add_binary(feature_shape, HloOpcode::kMultiply, mean, true_scale));

  auto shifted_normalized =
      add_binary(operand_shape, HloOpcode::kAdd,
                 add_binary(operand_shape, HloOpcode::kMultiply, operand,
                            feature_broadcast(true_scale)),
                 feature_broadcast(true_shift));

  int64_t instruction_count_after = computation_->instruction_count();
  CHECK_EQ(instruction_count_after,
           instruction_count_before + added_instructions.size());
  if (batch_norm->has_sharding()) {
    const HloSharding& sharding = batch_norm->sharding();
    optional<int64_t> unique_device = batch_norm->sharding_unique_device();
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
  TF_CHECK_OK(ReplaceInstruction(batch_norm, shifted_normalized));
  return absl::OkStatus();
}

absl::Status BatchNormExpanderVisitor::HandleBatchNormGrad(
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
    return absl::OkStatus();
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
  int64_t instruction_count_before = computation_->instruction_count();

  HloInstruction* activation = batch_norm->mutable_operand(0);
  const Shape activation_shape = activation->shape();
  PrimitiveType ptype = activation_shape.element_type();
  HloInstruction* scale = batch_norm->mutable_operand(1);
  const Shape feature_shape = scale->shape();
  HloInstruction* mean = batch_norm->mutable_operand(2);
  HloInstruction* variance = batch_norm->mutable_operand(3);
  HloInstruction* grad_output = batch_norm->mutable_operand(4);

  int64_t feature_index = batch_norm->feature_index();

  auto elements_per_feature =
      add(DynamicElementCountPerFeature(activation, feature_index, add));

  auto zero_literal = LiteralUtil::CreateR0(0.0f);
  TF_ASSIGN_OR_RETURN(zero_literal, zero_literal.Convert(ptype));
  auto zero = add(HloInstruction::CreateConstant(std::move(zero_literal)));

  auto epsilon_literal = LiteralUtil::CreateR0(batch_norm->epsilon());
  TF_ASSIGN_OR_RETURN(epsilon_literal, epsilon_literal.Convert(ptype));
  auto epsilon_scalar =
      add(HloInstruction::CreateConstant(std::move(epsilon_literal)));
  auto epsilon_activation = add(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeStaticShape(activation_shape), epsilon_scalar, {}));
  auto epsilon_feature = add(HloInstruction::CreateBroadcast(
      ShapeUtil::MakeStaticShape(feature_shape), epsilon_scalar, {}));

  std::vector<int64_t> dimensions_without_feature;
  const int64_t rank = activation_shape.rank();
  dimensions_without_feature.reserve(rank - 1);

  for (int64_t i = 0; i < rank; ++i) {
    if (i != feature_index) {
      dimensions_without_feature.push_back(i);
    }
  }

  auto activation_broadcast = [&](HloInstruction* hlo) -> HloInstruction* {
    Shape broadcast_shape = ShapeUtil::MakeStaticShape(activation_shape);
    broadcast_shape.set_dynamic_dimension(feature_index,
                                          hlo->shape().is_dynamic_dimension(0));
    return add(
        HloInstruction::CreateBroadcast(broadcast_shape, hlo, {feature_index}));
  };

  auto scale_broadcasted = activation_broadcast(scale);
  auto variance_broadcasted = activation_broadcast(variance);

  // E[X].
  auto mean_broadcasted = activation_broadcast(mean);

  // rsqrt[Var[X] + epsilon].
  auto rsqrt_var_add_epsilon_broadcasted =
      add(Rsqrt(add_binary(variance_broadcasted->shape(), HloOpcode::kAdd,
                           variance_broadcasted, epsilon_activation)));

  auto rsqrt_var_add_epsilon = add(Rsqrt(
      add_binary(feature_shape, HloOpcode::kAdd, variance, epsilon_feature)));

  // X - E[X].
  auto activation_minus_mean = add_binary(
      activation_shape, HloOpcode::kSubtract, activation, mean_broadcasted);

  // Grad[Y] * (X - E[X]).
  auto grad_output_times_activation_minus_mean =
      add_binary(activation_shape, HloOpcode::kMultiply, grad_output,
                 activation_minus_mean);

  HloComputation* add_reduce_computation =
      GetOrCreateScalarAddComputation(ptype);

  // sum(Grad[Y] * (X - E[X])).
  auto sum_grad_output_times_activation_minus_mean =
      add(HloInstruction::CreateReduce(
          feature_shape, grad_output_times_activation_minus_mean, zero,
          dimensions_without_feature, add_reduce_computation));

  // Grad[beta] = Sum(Grad[Y]).
  auto grad_beta = add(HloInstruction::CreateReduce(
      feature_shape, grad_output, zero, dimensions_without_feature,
      add_reduce_computation));

  // Grad[scale] = Sum(Grad[Y] * (X - E[X]) * rsqrt[Var[X] + epsilon]).
  auto grad_scale = add_binary(feature_shape, HloOpcode::kMultiply,
                               sum_grad_output_times_activation_minus_mean,
                               rsqrt_var_add_epsilon);

  // I2 = Sum(Grad[Y])
  auto i2 = activation_broadcast(grad_beta);

  // I3 = Sum(Grad[Y] * (X - E[X]))
  auto i3 = activation_broadcast(sum_grad_output_times_activation_minus_mean);

  // I4 = (X - E[X]) * I3
  auto i4 = add_binary(activation_shape, HloOpcode::kMultiply, i3,
                       activation_minus_mean);

  // I5 = I4 / (Var[X] + epsilon)
  auto i5 =
      add_binary(activation_shape, HloOpcode::kDivide, i4,
                 add_binary(variance_broadcasted->shape(), HloOpcode::kAdd,
                            variance_broadcasted, epsilon_activation));

  // scale * rsqrt[Var[X] + epsilon] * 1/N
  Shape scale_times_rsqrt_var_add_epsilon_shape = scale_broadcasted->shape();
  for (int64_t i = 0; i < rsqrt_var_add_epsilon_broadcasted->shape().rank();
       ++i) {
    if (rsqrt_var_add_epsilon_broadcasted->shape().is_dynamic_dimension(i)) {
      scale_times_rsqrt_var_add_epsilon_shape.set_dynamic_dimension(i, true);
    }
  }
  auto scale_times_rsqrt_var_add_epsilon =
      add_binary(scale_times_rsqrt_var_add_epsilon_shape, HloOpcode::kMultiply,
                 scale_broadcasted, rsqrt_var_add_epsilon_broadcasted);

  scale_times_rsqrt_var_add_epsilon =
      add(Mean(elements_per_feature, scale_times_rsqrt_var_add_epsilon, add));

  auto i1 = add_binary(grad_output->shape(), HloOpcode::kMultiply, grad_output,
                       add(HloInstruction::CreateBroadcast(
                           ShapeUtil::MakeStaticShape(activation_shape),
                           elements_per_feature, {})));

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
    int64_t instruction_count_after = computation_->instruction_count();
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

  return absl::OkStatus();
}

absl::StatusOr<bool> BatchNormExpander::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  XLA_VLOG_LINES(2, "BatchNormExpander::Run(), before:\n" + module->ToString());
  bool changed = false;
  for (HloComputation* computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (BatchNormExpanderVisitor::Run(computation, rewrite_training_op_,
                                      rewrite_inference_op_,
                                      rewrite_grad_op_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "BatchNormExpander::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
