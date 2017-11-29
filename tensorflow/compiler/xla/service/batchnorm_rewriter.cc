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

#include "tensorflow/compiler/xla/service/batchnorm_rewriter.h"

#include <algorithm>
#include <memory>
#include <numeric>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/service/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/service/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_opcode.h"
#include "tensorflow/compiler/xla/service/hlo_query.h"
#include "tensorflow/compiler/xla/service/shape_inference.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/window_util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// BatchNormRewriterVisitor traverses the HLO computation and rewrites BatchNorm
// operations into smaller operations.
class BatchNormRewriterVisitor : public DfsHloVisitorWithDefault {
 public:
  // Default visitor action is to do nothing and return OK.
  Status DefaultAction(HloInstruction* /*hlo_instruction*/) override {
    return Status::OK();
  }

  Status HandleBatchNormTraining(HloInstruction* batch_norm) override;

  // Runs the visitor on a computation.
  static bool Run(HloComputation* computation, bool rewrite_training_op,
                  bool rewrite_grad_op);

  // Returns whether any batch norm ops were rewritten.
  const bool changed() const { return changed_; }

  ~BatchNormRewriterVisitor() override = default;

 private:
  explicit BatchNormRewriterVisitor(HloComputation* computation,
                                    bool rewrite_training_op,
                                    bool rewrite_grad_op)
      : computation_(computation),
        rewrite_training_op_(rewrite_training_op),
        rewrite_grad_op_(rewrite_grad_op) {}

  HloComputation* GetScalarBinaryComputation(PrimitiveType primitive_type,
                                             HloOpcode opcode) {
    HloComputation::Builder b("scalar computation");
    auto scalar_lhs = b.AddInstruction(HloInstruction::CreateParameter(
        0, ShapeUtil::MakeShape(F32, {}), "scalar lhs"));
    auto scalar_rhs = b.AddInstruction(HloInstruction::CreateParameter(
        1, ShapeUtil::MakeShape(F32, {}), "scalar rhs"));
    auto scalar_op = b.AddInstruction(
        HloInstruction::CreateBinary(ShapeUtil::MakeShape(primitive_type, {}),
                                     opcode, scalar_lhs, scalar_rhs));
    return computation_->parent()->AddEmbeddedComputation(b.Build(scalar_op));
  }

  // Current HloComputation instance the BatchNormRewriter is
  // traversing.
  HloComputation* computation_;

  bool rewrite_training_op_;
  bool rewrite_grad_op_;

  // Whether rewrite has occurred.
  bool changed_ = false;

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
};

bool BatchNormRewriterVisitor::Run(HloComputation* computation,
                                   bool rewrite_training_op,
                                   bool rewrite_grad_op) {
  BatchNormRewriterVisitor visitor(computation,
                                   /*rewrite_training_op=*/rewrite_training_op,
                                   /*rewrite_grad_op=*/rewrite_grad_op);
  TF_CHECK_OK(computation->Accept(&visitor));
  return visitor.changed_;
}

Status BatchNormRewriterVisitor::HandleBatchNormTraining(
    HloInstruction* batch_norm) {
  if (!rewrite_training_op_) {
    return Status::OK();
  }
  // Expand batch norm training into smaller HLO ops.
  HloInstruction* operand = batch_norm->mutable_operand(0);
  const Shape operand_shape = operand->shape();
  int64 feature_index = batch_norm->feature_index();
  const int64 feature_count = operand_shape.dimensions(feature_index);
  const int64 size_in_elements = ShapeUtil::ElementsIn(operand_shape);
  auto elements_per_feature =
      computation_->AddInstruction(HloInstruction::CreateConstant(
          Literal::CreateR0<float>(size_in_elements / feature_count)));

  HloInstruction* scale = batch_norm->mutable_operand(1);
  HloInstruction* offset = batch_norm->mutable_operand(2);
  const Shape feature_shape = scale->shape();

  auto zero = computation_->AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0(0.0f)));

  auto epsilon = computation_->AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0(batch_norm->epsilon())));

  std::vector<int64> dimensions_without_feature;

  for (int64 i = 0; i < ShapeUtil::Rank(operand_shape); ++i) {
    if (i != feature_index) {
      dimensions_without_feature.push_back(i);
    }
  }

  auto scale_broadcasted = computation_->AddInstruction(
      HloInstruction::CreateBroadcast(operand_shape, scale, {feature_index}));

  auto offset_broadcasted = computation_->AddInstruction(
      HloInstruction::CreateBroadcast(operand_shape, offset, {feature_index}));

  HloComputation* add_reduce_computation =
      GetScalarBinaryComputation(F32, HloOpcode::kAdd);

  // X^2.
  auto operand_squared =
      computation_->AddInstruction(HloInstruction::CreateBinary(
          operand_shape, HloOpcode::kMultiply, operand, operand));
  // Sum[X].
  auto sum = computation_->AddInstruction(HloInstruction::CreateReduce(
      feature_shape, operand, zero, dimensions_without_feature,
      add_reduce_computation));

  // Sum[X^2].
  auto squared_sum = computation_->AddInstruction(HloInstruction::CreateReduce(
      feature_shape, operand_squared, zero, dimensions_without_feature,
      add_reduce_computation));

  // Fuse two parallel reduces together to improve performance.
  auto tuple = computation_->AddInstruction(
      HloInstruction::CreateTuple({sum, squared_sum}));

  auto fused = computation_->CreateFusionInstruction(
      {tuple, sum, squared_sum, operand_squared},
      HloInstruction::FusionKind::kInput);

  sum = computation_->AddInstruction(
      HloInstruction::CreateGetTupleElement(feature_shape, fused, 0));

  squared_sum = computation_->AddInstruction(
      HloInstruction::CreateGetTupleElement(feature_shape, fused, 1));

  // E[X].
  auto mean = computation_->AddInstruction(HloInstruction::CreateBinary(
      feature_shape, HloOpcode::kDivide, sum, elements_per_feature));

  auto mean_broadcasted = computation_->AddInstruction(
      HloInstruction::CreateBroadcast(operand_shape, mean, {feature_index}));

  // E[X^2].
  auto square_mean = computation_->AddInstruction(HloInstruction::CreateBinary(
      feature_shape, HloOpcode::kDivide, squared_sum, elements_per_feature));

  // E^2[X].
  auto mean_square = computation_->AddInstruction(HloInstruction::CreateBinary(
      feature_shape, HloOpcode::kMultiply, mean, mean));

  // Var[X].
  auto var = computation_->AddInstruction(HloInstruction::CreateBinary(
      feature_shape, HloOpcode::kSubtract, square_mean, mean_square));

  auto var_broadcasted = computation_->AddInstruction(
      HloInstruction::CreateBroadcast(operand_shape, var, {feature_index}));

  // Var[X] + epsilon.
  auto var_add_epsilon =
      computation_->AddInstruction(HloInstruction::CreateBinary(
          operand_shape, HloOpcode::kAdd, var_broadcasted, epsilon));

  auto neg_half = computation_->AddInstruction(
      HloInstruction::CreateConstant(Literal::CreateR0(-0.5f)));

  // 1 / Sqrt[Var[X] + epsilon].
  auto rsqrt_var_add_epsilon =
      computation_->AddInstruction(HloInstruction::CreateBinary(
          operand_shape, HloOpcode::kPower, var_add_epsilon, neg_half));

  // X - E[X].
  auto operand_minus_mean =
      computation_->AddInstruction(HloInstruction::CreateBinary(
          operand_shape, HloOpcode::kSubtract, operand, mean_broadcasted));

  // (X - E[X]) / Sqrt[Var[X] + epsilon].
  auto normalized = computation_->AddInstruction(
      HloInstruction::CreateBinary(operand_shape, HloOpcode::kMultiply,
                                   operand_minus_mean, rsqrt_var_add_epsilon));

  // (X - E[X]) / Sqrt[Var[X] + epsilon] * scale.
  auto scaled_normalized =
      computation_->AddInstruction(HloInstruction::CreateBinary(
          operand_shape, HloOpcode::kMultiply, normalized, scale_broadcasted));

  // (X - E[X]) / Sqrt[Var[X] + epsilon] * scale + offset.
  auto shifted_normalized = computation_->AddInstruction(
      HloInstruction::CreateBinary(operand_shape, HloOpcode::kAdd,
                                   scaled_normalized, offset_broadcasted));

  TF_CHECK_OK(ReplaceWithNewInstruction(
      batch_norm,
      HloInstruction::CreateTuple({shifted_normalized, mean, var})));
  return Status::OK();
}

StatusOr<bool> BatchNormRewriter::Run(HloModule* module) {
  XLA_VLOG_LINES(2, "BatchNormRewriter::Run(), before:\n" + module->ToString());
  bool changed = false;
  // Make a copy of the computations because we may add computations to the
  // module, invalidating iteration.
  std::vector<HloComputation*> computations;
  for (auto& comp : module->computations()) {
    computations.push_back(comp.get());
  }
  for (auto& comp : computations) {
    if (BatchNormRewriterVisitor::Run(comp, rewrite_training_op_,
                                      rewrite_grad_op_)) {
      changed = true;
    }
  }
  XLA_VLOG_LINES(2, "BatchNormRewriter::Run(), after:\n" + module->ToString());
  return changed;
}

}  // namespace xla
