/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/bitcast_remover.h"

#include <variant>

#include "tensorflow/compiler/xla/hlo/ir/dfs_hlo_visitor_with_default.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {
namespace {

class BitcastRemoverVisitor : public DfsHloRewriteVisitor {
 public:
  BitcastRemoverVisitor() = default;
  ~BitcastRemoverVisitor() override = default;

  Status HandleBitcast(HloInstruction* bitcast) override;
};

Status BitcastRemoverVisitor::HandleBitcast(HloInstruction* bitcast) {
  VLOG(3) << "Found a bitcast " << bitcast->ToString();
  TF_RET_CHECK(bitcast->shape().has_layout());

  HloInstruction* const operand = bitcast->mutable_operand(0);
  TF_RET_CHECK(operand->shape().has_layout());

  const ShapeUtil::BitcastDecomposition& decomposition =
      ShapeUtil::DecomposeBitcast(operand->shape(), bitcast->shape());

  if (std::holds_alternative<ShapeUtil::BitcastDecompositionReshape>(
          decomposition)) {
    HloInstruction* reshape = bitcast->AddInstruction(
        HloInstruction::CreateReshape(bitcast->shape(), operand));
    TF_RETURN_IF_ERROR(ReplaceInstruction(bitcast, reshape));
    VLOG(4) << "One reshape is enough: " << reshape->ToString();

    return OkStatus();
  }

  if (std::holds_alternative<ShapeUtil::BitcastDecompositionTranspose>(
          decomposition)) {
    const auto& decomposition_transpose =
        std::get<ShapeUtil::BitcastDecompositionTranspose>(decomposition);

    HloInstruction* transpose =
        bitcast->AddInstruction(HloInstruction::CreateTranspose(
            bitcast->shape(), operand, decomposition_transpose.transpose_dims));
    TF_RETURN_IF_ERROR(ReplaceInstruction(bitcast, transpose));
    VLOG(4) << "One transpose is enough: " << transpose->ToString();

    return OkStatus();
  }

  TF_RET_CHECK(std::holds_alternative<ShapeUtil::BitcastDecompositionTrt>(
      decomposition));
  const auto& decomposition_trt =
      std::get<ShapeUtil::BitcastDecompositionTrt>(decomposition);

  HloInstruction* replacement = operand;
  if (!decomposition_trt.IsTranspose1Identity()) {
    replacement = bitcast->AddInstruction(HloInstruction::CreateTranspose(
        decomposition_trt.transpose1_shape, replacement,
        decomposition_trt.transpose1_dims));
    VLOG(4) << "Transpose (1) needed: " << replacement->ToString();
  }

  replacement = bitcast->AddInstruction(HloInstruction::CreateReshape(
      decomposition_trt.reshape_shape, replacement));
  VLOG(4) << "Reshape needed: " << replacement->ToString();

  if (!decomposition_trt.IsTranspose2Identity()) {
    replacement = bitcast->AddInstruction(HloInstruction::CreateTranspose(
        bitcast->shape(), replacement, decomposition_trt.transpose2_dims));
    VLOG(4) << "Transpose (2) needed: " << replacement->ToString();
  }

  TF_RETURN_IF_ERROR(ReplaceInstruction(bitcast, replacement));

  return OkStatus();
}

}  // namespace

StatusOr<bool> BitcastRemover::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  return BitcastRemoverVisitor().RunOnModule(module, execution_threads);
}

}  // namespace xla
