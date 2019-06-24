/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/gemm_lowering.h"

#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/core/lib/core/errors.h"

namespace xla {
namespace gpu {

static double GetScalarConstantAsDouble(const Literal& literal) {
  switch (literal.shape().element_type()) {
    case F16:
      return static_cast<double>(literal.Get<Eigen::half>({}));
    case F32:
      return literal.Get<float>({});
    case F64:
      return literal.Get<double>({});
    default:
      LOG(FATAL) << "Unsupported type.";
  }
}

static const HloInstruction* StripTranspose(const HloInstruction& hlo) {
  if (hlo.IsRank2Transpose()) {
    return hlo.operand(0);
  }
  return &hlo;
}

struct GemmFusionConfiguration {
  double alpha;
  double beta;
  int lhs_op_pos;
  int rhs_op_pos;
  int bias_pos;

  explicit GemmFusionConfiguration(double alpha = 1.0, double beta = 0.0,
                                   int lhs_op_pos = 0, int rhs_op_pos = 1,
                                   int bias_pos = -1)
      : alpha(alpha),
        beta(beta),
        lhs_op_pos(lhs_op_pos),
        rhs_op_pos(rhs_op_pos),
        bias_pos(bias_pos) {}
};

static GemmFusionConfiguration GetGemmFusionConfig(HloInstruction* inst) {
  if (inst->opcode() == HloOpcode::kDot) {
    return GemmFusionConfiguration();
  }

  CHECK(inst->opcode() == HloOpcode::kFusion);
  CHECK_EQ(inst->fusion_kind(), HloInstruction::FusionKind::kOutput);
  const HloInstruction* output_fused_op = inst->fused_expression_root();
  double alpha_value = 1.0;
  const HloInstruction* bias = nullptr;
  int bias_pos = -1;
  const HloInstruction* dot = output_fused_op->operand(0);

  if (output_fused_op->opcode() == HloOpcode::kMultiply) {
    const HloInstruction* alpha = output_fused_op->operand(1);
    if (dot->opcode() != HloOpcode::kDot) {
      std::swap(dot, alpha);
    }
    if (alpha->opcode() == HloOpcode::kBroadcast) {
      alpha = alpha->operand(0);
    }
    if (alpha->opcode() == HloOpcode::kParameter) {
      alpha = inst->operand(alpha->parameter_number());
    }
    // TODO(b/74185543): Remove the following if block once we support fusion
    // with a non-constant as well. Then we will just always use the constant
    // on the device.
    if (alpha->opcode() == HloOpcode::kCopy) {
      alpha = alpha->operand(0);
    }
    alpha_value = GetScalarConstantAsDouble(alpha->literal());
  } else {
    // Fused bias add.
    CHECK_EQ(output_fused_op->opcode(), HloOpcode::kAdd);
    bias = output_fused_op->operand(1);
    if (dot->opcode() != HloOpcode::kDot) {
      std::swap(dot, bias);
    }
    bias_pos = bias->parameter_number();
    bias = inst->operand(bias_pos);
  }

  DCHECK(dot->opcode() == HloOpcode::kDot);
  const HloInstruction* lhs_parameter = StripTranspose(*dot->operand(0));
  const HloInstruction* rhs_parameter = StripTranspose(*dot->operand(1));
  DCHECK(lhs_parameter->opcode() == HloOpcode::kParameter &&
         rhs_parameter->opcode() == HloOpcode::kParameter);

  int lhs_pos = lhs_parameter->parameter_number();
  int rhs_pos = rhs_parameter->parameter_number();

  if (bias != nullptr) {
    return GemmFusionConfiguration(alpha_value, /*beta=*/1.0, lhs_pos, rhs_pos,
                                   bias_pos);
  } else {
    return GemmFusionConfiguration(alpha_value, /*beta=*/0.0, lhs_pos, rhs_pos);
  }

  LOG(FATAL) << "Unexpected GEMM fusion configuration";
}

static DotDimensionNumbers GetDimensionNumbers(
    const HloInstruction& hlo_instruction) {
  if (hlo_instruction.opcode() == HloOpcode::kDot) {
    return hlo_instruction.dot_dimension_numbers();
  }
  CHECK_EQ(hlo_instruction.opcode(), HloOpcode::kFusion);
  CHECK_EQ(hlo_instruction.fusion_kind(), HloInstruction::FusionKind::kOutput);
  CHECK(hlo_instruction.fused_expression_root()->opcode() == HloOpcode::kAdd ||
        hlo_instruction.fused_expression_root()->opcode() ==
            HloOpcode::kMultiply);
  // Try to find the dot inside the output fusion node.
  const HloInstruction* dot =
      hlo_instruction.fused_expression_root()->operand(0);
  if (dot->opcode() != HloOpcode::kDot) {
    dot = hlo_instruction.fused_expression_root()->operand(1);
  }
  CHECK_EQ(dot->opcode(), HloOpcode::kDot);

  return dot->dot_dimension_numbers();
}

static StatusOr<bool> RunOnInstruction(HloInstruction* inst) {
  TF_ASSIGN_OR_RETURN(GemmBackendConfig backend_config,
                      inst->backend_config<GemmBackendConfig>());

  int64 batch_size = std::accumulate(inst->shape().dimensions().begin(),
                                     inst->shape().dimensions().end() - 2, 1,
                                     std::multiplies<int64>());

  GemmFusionConfiguration gemm_config = GetGemmFusionConfig(inst);

  GemmBackendConfig new_config = backend_config;

  new_config.set_alpha(gemm_config.alpha);
  new_config.set_beta(gemm_config.beta);
  new_config.set_lhs_parameter_number(gemm_config.lhs_op_pos);
  new_config.set_rhs_parameter_number(gemm_config.rhs_op_pos);
  new_config.set_bias_parameter_number(gemm_config.bias_pos);
  *new_config.mutable_dot_dimension_numbers() = GetDimensionNumbers(*inst);
  new_config.set_batch_size(batch_size);

  TF_RETURN_IF_ERROR(inst->set_backend_config(new_config));
  return backend_config.SerializeAsString() != new_config.SerializeAsString();
}

static StatusOr<bool> RunOnComputation(HloComputation* computation) {
  bool changed = false;
  for (HloInstruction* instr : computation->instructions()) {
    if (ImplementedAsGemm(*instr)) {
      TF_ASSIGN_OR_RETURN(bool result, RunOnInstruction(instr));
      changed |= result;
    }
  }
  return changed;
}

StatusOr<bool> GemmLowering::Run(HloModule* module) {
  bool changed = false;
  for (HloComputation* computation : module->computations()) {
    TF_ASSIGN_OR_RETURN(bool result, RunOnComputation(computation));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
