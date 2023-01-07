/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/gpu/cudnn_fused_mha_rewriter.h"

#include <functional>
#include <string>

#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/cublas_cudnn.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_casting_utils.h"
#include "tensorflow/compiler/xla/service/hlo_creation_utils.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/stream_executor/dnn.pb.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/stream_executor_no_cuda.h"

namespace xla {
namespace gpu {
namespace {
namespace m = match;

bool IsFMHACustomCall(const HloInstruction* instr) {
  return instr->opcode() == HloOpcode::kCustomCall &&
         (instr->custom_call_target() == kCudnnfMHADefaultCallTarget ||
          instr->custom_call_target() ==
              kCudnnfMHAScaleBiasMaskSoftmaxCallTarget ||
          instr->custom_call_target() ==
              kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget ||
          instr->custom_call_target() == kCudnnfMHAScaleMaskSoftmaxCallTarget ||
          instr->custom_call_target() ==
              kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget ||
          instr->custom_call_target() == kCudnnfMHASoftmaxDropoutCallTarget);
}

StatusOr<bool> IsBatchedDot(const HloInstruction* gemm) {
  GemmBackendConfig config;
  TF_ASSIGN_OR_RETURN(config, gemm->backend_config<GemmBackendConfig>());
  const DotDimensionNumbers& dot_dims = config.dot_dimension_numbers();
  bool is_batch_dot = !dot_dims.lhs_batch_dimensions().empty() ||
                      !dot_dims.rhs_batch_dimensions().empty();
  return is_batch_dot;
}

bool IsBatchedMatmul(const HloInstruction* gemm) {
  return IsCublasGemm(*gemm) && IsBatchedDot(gemm).value();
}

// Give this instruction a more useful name than "custom-call.42".
Status SetName(HloModule* module, HloInstruction* fmha) {
  if (fmha->custom_call_target() == kCudnnfMHADefaultCallTarget) {
    module->SetAndUniquifyInstrName(fmha, "fmha-bmm-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() == kCudnnfMHASoftmaxDropoutCallTarget) {
    module->SetAndUniquifyInstrName(fmha, "fmha-bmm-softmax-dropout-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() == kCudnnfMHAScaleMaskSoftmaxCallTarget) {
    module->SetAndUniquifyInstrName(fmha, "fmha-bmm-scale-mask-softmax-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() ==
      kCudnnfMHAScaleMaskSoftmaxDropoutCallTarget) {
    module->SetAndUniquifyInstrName(fmha,
                                    "fmha-bmm-scale-mask-softmax-dropout-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() == kCudnnfMHAScaleBiasMaskSoftmaxCallTarget) {
    module->SetAndUniquifyInstrName(fmha,
                                    "fmha-bmm-scale-bias-mask-softmax-bmm");
    return OkStatus();
  }
  if (fmha->custom_call_target() ==
      kCudnnfMHAScaleBiasMaskSoftmaxDropoutCallTarget) {
    module->SetAndUniquifyInstrName(
        fmha, "fmha-bmm-scale-bias-mask-softmax-dropout-bmm");
    return OkStatus();
  }

  return InternalError(
      "Found invalid FMHA custom-call target while setting custom-call name");
}

StatusOr<bool> FuseBatchedMatmuls(HloComputation* comp) {
  bool changed = false;
  for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
    HloInstruction* bmm_1;
    HloInstruction* bmm_2;
    auto pattern =
        m::Op(&bmm_2)
            .WithPredicate(IsBatchedMatmul)
            .WithOperand(
                0, m::Op(&bmm_1).WithPredicate(IsBatchedMatmul).WithOneUse());
    if (!Match(instr, pattern)) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(auto config_bmm1,
                        bmm_1->backend_config<GemmBackendConfig>());
    TF_ASSIGN_OR_RETURN(auto config_bmm2,
                        bmm_2->backend_config<GemmBackendConfig>());

    CudnnfMHABackendConfig fmha_config;
    *fmha_config.mutable_bmm1_dot_dimension_numbers() =
        config_bmm1.dot_dimension_numbers();
    *fmha_config.mutable_bmm2_dot_dimension_numbers() =
        config_bmm2.dot_dimension_numbers();
    fmha_config.set_fmha_scale(1.0);
    fmha_config.set_dropout_rate(0.0);
    *fmha_config.mutable_intermediate_tensor_shape() = bmm_1->shape().ToProto();
    {
      auto* algorithm = fmha_config.mutable_algorithm();
      algorithm->set_algo_id(0);  // engine id
      algorithm->set_math_type(se::dnn::AlgorithmProto::TENSOR_OP_MATH);
      std::vector<int64_t> knob_ids = /* {0, 1} */{17, 24};
      std::vector<int64_t> knob_vals = {1, 0};
      for (int i = 0; i < knob_ids.size(); ++i) {
        (*algorithm->mutable_tuning_knobs())[knob_ids[i]] = knob_vals[i];
      }
      algorithm->set_is_cudnn_frontend(true);
      algorithm->mutable_workspace_size()->set_value(0);
    }
    HloInstruction* lhs_bmm1 = bmm_1->mutable_operand(0);
    HloInstruction* rhs_bmm1 = bmm_1->mutable_operand(1);
    HloInstruction* rhs_bmm2 = bmm_2->mutable_operand(1);
    const Shape& output_shape = bmm_2->shape();
    Shape call_shape = ShapeUtil::MakeTupleShape(
        {output_shape, ShapeUtil::MakeShape(U8, {0})});
    HloInstruction* fmha_call =
        comp->AddInstruction(HloInstruction::CreateCustomCall(
            call_shape, {lhs_bmm1, rhs_bmm1, rhs_bmm2},
            absl::string_view(kCudnnfMHADefaultCallTarget)));
    TF_RETURN_IF_ERROR(fmha_call->set_backend_config(fmha_config));
    TF_RETURN_IF_ERROR(SetName(bmm_1->GetModule(), fmha_call));
    TF_RETURN_IF_ERROR(comp->ReplaceWithNewInstruction(
        instr,
        HloInstruction::CreateGetTupleElement(instr->shape(), fmha_call, 0)));
    if (VLOG_IS_ON(2)) {
      VLOG(2) << "After CudnnFusedMHARewriter: \n"
              << comp->parent()->ToString();
    }
    changed = true;
  }
  return changed;
}
}  // namespace

StatusOr<bool> CudnnFusedMHARewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool any_changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    bool changed = false;
    TF_ASSIGN_OR_RETURN(changed, FuseBatchedMatmuls(comp));
    any_changed |= changed;
  }

  return any_changed;
}
}  // namespace gpu
}  // namespace xla
