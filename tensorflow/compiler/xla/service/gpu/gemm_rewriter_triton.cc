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

#include "tensorflow/compiler/xla/service/gpu/gemm_rewriter_triton.h"

#include <stack>
#include <vector>

#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"

namespace xla {
namespace gpu {
namespace {

// Data types that are tested to work in the Triton-based matmul emitter.
bool IsTritonSupportedInputType(
    PrimitiveType t, se::CudaComputeCapability cuda_compute_capability) {
  switch (t) {
    case PRED:
    case S8:
    case S32:
    case F16:
    case F32:
      return true;
    case BF16:
      return cuda_compute_capability.IsAtLeast(
          stream_executor::CudaComputeCapability::AMPERE);
    default:
      return false;
  }
}

// Filters data type conversions which should be fused into Triton GEMM.
bool IsTritonFusibleConvert(const HloInstruction *input,
                            se::CudaComputeCapability cuda_compute_capability) {
  // TODO(b/266862494): Can pick up almost any
  // convert, but if it's reducing the data volume it should rather be fused
  // to the output of the producer kernel. However not all operations support
  // output fusion - then it should be fused here anyway!
  return IsTritonSupportedInputType(input->operand(0)->shape().element_type(),
                                    cuda_compute_capability) &&
         ShapeUtil::ByteSizeOf(input->operand(0)->shape()) <=
             ShapeUtil::ByteSizeOf(input->shape());
}

// Extracts parts of HLO graph around dot() operations that can target
// the triton matmul emitter into fused computations.
class GemmRewriterTritonVisitor : public DfsHloRewriteVisitor {
 public:
  explicit GemmRewriterTritonVisitor(const se::CudaComputeCapability cc)
      : cuda_compute_capability_(cc) {}
  // Checks that a dot() should be targeting the triton-based matmul emitter;
  // if so - fuses all its compatible inputs and outputs as a new computation
  // and replaces the original dot() with a call to the computation.
  Status HandleDot(HloInstruction *dot) override {
    HloModule *mod = dot->parent()->parent();
    VLOG(10) << mod->ToString();
    if (!IsTritonHandledGEMM(*dot, cuda_compute_capability_)) {
      return OkStatus();
    }

    // TODO(b/266857789): also fuse convert(dot()) at output if present:
    // seen on s8xf32->bf16
    HloComputation::Builder builder(dot->name());
    // Original instruction -> fused one.
    absl::flat_hash_map<const HloInstruction *, HloInstruction *>
        old_to_new_mapping;
    std::vector<HloInstruction *> call_params;
    // Traverse and fuse dot() inputs bottom-up starting from direct operands.
    // If an input is not fusible stop there and make it a parameter of the new
    // fusion, otherwise put it onto stack and check its own inputs first.
    std::stack<const HloInstruction *> to_fuse;
    to_fuse.push(dot);
    while (!to_fuse.empty()) {
      bool top_is_ready_to_fuse = true;
      for (const HloInstruction *operand : to_fuse.top()->operands()) {
        if (!old_to_new_mapping.contains(operand) &&
            (operand->opcode() == HloOpcode::kBitcast ||
             (operand->opcode() == HloOpcode::kConvert &&
              IsTritonFusibleConvert(operand, cuda_compute_capability_)))) {
          to_fuse.push(operand);
          top_is_ready_to_fuse = false;
        }
      }
      if (top_is_ready_to_fuse) {
        std::vector<HloInstruction *> operands;
        for (HloInstruction *operand : to_fuse.top()->operands()) {
          const auto it = old_to_new_mapping.find(operand);
          if (it != old_to_new_mapping.end()) {
            operands.push_back(it->second);
          } else {
            operands.push_back(
                builder.AddInstruction(HloInstruction::CreateParameter(
                    call_params.size(), operand->shape(),
                    absl::StrCat("parameter_", call_params.size()))));
            call_params.push_back(operand);
          }
        }
        old_to_new_mapping[to_fuse.top()] =
            builder.AddInstruction(to_fuse.top()->CloneWithNewOperands(
                to_fuse.top()->shape(), operands));
        to_fuse.pop();
      }
    }
    HloComputation *custom_call_computation =
        dot->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                            /*is_entry=*/false);
    HloInstruction *dot_custom_call =
        dot->parent()->AddInstruction(HloInstruction::CreateCustomCall(
            dot->shape(), call_params, custom_call_computation,
            kTritonCallTarget));
    if (dot->IsRoot()) {
      dot->parent()->set_root_instruction(dot_custom_call);
      TF_RETURN_IF_ERROR(
          dot->parent()->RemoveInstructionAndUnusedOperands(dot));
    } else {
      TF_RETURN_IF_ERROR(
          dot->parent()->ReplaceInstruction(dot, dot_custom_call));
    }
    VLOG(10) << custom_call_computation->ToString();
    return OkStatus();
  }

 private:
  se::CudaComputeCapability cuda_compute_capability_;
};

StatusOr<bool> RunOnComputation(
    HloComputation *computation,
    se::CudaComputeCapability cuda_compute_capability) {
  GemmRewriterTritonVisitor visitor(cuda_compute_capability);
  TF_RETURN_IF_ERROR(computation->Accept(&visitor));
  return visitor.changed();
}

}  // anonymous namespace

bool IsTritonHandledGEMM(const HloInstruction &dot,
                         se::CudaComputeCapability cuda_compute_capability) {
  if (dot.opcode() != HloOpcode::kDot) {
    return false;
  }

  // TODO(b/266860366): Support batch dimensions.
  if (dot.dot_dimension_numbers().lhs_batch_dimensions_size()) {
    return false;
  }

  auto supported_output_type = [&](const PrimitiveType t) {
    switch (t) {
      case F16:
      case F32:
        return true;
      case BF16:
        return cuda_compute_capability.IsAtLeast(
            stream_executor::CudaComputeCapability::AMPERE);
      default:
        return false;
    }
  };

  // TODO(b/266862493): Support more output types.
  if (!supported_output_type(dot.shape().element_type())) {
    return false;
  }

  // Traverse HLO graph looking for its part that both can be fused
  // and is worth fusing.
  auto is_triton_fusible_input = [&](const HloInstruction *input) {
    while (true) {
      if (!IsTritonSupportedInputType(input->shape().element_type(),
                                      cuda_compute_capability)) {
        return false;
      }
      if (input->GetModule()
              ->config()
              .debug_options()
              .xla_gpu_triton_gemm_any()) {
        return true;
      }
      switch (input->opcode()) {
        case HloOpcode::kBitcast:
          input = input->operand(0);
          continue;
        case HloOpcode::kConvert:
          return IsTritonFusibleConvert(input, cuda_compute_capability);
        default:
          return false;
      }
    }
  };

  return is_triton_fusible_input(dot.operand(0)) ||
         is_triton_fusible_input(dot.operand(1));

  // TODO(b/266857789): either check that no output fusion (axpy, relu etc)
  // is expected or actually support it.
}

StatusOr<bool> GemmRewriterTriton::Run(
    HloModule *module,
    const absl::flat_hash_set<absl::string_view> &execution_threads) {
  bool changed = false;
  for (HloComputation *computation :
       module->MakeNonfusionComputations(execution_threads)) {
    if (computation->IsCustomCallComputation() &&
        IsTritonCustomCall(*computation->CustomCallInstruction())) {
      continue;
    }
    TF_ASSIGN_OR_RETURN(
        bool result, RunOnComputation(computation, cuda_compute_capability_));
    changed |= result;
  }
  return changed;
}

}  // namespace gpu
}  // namespace xla
