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
