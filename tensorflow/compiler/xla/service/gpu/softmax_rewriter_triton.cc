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

#include "tensorflow/compiler/xla/service/gpu/softmax_rewriter_triton.h"

#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_module.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_opcode.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/service/gpu/backend_configs.pb.h"
#include "tensorflow/compiler/xla/service/gpu/ir_emission_utils.h"
#include "tensorflow/compiler/xla/service/pattern_matcher.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/tsl/platform/errors.h"

namespace xla::gpu {
namespace {
namespace m = ::xla::match;
bool HasDefaultLayout(const Shape& shape) {
  return shape.has_layout() &&
         LayoutUtil::IsMonotonicWithDim0Major(shape.layout());
}

bool IsSupportedReductionComputation(HloComputation* computation) {
  static const absl::flat_hash_set<HloOpcode>* const kSupportedOpcodes =
      new absl::flat_hash_set<HloOpcode>{HloOpcode::kAdd, HloOpcode::kMaximum};

  HloInstruction* root = computation->root_instruction();
  if (root->operand_count() != 2 ||
      root->operand(0)->opcode() != HloOpcode::kParameter ||
      root->operand(1)->opcode() != HloOpcode::kParameter) {
    return false;
  }
  return kSupportedOpcodes->contains(root->opcode());
}

std::optional<HloInstruction*> MatchesExactSoftmaxPattern(
    HloInstruction* instr) {
  // Return the producer found by the following pattern if it matches:
  //
  // producer
  // |   \
  // |  reduce_max
  // |     |
  // |  broadcast
  // |   /
  // subtract
  // |
  // exponential
  // |   \
  // |  reduce_sum
  // |     |
  // |  broadcast
  // |   /
  // divide  // (instr parameter)
  //
  // where both reductions occur only on the last axis. It is also assumed that
  // the physical layouts are normalized.
  HloInstruction* left_exponential;
  HloInstruction* right_exponential;
  HloInstruction* left_producer;
  HloInstruction* right_producer;

  if (!HasDefaultLayout(instr->shape())) return std::nullopt;

  // Lower diamond
  if (!Match(instr,
             m::Divide(
                 m::Exp(&left_exponential, m::Op()),
                 m::Broadcast(
                     m::Reduce(m::Exp(&right_exponential, m::Op()), m::Op())
                         .WithPredicate([](const HloInstruction* reduce) {
                           HloComputation* reducer = reduce->to_apply();
                           return (IsSupportedReductionComputation(reducer) &&
                                   reducer->root_instruction()->opcode() ==
                                       HloOpcode::kAdd &&
                                   reduce->dimensions().size() == 1 &&
                                   reduce->dimensions()[0] !=
                                       reduce->shape().rank() - 1);
                         })
                         .WithOneUse())
                     .WithOneUse()))) {
    return std::nullopt;
  }

  if (left_exponential != right_exponential ||
      left_exponential->user_count() != 2)
    return std::nullopt;

  // Upper diamond
  if (!Match(left_exponential->mutable_operand(0),
             m::Subtract(
                 m::Op(&left_producer),
                 m::Broadcast(
                     m::Reduce(m::Op(&right_producer), m::Op())
                         .WithPredicate([](const HloInstruction* reduce) {
                           HloComputation* reducer = reduce->to_apply();
                           return (IsSupportedReductionComputation(reducer) &&
                                   reducer->root_instruction()->opcode() ==
                                       HloOpcode::kMaximum &&
                                   reduce->dimensions().size() == 1 &&
                                   reduce->dimensions()[0] !=
                                       reduce->shape().rank() - 1);
                         })
                         .WithOneUse())
                     .WithOneUse())
                 .WithOneUse())) {
    return std::nullopt;
  }

  if (left_producer != right_producer || left_producer->user_count() != 2)
    return std::nullopt;

  return left_producer;
}

Status FuseSoftmax(HloInstruction* root, HloInstruction* producer) {
  std::string suggested_name = "triton_softmax";
  HloComputation::Builder builder(absl::StrCat(suggested_name, "_computation"));
  // Original instruction -> fused one.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*>
      old_to_new_mapping;

  old_to_new_mapping[producer] = builder.AddInstruction(
      HloInstruction::CreateParameter(0, producer->shape(), "parameter_0"));

  std::function<void(const HloInstruction*)> create_computation =
      [&](const HloInstruction* instr) -> void {
    if (old_to_new_mapping.contains(instr)) {
      return;
    }
    std::vector<HloInstruction*> new_operands;
    for (const HloInstruction* operand : instr->operands()) {
      create_computation(operand);
      new_operands.push_back(old_to_new_mapping[operand]);
    }
    old_to_new_mapping[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), new_operands));
  };
  create_computation(root);

  HloComputation* computation =
      root->GetModule()->AddComputationAndUnifyNamesAndIds(builder.Build(),
                                                           /*is_entry=*/false);

  HloInstruction* softmax_fusion =
      root->parent()->AddInstruction(HloInstruction::CreateFusion(
          root->shape(), HloInstruction::FusionKind::kCustom,
          std::vector<HloInstruction*>({producer}), computation));

  softmax_fusion->GetModule()->SetAndUniquifyInstrName(softmax_fusion,
                                                       suggested_name);

  TF_ASSIGN_OR_RETURN(auto backend_config,
                      softmax_fusion->backend_config<FusionBackendConfig>());
  backend_config.set_kind(std::string(kTritonSoftmaxFusionKind));
  TF_RETURN_IF_ERROR(softmax_fusion->set_backend_config(backend_config));

  if (root->IsRoot()) {
    root->parent()->set_root_instruction(softmax_fusion);
    TF_RETURN_IF_ERROR(
        root->parent()->RemoveInstructionAndUnusedOperands(root));
  } else {
    TF_RETURN_IF_ERROR(
        root->parent()->ReplaceInstruction(root, softmax_fusion));
  }

  VLOG(5) << softmax_fusion->ToString();
  return OkStatus();
}

}  // anonymous namespace

StatusOr<bool> SoftmaxRewriterTriton::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // TODO(b/281980675): allow pattern matching more than a vanilla Softmax.

  std::vector<std::pair<HloInstruction*, HloInstruction*>>
      softmax_root_and_producer_pairs;

  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    if (comp->IsCustomCallComputation()) {
      continue;
    }
    for (HloInstruction* instr : comp->MakeInstructionPostOrder()) {
      PrimitiveType element_ty = instr->shape().element_type();
      // TODO(b/281980675): ensure that code generation also works well for FP8
      // and BF16. This fails for the moment due to these data types requiring
      // float normalization.
      if (element_ty != F16 && element_ty != F32 && element_ty != F64) {
        continue;
      }

      if (auto producer = MatchesExactSoftmaxPattern(instr)) {
        softmax_root_and_producer_pairs.push_back(
            std::make_pair(instr, producer.value()));
      }
    }
  }
  if (softmax_root_and_producer_pairs.empty()) {
    return false;
  }

  for (auto [root, producer] : softmax_root_and_producer_pairs) {
    TF_RET_CHECK(FuseSoftmax(root, producer).ok());
  }
  return true;
}
}  // namespace xla::gpu
