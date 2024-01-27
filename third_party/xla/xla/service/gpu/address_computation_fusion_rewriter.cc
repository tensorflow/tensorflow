/* Copyright 2024 The OpenXLA Authors.

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
#include "xla/service/gpu/address_computation_fusion_rewriter.h"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <optional>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/cublas_cudnn.h"
#include "xla/service/gpu/hlo_traversal.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/status_macros.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace gpu {

namespace {

bool IsNoOp(const HloInstruction* hlo) {
  return HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kTuple,
                          HloOpcode::kGetTupleElement>(hlo);
}

absl::InlinedVector<HloInstruction*, 8> GetSlicedOperandChains(
    const HloInstruction* instr) {
  absl::InlinedVector<HloInstruction*, 8> sliced_operand_chains = {
      const_cast<HloInstruction*>(instr)};
  auto fusion = HloFusionAdaptor::ForComputation(instr->parent());
  for (auto* operand : instr->operands()) {
    absl::InlinedVector<HloInstruction*, 4> maybe_sliced_operand_chain;
    auto maybe_slice_adaptor =
        HloFindIf({HloInstructionAdaptor(*operand)}, *fusion, [&](auto node) {
          const HloInstruction* cur = &node.instruction();
          maybe_sliced_operand_chain.push_back(
              const_cast<HloInstruction*>(cur));
          // TODO(vuson): lift the first restriction by considering fusing other
          // uses of the operand to reuse the address computation. Only worth it
          // if other uses are also custom calls though.
          // TODO(vuson): lift the second restriction by considering fusing the
          // non-noop instructions to the computation if possible.
          return cur->user_count() > 1 || !IsNoOp(cur) ||
                 IsContiguousSlice(*cur);
        });
    if (maybe_slice_adaptor == std::nullopt) continue;
    const auto& maybe_slice_instr = maybe_slice_adaptor->instruction();
    if (IsContiguousSlice(maybe_slice_instr)) {
      sliced_operand_chains.insert(sliced_operand_chains.end(),
                                   maybe_sliced_operand_chain.begin(),
                                   maybe_sliced_operand_chain.end());
    }
  }
  return sliced_operand_chains;
}

absl::InlinedVector<HloInstruction*, 4> GetPatternCaptures(
    absl::Span<HloInstruction* const> matched) {
  absl::InlinedVector<HloInstruction*, 4> captures;

  absl::flat_hash_set<HloInstruction*> instructions_set(matched.begin(),
                                                        matched.end());

  for (HloInstruction* instr : matched) {
    for (HloInstruction* operand : instr->operands()) {
      if (!instructions_set.contains(operand) &&
          absl::c_find(captures, operand) == captures.end()) {
        captures.emplace_back(operand);
      }
    }
  }

  return captures;
}

absl::StatusOr<HloComputation*> CreateFusionBody(
    HloModule* module, absl::Span<HloInstruction* const> matched,
    absl::Span<HloInstruction* const> captures) {
  HloComputation::Builder builder("address-computation");

  // A mapping from original instructions to instructions in the fusion body.
  absl::flat_hash_map<const HloInstruction*, HloInstruction*> instr_mapping;

  auto mapped_operands = [&](HloInstruction* instr) {
    absl::InlinedVector<HloInstruction*, 4> operands;
    for (HloInstruction* operand : instr->operands()) {
      operands.push_back(instr_mapping.at(operand));
    }
    return operands;
  };

  // For every captured value create a parameter instruction in the computation
  // body and set up instruction mapping.
  for (const HloInstruction* capture : captures) {
    int64_t index = instr_mapping.size();
    instr_mapping[capture] =
        builder.AddInstruction(HloInstruction::CreateParameter(
            index, capture->shape(), absl::StrCat("p", index)));
  }

  // Instructions in the pattern are already topologically sorted, as we visited
  // them following use-def chain, then reverse the list.
  for (HloInstruction* instr : matched) {
    instr_mapping[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), mapped_operands(instr)));
  }

  HloInstruction* root = builder.last_added_instruction();

  // If the custom call requires a workspace we wrap the produced values with a
  // root tuple of "real" result and a workspace.
  if (root->shape().IsTuple()) {
    TF_RET_CHECK(root->shape().tuple_shapes_size() == 2);
    HloInstruction* result =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(root, 0));
    HloInstruction* workspace =
        builder.AddInstruction(HloInstruction::CreateGetTupleElement(root, 1));
    builder.AddInstruction(HloInstruction::CreateTuple({result, workspace}));
  }

  return module->AddComputationAndUnifyNamesAndIds(builder.Build(), false);
}

absl::StatusOr<HloInstruction*> CreateFusionInstruction(
    HloModule* module, HloInstruction* orig,
    absl::Span<HloInstruction* const> captures, HloComputation* body) {
  HloComputation* parent = orig->parent();

  // Add a fusion operation calling outlined fusion computation.
  HloInstruction* fusion = parent->AddInstruction(HloInstruction::CreateFusion(
      body->root_instruction()->shape(), HloInstruction::FusionKind::kCustom,
      captures, body));
  module->SetAndUniquifyInstrName(fusion, "address_computation");

  // Set backends config to a matched custom fusion config.
  GpuBackendConfig gpu_config;
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind("__custom_fusion");
  CustomFusionConfig config;
  config.set_name("address_computation");
  *backend_config.mutable_custom_fusion_config() = config;
  TF_RETURN_IF_ERROR(fusion->set_backend_config(std::move(gpu_config)));

  return fusion;
}

}  // namespace

absl::StatusOr<bool> AddressComputationFusionRewriter::Run(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  auto instructions = module->entry_computation()->MakeInstructionPostOrder();
  bool changed = false;

  absl::flat_hash_map<HloInstruction*, absl::InlinedVector<HloInstruction*, 8>>
      matches;

  // Collect all potential custom call matches in the non-fusion computations.
  for (HloComputation* computation : module->computations()) {
    if (computation->IsFusionComputation()) continue;
    for (HloInstruction* instr : computation->instructions()) {
      if (IsLegacyCublasMatmul(*instr)) {
        auto sliced_operand_chains = GetSlicedOperandChains(instr);
        if (!(sliced_operand_chains.size() == 1 &&
              sliced_operand_chains.front() == instr)) {
          matches[instr] = std::move(sliced_operand_chains);
        }
      }
    }
  }

  for (auto& kv : matches) {
    auto captures = GetPatternCaptures(kv.second);
    std::reverse(kv.second.begin(), kv.second.end());
    TF_ASSIGN_OR_RETURN(HloComputation * fusion_body,
                        CreateFusionBody(module, kv.second, captures));
    TF_ASSIGN_OR_RETURN(
        HloInstruction * fusion,
        CreateFusionInstruction(module, kv.first, captures, fusion_body));
    HloComputation* parent = kv.first->parent();
    TF_RETURN_IF_ERROR(parent->ReplaceInstruction(kv.first, fusion));
    changed = true;
  }

  return changed;
}

}  // namespace gpu
}  // namespace xla
