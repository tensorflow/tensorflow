/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/backends/gpu/transforms/dynamic_slice_fusion_rewriter_v2.h"

#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/tsl/platform/status_macros.h"
#include "xla/backends/gpu/transforms/dynamic_slice_fusion.h"
#include "xla/hlo/ir/hlo_casting_utils.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/gpu/backend_configs.pb.h"
#include "xla/service/gpu/gpu_constants.h"
#include "xla/service/gpu/ir_emission_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::gpu {
namespace {

//===----------------------------------------------------------------------===//
// Data structures
//===----------------------------------------------------------------------===//

// A hero operand path: slice → noops → hero. Describes a Slice or
// DynamicSlice instruction feeding the hero through zero or more no-op
// instructions (bitcasts, GTEs).
struct SlicedParameter {
  // The Slice or DynamicSlice instruction at the start of the chain.
  HloInstruction* slice;

  // Bitcasts/GTEs between the slice and the hero (topological order).
  std::vector<HloInstruction*> noops;
};

// A hero result path: hero → [GTE] → [bitcasts] → DUS. Describes how one hero
// output flows through an optional GTE and bitcasts into a DynamicUpdateSlice
// that writes it into a target buffer. The `noops` vector holds the full chain
// (GTE first, then bitcasts). For passthrough results (tuple outputs without
// DUS), update_slice is nullptr and noops contains only the GTE.
//
// For flat tuple-producing heroes, there is one SlicedResult per tuple element.
struct SlicedResult {
  // Flat tuple element index within the hero's output shape. Matches
  // DynamicSliceFusion::Result::result_number. 0 for non-tuple heroes.
  int64_t result_number = 0;

  // Instructions between the hero output and the DUS: GTE (for tuple heroes)
  // followed by bitcasts. Empty when the leaf has no users in the original HLO
  // (dead output) or for non-tuple heroes without bitcasts.
  std::vector<HloInstruction*> noops;

  // The DynamicUpdateSlice at the end of the chain. nullptr for passthrough
  // results (tuple outputs that don't flow through a DUS).
  HloInstruction* update_slice = nullptr;
};

//===----------------------------------------------------------------------===//
// Helpers
//===----------------------------------------------------------------------===//

bool IsNoOp(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kBitcast, HloOpcode::kGetTupleElement>(
      instr);
}

bool IsSlicingBoundary(const HloInstruction* instr) {
  return HloPredicateIsOp<HloOpcode::kSlice, HloOpcode::kDynamicSlice,
                          HloOpcode::kDynamicUpdateSlice>(instr);
}

bool HasDynamicSliceConfig(const HloInstruction* instr) {
  auto config = instr->backend_config<GpuBackendConfig>();
  if (!config.ok()) {
    return false;
  }
  return config->has_dynamic_slice_config();
}

bool IsAlignedSlice(const HloInstruction* instr) {
  if (!IsContiguousSlice(*instr)) {
    return false;
  }

  const Shape& slice_shape = instr->opcode() == HloOpcode::kDynamicUpdateSlice
                                 ? instr->operand(1)->shape()
                                 : instr->shape();

  auto byte_strides = ShapeUtil::ByteStrides(slice_shape);
  if (!byte_strides.has_value()) {
    return false;
  }

  int64_t slice_bytes = ShapeUtil::ByteSizeOfElements(slice_shape);
  return slice_bytes % kXlaAllocatedBufferAlignBytes == 0;
}

bool HasSupportedShapes(const HloInstruction* hero) {
  for (const HloInstruction* operand : hero->operands()) {
    if (operand->shape().IsTuple()) {
      LOG(WARNING) << "DynamicSliceFusionRewriterV2: skipping " << hero->name()
                   << " because operand " << operand->name() << " is a tuple";
      return false;
    }
  }
  if (hero->shape().IsTuple()) {
    for (const Shape& tuple_shape : hero->shape().tuple_shapes()) {
      if (tuple_shape.IsTuple()) {
        LOG(WARNING) << "DynamicSliceFusionRewriterV2: skipping "
                     << hero->name()
                     << " because nested tuple results are not supported";
        return false;
      }
    }
  }
  return true;
}

//===----------------------------------------------------------------------===//
// Resolve sliced parameters
//===----------------------------------------------------------------------===//

using OptLevel = DynamicSliceFusionRewriterV2::OptLevel;
using CaptureSlice = DynamicSliceFusionRewriterV2::CaptureSlice;
using CaptureUpdateSlice = DynamicSliceFusionRewriterV2::CaptureUpdateSlice;

std::optional<SlicedParameter> ResolveSlicedParameter(HloInstruction* operand,
                                                      OptLevel opt_level) {
  HloInstruction* current = operand;

  // Walk backward through no-ops to find the slice. In O2 mode, also look
  // through tuple→GTE barriers: if a chain of GTEs resolves through
  // corresponding tuple instructions, we skip the entire GTE/tuple tower and
  // continue from the innermost tuple operand. These are NOT added to noops.
  std::vector<HloInstruction*> noops;
  while (IsNoOp(current)) {
    if (opt_level == OptLevel::kO2 &&
        current->opcode() == HloOpcode::kGetTupleElement) {
      // Collect a chain of GTEs, then try to resolve through
      // opt-barriers and tuples back to the original operand.
      std::vector<int64_t> indices;
      HloInstruction* probe = current;
      while (probe->opcode() == HloOpcode::kGetTupleElement) {
        indices.push_back(
            Cast<HloGetTupleElementInstruction>(probe)->tuple_index());
        probe = probe->mutable_operand(0);
      }
      // Skip through optimization barriers (they pass tuples through
      // unchanged).
      while (probe->opcode() == HloOpcode::kOptimizationBarrier) {
        probe = probe->mutable_operand(0);
      }
      // Walk the tuple chain: each index peels one tuple layer.
      bool resolved = true;
      for (int64_t idx : indices) {
        if (probe->opcode() != HloOpcode::kTuple) {
          resolved = false;
          break;
        }
        probe = probe->mutable_operand(idx);
      }
      if (resolved) {
        current = probe;
        continue;
      }
    }
    noops.push_back(current);
    current = current->mutable_operand(0);
  }

  if (auto* slice = DynCast<HloSliceInstruction>(current)) {
    if (!IsAlignedSlice(slice)) {
      return std::nullopt;
    }
    absl::c_reverse(noops);
    return SlicedParameter{slice, std::move(noops)};
  }

  if (auto* ds = DynCast<HloDynamicSliceInstruction>(current)) {
    if (!HasDynamicSliceConfig(ds)) {
      return std::nullopt;
    }
    if (!IsAlignedSlice(ds)) {
      return std::nullopt;
    }
    absl::c_reverse(noops);
    return SlicedParameter{ds, std::move(noops)};
  }

  return std::nullopt;
}

std::vector<SlicedParameter> ResolveSlicedParameters(
    HloInstruction* hero, OptLevel opt_level,
    const CaptureSlice& capture_slice) {
  std::vector<SlicedParameter> result;
  for (int64_t i = 0; i < hero->operand_count(); ++i) {
    HloInstruction* operand = hero->mutable_operand(i);
    auto param = ResolveSlicedParameter(operand, opt_level);
    if (!param.has_value()) {
      continue;
    }
    if (!capture_slice(hero, i, param->slice)) {
      continue;
    }
    // In O2 mode, the chain output may differ from the hero's operand when
    // we looked through a tuple/GTE barrier. Replace the hero's operand so
    // BuildFusionPlan sees a connected graph. Safe because the hero will be
    // replaced by the fusion.
    HloInstruction* chain_output =
        param->noops.empty() ? param->slice : param->noops.back();
    if (chain_output != operand) {
      CHECK_OK(hero->ReplaceOperandWith(i, chain_output));
    }
    result.push_back(std::move(*param));
  }
  return result;
}

//===----------------------------------------------------------------------===//
// Resolve sliced results
//===----------------------------------------------------------------------===//

std::optional<SlicedResult> ResolveSlicedResult(HloInstruction* user) {
  HloInstruction* current = user;

  // Walk forward through no-ops to find the DUS.
  std::vector<HloInstruction*> noops;
  while (IsNoOp(current)) {
    if (current->user_count() != 1) {
      return std::nullopt;
    }
    noops.push_back(current);
    current = current->users().front();
  }

  auto* dus = DynCast<HloDynamicUpdateSliceInstruction>(current);
  if (dus == nullptr) {
    return std::nullopt;
  }

  if (!HasDynamicSliceConfig(dus)) {
    return std::nullopt;
  }

  if (!IsAlignedSlice(dus)) {
    return std::nullopt;
  }

  // DUS must flow into the root: either IS the root, or feeds root tuple.
  HloComputation* parent = dus->parent();
  HloInstruction* root = parent->root_instruction();
  if (dus != root) {
    if (root->opcode() != HloOpcode::kTuple) {
      return std::nullopt;
    }
    bool feeds_root =
        absl::c_any_of(root->operands(),
                       [dus](const HloInstruction* op) { return op == dus; });
    if (!feeds_root) {
      return std::nullopt;
    }
  }

  return SlicedResult{0, std::move(noops), dus};
}

// Finds the GTE user of `hero` that extracts the given tuple index.
HloInstruction* FindGte(HloInstruction* hero, int64_t tuple_index) {
  for (HloInstruction* user : hero->users()) {
    auto* gte = DynCast<HloGetTupleElementInstruction>(user);
    if (gte != nullptr && gte->tuple_index() == tuple_index) {
      return gte;
    }
  }
  return nullptr;
}

// Walks forward from `gte` to find a DUS chain. If found, returns a
// SlicedResult with the GTE prepended into noops.
std::optional<SlicedResult> ResolveLeafDus(HloInstruction* gte) {
  for (HloInstruction* user : gte->users()) {
    if (auto sliced = ResolveSlicedResult(user)) {
      sliced->noops.insert(sliced->noops.begin(), gte);
      return sliced;
    }
  }
  return std::nullopt;
}

// Resolves the sliced result for a non-tuple hero.
std::vector<SlicedResult> ResolveNonTupleSlicedResult(
    HloInstruction* hero, const CaptureUpdateSlice& capture_update_slice) {
  for (HloInstruction* user : hero->users()) {
    if (auto sliced = ResolveSlicedResult(user)) {
      if (!capture_update_slice(hero, std::nullopt, sliced->update_slice)) {
        continue;
      }
      return std::vector<SlicedResult>{std::move(*sliced)};
    }
  }
  return {};
}

// Resolves sliced results for a flat tuple-producing hero. Returns one
// SlicedResult per tuple element.
std::vector<SlicedResult> ResolveTupleSlicedResults(
    HloInstruction* hero, const CaptureUpdateSlice& capture_update_slice) {
  const int64_t tuple_size = hero->shape().tuple_shapes().size();

  struct LeafInfo {
    int64_t result_number;
    HloInstruction* gte;
    std::optional<SlicedResult> sliced_update;
  };

  std::vector<LeafInfo> leaf_infos;
  leaf_infos.reserve(tuple_size);
  for (int64_t i = 0; i < tuple_size; ++i) {
    HloInstruction* gte = FindGte(hero, i);
    std::optional<SlicedResult> sliced_update;
    if (gte != nullptr) {
      sliced_update = ResolveLeafDus(gte);
    }
    leaf_infos.push_back({i, gte, std::move(sliced_update)});
  }

  std::vector<SlicedResult> result;
  result.reserve(tuple_size);
  for (auto& info : leaf_infos) {
    if (info.sliced_update.has_value() &&
        capture_update_slice(hero, info.result_number,
                             info.sliced_update->update_slice)) {
      info.sliced_update->result_number = info.result_number;
      result.push_back(std::move(*info.sliced_update));
      continue;
    }

    SlicedResult passthrough;
    passthrough.result_number = info.result_number;
    if (info.gte != nullptr) {
      passthrough.noops.push_back(info.gte);
    }
    result.push_back(std::move(passthrough));
  }
  return result;
}

std::vector<SlicedResult> ResolveSlicedResults(
    HloInstruction* hero, const CaptureUpdateSlice& capture_update_slice) {
  if (hero->shape().IsTuple()) {
    return ResolveTupleSlicedResults(hero, capture_update_slice);
  }
  return ResolveNonTupleSlicedResult(hero, capture_update_slice);
}

//===----------------------------------------------------------------------===//
// Resolve offset expressions
//===----------------------------------------------------------------------===//

// Follows `instr` operands to collect instructions that define dynamic (update)
// slice offset expression: scalar operations computing integer offsets.
void AddOffsetExpressionInstructions(
    HloInstruction* instr, std::vector<HloInstruction*>& clone_instructions,
    absl::flat_hash_set<HloInstruction*>& clone_instruction_set,
    absl::flat_hash_set<HloInstruction*>& offset_instruction_set) {
  if (HloPredicateIsOp<HloOpcode::kConstant, HloOpcode::kParameter>(instr) ||
      IsSlicingBoundary(instr) || !DynamicSliceFusion::Offset::IsExpr(instr)) {
    return;
  }

  if (!clone_instruction_set.insert(instr).second) {
    return;
  }

  offset_instruction_set.insert(instr);
  for (HloInstruction* operand : instr->operands()) {
    AddOffsetExpressionInstructions(operand, clone_instructions,
                                    clone_instruction_set,
                                    offset_instruction_set);
  }

  clone_instructions.push_back(instr);
}

void AddDynamicSliceOffsetExpressionInstructions(
    HloInstruction* instr, int64_t first_offset_operand,
    std::vector<HloInstruction*>& clone_instructions,
    absl::flat_hash_set<HloInstruction*>& clone_instruction_set,
    absl::flat_hash_set<HloInstruction*>& offset_instruction_set) {
  for (int64_t i = first_offset_operand; i < instr->operand_count(); ++i) {
    AddOffsetExpressionInstructions(instr->mutable_operand(i),
                                    clone_instructions, clone_instruction_set,
                                    offset_instruction_set);
  }
}

//===----------------------------------------------------------------------===//
// Build fusion plan
//===----------------------------------------------------------------------===//

// Complete plan for creating a dynamic-slice fusion from a hero instruction.
// Contains all instructions to include in the fusion body and the external
// operands to pass to the fusion.
struct DynamicSliceFusionPlan {
  // External operands needed by cloned instructions. These are operands not
  // cloned into the fusion body, such as original buffers and scalar offset
  // expression leaves.
  std::vector<HloInstruction*> external_operands;

  // All instructions to clone into the fusion body, in dependency order. This
  // includes offset expression interiors, slices, noops, the hero, result
  // noops, and DUS instructions. CreateFusionBody builds the final root from
  // these cloned instructions.
  std::vector<HloInstruction*> clone_instructions;
};

std::optional<DynamicSliceFusionPlan> BuildFusionPlan(
    HloInstruction* hero, absl::Span<const SlicedParameter> sliced_params,
    absl::Span<const SlicedResult> sliced_results) {
  bool has_any_slice = !sliced_params.empty() ||
                       absl::c_any_of(sliced_results, [](const auto& r) {
                         return r.update_slice != nullptr;
                       });
  if (!has_any_slice) {
    return std::nullopt;
  }

  // Instructions to clone into the fusion body, in dependency order.
  std::vector<HloInstruction*> clone_instructions;
  // Membership set for clone_instructions, used to avoid duplicate clones.
  absl::flat_hash_set<HloInstruction*> clone_instruction_set;
  // Subset of cloned instructions that are part of index offset expressions.
  absl::flat_hash_set<HloInstruction*> offset_instruction_set;

  // Instructions whose operands are scanned, in order, to discover external
  // operands that become fusion parameters.
  std::vector<HloInstruction*> external_operand_scan_roots;

  auto add_clone_instruction = [&](HloInstruction* instr) {
    if (clone_instruction_set.insert(instr).second) {
      clone_instructions.push_back(instr);
    }
  };

  // Add sliced parameter paths (topological: slice → noops → hero).
  for (const SlicedParameter& param : sliced_params) {
    if (param.slice->opcode() == HloOpcode::kDynamicSlice) {
      AddDynamicSliceOffsetExpressionInstructions(
          param.slice, 1, clone_instructions, clone_instruction_set,
          offset_instruction_set);
    }
    add_clone_instruction(param.slice);
    external_operand_scan_roots.push_back(param.slice);
    for (HloInstruction* noop : param.noops) {
      add_clone_instruction(noop);
      external_operand_scan_roots.push_back(noop);
    }
  }

  add_clone_instruction(hero);
  external_operand_scan_roots.push_back(hero);

  // Add sliced result paths (topological: hero → noops → DUS).
  // For passthrough results (no DUS), add the GTE chain so the hero output
  // is accessible inside the fusion.
  for (const auto& result : sliced_results) {
    for (HloInstruction* noop : result.noops) {
      add_clone_instruction(noop);
      external_operand_scan_roots.push_back(noop);
    }
    if (result.update_slice != nullptr) {
      AddDynamicSliceOffsetExpressionInstructions(
          result.update_slice, 2, clone_instructions, clone_instruction_set,
          offset_instruction_set);
      add_clone_instruction(result.update_slice);
      external_operand_scan_roots.push_back(result.update_slice);
    }
  }

  // Sink constants that are part of offset expressions into the fusion body
  // instead of capturing them as parameters. Other constants must remain fusion
  // parameters because the dynamic-slice fusion emitter expects hero operands
  // and DUS targets to be backed by fusion parameters.
  auto is_offset_use = [&](HloInstruction* instr, int64_t operand_index) {
    if (offset_instruction_set.contains(instr)) {
      return true;
    }
    return (instr->opcode() == HloOpcode::kDynamicSlice &&
            operand_index >= 1) ||
           (instr->opcode() == HloOpcode::kDynamicUpdateSlice &&
            operand_index >= 2);
  };

  std::vector<HloInstruction*> offset_constants;
  absl::flat_hash_set<HloInstruction*> offset_constant_set;
  absl::flat_hash_set<HloInstruction*> non_offset_constant_set;

  for (HloInstruction* instr : clone_instructions) {
    for (int64_t i = 0; i < instr->operand_count(); ++i) {
      HloInstruction* operand = instr->mutable_operand(i);
      if (!HloPredicateIsOp<HloOpcode::kConstant>(operand)) {
        continue;
      }
      if (is_offset_use(instr, i)) {
        if (offset_constant_set.insert(operand).second) {
          offset_constants.push_back(operand);
        }
      } else {
        non_offset_constant_set.insert(operand);
      }
    }
  }

  std::vector<HloInstruction*> constants;
  for (HloInstruction* constant : offset_constants) {
    if (!non_offset_constant_set.contains(constant) &&
        clone_instruction_set.insert(constant).second) {
      constants.push_back(constant);
    }
  }

  clone_instructions.insert(clone_instructions.begin(), constants.begin(),
                            constants.end());

  // Collect external operands by scanning cloned slice/hero/update path
  // instructions in order. Recurse only through cloned offset expressions, so
  // their scalar leaves become fusion parameters without disturbing the
  // buffer-first parameter order.
  std::vector<HloInstruction*> external_operands;
  absl::flat_hash_set<HloInstruction*> external_operand_set;
  absl::flat_hash_set<HloInstruction*> visited_offset_instruction_set;

  auto collect_external_operand = [&](auto& self, HloInstruction* operand) {
    // Any non-cloned operand is external to the fusion body and must become a
    // fusion parameter.
    if (!clone_instruction_set.contains(operand)) {
      if (external_operand_set.insert(operand).second) {
        external_operands.push_back(operand);
      }
      return;
    }
    // Only cloned offset expressions are transparent for this traversal; other
    // cloned instructions are already represented inside the fusion body.
    if (!offset_instruction_set.contains(operand)) {
      return;
    }
    if (!visited_offset_instruction_set.insert(operand).second) {
      return;
    }
    // Recurse through offset expressions to find scalar leaves.
    for (HloInstruction* offset_operand : operand->operands()) {
      self(self, offset_operand);
    }
  };

  for (HloInstruction* instr : external_operand_scan_roots) {
    for (HloInstruction* operand : instr->operands()) {
      collect_external_operand(collect_external_operand, operand);
    }
  }

  return DynamicSliceFusionPlan{std::move(external_operands),
                                std::move(clone_instructions)};
}

//===----------------------------------------------------------------------===//
// Create fusion
//===----------------------------------------------------------------------===//

absl::StatusOr<HloComputation*> CreateFusionBody(
    HloModule* module, const DynamicSliceFusionPlan& plan,
    absl::Span<const SlicedResult> sliced_results, HloInstruction* hero) {
  HloComputation::Builder builder("dynamic-slice-fusion");

  absl::flat_hash_map<const HloInstruction*, HloInstruction*> instr_mapping;
  instr_mapping.reserve(plan.external_operands.size() +
                        plan.clone_instructions.size());

  // Create fusion parameters for external operands.
  for (const HloInstruction* operand : plan.external_operands) {
    int64_t index = instr_mapping.size();
    instr_mapping[operand] =
        builder.AddInstruction(HloInstruction::CreateParameter(
            index, operand->shape(), absl::StrCat("p", index)));
  }

  auto mapped_operands = [&](HloInstruction* instr) {
    absl::InlinedVector<HloInstruction*, 4> operands;
    operands.reserve(instr->operand_count());
    for (HloInstruction* operand : instr->operands()) {
      operands.push_back(instr_mapping.at(operand));
    }
    return operands;
  };

  // Clone instructions in topological order.
  for (HloInstruction* instr : plan.clone_instructions) {
    instr_mapping[instr] = builder.AddInstruction(
        instr->CloneWithNewOperands(instr->shape(), mapped_operands(instr)));
  }

  // Build root tuple when there are multiple results (DUS and/or passthrough).
  if (sliced_results.size() > 1) {
    HloInstruction* cloned_hero = instr_mapping.at(hero);
    std::vector<HloInstruction*> tuple_operands;
    tuple_operands.reserve(sliced_results.size());
    for (const auto& result : sliced_results) {
      if (result.update_slice != nullptr) {
        tuple_operands.push_back(instr_mapping.at(result.update_slice));
      } else if (!result.noops.empty()) {
        tuple_operands.push_back(instr_mapping.at(result.noops.back()));
      } else {
        // Dead output: create a GTE from the cloned hero to extract this tuple
        // element so the fusion output has a buffer slot for it.
        tuple_operands.push_back(
            builder.AddInstruction(HloInstruction::CreateGetTupleElement(
                cloned_hero, result.result_number)));
      }
    }
    builder.AddInstruction(HloInstruction::CreateTuple(tuple_operands));
  }

  return module->AddComputationAndUnifyNamesAndIds(builder.Build(), false);
}

absl::Status SetDynamicSliceFusionBackendConfig(HloInstruction* fusion) {
  GpuBackendConfig gpu_config;
  FusionBackendConfig& backend_config =
      *gpu_config.mutable_fusion_backend_config();
  backend_config.set_kind("__custom_fusion");
  CustomFusionConfig config;
  config.set_name(std::string(kDynamicSliceFusionConfigName));
  *backend_config.mutable_custom_fusion_config() = config;
  return fusion->set_backend_config(std::move(gpu_config));
}

//===----------------------------------------------------------------------===//
// Rewrite sync hero
//===----------------------------------------------------------------------===//

absl::StatusOr<bool> RewriteHero(
    HloModule* module, HloInstruction* hero,
    absl::Span<const SlicedParameter> sliced_params,
    absl::Span<const SlicedResult> sliced_results) {
  auto plan = BuildFusionPlan(hero, sliced_params, sliced_results);
  if (!plan.has_value()) {
    return false;
  }

  ASSIGN_OR_RETURN(HloComputation * fusion_body,
                   CreateFusionBody(module, *plan, sliced_results, hero));

  HloComputation* parent = hero->parent();
  HloInstruction* fusion = parent->AddInstruction(
      HloInstruction::CreateFusion(fusion_body->root_instruction()->shape(),
                                   HloInstruction::FusionKind::kCustom,
                                   plan->external_operands, fusion_body));
  module->SetAndUniquifyInstrName(fusion, "dynamic_slice_fusion");
  RETURN_IF_ERROR(SetDynamicSliceFusionBackendConfig(fusion));

  if (sliced_results.size() > 1) {
    bool any_result_replaced = false;
    for (int64_t i = 0; i < sliced_results.size(); ++i) {
      auto* gte = parent->AddInstruction(
          HloInstruction::CreateGetTupleElement(fusion, i));
      if (sliced_results[i].update_slice != nullptr) {
        RETURN_IF_ERROR(
            parent->ReplaceInstruction(sliced_results[i].update_slice, gte));
        any_result_replaced = true;
      } else if (!sliced_results[i].noops.empty()) {
        HloInstruction* original_leaf = sliced_results[i].noops.back();
        RETURN_IF_ERROR(parent->ReplaceInstruction(original_leaf, gte));
        any_result_replaced = true;
      }
    }
    if (!any_result_replaced) {
      RETURN_IF_ERROR(parent->ReplaceInstruction(hero, fusion));
    }
  } else if (sliced_results.size() == 1) {
    if (sliced_results[0].update_slice != nullptr) {
      RETURN_IF_ERROR(
          parent->ReplaceInstruction(sliced_results[0].update_slice, fusion));
    } else {
      RETURN_IF_ERROR(parent->ReplaceInstruction(hero, fusion));
    }
  } else {
    RETURN_IF_ERROR(parent->ReplaceInstruction(hero, fusion));
  }

  return true;
}

}  // namespace

//===----------------------------------------------------------------------===//
// RunImpl
//===----------------------------------------------------------------------===//

absl::StatusOr<bool> DynamicSliceFusionRewriterV2::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;

  std::vector<HloComputation*> computations =
      module->MakeNonfusionComputations(execution_threads);
  for (HloComputation* computation : computations) {
    std::vector<HloInstruction*> heroes;
    for (HloInstruction* candidate : computation->instructions()) {
      if (!options_.predicate(candidate)) {
        continue;
      }
      if (candidate->opcode() == HloOpcode::kAsyncStart) {
        return absl::InvalidArgumentError(absl::StrCat(
            "DynamicSliceFusionRewriterV2 predicate must not match "
            "async-start instructions, but matched: ",
            candidate->name()));
      }
      if (HasSupportedShapes(candidate)) {
        heroes.push_back(candidate);
      }
    }

    for (HloInstruction* hero : heroes) {
      auto sliced_params = ResolveSlicedParameters(hero, options_.opt_level,
                                                   options_.capture_slice);
      auto sliced_results =
          ResolveSlicedResults(hero, options_.capture_update_slice);
      ASSIGN_OR_RETURN(
          bool hero_changed,
          RewriteHero(module, hero, sliced_params, sliced_results));
      changed |= hero_changed;
    }
  }

  return changed;
}

}  // namespace xla::gpu
