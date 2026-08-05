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

#include "xla/hlo/tools/comparison/original_tensor_summary_calculator.h"

#include <algorithm>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/str_split.h"
#include "absl/strings/string_view.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/hlo/tools/comparison/comparison_service.pb.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla::numerics::comparison {

using tensor_transformation::Reshape;
using tensor_transformation::TensorTransformation;
using tensor_transformation::Unshard;

namespace {

// Parses a string representation of a scope instruction (e.g., "loop#3") into
// a ScopeInstruction object.
absl::StatusOr<ScopeInstruction> ParseScopeInstruction(
    absl::string_view scope_str) {
  std::pair<absl::string_view, absl::string_view> parts =
      absl::StrSplit(scope_str, absl::MaxSplits('#', 1));
  int64_t iteration_index = 0;
  if (!parts.second.empty()) {
    if (parts.second == "*") {
      iteration_index = -1;
    } else if (!absl::SimpleAtoi(parts.second, &iteration_index)) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid iteration index: ", parts.second));
    }
  }
  return ScopeInstruction::Create(parts.first, iteration_index);
}

// Parses a string like "scope1/scope2/instruction_name" into a
// RelativeScopedTensorKey.
absl::StatusOr<RelativeScopedTensorKey>
ParseInstructionNameAsRelativeScopedTensorKey(
    absl::string_view instruction_name, const xla::ShapeIndex& shape_index) {
  std::vector<absl::string_view> parts = absl::StrSplit(instruction_name, '/');
  if (parts.empty()) {
    return absl::InvalidArgumentError(
        "Empty instruction name for original array");
  }
  std::vector<ScopeInstruction> scope_instructions;
  if (parts.size() > 1) {
    scope_instructions.reserve(parts.size() - 1);
    for (int64_t i = 0; i < parts.size() - 1; ++i) {
      ASSIGN_OR_RETURN(ScopeInstruction scope_instr,
                       ParseScopeInstruction(parts[i]));
      scope_instructions.push_back(std::move(scope_instr));
    }
  }
  return RelativeScopedTensorKey::Create(
      TensorKey::Create(parts.back(), shape_index),
      std::move(scope_instructions));
}

absl::StatusOr<TensorTransformation> ParseRecoveryModule(
    const HloModule& recovery_module) {
  HloComputation* comp = recovery_module.entry_computation();
  if (comp->num_parameters() != 1) {
    return absl::InvalidArgumentError("Recovery module must have 1 parameter");
  }
  const HloInstruction* param = comp->parameter_instruction(0);
  if (param->has_sharding()) {
    if (param->sharding().IsManual()) {
      return absl::InvalidArgumentError("Manual sharding is not supported yet");
    }
    return Unshard{/*original_dimensions=*/std::vector<int64_t>(
                       comp->root_instruction()->shape().dimensions().begin(),
                       comp->root_instruction()->shape().dimensions().end()),
                   /*sharding=*/param->sharding()};
  }

  HloInstruction const* instruction = comp->root_instruction();
  while (instruction->opcode() == HloOpcode::kBitcast ||
         instruction->opcode() == HloOpcode::kCopy) {
    instruction = instruction->operand(0);
  }

  if (instruction->opcode() == HloOpcode::kReshape &&
      instruction->operand(0)->opcode() == HloOpcode::kParameter) {
    return Reshape{/*output_dimensions=*/std::vector<int64_t>(
        comp->root_instruction()->shape().dimensions().begin(),
        comp->root_instruction()->shape().dimensions().end())};
  }

  return absl::UnimplementedError(
      absl::StrCat("Unsupported recovery module with root operand ",
                   HloOpcodeString(instruction->opcode())));
}

void BuildTransformationChain(
    const OriginalArray& placeholder,
    const absl::flat_hash_map<
        OriginalArray, std::vector<std::pair<OriginalArray, const HloModule*>>>&
        placeholder_to_recoverables,
    absl::flat_hash_set<OriginalArray>& visited_placeholders,
    std::vector<OriginalTensorSummaryCalculator::OriginalTensorInfo>& results) {
  if (!visited_placeholders.insert(placeholder).second) {
    LOG(ERROR) << "Cycle detected in recovery transformation chain involving "
                  "placeholder: "
               << placeholder.ToString();
    return;
  }
  auto it = placeholder_to_recoverables.find(placeholder);
  if (it == placeholder_to_recoverables.end()) {
    visited_placeholders.erase(placeholder);
    return;
  }
  for (const auto& [goal, module] : it->second) {
    std::optional<TensorTransformation> transformation;
    if (module != nullptr) {
      auto trans_or = ParseRecoveryModule(*module);
      if (!trans_or.ok()) {
        LOG(ERROR) << "Failed to parse recovery module for " << goal.ToString()
                   << ": " << trans_or.status();
        continue;
      }
      transformation = *trans_or;
    }

    std::vector<OriginalTensorSummaryCalculator::OriginalTensorInfo>
        downstream_results;
    BuildTransformationChain(goal, placeholder_to_recoverables,
                             visited_placeholders, downstream_results);

    if (!absl::StrContains(goal.instruction_name, "__ovp")) {
      results.push_back(
          {/*original_scoped_tensor_key=*/RelativeScopedTensorKey::FromString(
               goal.instruction_name, goal.shape_index),
           /*tensor_transformation=*/
           transformation.has_value()
               ? std::make_shared<TensorTransformation>(*transformation)
               : nullptr});
    }

    if (!transformation.has_value()) {
      results.insert(results.end(), downstream_results.begin(),
                     downstream_results.end());
    } else {
      for (const auto& downstream_info : downstream_results) {
        TensorTransformation chained_trans = *transformation;
        std::visit(
            [&](auto& value) {
              value.continuation = downstream_info.tensor_transformation;
            },
            chained_trans);
        results.push_back(
            {/*original_scoped_tensor_key=*/
             downstream_info.original_scoped_tensor_key,
             /*tensor_transformation=*/
             std::make_shared<TensorTransformation>(chained_trans)});
      }
    }
  }
  visited_placeholders.erase(placeholder);
}

absl::flat_hash_map<std::string, std::string>
GetUniqueCallLikeInstructionNameByContainedInstructionName(
    const HloModule& module) {
  absl::flat_hash_map<const HloComputation*, std::vector<absl::string_view>>
      caller_instruction_names_by_computation;
  // Add a special entry for the root computation so that our matching logic can
  // properly handle root computation as well.
  caller_instruction_names_by_computation[module.entry_computation()] = {
      "<root>"};

  for (HloComputation* comp : module.computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (IsCallLike(*instr)) {
        for (const HloComputation* called_computation :
             instr->called_computations()) {
          caller_instruction_names_by_computation[called_computation].push_back(
              instr->name());
        }
      }
    }
  }

  absl::flat_hash_map<std::string, std::string>
      unique_caller_by_contained_instruction;
  for (HloComputation* comp : module.computations()) {
    auto it = caller_instruction_names_by_computation.find(comp);
    if (it != caller_instruction_names_by_computation.end() &&
        it->second.size() == 1) {
      absl::string_view unique_caller_name = it->second.front();
      for (HloInstruction* instr : comp->instructions()) {
        unique_caller_by_contained_instruction[instr->name()] =
            std::string(unique_caller_name);
      }
    }
  }
  return unique_caller_by_contained_instruction;
}

void PopulateCallMapForCallLikeInstruction(
    const HloInstruction& optimized_instr,
    absl::flat_hash_map<std::string, std::vector<ScopeInstruction>>& call_map,
    const absl::flat_hash_map<std::string, std::string>&
        original_unique_caller_by_contained_instruction_name,
    absl::flat_hash_map<std::string, std::string>&
        original_instruction_name_by_optimized_instruction_name) {
  if (optimized_instr.original_value()) {
    auto scope_instructions_str =
        optimized_instr.original_value()->GetOriginalCallLikeInstructions();
    if (scope_instructions_str.has_value()) {
      std::vector<std::string> scope_names =
          absl::StrSplit(*scope_instructions_str, '/');
      std::vector<ScopeInstruction> scope_instructions;
      scope_instructions.reserve(scope_names.size());
      for (const auto& scope_name : scope_names) {
        scope_instructions.push_back(ScopeInstruction::FromString(scope_name));
      }
      call_map[optimized_instr.name()] = std::move(scope_instructions);
      return;
    }
  }
  // If original value tracking does not tell us the original call-like
  // instructions, we try to establish the connection through instructions in
  // the called computations.
  // The heuristic works as follows:
  // 1. Iterate through all instructions within the computations called by
  //    `optimized_instr`.
  // 2. For each such instruction, find its corresponding instruction name in
  //    the original module.
  // 3. Find the computation in the original module that contains this original
  //    instruction.
  // 4. Find all callers of this original computation.
  // 5. If there is exactly one caller instruction in the original module, we
  //    assume this single caller is the original call-like instruction
  //    that corresponds to `optimized_instr`.
  std::vector<const HloComputation*> computations_to_process;
  for (const HloComputation* called_computation :
       optimized_instr.called_computations()) {
    computations_to_process.push_back(called_computation);
  }

  while (!computations_to_process.empty()) {
    const HloComputation* current_comp = computations_to_process.back();
    computations_to_process.pop_back();

    for (const HloInstruction* instr_in_called_computation :
         current_comp->instructions()) {
      if (instr_in_called_computation->opcode() == HloOpcode::kFusion) {
        computations_to_process.push_back(
            instr_in_called_computation->fused_instructions_computation());
      }

      std::string original_instr_name;
      // Try to find the original instruction name for
      // `instr_in_called_computation`.
      auto it = original_instruction_name_by_optimized_instruction_name.find(
          instr_in_called_computation->name());
      if (it == original_instruction_name_by_optimized_instruction_name.end()) {
        // If not found in the direct map, check if it's a call-like instruction
        // whose mapping is already in call_map.
        auto call_map_it = call_map.find(instr_in_called_computation->name());
        if (call_map_it == call_map.end() || call_map_it->second.empty()) {
          continue;  // No original instruction found for this one.
        }
        original_instr_name = call_map_it->second.front().instruction_name;
      } else {
        original_instr_name = it->second;
      }

      // Find the unique caller of the computation containing
      // `original_instr_name`.
      auto unique_caller_it =
          original_unique_caller_by_contained_instruction_name.find(
              original_instr_name);
      if (unique_caller_it !=
          original_unique_caller_by_contained_instruction_name.end()) {
        call_map[optimized_instr.name()] = {
            ScopeInstruction::FromString(unique_caller_it->second)};
        return;
      }
    }
  }
}

bool AreCallersEquivalent(
    absl::string_view opt_caller, absl::string_view orig_caller,
    const absl::flat_hash_map<std::string, std::vector<ScopeInstruction>>&
        call_map) {
  if (opt_caller == "<root>") {
    return orig_caller == "<root>";
  }
  if (orig_caller == "<root>") {
    return opt_caller == "<root>";
  }
  auto it = call_map.find(opt_caller);
  return it != call_map.end() && !it->second.empty() &&
         it->second.back().instruction_name == orig_caller;
}
}  // namespace
/* static */ absl::StatusOr<
    std::pair<std::unique_ptr<OriginalTensorSummaryCalculator>,
              OriginalTensorSummaryCalculator::CreationMetrics>>
OriginalTensorSummaryCalculator::Create(
    HloModule* optimized_module, HloModule* original_module,
    OriginalTensorSummaryCallback&& on_original_tensor_summary_ready,
    std::optional<::absl::Duration> log_interval) {
  CreationMetrics creation_metrics;
  absl::flat_hash_map<TensorKey, std::vector<int64_t>>
      optimized_tensor_dimensions;
  absl::flat_hash_map<std::string, std::vector<ScopeInstruction>> call_map;
  absl::flat_hash_map<TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>
      original_tensor_by_optimized_tensor_key;

  absl::flat_hash_map<OriginalArray,
                      std::vector<std::pair<OriginalArray, const HloModule*>>>
      placeholder_to_recoverables;
  for (const auto& [goal, recovery_pair] :
       optimized_module->original_value_recovery_table()) {
    placeholder_to_recoverables[recovery_pair.first].push_back(
        {goal, recovery_pair.second.get()});
  }

  int64_t total_ovp_placeholders = 0;
  for (HloComputation* comp : optimized_module->computations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->original_value() == nullptr) {
        continue;
      }
      for (const auto& [shape_idx, original_array_opt] :
           instr->original_value()->original_arrays()) {
        if (original_array_opt.has_value() &&
            absl::StrContains(original_array_opt->instruction_name, "__ovp")) {
          total_ovp_placeholders++;
        }
      }
    }
  }

  {
    ProgressReporter progress_reporter("Building transformation chains",
                                       total_ovp_placeholders,
                                       /*use_percent=*/false, log_interval);
    for (HloComputation* comp : optimized_module->computations()) {
      for (HloInstruction* instr : comp->instructions()) {
        if (instr->original_value() == nullptr) {
          continue;
        }
        for (const auto& [shape_idx, original_array_opt] :
             instr->original_value()->original_arrays()) {
          if (!original_array_opt.has_value()) {
            continue;
          }
          const OriginalArray& oa = *original_array_opt;
          TensorKey optimized_key{
              /*instruction_name=*/std::string(instr->name()),
              /*shape_index=*/shape_idx};
          if (absl::StrContains(oa.instruction_name, "__ovp")) {
            progress_reporter.Report();
            std::vector<OriginalTensorInfo> results;
            absl::flat_hash_set<OriginalArray> visited_placeholders;
            BuildTransformationChain(oa, placeholder_to_recoverables,
                                     visited_placeholders, results);
            for (const auto& info : results) {
              original_tensor_by_optimized_tensor_key[optimized_key].push_back(
                  info);
            }
          } else {
            absl::StatusOr<RelativeScopedTensorKey> relative_key_or =
                ParseInstructionNameAsRelativeScopedTensorKey(
                    oa.instruction_name, oa.shape_index);
            if (!relative_key_or.ok()) {
              LOG(ERROR) << "Failed to parse original array instruction name '"
                         << oa.instruction_name << "' on instruction "
                         << instr->name() << ": " << relative_key_or.status();
              continue;
            }
            original_tensor_by_optimized_tensor_key[optimized_key].push_back(
                {/*original_scoped_tensor_key=*/*std::move(relative_key_or),
                 /*tensor_transformation=*/nullptr});
          }
        }
      }
    }
  }

  absl::flat_hash_map<std::string, std::string>
      original_instruction_name_by_optimized_instruction_name;

  // NOLINTNEXTLINE
  for (const auto& [key, infos] : original_tensor_by_optimized_tensor_key) {
    creation_metrics.original_module_recoverable_tensor_count += infos.size();
    for (const auto& info : infos) {
      creation_metrics.recoverable_tensor_keys.insert(
          info.original_scoped_tensor_key.tensor_key);
    }
    if (!infos.empty()) {
      // We use the first element of `infos` because all original tensor keys
      // associated with the same optimized tensor key will share the same
      // containing computation in the original module. This is because original
      // value tracking does not cross computation boundaries.

      auto& first_original_key = infos[0].original_scoped_tensor_key;
      if (first_original_key.scope_instructions.empty()) {
        // If there are no scope instructions, it means the original tensor
        // was not inside a call-like operation (e.g., call, while). The
        // original instruction name is simply the one in the tensor_key.
        original_instruction_name_by_optimized_instruction_name
            [key.instruction_name] =
                first_original_key.tensor_key.instruction_name;
      } else {
        // If there are scope instructions, the original tensor was inside
        // one or more call-like operations. The *first* scope instruction
        // represents the outermost call-like instruction in the original
        // module that contains the tensor. For example, if an original
        // instruction `call.1/add.2` is inlined to become `add.3` in the
        // optimized module, the `scope_instructions` would start with `call.1`.
        // When mapping a caller of the computation containing `add.3`, we
        // want to map it to the caller of the computation containing `call.1`,
        // not to `call.1` itself. Thus, we take the name of the first
        // scope instruction as the effective "containing" instruction in the
        // original module.
        original_instruction_name_by_optimized_instruction_name
            [key.instruction_name] =
                first_original_key.scope_instructions.front().instruction_name;
      }
    }
  }

  absl::flat_hash_map<std::string, std::string>
      original_unique_caller_by_contained_instruction_name =
          GetUniqueCallLikeInstructionNameByContainedInstructionName(
              *original_module);

  int64_t call_like_instr_count = 0;
  for (HloComputation* comp : optimized_module->MakeComputationPostOrder()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (IsCallLike(*instr)) {
        call_like_instr_count++;
      }
    }
  }

  // This reverse topological sort is needed to ensure that call_map is
  // populated from leaf to root, so that in
  // `PopulateCallMapForCallLikeInstruction` where we populate the map through
  // heuristics, we can leverage the call map containing the inner call-like
  // instructions when populating the map for the outer call-like instructions.
  {
    ProgressReporter progress_reporter("Populating call map",
                                       call_like_instr_count,
                                       /*use_percent=*/false, log_interval);
    for (HloComputation* comp : optimized_module->MakeComputationPostOrder()) {
      for (HloInstruction* instr : comp->instructions()) {
        creation_metrics.optimized_module_tensor_count +=
            ShapeUtil::GetLeafCount(instr->shape());
        bool has_original_array = false;
        if (instr->original_value()) {
          for (const auto& [shape_idx, original_array_opt] :
               instr->original_value()->original_arrays()) {
            if (original_array_opt.has_value()) {
              has_original_array = true;
              const Shape& subshape =
                  ShapeUtil::GetSubshape(instr->shape(), shape_idx);
              if (subshape.IsArray()) {
                TensorKey key{/*instruction_name=*/std::string(instr->name()),
                              /*shape_index=*/shape_idx};
                optimized_tensor_dimensions[key] = std::vector<int64_t>(
                    subshape.dimensions().begin(), subshape.dimensions().end());
              }
            }
          }
        }

        if (IsCallLike(*instr)) {
          progress_reporter.Report();
          creation_metrics.optimized_module_call_like_instr_count++;
          if (has_original_array) {
            creation_metrics
                .optimized_module_call_like_instr_with_original_value_count++;
          }
          PopulateCallMapForCallLikeInstruction(
              *instr, call_map,
              original_unique_caller_by_contained_instruction_name,
              original_instruction_name_by_optimized_instruction_name);
        }
      }
    }
  }
  creation_metrics.optimized_module_tensor_with_original_array_count =
      optimized_tensor_dimensions.size();

  absl::flat_hash_map<std::string, std::string>
      optimized_unique_caller_by_contained_instruction_name =
          GetUniqueCallLikeInstructionNameByContainedInstructionName(
              *optimized_module);

  // Reconcile call map.
  // The call map is populated using original value tracking and heuristics.
  // The heuristic-based population can be incorrect for inlined call-like
  // instructions. For example, if `call.1/call.2/add.3` in the original module
  // becomes `call.3/add.4` in the optimized module (where `call.1` is inlined)
  // and `add.4` is mapped to `add.3`, the heuristic might map `call.3` to
  // `call.2` because `call.2` is the caller of the computation containing
  // `add.3`.
  // This reconciliation step corrects such inaccuracies by comparing the caller
  // of the optimized instruction with the caller of the mapped original
  // instruction. If they do not match, it prepends the caller of the original
  // instruction to the mapped scope instructions and repeats the comparison
  // until the callers match or the root is reached.
  absl::flat_hash_set<std::string> unreconciled_keys;
  {
    ProgressReporter progress_reporter("Reconciling call map", call_map.size(),
                                       /*use_percent=*/false, log_interval);
    // NOLINTNEXTLINE
    for (auto& [opt_instr_name, scope_instructions] : call_map) {
      progress_reporter.Report();
      if (scope_instructions.empty()) {
        continue;
      }
      constexpr int kMaxPrepends = 100;  // Protection against infinite loop.
      bool reconciled = false;
      for (int i = 0; i < kMaxPrepends; ++i) {
        auto opt_caller_it =
            optimized_unique_caller_by_contained_instruction_name.find(
                opt_instr_name);
        if (opt_caller_it ==
            optimized_unique_caller_by_contained_instruction_name.end()) {
          break;  // Cannot find opt caller, cannot reconcile.
        }
        const std::string& opt_caller = opt_caller_it->second;

        const std::string& orig_instr_name =
            scope_instructions.front().instruction_name;
        auto orig_caller_it =
            original_unique_caller_by_contained_instruction_name.find(
                orig_instr_name);
        if (orig_caller_it ==
            original_unique_caller_by_contained_instruction_name.end()) {
          break;  // Cannot find orig caller, cannot reconcile.
        }
        const std::string& orig_caller = orig_caller_it->second;

        if (AreCallersEquivalent(opt_caller, orig_caller, call_map)) {
          reconciled = true;
          break;  // Callers match, reconciliation for this entry is done.
        }

        // If callers do not match, prepend orig_caller to scope instructions.
        scope_instructions.insert(scope_instructions.begin(),
                                  ScopeInstruction::FromString(orig_caller));
        if (orig_caller == "<root>") {
          // We prepended root but it didn't match opt_caller in
          // AreCallersEquivalent. This means opt_caller is not "<root>", so
          // this is unreconcilable.
          break;
        }
      }
      if (!reconciled) {
        unreconciled_keys.insert(opt_instr_name);
      }
    }
  }

  // NOLINTNEXTLINE
  for (const std::string& key : unreconciled_keys) {
    LOG(WARNING)
        << "Failed to reconcile call map entry for optimized instruction '"
        << key << "'. Removing it from call map.";
    call_map.erase(key);
  }

  return std::make_pair(
      std::make_unique<OriginalTensorSummaryCalculator>(
          std::make_shared<
              const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>(
              std::move(optimized_tensor_dimensions)),
          std::make_shared<const absl::flat_hash_map<
              std::string, std::vector<ScopeInstruction>>>(std::move(call_map)),
          std::make_shared<const absl::flat_hash_map<
              TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>(
              std::move(original_tensor_by_optimized_tensor_key)),
          std::move(on_original_tensor_summary_ready), log_interval),
      creation_metrics);
}

namespace {
using FloatBlockSummary = ::xla::comparison::FloatBlockSummary;
using FloatSummary = ::xla::comparison::FloatSummary;
using DimSplitSpec = ::xla::comparison::DimSplitSpec;

// Combines block summaries from multiple shards into a single summary for the
// unsharded tensor. This function aims to preserve the block structure from
// shard-level summaries. If a dimension is partitioned by sharding, the block
// structure along that dimension is extended to cover the full tensor. If a
// dimension is not partitioned by sharding (i.e., it is replicated across
// shards), the corresponding block summaries from each shard would be identical
// and hence the first shard's block summaries are returned.
//
// For example, consider a [10, 20] tensor sharded into 2 shards along the
// second dimension, and replicated 4 times along the first dimension,
//  resulting in each shard having shape [10, 10].
// In each shard, the summaries are split the first dimension (which is
// replicated) into 5 blocks. Now the final summary will have totally 5 * 2 = 10
// blocks, where summaries from one of the 4 repilcas is used (since the other
// 3 replicas are identical). And the 5 blocks from each shard are concatenated
// along the second dimension with each pair of blocks perfectly aligned.
// The block indices will be adjusted based on the shard's position.
//
// Note that when the sharding includes multiple replicas, the input
// `shard_summaries` should only contain summaries for shards in the first
// replica. All the other replicas with identical data have been dropped in
// an earlier stage.
//
// If the sharding contains a manual subgroup dimension, shard summaries are
// grouped by their manual subgroup index. Each group is combined into a
// `FloatSummary`, and the function returns a vector containing one summary for
// each manual subgroup. If there is no manual sharding, a vector containing a
// single combined `FloatSummary` is returned.
absl::StatusOr<std::vector<::xla::comparison::FloatSummary>>
CombineShardSummaries(
    absl::Span<const OriginalTensorSummaryCalculator::ShardTensorSummary>
        shard_summaries,
    const tensor_transformation::Unshard& unshard_transform,
    absl::Span<const int64_t> optimized_tensor_dims) {
  if (shard_summaries.empty()) {
    return absl::InvalidArgumentError("No shard summaries provided.");
  }
  // If the tensor is replicated or tile-maximal (sharded on one device),
  // there should be only one shard summary, which is the summary of the
  // entire tensor.
  if (unshard_transform.sharding.IsReplicatedOrSingleDevice()) {
    if (shard_summaries.size() != 1) {
      return absl::InternalError(
          "Replicated or tile-maximal sharding should only have one shard "
          "summary.");
    }
    return std::vector<FloatSummary>{shard_summaries[0].summary};
  }

  // If there are multiple shards, all shard summaries must have the same
  // split spec, i.e. they must be split in the same way.
  if (shard_summaries.size() > 1) {
    for (int64_t i = 1; i < shard_summaries.size(); ++i) {
      if (shard_summaries[i].summary.split_spec !=
          shard_summaries[0].summary.split_spec) {
        return absl::InvalidArgumentError(
            "All shard summaries must have the same split spec.");
      }
    }
  }

  // Build a map from device ID to tile index in the tile assignment.
  absl::flat_hash_map<int64_t, std::vector<int64_t>> device_to_tile_index;
  unshard_transform.sharding.EachTile(
      [&](absl::Span<const int64_t> index, int64_t device) {
        device_to_tile_index[device] = {index.begin(), index.end()};
      });

  // If manual sharding is used, identify the manual dimension.
  int64_t manual_dim = -1;
  if (unshard_transform.sharding.IsManualSubgroup()) {
    manual_dim = unshard_transform.sharding.SubgroupManualDim();
    CHECK_NE(manual_dim, -1);
  }

  // If there is a manual dimension, group shard summaries by manual subgroup
  // index. Otherwise, all summaries belong to a single group.
  int64_t num_manual_groups = 1;
  if (manual_dim != -1) {
    num_manual_groups = unshard_transform.sharding.dimension(manual_dim);
  }

  std::vector<std::vector<OriginalTensorSummaryCalculator::ShardTensorSummary>>
      grouped_summaries(num_manual_groups);
  if (manual_dim != -1) {
    for (const auto& summary : shard_summaries) {
      grouped_summaries[device_to_tile_index.at(
                            summary.logical_shard_id)[manual_dim]]
          .push_back(summary);
    }
  } else {
    grouped_summaries[0] = {shard_summaries.begin(), shard_summaries.end()};
  }

  std::vector<FloatSummary> result_summaries;
  for (const auto& group : grouped_summaries) {
    if (group.empty()) {
      continue;
    }

    FloatSummary combined_summary;
    // Build a map from dimension index to {block_count, split_spec_index}
    // for dimensions that are split within each shard.
    absl::flat_hash_map<int64_t, std::pair<int64_t, int64_t>>
        shard_split_dim_info;
    for (int i = 0; i < group[0].summary.split_spec.size(); ++i) {
      const auto& spec = group[0].summary.split_spec[i];
      shard_split_dim_info[spec.dim_index] = {spec.block_count, i};
    }

    // Combine split specs from all shards into a single split spec for the
    // unsharded tensor.
    std::vector<DimSplitSpec> combined_split_spec_vec;
    for (int64_t i = 0; i < optimized_tensor_dims.size(); ++i) {
      int64_t tile_dim = unshard_transform.sharding.dimension(i);
      if (auto it = shard_split_dim_info.find(i);
          it != shard_split_dim_info.end()) {
        // If dimension `i` is split within each shard, the total number of
        // blocks is block_count_in_shard * number_of_shards_in_dim_i.
        combined_split_spec_vec.push_back(
            {/*dim_index=*/i, /*block_count=*/it->second.first * tile_dim});
      } else if (tile_dim > 1) {
        // If dimension `i` is not split within each shard but sharded over
        // multiple devices, each shard becomes a block.
        combined_split_spec_vec.push_back(
            {/*dim_index=*/i, /*block_count=*/tile_dim});
      }
    }
    std::sort(combined_split_spec_vec.begin(), combined_split_spec_vec.end(),
              [](const DimSplitSpec& a, const DimSplitSpec& b) {
                return a.dim_index < b.dim_index;
              });
    combined_summary.split_spec = combined_split_spec_vec;

    // Combine block summaries from all shards.
    for (const auto& shard_summary : group) {
      const std::vector<int64_t>& tile_indices =
          device_to_tile_index.at(shard_summary.logical_shard_id);
      for (const auto& block_summary : shard_summary.summary.block_summaries) {
        FloatBlockSummary new_block_summary = block_summary;
        new_block_summary.block_indices.resize(
            combined_summary.split_spec.size());
        // For each block in the shard summary, calculate its new block index in
        // the combined summary.
        for (int i = 0; i < combined_summary.split_spec.size(); ++i) {
          int64_t dim_index = combined_summary.split_spec[i].dim_index;
          int64_t tile_index_for_dim = tile_indices[dim_index];
          if (auto it = shard_split_dim_info.find(dim_index);
              it != shard_split_dim_info.end()) {
            auto [block_count_in_shard, split_spec_idx] = it->second;
            new_block_summary.block_indices[i] =
                tile_index_for_dim * block_count_in_shard +
                block_summary.block_indices[split_spec_idx];
          } else {
            new_block_summary.block_indices[i] = tile_index_for_dim;
          }
        }
        combined_summary.block_summaries.push_back(new_block_summary);
      }
    }
    std::sort(combined_summary.block_summaries.begin(),
              combined_summary.block_summaries.end(),
              [](const FloatBlockSummary& a, const FloatBlockSummary& b) {
                return a.block_indices < b.block_indices;
              });
    result_summaries.push_back(combined_summary);
  }
  return result_summaries;
}
}  // namespace

OriginalTensorSummaryCalculator::OriginalTensorSummaryCalculator(
    std::shared_ptr<const absl::flat_hash_map<TensorKey, std::vector<int64_t>>>
        optimized_tensor_dimensions,
    std::shared_ptr<
        const absl::flat_hash_map<std::string, std::vector<ScopeInstruction>>>
        call_map,
    std::shared_ptr<const absl::flat_hash_map<
        TensorKey, absl::InlinedVector<OriginalTensorInfo, 1>>>
        original_tensor_by_optimized_tensor_key,
    OriginalTensorSummaryCallback&& on_original_tensor_summary_ready,
    std::optional<::absl::Duration> log_interval)
    : optimized_tensor_dimensions_(std::move(optimized_tensor_dimensions)),
      call_map_(std::move(call_map)),
      original_tensor_by_optimized_tensor_key_(
          std::move(original_tensor_by_optimized_tensor_key)),
      on_original_tensor_summary_ready_(
          std::move(on_original_tensor_summary_ready)),
      log_interval_(log_interval) {
  PopulateExpectedShardIds();
}

void OriginalTensorSummaryCalculator::PopulateExpectedShardIds() {
  for (const auto& [optimized_key, original_infos] :
       // NOLINTNEXTLINE
       *original_tensor_by_optimized_tensor_key_) {
    for (const auto& original_info : original_infos) {
      const tensor_transformation::TensorTransformation* transform =
          original_info.tensor_transformation.get();
      const tensor_transformation::Unshard* unshard = nullptr;
      while (transform) {
        if (const auto* u =
                std::get_if<tensor_transformation::Unshard>(transform)) {
          unshard = u;
          break;
        }
        transform = std::visit(
            [](const auto& t) { return t.continuation.get(); }, *transform);
      }

      if (unshard) {
        absl::flat_hash_set<int64_t> shard_ids;
        HloSharding tile_based_sharding =
            unshard->sharding.UseNamedShardingLeaf()
                ? xla::HloSharding::V3ToV2Sharding(
                      unshard->sharding.named_sharding())
                : unshard->sharding;
        if (!tile_based_sharding.IsReplicated() &&
            !tile_based_sharding.IsReplicatedOrSingleDevice()) {
          if (tile_based_sharding.IsManualSubgroup()) {
            const int64_t replication_dim =
                tile_based_sharding.SubgroupReplicationDim();
            tile_based_sharding.EachTile(
                [&](absl::Span<const int64_t> index, int64_t device) {
                  if (replication_dim == -1 || index[replication_dim] == 0) {
                    shard_ids.insert(device);
                  }
                });
          } else {
            const auto& device_assignment =
                tile_based_sharding.device_assignment();
            shard_ids.insert(device_assignment.array().begin(),
                             device_assignment.array().end());
          }
        } else {
          // Replicated or single device, only shard 0 is needed.
          shard_ids.insert(0);
        }
        expected_shard_ids_by_corresponding_tensor_key_pair_[std::make_pair(
            optimized_key, original_info.original_scoped_tensor_key)] =
            std::move(shard_ids);
      }
    }
  }
}

absl::StatusOr<AbsoluteScopedTensorKey>
OriginalTensorSummaryCalculator::ConstructOriginalTensorKey(
    absl::Span<const ScopeInstruction> optimized_root_scopes,
    const RelativeScopedTensorKey& relative_original_key) const {
  std::vector<ScopeInstruction> combined_scopes;
  for (const auto& scope : optimized_root_scopes) {
    if (auto it = call_map_->find(scope.instruction_name);
        it != call_map_->end()) {
      const std::vector<ScopeInstruction>& mapped_original_scopes = it->second;
      combined_scopes.insert(combined_scopes.end(),
                             mapped_original_scopes.begin(),
                             mapped_original_scopes.end());
      if (mapped_original_scopes.empty()) {
        continue;
      }
      if (scope.iteration_index != 0) {
        // If there is only one mapped original scope instruction, we assume it
        // is the while loop and set its iteration index to the actual iteration
        // index at runtime.
        if (mapped_original_scopes.size() == 1) {
          combined_scopes.back().iteration_index = scope.iteration_index;
          continue;
        }
        bool found_while_loop = false;
        for (int64_t i = 0; i < mapped_original_scopes.size(); ++i) {
          auto& original_scope =
              combined_scopes[combined_scopes.size() - i - 1];
          // -2 indicates the iteration index of the scope instruction should be
          // replaced by the actual iteration index at runtime.
          if (original_scope.iteration_index == -2) {
            original_scope.iteration_index = scope.iteration_index;
            found_while_loop = true;
            break;
          }
        }

        if (!found_while_loop) {
          // Fallback 1: Check if any of the added scopes start with "while.".
          for (int64_t i = 0; i < mapped_original_scopes.size(); ++i) {
            auto& original_scope =
                combined_scopes[combined_scopes.size() -
                                mapped_original_scopes.size() + i];
            if (absl::StartsWith(original_scope.instruction_name, "while.")) {
              original_scope.iteration_index = scope.iteration_index;
              found_while_loop = true;
              LOG(WARNING)
                  << "While loop not found for optimized scope instruction: "
                  << scope.instruction_name
                  << ". The mapped original scope instructions are: "
                  << absl::StrJoin(
                         mapped_original_scopes, "/",
                         [](std::string* out, const ScopeInstruction& s) {
                           out->append(s.ToString());
                         })
                  << ". Using instruction " << original_scope.instruction_name
                  << " as the while instruction because it looks like a while "
                     "instruction.";
              break;
            }
          }
        }

        if (!found_while_loop) {
          // Fallback 2: If we don't find a while loop, we just set the
          // iteration index to the last scope's iteration index.
          LOG(ERROR) << "While loop not found for optimized scope instruction: "
                     << scope.instruction_name
                     << ". The mapped original scope instructions are: "
                     << absl::StrJoin(
                            mapped_original_scopes, "/",
                            [](std::string* out, const ScopeInstruction& s) {
                              out->append(s.ToString());
                            })
                     << ". Treating the last scope instruction as the while "
                        "instruction.";
          combined_scopes.back().iteration_index = scope.iteration_index;
        }
      }
    } else {
      if (!missing_call_map_instr_names_.contains(scope.instruction_name)) {
        LOG(WARNING)
            << "Failed to find original call-like instruction for optimized "
               "call-like instruction '"
            << scope.instruction_name
            << "' when trying to map the callstack for original instruction '"
            << relative_original_key.tensor_key.instruction_name
            << "'. Will guess the corresponding call-like instruction for "
               "comparison.";
        missing_call_map_instr_names_.insert(
            std::string(scope.instruction_name));
      }
      combined_scopes.push_back(ScopeInstruction::Create(
          // Add a question mark to the end of the instruction name to indicate
          // that this call-like instruction is not found in the call map and
          // the instruction name is from the optimized HLO module, rather than
          // the original HLO module.
          absl::StrCat(scope.instruction_name, "?"), scope.iteration_index));
    }
  }
  combined_scopes.insert(combined_scopes.end(),
                         relative_original_key.scope_instructions.begin(),
                         relative_original_key.scope_instructions.end());
  return AbsoluteScopedTensorKey{
      /*scope_instructions=*/std::move(combined_scopes),
      /*tensor_key=*/relative_original_key.tensor_key};
}

absl::Status OriginalTensorSummaryCalculator::ProcessCompletedShards(
    const AbsoluteScopedTensorKey& optimized_tensor_position,
    const AbsoluteScopedTensorKey& original_tensor_key,
    const OriginalTensorInfo& original_tensor_info,
    const std::vector<ShardTensorSummary>& shard_summaries) {
  if (shard_summaries.empty()) {
    return absl::InternalError("No shard summaries to process.");
  }

  const tensor_transformation::TensorTransformation* transform =
      original_tensor_info.tensor_transformation.get();
  std::vector<::xla::comparison::FloatSummary> summaries;
  std::shared_ptr<const tensor_transformation::TensorTransformation>
      pending_transformation;

  auto opt_dims_it =
      optimized_tensor_dimensions_->find(optimized_tensor_position.tensor_key);
  if (opt_dims_it == optimized_tensor_dimensions_->end()) {
    return absl::NotFoundError(
        "Optimized tensor dimensions not found for key: " +
        optimized_tensor_position.tensor_key.instruction_name);
  }
  const std::vector<int64_t>& optimized_dims = opt_dims_it->second;

  const tensor_transformation::Unshard* unshard = nullptr;
  const tensor_transformation::TensorTransformation* transform_it = transform;
  while (transform_it) {
    if (const auto* u =
            std::get_if<tensor_transformation::Unshard>(transform_it)) {
      unshard = u;
      break;
    }
    transform_it = std::visit(
        [](const auto& t) { return t.continuation.get(); }, *transform_it);
  }

  std::vector<int64_t> current_shape = optimized_dims;
  if (unshard == nullptr) {
    if (shard_summaries.size() != 1) {
      return absl::InternalError(
          "Multiple shards found without an Unshard transformation.");
    }
    summaries.push_back(shard_summaries[0].summary);
    pending_transformation = original_tensor_info.tensor_transformation;
  } else {
    std::vector<ShardTensorSummary> current_shard_summaries = shard_summaries;
    transform_it = transform;
    while (transform_it) {
      if (const auto* reshape =
              std::get_if<tensor_transformation::Reshape>(transform_it)) {
        std::vector<ShardTensorSummary> reshaped_summaries;
        reshaped_summaries.reserve(current_shard_summaries.size());
        for (const auto& shard_summary : current_shard_summaries) {
          reshaped_summaries.push_back(
              {/*logical_shard_id=*/shard_summary.logical_shard_id,
               /*summary=*/ApplyReshapeToSummary(
                   shard_summary.summary, absl::MakeSpan(current_shape),
                   absl::MakeSpan(reshape->output_dimensions))});
        }
        current_shard_summaries = std::move(reshaped_summaries);
        current_shape = reshape->output_dimensions;
        transform_it = reshape->continuation.get();
      } else if (const auto* u = std::get_if<tensor_transformation::Unshard>(
                     transform_it)) {
        if (u != unshard) {
          return absl::InternalError("Multiple Unshard transformations found.");
        }
        ASSIGN_OR_RETURN(
            summaries,
            CombineShardSummaries(current_shard_summaries, *u, current_shape));
        current_shape = u->original_dimensions;
        pending_transformation = u->continuation;
        transform_it = nullptr;  // Stop processing.
      } else {
        return absl::InternalError("Unknown transformation type.");
      }
    }
  }

  OriginalTensorSummary original_summary = {/*dimensions=*/current_shape,
                                            /*summaries=*/summaries};

  const tensor_transformation::TensorTransformation* pending_transform =
      pending_transformation.get();
  std::shared_ptr<const tensor_transformation::TensorTransformation>
      remaining_transformation;
  if (pending_transform) {
    remaining_transformation =
        std::shared_ptr<const tensor_transformation::TensorTransformation>(
            original_tensor_info.tensor_transformation, pending_transform);
  }

  VLOG(1) << "\nProcessCompletedShards: " << original_tensor_key.ToString()
          << "\nremaining transformation: "
          << (remaining_transformation ? absl::StrCat(*remaining_transformation)
                                       : "null")
          << "\noriginal tensor summary: " << original_summary.ToDebugString()
          << "============================================================\n";
  return on_original_tensor_summary_ready_(
      original_tensor_key, remaining_transformation, original_summary);
}

absl::Status OriginalTensorSummaryCalculator::ProcessShardSummary(
    const AbsoluteScopedTensorKey& optimized_tensor_position,
    const ShardTensorSummary& tensor_shard_summary) {
  VLOG(1) << "\nProcessShardSummary: " << optimized_tensor_position.ToString()
          << "\ntensor shard summary: " << tensor_shard_summary.ToDebugString()
          << "------------------------------------------------------------\n";
  processing_metrics_.received_optimized_tensor_shard_count++;
  auto it = original_tensor_by_optimized_tensor_key_->find(
      optimized_tensor_position.tensor_key);
  if (it == original_tensor_by_optimized_tensor_key_->end()) {
    return absl::NotFoundError(absl::StrCat(
        "No original tensor info found for optimized tensor position: ",
        optimized_tensor_position));
  }
  absl::Span<const OriginalTensorInfo> original_tensor_infos = it->second;
  processing_metrics_.processed_original_tensor_shard_count +=
      original_tensor_infos.size();

  for (const auto& original_tensor_info : original_tensor_infos) {
    ASSIGN_OR_RETURN(AbsoluteScopedTensorKey original_tensor_key,
                     ConstructOriginalTensorKey(
                         optimized_tensor_position.scope_instructions,
                         original_tensor_info.original_scoped_tensor_key));
    if (completed_tensor_keys_.find(original_tensor_key) !=
        completed_tensor_keys_.end()) {
      continue;
    }

    auto expected_shards =
        expected_shard_ids_by_corresponding_tensor_key_pair_.find(
            std::make_pair(optimized_tensor_position.tensor_key,
                           original_tensor_info.original_scoped_tensor_key));
    if (expected_shards ==
        expected_shard_ids_by_corresponding_tensor_key_pair_.end()) {
      // No unshard, process immediately
      RETURN_IF_ERROR(
          ProcessCompletedShards(optimized_tensor_position, original_tensor_key,
                                 original_tensor_info, {tensor_shard_summary}));
      completed_tensor_keys_.emplace(original_tensor_key);
      processing_metrics_.completed_original_tensor_count++;
      continue;
    }
    // Collect shards
    auto& received_tensor_shards =
        received_tensor_shards_by_optimized_key_[optimized_tensor_position];
    if (absl::c_none_of(received_tensor_shards,
                        [&](const ShardTensorSummary& received_shard) {
                          return received_shard.logical_shard_id ==
                                 tensor_shard_summary.logical_shard_id;
                        })) {
      received_tensor_shards.push_back(tensor_shard_summary);
    }

    // Only continue process shards if all the required shards for this
    // original tensor are collected.
    if (absl::c_all_of(expected_shards->second, [&](int64_t expected_shard_id) {
          return absl::c_any_of(received_tensor_shards,
                                [&](const ShardTensorSummary& received_shard) {
                                  return received_shard.logical_shard_id ==
                                         expected_shard_id;
                                });
        })) {
      RETURN_IF_ERROR(
          ProcessCompletedShards(optimized_tensor_position, original_tensor_key,
                                 original_tensor_info, received_tensor_shards));
      completed_tensor_keys_.emplace(original_tensor_key);
      processing_metrics_.completed_original_tensor_count++;
    }
  }

  bool all_completed = true;
  for (const auto& original_tensor_info : original_tensor_infos) {
    ASSIGN_OR_RETURN(AbsoluteScopedTensorKey original_tensor_key,
                     ConstructOriginalTensorKey(
                         optimized_tensor_position.scope_instructions,
                         original_tensor_info.original_scoped_tensor_key));
    if (completed_tensor_keys_.find(original_tensor_key) ==
        completed_tensor_keys_.end()) {
      all_completed = false;
      break;
    }
  }
  if (all_completed) {
    received_tensor_shards_by_optimized_key_.erase(optimized_tensor_position);
    processing_metrics_.completed_optimized_tensor_count++;
  }

  return absl::OkStatus();
}

std::unique_ptr<OriginalTensorSummaryCalculator>
OriginalTensorSummaryCalculator::CloneWithCallback(
    OriginalTensorSummaryCallback&& on_original_tensor_summary_ready) const {
  return std::make_unique<OriginalTensorSummaryCalculator>(
      optimized_tensor_dimensions_, call_map_,
      original_tensor_by_optimized_tensor_key_,
      std::move(on_original_tensor_summary_ready));
}

OriginalTensorSummaryCalculator::ProcessingMetrics
OriginalTensorSummaryCalculator::GetProcessingMetrics() const {
  ProcessingMetrics result = processing_metrics_;
  result.incomplete_optimized_tensor_count =
      received_tensor_shards_by_optimized_key_.size();
  result.incomplete_original_tensor_count = 0;
  for (const auto& [optimized_key, shards] :
       // NOLINTNEXTLINE
       received_tensor_shards_by_optimized_key_) {
    auto it = original_tensor_by_optimized_tensor_key_->find(
        optimized_key.tensor_key);
    if (it == original_tensor_by_optimized_tensor_key_->end()) {
      continue;
    }
    for (const auto& original_info : it->second) {
      absl::StatusOr<AbsoluteScopedTensorKey> original_tensor_key =
          ConstructOriginalTensorKey(optimized_key.scope_instructions,
                                     original_info.original_scoped_tensor_key);
      if (!original_tensor_key.ok()) {
        continue;
      }
      if (completed_tensor_keys_.find(*original_tensor_key) ==
          completed_tensor_keys_.end()) {
        result.incomplete_original_tensor_count++;
      }
    }
  }
  return result;
}

std::string OriginalTensorSummaryCalculator::DumpHloDerivedData() const {
  std::string result;
  absl::StrAppend(&result, "Optimized Tensor Dimensions:\n");
  std::vector<TensorKey> sorted_optimized_keys;
  sorted_optimized_keys.reserve(optimized_tensor_dimensions_->size());
  // NOLINTNEXTLINE
  for (const auto& [key, dims] : *optimized_tensor_dimensions_) {
    sorted_optimized_keys.push_back(key);
  }
  std::sort(sorted_optimized_keys.begin(), sorted_optimized_keys.end(),
            [](const TensorKey& a, const TensorKey& b) {
              if (a.instruction_name != b.instruction_name) {
                return a.instruction_name < b.instruction_name;
              }
              return a.shape_index < b.shape_index;
            });
  for (const auto& key : sorted_optimized_keys) {
    const auto& dims = optimized_tensor_dimensions_->at(key);
    absl::StrAppend(&result, "  ", key.ToString(), ": [",
                    absl::StrJoin(dims, ", "), "]\n");
  }

  absl::StrAppend(&result, "\nCall Map:\n");
  std::vector<std::string> sorted_call_keys;
  // NOLINTNEXTLINE
  for (const auto& [instr_name, scopes] : *call_map_) {
    sorted_call_keys.push_back(instr_name);
  }
  std::sort(sorted_call_keys.begin(), sorted_call_keys.end());
  for (const auto& instr_name : sorted_call_keys) {
    const auto& scopes = call_map_->at(instr_name);
    std::vector<std::string> scope_strs;
    scope_strs.reserve(scopes.size());
    for (const auto& scope : scopes) {
      scope_strs.push_back(scope.ToString());
    }
    absl::StrAppend(&result, "  ", instr_name, ": [",
                    absl::StrJoin(scope_strs, "/"), "]\n");
  }

  absl::StrAppend(&result, "\nOriginal Tensor by Optimized Tensor Key:\n");
  std::vector<TensorKey> sorted_original_keys;
  sorted_original_keys.reserve(
      original_tensor_by_optimized_tensor_key_->size());
  // NOLINTNEXTLINE
  for (const auto& [key, infos] : *original_tensor_by_optimized_tensor_key_) {
    sorted_original_keys.push_back(key);
  }
  std::sort(sorted_original_keys.begin(), sorted_original_keys.end(),
            [](const TensorKey& a, const TensorKey& b) {
              if (a.instruction_name != b.instruction_name) {
                return a.instruction_name < b.instruction_name;
              }
              return a.shape_index < b.shape_index;
            });
  for (const auto& key : sorted_original_keys) {
    const auto& infos = original_tensor_by_optimized_tensor_key_->at(key);
    absl::StrAppend(&result, "  ", key.ToString(), ":\n");
    for (const auto& info : infos) {
      absl::StrAppend(&result, "    ",
                      info.original_scoped_tensor_key.ToString());
      if (info.tensor_transformation) {
        absl::StrAppend(&result, " via ", *info.tensor_transformation);
      } else {
        absl::StrAppend(&result, " via no transformation");
      }
      absl::StrAppend(&result, "\n");
    }
  }
  return result;
}

}  // namespace xla::numerics::comparison
