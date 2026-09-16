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

#include "xla/hlo/tools/comparison/original_tensor_summary_comparator.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_key_matcher.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/comparison/tensor_summary_util.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"
#include "xla/hlo/tools/hlo_diff/utils/bidirectional_map.h"
#include "xla/shape_util.h"

namespace xla::numerics::comparison {

using TensorTransformation = tensor_transformation::TensorTransformation;
using DimSplitSpec = ::xla::comparison::DimSplitSpec;
using FloatSummary = ::xla::comparison::FloatSummary;
using FloatBlockSummary = ::xla::comparison::FloatBlockSummary;

namespace {
// Extracts iteration indices from a ScopedTensorKey.
std::vector<int64_t> GetIterationIndices(const ScopedTensorKey& key) {
  std::vector<int64_t> indices;
  indices.reserve(key.scope_instructions.size());
  for (const auto& scope : key.scope_instructions) {
    indices.push_back(scope.iteration_index);
  }
  return indices;
}

inline bool IterationIndicesMatch(absl::Span<const int64_t> indices1,
                                  absl::Span<const int64_t> indices2) {
  if (indices1.size() != indices2.size()) {
    return false;
  }
  for (size_t i = 0; i < indices1.size(); ++i) {
    if (indices1[i] != indices2[i] && indices1[i] != -1 && indices2[i] != -1) {
      return false;
    }
  }
  return true;
}

// Given two chains of recovering transformations, finds the outermost
// transformation that is common to both chains.
// If transformations are A->B->C and D->B->C, where C is the last
// transformation in both chains, B is the first common transformation,
// and A and D are transformations unique to each chain, this function will
// return B. If chains are completely different, returns nullptr.
std::shared_ptr<const TensorTransformation> GetCommonContinuation(
    std::shared_ptr<const TensorTransformation> baseline_tensor_key,
    std::shared_ptr<const TensorTransformation> target_tensor_key) {
  std::vector<std::shared_ptr<const TensorTransformation>> baseline_path;
  for (auto curr = baseline_tensor_key; curr != nullptr;
       curr = GetContinuation(curr.get())) {
    baseline_path.push_back(curr);
  }
  std::vector<std::shared_ptr<const TensorTransformation>> target_path;
  for (auto curr = target_tensor_key; curr != nullptr;
       curr = GetContinuation(curr.get())) {
    target_path.push_back(curr);
  }

  int i = baseline_path.size() - 1;
  int j = target_path.size() - 1;
  std::shared_ptr<const TensorTransformation> common_continuation = nullptr;
  while (i >= 0 && j >= 0 &&
         EqualsWithoutContinuation(*baseline_path[i], *target_path[j])) {
    common_continuation = baseline_path[i];
    --i;
    --j;
  }
  return common_continuation;
}
}  // namespace

absl::StatusOr<std::tuple<std::unique_ptr<OriginalTensorSummaryComparator>,
                          OriginalTensorSummaryComparator::CreationMetrics,
                          hlo_diff::HloGumgraphDiffResults>>
OriginalTensorSummaryComparator::Create(
    const HloModule* baseline_module, const HloModule* target_module,
    absl::string_view baseline_recovered_tensor_summaries_file,
    absl::string_view target_recovered_tensor_summaries_file,
    OriginalTensorSummaryComparisonCallback
        on_original_tensor_summary_comparison_ready) {
  BidirectionalMap<std::string, std::string, std::monostate> hlo_diff_bimap;
  ABSL_ASSIGN_OR_RETURN(hlo_diff::HloGumgraphDiffResults diff_results,
                   hlo_diff::ComputeDiff(*baseline_module, *target_module));
  CreationMetrics creation_metrics;
  creation_metrics.baseline_tensor_count = 0;
  for (const HloComputation* comp : baseline_module->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      creation_metrics.baseline_tensor_count +=
          ShapeUtil::GetLeafCount(instr->shape());
    }
  }
  for (const HloComputation* comp : target_module->computations()) {
    for (const HloInstruction* instr : comp->instructions()) {
      creation_metrics.target_tensor_count +=
          ShapeUtil::GetLeafCount(instr->shape());
    }
  }
  creation_metrics.unchanged_tensor_pair_count =
      diff_results.diff_result->unchanged_instructions.size();
  creation_metrics.changed_tensor_pair_count =
      diff_results.diff_result->changed_instructions.size();
  for (const auto& instructions :
       {diff_results.diff_result->unchanged_instructions,
        diff_results.diff_result->changed_instructions}) {
    // NOLINTNEXTLINE
    for (const auto& [baseline_inst, target_inst] : instructions) {
      // It doesn't make sense to compare tensors with different shapes. So
      // we just ignore such cases.
      if (ShapeUtil::CompatibleIgnoringElementType(baseline_inst->shape(),
                                                   target_inst->shape())) {
        hlo_diff_bimap.Insert(std::string(baseline_inst->name()),
                              std::string(target_inst->name()));
      }
    }
  }
  ABSL_ASSIGN_OR_RETURN(
      auto key_matcher,
      OriginalTensorSummaryKeyMatcher::Create(
          std::make_shared<
              const BidirectionalMap<std::string, std::string, std::monostate>>(
              std::move(hlo_diff_bimap)),
          baseline_recovered_tensor_summaries_file,
          target_recovered_tensor_summaries_file));
  return std::make_tuple(
      std::make_unique<OriginalTensorSummaryComparator>(
          std::move(key_matcher),
          std::move(on_original_tensor_summary_comparison_ready)),
      creation_metrics, std::move(diff_results));
}

absl::Status
OriginalTensorSummaryComparator::ProcessOriginalTensorSummaryInternal(
    AbsoluteScopedTensorKey baseline_tensor_key,
    std::shared_ptr<const TensorTransformation> baseline_pending_transformation,
    OriginalTensorSummary const* baseline_tensor_summary,
    AbsoluteScopedTensorKey target_tensor_key,
    std::shared_ptr<const TensorTransformation> target_pending_transformation,
    OriginalTensorSummary const* target_tensor_summary) {
  auto common_continuation = GetCommonContinuation(
      baseline_pending_transformation, target_pending_transformation);
  std::optional<OriginalTensorSummary> transformed_baseline_summary;
  if (baseline_tensor_summary != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        transformed_baseline_summary,
        ApplyNonUnshardTensorTransformationToSummary(
            *baseline_tensor_summary, baseline_pending_transformation.get(),
            common_continuation.get()));
  }
  std::optional<OriginalTensorSummary> transformed_target_summary;
  if (target_tensor_summary != nullptr) {
    ABSL_ASSIGN_OR_RETURN(
        transformed_target_summary,
        ApplyNonUnshardTensorTransformationToSummary(
            *target_tensor_summary, target_pending_transformation.get(),
            common_continuation.get()));
  }
  if (baseline_tensor_summary == nullptr || target_tensor_summary == nullptr) {
    return on_original_tensor_summary_comparison_ready_(
        common_continuation, baseline_tensor_key,
        transformed_baseline_summary.has_value()
            ? &*transformed_baseline_summary
            : nullptr,
        target_tensor_key,
        transformed_target_summary.has_value() ? &*transformed_target_summary
                                               : nullptr);
  }
  ABSL_ASSIGN_OR_RETURN((auto [aligned_baseline_summary, aligned_target_summary]),
                   AlignTensorSummaries(*transformed_baseline_summary,
                                        *transformed_target_summary));
  return on_original_tensor_summary_comparison_ready_(
      common_continuation, baseline_tensor_key, &aligned_baseline_summary,
      target_tensor_key, &aligned_target_summary);
}

/**static*/ OriginalTensorSummaryComparator::IndexlessScopedTensorKey
OriginalTensorSummaryComparator::GetIndexlessScopedTensorKey(
    const ScopedTensorKey& key) {
  IndexlessScopedTensorKey indexless_key;
  indexless_key.scope_instruction_names.reserve(key.scope_instructions.size());
  for (const auto& scope : key.scope_instructions) {
    indexless_key.scope_instruction_names.push_back(scope.instruction_name);
  }
  indexless_key.tensor_key = key.tensor_key;
  return indexless_key;
}

absl::Status OriginalTensorSummaryComparator::ProcessOriginalTensorSummary(
    ComparisonVariant variant,
    const AbsoluteScopedTensorKey& original_tensor_key,
    std::shared_ptr<const TensorTransformation> pending_transformation,
    const OriginalTensorSummary& original_tensor_summary) {
  if (!original_tensor_summary.dimensions.empty()) {
    if (variant == ComparisonVariant::kBaseline) {
      ++processing_metrics_.received_baseline_tensor_summaries;
    } else {
      ++processing_metrics_.received_target_tensor_summaries;
    }
  }
  std::optional<AbsoluteScopedTensorKey> key_in_other_variant =
      key_matcher_->FindMatchingKey(original_tensor_key, variant);
  if (!key_in_other_variant.has_value()) {
    if (!original_tensor_summary.dimensions.empty()) {
      if (variant == ComparisonVariant::kBaseline) {
        ++processing_metrics_.untranslatable_baseline_tensor_summaries;
      } else {
        ++processing_metrics_.untranslatable_target_tensor_summaries;
      }
    }
    return absl::OkStatus();
  }

  auto* current_variant_pending_summaries_map =
      variant == ComparisonVariant::kBaseline ? &baseline_pending_summaries_
                                              : &target_pending_summaries_;
  auto* other_variant_pending_summaries_map =
      variant == ComparisonVariant::kBaseline ? &target_pending_summaries_
                                              : &baseline_pending_summaries_;

  const auto indices_in_other_variant =
      GetIterationIndices(*key_in_other_variant);
  const auto indexless_key_in_other_variant =
      GetIndexlessScopedTensorKey(*key_in_other_variant);

  if (auto pending_it = other_variant_pending_summaries_map->find(
          indexless_key_in_other_variant);
      pending_it != other_variant_pending_summaries_map->end()) {
    PendingSummaries& pending_summaries = pending_it->second;
    bool processed = false;
    auto current_indices = GetIterationIndices(original_tensor_key);
    bool current_has_wildcard = absl::c_linear_search(current_indices, -1);

    auto process_match =
        [&](const PendingComparisonTensorSummary& match) -> absl::Status {
      processed = true;
      if (!original_tensor_summary.dimensions.empty()) {
        ++processing_metrics_.compared_pairs_count;
      }
      if (variant == ComparisonVariant::kBaseline) {
        return ProcessOriginalTensorSummaryInternal(
            original_tensor_key, pending_transformation,
            &original_tensor_summary, match.original_key,
            match.pending_transformation, &match.original_tensor_summary);
      }
      return ProcessOriginalTensorSummaryInternal(
          match.original_key, match.pending_transformation,
          &match.original_tensor_summary, original_tensor_key,
          pending_transformation, &original_tensor_summary);
    };

    // If key in other variant has no wildcards, look in non-wildcard table
    // first.
    bool other_has_wildcard =
        absl::c_linear_search(indices_in_other_variant, -1);
    if (!other_has_wildcard) {
      if (auto it = pending_summaries.non_wildcard_summaries.find(
              indices_in_other_variant);
          it != pending_summaries.non_wildcard_summaries.end()) {
        ABSL_RETURN_IF_ERROR(process_match(it->second));
        if (!current_has_wildcard) {
          pending_summaries.non_wildcard_summaries.erase(it);
        }
      }
    } else {
      // If key in other variant has wildcards, look in wildcard table first.
      if (auto it = pending_summaries.wildcard_summaries.find(
              indices_in_other_variant);
          it != pending_summaries.wildcard_summaries.end()) {
        ABSL_RETURN_IF_ERROR(process_match(it->second));
        if (!current_has_wildcard && !absl::c_linear_search(it->first, -1)) {
          pending_summaries.wildcard_summaries.erase(it);
        }
      }
    }

    auto try_wildcard_match = [&](auto& summaries_map) -> absl::StatusOr<bool> {
      // NOLINTNEXTLINE
      for (auto it = summaries_map.begin(); it != summaries_map.end(); ++it) {
        if (IterationIndicesMatch(it->first, indices_in_other_variant)) {
          ABSL_RETURN_IF_ERROR(process_match(it->second));
          if (!current_has_wildcard && !absl::c_linear_search(it->first, -1)) {
            summaries_map.erase(it);
          }
          return true;
        }
      }
      return false;
    };

    if (!processed) {
      // No exact match found, try wildcard matching starting with wildcard
      // table.
      bool matched = false;
      ABSL_ASSIGN_OR_RETURN(
          matched, try_wildcard_match(pending_summaries.wildcard_summaries));
      if (!matched) {
        ABSL_ASSIGN_OR_RETURN(
            matched,
            try_wildcard_match(pending_summaries.non_wildcard_summaries));
      }
    }

    if (processed) {
      if (pending_summaries.non_wildcard_summaries.empty() &&
          pending_summaries.wildcard_summaries.empty()) {
        other_variant_pending_summaries_map->erase(pending_it);
      }
      return absl::OkStatus();
    }
  }

  // No match found, add to current variant's pending summaries.
  auto indices = GetIterationIndices(original_tensor_key);
  auto indexless_key = GetIndexlessScopedTensorKey(original_tensor_key);
  PendingSummaries& pending_summaries =
      (*current_variant_pending_summaries_map)[indexless_key];
  if (absl::c_linear_search(indices, -1)) {
    pending_summaries.wildcard_summaries[indices] = {
        pending_transformation, original_tensor_summary, original_tensor_key};
  } else {
    pending_summaries.non_wildcard_summaries[indices] = {
        pending_transformation, original_tensor_summary, original_tensor_key};
  }
  return absl::OkStatus();
}

std::unique_ptr<OriginalTensorSummaryComparator>
OriginalTensorSummaryComparator::CloneWithCallback(
    OriginalTensorSummaryComparisonCallback&&
        on_original_tensor_summary_comparison_ready) const {
  // Here we just use the same matcher because the absolute tensor keys should
  // be the same across replicas thanks to SPMD.
  return std::make_unique<OriginalTensorSummaryComparator>(
      key_matcher_, std::move(on_original_tensor_summary_comparison_ready));
}

OriginalTensorSummaryComparator::ProcessingMetrics
OriginalTensorSummaryComparator::GetProcessingMetrics() const {
  return processing_metrics_;
}

absl::Status OriginalTensorSummaryComparator::FinishComparison() {
  for (const auto& [indexless_key, pending_summaries] :
       baseline_pending_summaries_) {  // NOLINT
    for (const auto& [indices, pending_comparison] :
         pending_summaries.non_wildcard_summaries) {  // NOLINT
      auto target_key = key_matcher_->FindMatchingKey(
          pending_comparison.original_key, ComparisonVariant::kBaseline);
      if (!target_key.has_value()) {
        continue;
      }
      ABSL_RETURN_IF_ERROR(ProcessOriginalTensorSummaryInternal(
          pending_comparison.original_key,
          pending_comparison.pending_transformation,
          &pending_comparison.original_tensor_summary, *target_key, nullptr,
          nullptr));
    }
  }
  baseline_pending_summaries_.clear();
  for (const auto& [indexless_key, pending_summaries] :
       target_pending_summaries_) {  // NOLINT
    for (const auto& [indices, pending_comparison] :
         pending_summaries.non_wildcard_summaries) {  // NOLINT
      auto baseline_key = key_matcher_->FindMatchingKey(
          pending_comparison.original_key, ComparisonVariant::kTarget);
      if (!baseline_key.has_value()) {
        continue;
      }
      ABSL_RETURN_IF_ERROR(ProcessOriginalTensorSummaryInternal(
          *baseline_key, nullptr, nullptr, pending_comparison.original_key,
          pending_comparison.pending_transformation,
          &pending_comparison.original_tensor_summary));
    }
  }
  target_pending_summaries_.clear();
  return absl::OkStatus();
}

}  // namespace xla::numerics::comparison
