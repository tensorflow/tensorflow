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

#ifndef XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_COMPARATOR_H_
#define XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_COMPARATOR_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_key_matcher.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"

namespace xla::numerics::comparison {

// Compares tensor summaries from two HLO modules (baseline and target).
//
// This class receives tensor summaries from baseline and target computations
// via `ProcessOriginalTensorSummary`. It uses HLO diff to identify
// corresponding instructions in the baseline and target HLO modules. When
// summaries for corresponding tensors from both baseline and target are
// available, it aligns them by applying recovering transformations and invokes
// a user-provided callback with the aligned summaries.
class OriginalTensorSummaryComparator {
 public:
  struct CreationMetrics {
    // The total number of tensors in the baseline module. Note that this is
    // not the number of instructions in the module. Instead, it counts arrays
    // in tuple-shaped instructions individually.
    int64_t baseline_tensor_count = 0;
    // The total number of tensors in the target module.
    int64_t target_tensor_count = 0;
    // The number of tensor pairs that are unchanged between baseline and
    // target.
    int64_t unchanged_tensor_pair_count = 0;
    // The number of tensor pairs that are changed between baseline and
    // target.
    int64_t changed_tensor_pair_count = 0;
  };

  struct ProcessingMetrics {
    // The total number of baseline tensor summaries received.
    int64_t received_baseline_tensor_summaries = 0;
    // The total number of target tensor summaries received.
    int64_t received_target_tensor_summaries = 0;
    // The number of baseline tensor summaries that cannot be translated to
    // target tensor summaries because HLO diff did not find a pair for the
    // instruction.
    int64_t untranslatable_baseline_tensor_summaries = 0;
    // The number of target tensor summaries that cannot be translated to
    // baseline tensor summaries because HLO diff did not find a pair for the
    // instruction.
    int64_t untranslatable_target_tensor_summaries = 0;
    // The number of pairs of tensor summaries that have been compared.
    int64_t compared_pairs_count = 0;
  };

  // Callback to be called when tensor summaries from baseline and target are
  // ready for comparison. The summaries are aligned to a common transformation
  // point.
  using OriginalTensorSummaryComparisonCallback = std::function<absl::Status(
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>;

  OriginalTensorSummaryComparator(
      std::shared_ptr<OriginalTensorSummaryKeyMatcher> key_matcher,
      OriginalTensorSummaryComparisonCallback&&
          on_original_tensor_summary_comparison_ready)
      : key_matcher_(std::move(key_matcher)),
        on_original_tensor_summary_comparison_ready_(
            std::move(on_original_tensor_summary_comparison_ready)) {}

  // Not copyable.
  OriginalTensorSummaryComparator(const OriginalTensorSummaryComparator&) =
      delete;
  OriginalTensorSummaryComparator& operator=(
      const OriginalTensorSummaryComparator&) = delete;

  // Movable.
  OriginalTensorSummaryComparator(OriginalTensorSummaryComparator&&) = default;
  OriginalTensorSummaryComparator& operator=(
      OriginalTensorSummaryComparator&&) = default;

  // Creates an OriginalTensorSummaryComparator by computing the HLO diff
  // between baseline and target modules to establish instruction mappings.
  static absl::StatusOr<
      std::tuple<std::unique_ptr<OriginalTensorSummaryComparator>,
                 CreationMetrics, hlo_diff::HloGumgraphDiffResults>>
  Create(const HloModule* baseline_module, const HloModule* target_module,
         absl::string_view baseline_recovered_tensor_summaries_file,
         absl::string_view target_recovered_tensor_summaries_file,
         OriginalTensorSummaryComparisonCallback
             on_original_tensor_summary_comparison_ready);

  // Processes a tensor summary for a given original tensor position. This
  // function should be called for each original tensor summary for each
  // variant. When both baseline and target tensor summaries are ready according
  // to the HLO diff, the callback
  // `on_original_tensor_summary_comparison_ready_` is called.
  absl::Status ProcessOriginalTensorSummary(
      ComparisonVariant variant,
      const AbsoluteScopedTensorKey& original_tensor_key,
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          pending_transformation,
      const OriginalTensorSummary& original_tensor_summary);

  // Clones the comparator with a new callback. Note that the internal
  // states are not cloned. Only the HLO diff bimap is cloned.
  std::unique_ptr<OriginalTensorSummaryComparator> CloneWithCallback(
      OriginalTensorSummaryComparisonCallback&&
          on_original_tensor_summary_comparison_ready) const;

  ProcessingMetrics GetProcessingMetrics() const;

  // Finishes the comparison by processing any remaining pending tensor
  // summaries.
  absl::Status FinishComparison();

 private:
  // IndexlessScopedTensorKey is like ScopedTensorKey, but without iteration
  // indices. This is useful when looking up pending summaries regardless of
  // iteration indices, in order to support wildcard iteration indices (i.e.,
  // -1) in ScopedInstruction.
  struct IndexlessScopedTensorKey {
    std::vector<std::string> scope_instruction_names;
    TensorKey tensor_key;

    bool operator==(const IndexlessScopedTensorKey& other) const {
      return scope_instruction_names == other.scope_instruction_names &&
             tensor_key == other.tensor_key;
    }
    bool operator!=(const IndexlessScopedTensorKey& other) const {
      return !(*this == other);
    }
    template <typename H>
    friend H AbslHashValue(H h, const IndexlessScopedTensorKey& key) {
      return H::combine(std::move(h), key.scope_instruction_names,
                        key.tensor_key);
    }
    template <typename Sink>
    friend void AbslStringify(Sink& sink, const IndexlessScopedTensorKey& key) {
      absl::Format(&sink, "%s/%v",
                   absl::StrJoin(key.scope_instruction_names, "/"),
                   key.tensor_key);
    }
    friend std::ostream& operator<<(std::ostream& os,
                                    const IndexlessScopedTensorKey& key) {
      return os << absl::StrCat(key);
    }
  };
  // Holds a tensor summary and its common pending transformation that are not
  // applied to the summary.
  struct PendingComparisonTensorSummary {
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        pending_transformation;
    OriginalTensorSummary original_tensor_summary;
    AbsoluteScopedTensorKey original_key;
  };

  struct PendingSummaries {
    // iteration_indices -> PendingComparisonTensorSummary for keys without any
    // wildcards.
    absl::flat_hash_map<std::vector<int64_t>, PendingComparisonTensorSummary>
        non_wildcard_summaries;
    // iteration_indices -> PendingComparisonTensorSummary for keys with at
    // least one wildcard.
    absl::flat_hash_map<std::vector<int64_t>, PendingComparisonTensorSummary>
        wildcard_summaries;
  };

  // Creates an IndexlessScopedTensorKey from a ScopedTensorKey by removing
  // iteration indices.
  static IndexlessScopedTensorKey GetIndexlessScopedTensorKey(
      const ScopedTensorKey& key);

  // Helper function to align the two tensor summaries and call the callback.
  absl::Status ProcessOriginalTensorSummaryInternal(
      AbsoluteScopedTensorKey baseline_tensor_key,
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          baseline_pending_transformation,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          target_pending_transformation,
      OriginalTensorSummary const* target_tensor_summary);

  std::shared_ptr<OriginalTensorSummaryKeyMatcher> key_matcher_;

  // Callback to be called when tensor summaries from baseline and target are
  // ready for comparison.
  OriginalTensorSummaryComparisonCallback
      on_original_tensor_summary_comparison_ready_;

  // Tensor summaries from baseline/target that are waiting for their
  // counterparts from target/baseline to be processed.
  absl::flat_hash_map<IndexlessScopedTensorKey, PendingSummaries>
      baseline_pending_summaries_;
  absl::flat_hash_map<IndexlessScopedTensorKey, PendingSummaries>
      target_pending_summaries_;

  ProcessingMetrics processing_metrics_;
};

}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_ORIGINAL_TENSOR_SUMMARY_COMPARATOR_H_
