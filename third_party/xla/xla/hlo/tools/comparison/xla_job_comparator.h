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

#ifndef XLA_HLO_TOOLS_COMPARISON_XLA_JOB_COMPARATOR_H_
#define XLA_HLO_TOOLS_COMPARISON_XLA_JOB_COMPARATOR_H_

#include <functional>
#include <memory>
#include <tuple>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_comparator.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"

namespace xla::numerics::comparison {

// Compares tensor summaries between two XLA jobs (baseline and target), which
// may have different device assignments. This class manages per-replica
// comparison states and uses device assignments to route incoming tensor shard
// summaries to the correct replica for processing. This allows comparison of
// jobs even if they are run on different device topologies or with different
// device assignments, as long as they have the same number of replicas.
class XlaJobComparator {
 public:
  struct CreationMetrics {
    OriginalTensorSummaryComparator::CreationMetrics comparator_metrics;
  };

  struct ProcessingMetrics {
    OriginalTensorSummaryComparator::ProcessingMetrics comparator_metrics;
  };

  // Callback to be invoked when tensor summaries from baseline and target for a
  // specific replica are ready for comparison. The summaries are aligned to a
  // common transformation point.
  using XlaJobComparatorCallback = std::function<absl::Status(
      // The replica ID identifying the replica in data parallelism. This is the
      // same as the replica ID in DeviceAssignment::LogicalId::replica_id.
      int replica_id,
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          pending_transformation,
      AbsoluteScopedTensorKey baseline_tensor_key,
      OriginalTensorSummary const* baseline_tensor_summary,
      AbsoluteScopedTensorKey target_tensor_key,
      OriginalTensorSummary const* target_tensor_summary)>;

  // Creates an XlaJobComparator. It initializes comparison states for each
  // replica. It requires device assignments and HLO modules for both baseline
  // and target jobs. Returns an error if replica counts are missing or mismatch
  // between baseline and target.
  static absl::StatusOr<std::tuple<XlaJobComparator, CreationMetrics,
                                   hlo_diff::HloGumgraphDiffResults>>
  Create(int replica_count, HloModule* absl_nonnull baseline_original_module,
         HloModule* absl_nonnull target_original_module,
         absl::string_view baseline_recovered_tensor_summaries_file,
         absl::string_view target_recovered_tensor_summaries_file,
         XlaJobComparatorCallback&& callback);

  // Processes an original tensor summary for a specific replica. The tensor
  // summary should be recovered by a XlaJobRecoverer.
  absl::Status ProcessOriginalTensorSummary(
      ComparisonVariant variant, int replica_id,
      const AbsoluteScopedTensorKey& key,
      std::shared_ptr<const tensor_transformation::TensorTransformation>
          transformation,
      const OriginalTensorSummary& summary);

  std::vector<ProcessingMetrics> GetProcessingMetrics() const;

  absl::Status FinishComparison();

 private:
  XlaJobComparator(
      HloModule* baseline_original_module, HloModule* target_original_module,
      std::vector<std::unique_ptr<OriginalTensorSummaryComparator>>&&
          comparators,
      XlaJobComparatorCallback&& callback);

  HloModule* baseline_original_module_;
  HloModule* target_original_module_;
  std::vector<std::unique_ptr<OriginalTensorSummaryComparator>> comparators_;
  XlaJobComparatorCallback callback_;
};
}  // namespace xla::numerics::comparison

#endif  // XLA_HLO_TOOLS_COMPARISON_XLA_JOB_COMPARATOR_H_
