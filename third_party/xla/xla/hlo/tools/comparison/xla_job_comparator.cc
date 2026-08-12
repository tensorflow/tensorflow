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

#include "xla/hlo/tools/comparison/xla_job_comparator.h"

#include <cstdint>
#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_comparator.h"
#include "xla/hlo/tools/comparison/original_tensor_summary_utils.h"
#include "xla/hlo/tools/hlo_diff/hlo_gumgraph_diff.h"

namespace xla::numerics::comparison {

XlaJobComparator::XlaJobComparator(
    HloModule* baseline_original_module, HloModule* target_original_module,
    std::vector<std::unique_ptr<OriginalTensorSummaryComparator>>&& comparators,
    XlaJobComparatorCallback&& callback)
    : baseline_original_module_(baseline_original_module),
      target_original_module_(target_original_module),
      comparators_(std::move(comparators)),
      callback_(std::move(callback)) {}

/*static */ absl::StatusOr<
    std::tuple<XlaJobComparator, XlaJobComparator::CreationMetrics,
               hlo_diff::HloGumgraphDiffResults>>
XlaJobComparator::Create(
    int replica_count, HloModule* absl_nonnull baseline_original_module,
    HloModule* absl_nonnull target_original_module,
    absl::string_view baseline_recovered_tensor_summaries_file,
    absl::string_view target_recovered_tensor_summaries_file,
    XlaJobComparatorCallback&& callback) {
  std::vector<std::unique_ptr<OriginalTensorSummaryComparator>> comparators;
  comparators.resize(replica_count);

  CreationMetrics creation_metrics;
  ABSL_ASSIGN_OR_RETURN(
      (auto [comparator, metrics, diff_results]),
      OriginalTensorSummaryComparator::Create(
          baseline_original_module, target_original_module,
          baseline_recovered_tensor_summaries_file,
          target_recovered_tensor_summaries_file,
          [callback](
              std::shared_ptr<const tensor_transformation::TensorTransformation>
                  pending_transformation,
              AbsoluteScopedTensorKey baseline_tensor_key,
              OriginalTensorSummary const* baseline_tensor_summary,
              AbsoluteScopedTensorKey target_tensor_key,
              OriginalTensorSummary const* target_tensor_summary) {
            return callback(0, pending_transformation, baseline_tensor_key,
                            baseline_tensor_summary, target_tensor_key,
                            target_tensor_summary);
          }));
  creation_metrics.comparator_metrics = std::move(metrics);
  comparators[0] = std::move(comparator);

  for (int64_t i = 1; i < replica_count; ++i) {
    comparators[i] = comparators[0]->CloneWithCallback(
        [i, callback](
            std::shared_ptr<const tensor_transformation::TensorTransformation>
                pending_transformation,
            AbsoluteScopedTensorKey baseline_tensor_key,
            OriginalTensorSummary const* baseline_tensor_summary,
            AbsoluteScopedTensorKey target_tensor_key,
            OriginalTensorSummary const* target_tensor_summary) {
          return callback(i, pending_transformation, baseline_tensor_key,
                          baseline_tensor_summary, target_tensor_key,
                          target_tensor_summary);
        });
  }

  XlaJobComparator job_comparator(baseline_original_module,
                                  target_original_module,
                                  std::move(comparators), std::move(callback));
  return std::make_tuple(std::move(job_comparator), creation_metrics,
                         std::move(diff_results));
}

absl::Status XlaJobComparator::ProcessOriginalTensorSummary(
    ComparisonVariant variant, int replica_id,
    const AbsoluteScopedTensorKey& key,
    std::shared_ptr<const tensor_transformation::TensorTransformation>
        transformation,
    const OriginalTensorSummary& summary) {
  if (replica_id < 0 || replica_id >= comparators_.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Invalid replica_id: ", replica_id));
  }
  return comparators_[replica_id]->ProcessOriginalTensorSummary(
      variant, key, transformation, summary);
}

std::vector<XlaJobComparator::ProcessingMetrics>
XlaJobComparator::GetProcessingMetrics() const {
  std::vector<ProcessingMetrics> processing_metrics;
  processing_metrics.reserve(comparators_.size());
  for (const auto& comparator : comparators_) {
    processing_metrics.push_back(ProcessingMetrics{
        /*comparator_metrics=*/comparator->GetProcessingMetrics()});
  }
  return processing_metrics;
}

absl::Status XlaJobComparator::FinishComparison() {
  for (auto& comparator : comparators_) {
    ABSL_RETURN_IF_ERROR(comparator->FinishComparison());
  }
  return absl::OkStatus();
}

}  // namespace xla::numerics::comparison
