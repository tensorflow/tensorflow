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

#include "xla/hlo/utils/hlo_original_value_analyzer.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/utils/hlo_original_value_analysis.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"

namespace xla {

HloOriginalValueAnalyzer::HloOriginalValueAnalyzer(
    std::shared_ptr<const HloOriginalValueAnalysis> analysis,
    std::function<void(const TensorKey&)> on_mark_optimized_tensor)
    : analysis_(std::move(analysis)),
      on_mark_optimized_tensor_(std::move(on_mark_optimized_tensor)) {}

bool HloOriginalValueAnalyzer::
    MarkOptimizedTensorAndCheckWhetherLoggingRequested(
        const TensorKey& optimized_tensor_key) {
  bool requested = false;
  const auto& map = analysis_->original_tensor_by_optimized_tensor_key();
  auto it = map.find(optimized_tensor_key);
  if (it != map.end()) {
    for (const auto& info : it->second) {
      if (analysis_->requested_original_arrays().contains(
              info.original_array)) {
        requested = true;
        logged_original_arrays_.insert(info.original_array);
      }
    }
  }
  if (requested && on_mark_optimized_tensor_) {
    on_mark_optimized_tensor_(optimized_tensor_key);
  }
  return requested;
}

std::vector<OriginalArray>
HloOriginalValueAnalyzer::GetRequestedButNotLoggedOriginalArrays() const {
  std::vector<OriginalArray> not_logged;
  for (const auto& [requested, _] : analysis_->requested_original_arrays()) {
    if (!logged_original_arrays_.contains(requested)) {
      not_logged.push_back(requested);
    }
  }
  std::sort(not_logged.begin(), not_logged.end(),
            [](const OriginalArray& a, const OriginalArray& b) {
              return a.ToString() < b.ToString();
            });
  return not_logged;
}

}  // namespace xla
