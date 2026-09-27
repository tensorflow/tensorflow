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

#ifndef XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_ANALYZER_H_
#define XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_ANALYZER_H_

#include <functional>
#include <memory>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "xla/hlo/ir/hlo_original_value.h"
#include "xla/hlo/utils/hlo_original_value_analysis.h"
#include "xla/hlo/utils/hlo_original_value_analyzer_utils.h"

namespace xla {

class HloOriginalValueAnalyzer {
 public:
  explicit HloOriginalValueAnalyzer(
      std::shared_ptr<const HloOriginalValueAnalysis> analysis,
      std::function<void(const TensorKey&)> on_mark_optimized_tensor = nullptr);

  // Returns true if this optimized tensor corresponds to one or more original
  // tensors that have a non-empty callback_id in `DebugAttributes`. Also
  // internally tracks that the given original tensor will be logged to
  // reconstruct its value.
  bool MarkOptimizedTensorAndCheckWhetherLoggingRequested(
      const TensorKey& optimized_tensor_key);

  // Returns original arrays that were requested to be logged but have no
  // optimized tensors logged to reconstruct them (not recoverable).
  std::vector<OriginalArray> GetRequestedButNotLoggedOriginalArrays() const;

 private:
  std::shared_ptr<const HloOriginalValueAnalysis> analysis_;
  absl::flat_hash_set<OriginalArray> logged_original_arrays_;
  std::function<void(const TensorKey&)> on_mark_optimized_tensor_;
};

}  // namespace xla

#endif  // XLA_HLO_UTILS_HLO_ORIGINAL_VALUE_ANALYZER_H_
