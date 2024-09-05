/* Copyright 2021 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_COMPILE_TIME_CAP_H_
#define XLA_SERVICE_COMPILE_TIME_CAP_H_
#include <algorithm>

#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"

namespace xla {
// Provide a common way to bound compiler analyses that potentially have
// overhead that is non-linear to the number of instructions in a module.
class BoundNonLinearCompilerAnalysis {
 public:
  // Sampling_rate specifies the proportion of all instructions expected to be
  // analyzed. e.g., if sampling_rate_=2, then every other instructions are
  // expected to be analyzed. If sample_rate <= 0, the analysis will be always
  // allowed to complete. Each analysis is allowed at least a constant number of
  // abstract cost units, before it is considered for early termination.
  explicit BoundNonLinearCompilerAnalysis(HloModule* m,
                                          absl::string_view pass_name,
                                          std::optional<int64_t> sampling_rate)
      : analysis_allowance_(
            (!sampling_rate.has_value() || sampling_rate.value() <= 0 ||
             m->config().GetAnalysisAllowance(pass_name) < 0)
                ? -1
                : std::max(m->config().GetAnalysisAllowance(pass_name),
                           m->instruction_count() / sampling_rate.value())) {}
  // Return whether the cost is deducted successfully. If not, the analysis
  // should be terminated as its overhead is too high.
  bool DeductCost(int64_t cost_now) {
    if (analysis_allowance_ > 0 && cost_now > 0) {
      analysis_allowance_ -= cost_now;
      if (analysis_allowance_ < 0) {
        analysis_allowance_ = 0;
      }
    }
    return analysis_allowance_ != 0;
  }

  bool ContinueAnalysis() const { return analysis_allowance_ != 0; }
  int64_t analysis_allowance() const { return analysis_allowance_; }

 private:
  int64_t analysis_allowance_;
};

};  // namespace xla

#endif  // XLA_SERVICE_COMPILE_TIME_CAP_H_
