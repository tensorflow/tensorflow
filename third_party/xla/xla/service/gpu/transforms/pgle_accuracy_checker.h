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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_PGLE_ACCURACY_CHECKER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_PGLE_ACCURACY_CHECKER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/service/profile_guided_latency_estimator.h"

namespace xla::gpu {

// This pass checks the accuracy of the input feedback-driven optimization (FDO)
// profile. If any non-NOP instruction from the given HloModule is not present
// in the profile this pass fails.
class PGLEAccuracyChecker : public HloModulePass {
 public:
  explicit PGLEAccuracyChecker(ProfileGuidedLatencyEstimator& pgle_estimator)
      : pgle_estimator_(pgle_estimator) {}
  absl::string_view name() const override { return "pgle-accuracy-checker"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  ProfileGuidedLatencyEstimator& pgle_estimator_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_PGLE_ACCURACY_CHECKER_H_
