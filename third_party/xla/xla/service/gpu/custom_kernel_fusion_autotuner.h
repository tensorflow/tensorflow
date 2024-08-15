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
#ifndef XLA_SERVICE_GPU_CUSTOM_KERNEL_FUSION_AUTOTUNER_H_
#define XLA_SERVICE_GPU_CUSTOM_KERNEL_FUSION_AUTOTUNER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/autotuning.pb.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/gpu/autotuner_util.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/xla.pb.h"

namespace xla {
namespace gpu {

// Find best custom kernel for custom kernel fusions.
class CustomKernelFusionAutotuner : public HloModulePass {
 public:
  explicit CustomKernelFusionAutotuner(const AutotuneConfig& config)
      : config_(config) {}

  absl::string_view name() const override {
    return "custom_kernel-fusion-autotuner";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const AutotuneConfig config_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_CUSTOM_KERNEL_FUSION_AUTOTUNER_H_
