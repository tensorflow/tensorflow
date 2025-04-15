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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_NEST_GEMM_FUSION_H_
#define XLA_SERVICE_GPU_TRANSFORMS_NEST_GEMM_FUSION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla::gpu {

// Rewrites Triton GEMM fusions to generic Triton fusions. Any other fusions are
// left unchanged.
//
// Fusions with kind kCustom and fusion_backend_config.kind "__triton_gemm" are
// rewritten to fusion_backend_config.kind
// "__triton_nested_fusion_gemm".
//
// While this new fusion kind is supported by generic triton emitter we want
// to distinguish it from "__triton" as we don't want other passes to modify the
// resulting fusions.
//
// The fusion's backend config is set to a BlockLevelFusionConfig, derived from
// a previously set TritonGemmConfig.
//
// The operands of the dot (including their prologues) are fused into two new
// nested fusions, each with their own BlockLevelFusionConfig.
class NestGemmFusion : public HloModulePass {
 public:
  explicit NestGemmFusion(const se::GpuComputeCapability& compute_capability)
      : compute_capability_(compute_capability) {}

  absl::string_view name() const override { return "nest_gemm_fusion"; }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const se::GpuComputeCapability compute_capability_;
};

}  // namespace xla::gpu

#endif  // XLA_SERVICE_GPU_TRANSFORMS_NEST_GEMM_FUSION_H_
