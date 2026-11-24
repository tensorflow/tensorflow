/* Copyright 2023 The OpenXLA Authors.

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
#ifndef XLA_BACKENDS_GPU_HLO_TRANSFORMS_FUSION_GEMM_FUSION_H_
#define XLA_BACKENDS_GPU_HLO_TRANSFORMS_FUSION_GEMM_FUSION_H_

// This file contains the code for fusing dots and other operations into Triton
// GEMM fusions.

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Rewrite compatible dot() calls into custom calls with fused computations
// that target Triton-based matmul emitter.
class GemmFusion : public HloModulePass {
 public:
  explicit GemmFusion(const se::DeviceDescription& device_description)
      : device_description_(device_description) {}
  absl::string_view name() const override { return "triton-gemm-rewriter"; }

  // Every parameter requires a separate piece of shared memory for asynchronous
  // loads. Multiple parameters are approximately equivalent to multiple
  // pipeline stages.
  // Note: This has been tuned specifically for GEMMs, where pipelining with
  // more than 4 stages has been shown to rarely be practical.
  // Note: The limit doesn't apply to the epilogue of the fusion.
  static constexpr int kMaxParameterPerDotOperand = 4;

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::DeviceDescription device_description_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_HLO_TRANSFORMS_FUSION_GEMM_FUSION_H_
