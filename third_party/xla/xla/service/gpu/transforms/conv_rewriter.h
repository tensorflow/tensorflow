/* Copyright 2018 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_CONV_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_CONV_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites plain convolutions, backwards-filter convolutions, and
// backwards-input convolutions into CustomCall HLOs that call into
// Cudnn/Miopen.
//
// This pass does not fuse other ops into the convolution. Instead, specific
// patterns of ops will be matched and fused into the custom call in
// CudnnFusedConvRewriter.

class ConvRewriter : public HloModulePass {
 public:
  explicit ConvRewriter(const se::GpuComputeCapability& compute_capability)
      : compute_capability_(compute_capability) {};

  absl::string_view name() const override { return "conv-rewriter"; }

  static bool ConvIsLowerable(HloInstruction* conv);

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  se::GpuComputeCapability compute_capability_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_CONV_REWRITER_H_
