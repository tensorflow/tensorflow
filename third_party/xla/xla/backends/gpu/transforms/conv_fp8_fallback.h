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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_CONV_FP8_FALLBACK_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_CONV_FP8_FALLBACK_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Replaces `fusion` with a clone whose F8 shapes are BF16: F8 operands are
// converted to BF16 outside the fusion, F8 (tuple-)outputs are converted
// back to the original F8 types. Returns the BF16 fusion instruction. Used
// by ConvFp8Fallback after cuDNN probing; exposed for tests.
absl::StatusOr<HloFusionInstruction*> RewriteFp8FusionToBf16(
    HloFusionInstruction* fusion);

// Rewrites FP8 cuDNN convolution fusions (__cudnn$fusion produced by
// ConvFusionRewriter) to BF16 when cuDNN has no execution plan for the
// specific FP8 configuration (dimensions, strides, groups, etc.) on the
// target GPU; today such fusions fail with a hard error when the autotuner
// enumerates their plans.
//
// The pass probes cuDNN devicelessly (CuDnnFusionCompiler::
// SupportsFusionDeviceless) using only the target's
// stream_executor::DeviceDescription, so it works identically during JIT and
// AOT compilation. If an FP8 conv fusion has no plans but its BF16
// equivalent does, the fusion is replaced by a clone whose F8 shapes are
// BF16, with converts on the F8 operands and converts back to the original
// F8 types on the outputs.
//
// The pass must run after ConvFusionRewriter (which creates the fusions;
// rewriting the convolution earlier is not robust because inserted converts
// are themselves fusable and would be folded back into an FP8 fusion), and
// before the autotuner. Probe verdicts other than kUnsupported for the FP8
// fusion — including kUnknown ones (a cuDNN runtime too old for the target,
// graphs the deviceless probe cannot model, cuDNN frontend failures) — leave
// the fusion unchanged, and the rewrite only fires if the BF16 replacement
// probes as supported.
class ConvFp8Fallback : public HloModulePass {
 public:
  explicit ConvFp8Fallback(
      stream_executor::DeviceDescription device_description)
      : device_description_(std::move(device_description)) {}

  absl::string_view name() const override { return "conv-fp8-fallback"; }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  stream_executor::DeviceDescription device_description_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_CONV_FP8_FALLBACK_H_
