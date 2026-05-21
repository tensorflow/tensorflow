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

#ifndef XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_FUSION_REWRITER_V2_H_
#define XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_FUSION_REWRITER_V2_H_

#include <cstdint>
#include <optional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/functional/any_invocable.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/platform_id.h"

namespace xla::gpu {

// Dynamic slice fusion rewriter V2 wraps a hero instruction that reads from
// sliced buffers (via Slice or DynamicSlice) and/or writes into sliced buffers
// (via DynamicUpdateSlice) into a custom fusion.
//
// Requires that the DynamicSliceAnnotator pass has run first to annotate
// DynamicSlice/DynamicUpdateSlice instructions with DynamicSliceConfig.
//
// The pass is configured with options that select hero instructions and decide
// which sliced input/output edges to pull into each fusion body.
//
class DynamicSliceFusionRewriterV2 : public HloModulePass {
 public:
  // Selects instructions that are legal hero candidates for dynamic-slice
  // fusion wrapping. It is evaluated before input/output slicing analysis.
  using Predicate = absl::AnyInvocable<bool(const HloInstruction*) const>;

  // Called after the pass has resolved an aligned Slice/DynamicSlice chain
  // feeding one hero operand. Return true to move that slice chain into the
  // fusion body; return false to leave it outside. For O2 tuple/GTE
  // look-through, rejecting an input also prevents the temporary hero operand
  // rewrite for that input.
  using CaptureSlice =
      absl::AnyInvocable<bool(const HloInstruction* hero, int64_t operand_index,
                              const HloInstruction* slice) const>;

  // Called after the pass has resolved an aligned DynamicUpdateSlice chain
  // consuming one hero result. `result_index` is absent for non-tuple hero
  // results and contains the flat tuple result number for tuple hero results.
  // Return true to move that update chain into the fusion body; return false
  // to leave it outside.
  using CaptureUpdateSlice = absl::AnyInvocable<bool(
      const HloInstruction* hero, std::optional<int64_t> result_index,
      const HloInstruction* dynamic_update_slice) const>;

  enum class OptLevel {
    // Follow a sequence of no-op (bitcast, tuple, gte) operations to find the
    // the sliced source
    kO1,
    // Aggressive optimization that passes through optimization barriers to find
    // the sliced source.
    kO2,
  };

  struct Options {
    Predicate predicate = [](auto...) { return false; };
    CaptureSlice capture_slice = [](auto...) { return true; };
    CaptureUpdateSlice capture_update_slice = [](auto...) { return true; };
    OptLevel opt_level = OptLevel::kO1;
  };

  DynamicSliceFusionRewriterV2(stream_executor::PlatformId platform_id,
                               Options options)
      : platform_id_(std::move(platform_id)), options_(std::move(options)) {}

  absl::string_view name() const override {
    return "dynamic-slice-fusion-rewriter-v2";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  stream_executor::PlatformId platform_id_;
  Options options_;
};

}  // namespace xla::gpu

#endif  // XLA_BACKENDS_GPU_TRANSFORMS_DYNAMIC_SLICE_FUSION_REWRITER_V2_H_
