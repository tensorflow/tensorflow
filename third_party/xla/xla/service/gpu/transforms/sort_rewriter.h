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

#ifndef XLA_SERVICE_GPU_TRANSFORMS_SORT_REWRITER_H_
#define XLA_SERVICE_GPU_TRANSFORMS_SORT_REWRITER_H_

#include <string>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/stream_executor/device_description.h"

namespace xla {
namespace gpu {

// Rewrites sort operations into CustomCall HLOs that call into CUB.
// Only a subset of shapes is supported - either a single tensor with a simple
// compare function or a pair of tensors where keys are unsigned integers.

class SortRewriter : public HloModulePass {
 public:
  explicit SortRewriter(const se::DeviceDescription& device_description,
                        std::string platform_name)
      : device_description_(device_description),
        platform_name_(std::move(platform_name)) {}
  absl::string_view name() const override { return "sort-rewriter"; }

  enum class Mode {
    kAuto,   // Decide whether to rewrite compatible sorts based on a heuristic.
    kAlways  // Always rewrite compatible sorts. Used for testing.
  };

  // CUB radix sort is slower than XLA sort on small shapes, so do not rewrite
  // tensors with sizes below this limit.
  static Mode SortMode() { return sort_mode_; }
  static void SetSortModeForTestingOnly(Mode sort_mode) {
    // We need to be able to force rewrites for testing for arbitrary shapes.
    // This enables the tests to run and compare against the reference
    // interpreter, which is quite slow and needs smaller shapes that would
    // normally not be rewritten.
    sort_mode_ = sort_mode;
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> RunOnInstruction(HloSortInstruction* sort_op);
  absl::StatusOr<bool> RunOnComputation(HloComputation* computation);

  static inline Mode sort_mode_ = Mode::kAuto;
  se::DeviceDescription device_description_;
  std::string platform_name_;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_SORT_REWRITER_H_
