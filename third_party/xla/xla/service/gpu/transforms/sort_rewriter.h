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

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
namespace gpu {

// Rewrites sort operations into CustomCall HLOs that call into CUB.
// Only a subset of shapes is supported - either a single tensor with a simple
// compare function or a pair of tensors where keys are unsigned integers.

class SortRewriter : public HloModulePass {
 public:
  absl::string_view name() const override { return "sort-rewriter"; }

  // CUB radix sort is slower than XLA sort on small shapes, so do not rewrite
  // tensors with sizes below this limit.
  static int SortSizeThreshold() { return sort_size_threshold_; }
  static void SetSortSizeThresholdForTestingOnly(int threshold) {
    // We need to be able to reduce the threshold for testing, so that the tests
    // can run and compare against the reference interpreter, which is quite
    // slow.
    sort_size_threshold_ = threshold;
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  absl::StatusOr<bool> RunOnInstruction(HloSortInstruction* sort_op);
  absl::StatusOr<bool> RunOnComputation(HloComputation* computation);

  static inline int sort_size_threshold_ = 16385;
};

}  // namespace gpu
}  // namespace xla

#endif  // XLA_SERVICE_GPU_TRANSFORMS_SORT_REWRITER_H_
