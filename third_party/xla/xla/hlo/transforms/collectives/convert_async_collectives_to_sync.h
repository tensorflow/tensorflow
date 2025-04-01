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

#ifndef XLA_HLO_TRANSFORMS_COLLECTIVES_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_
#define XLA_HLO_TRANSFORMS_COLLECTIVES_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_

#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/util.h"

namespace xla {

// Convert asynchronous collectives to synchronous (after HLO scheduling) if
// there are no compute operations overlapping with them.

class ConvertAsyncCollectivesToSync : public HloModulePass {
 public:
  explicit ConvertAsyncCollectivesToSync(HloPredicate is_nop = {})
      : is_nop_(is_nop) {}
  absl::string_view name() const override {
    return "convert-async-collectives-to-sync";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  virtual absl::Status ConvertAsyncInstructionsToSync(
      HloComputation* computation,
      absl::Span<const std::pair<HloInstruction*, HloInstruction*>> async_pairs)
      const {
    return ReplaceAsyncInstructionsWithSync(computation, async_pairs);
  }

  // Helper utility to replace a list of pairs of async-start/done ops in a
  // computation with their synchronous variants and update the schedule.
  static absl::Status ReplaceAsyncInstructionsWithSync(
      HloComputation* computation,
      absl::Span<const std::pair<HloInstruction*, HloInstruction*>>
          async_pairs);

  static constexpr char kAsyncCollectiveNameAttributeName[] =
      "async_collective_name";

 private:
  absl::StatusOr<bool> RunOnComputation(HloComputation* computation);
  HloPredicate is_nop_;
};
}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_COLLECTIVES_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_
