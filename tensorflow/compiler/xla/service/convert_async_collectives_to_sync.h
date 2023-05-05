/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_

#include <utility>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/hlo/ir/hlo_computation.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"

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
  StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  virtual Status ConvertAsyncInstructionsToSync(
      HloComputation* computation,
      absl::Span<const std::pair<HloInstruction*, HloInstruction*>> async_pairs)
      const {
    return ReplaceAsyncInstructionsWithSync(computation, async_pairs);
  }

  // Helper utility to replace a list of pairs of async-start/done ops in a
  // computation with their synchronous variants and update the schedule.
  static Status ReplaceAsyncInstructionsWithSync(
      HloComputation* computation,
      absl::Span<const std::pair<HloInstruction*, HloInstruction*>>
          async_pairs);

  static constexpr char kAsyncCollectiveNameAttributeName[] =
      "async_collective_name";

 private:
  StatusOr<bool> RunOnComputation(HloComputation* computation);
  HloPredicate is_nop_;
};
}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_CONVERT_ASYNC_COLLECTIVES_TO_SYNC_H_
