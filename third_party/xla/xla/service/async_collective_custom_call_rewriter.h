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

#ifndef XLA_SERVICE_ASYNC_COLLECTIVE_CUSTOM_CALL_REWRITER_H_
#define XLA_SERVICE_ASYNC_COLLECTIVE_CUSTOM_CALL_REWRITER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

class AsyncComputeOnHelper {
 public:
  virtual ~AsyncComputeOnHelper() = default;
  virtual absl::Status AddBackendSpecializations(
      HloInstruction* custom_call_start, HloInstruction* async_start) {
    return absl::OkStatus();
  }
};

// Convert fake async collective custom calls to async collective instructions.
// The fake async collective custom calls are temporary workaround to unblock
// the development of async collective operations. This pass should be removed
// as soon as possible.
class AsyncCollectiveCustomCallRewriter : public HloModulePass {
 public:
  explicit AsyncCollectiveCustomCallRewriter(
      bool use_legacy_collectives = false,
      AsyncComputeOnHelper* compute_on_helper = nullptr)
      : use_legacy_collectives_(use_legacy_collectives),
        compute_on_helper_(compute_on_helper) {}

  absl::string_view name() const override {
    return "async-collective-custom-call-rewriter";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  absl::StatusOr<bool> ProcessPair(
      HloComputation* computation, HloInstruction* start_call,
      HloInstruction* done_call,
      absl::Span<const hlo_instruction_utils::async::AsyncTraceStep>
          forward_path,
      bool use_legacy_collectives);

  absl::StatusOr<bool> ProcessComputeOn(
      HloComputation* computation, HloInstruction* start_call,
      HloInstruction* done_call,
      absl::Span<const hlo_instruction_utils::async::AsyncTraceStep>
          forward_path,
      bool use_legacy_collectives);

 private:
  bool use_legacy_collectives_;
  AsyncComputeOnHelper* compute_on_helper_;
};

}  // namespace xla

#endif  // XLA_SERVICE_ASYNC_COLLECTIVE_CUSTOM_CALL_REWRITER_H_
