/* Copyright 2020 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_WHILE_LOOP_ALL_REDUCE_CODE_MOTION_H_
#define XLA_SERVICE_WHILE_LOOP_ALL_REDUCE_CODE_MOTION_H_

#include "absl/status/statusor.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// HLO pass that rewrites while loops to sink all-reduces that are only
// accumulated into a buffer and not otherwise used in the loop body.
// An all-reduce instruction can be sinked if its result is only added
// to a number of accumulation buffers, and the accumulation buffers are not
// used inside the loop.
//
// Pattern before this pass:
// a = ...
// while:
//   b = ...
//   c = all-reduce(b)
//   a += c
// Pattern after this pass:
// a = ...
// d = 0
// while:
//   b = ...
//   d += b
// e = all-reduce(d)
// a += e
class WhileLoopAllReduceCodeMotion : public HloModulePass {
 public:
  explicit WhileLoopAllReduceCodeMotion(bool enable_reduce_scatter = false,
                                        bool run_setup_passes = false)
      : enable_reduce_scatter_(enable_reduce_scatter),
        run_setup_passes_(run_setup_passes) {}
  ~WhileLoopAllReduceCodeMotion() override = default;

  static constexpr absl::string_view kName =
      "while-loop-all-reduce-code-motion";
  absl::string_view name() const override { return kName; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  const bool enable_reduce_scatter_;

  // Whether to run passes that may setup the add(all-reduce/reduce-scatter,
  // accumulation_buffer) pattern.
  const bool run_setup_passes_;
};
}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_ALL_REDUCE_CODE_MOTION_H_
