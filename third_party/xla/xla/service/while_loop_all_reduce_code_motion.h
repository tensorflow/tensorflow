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

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// HLO pass that sinks all-reduce instructions out of while loop bodies.
// An all-reduce is sunk if its result is ONLY used to accumulate into buffers,
// and those buffers are not used otherwise inside the loop.
//
// Pattern before this pass:
// a = ...
// while:
//   b = ...
//   c = all-reduce(b)
//   a += c
//
// Pattern after this pass:
// a = ...
// d = 0
// while:
//   b = ...
//   d += b
// e = all-reduce(d)
// a += e
//
// Additionally sinks all-reduces that are scattered into the loop output,
// without being used in the loop body.
// Supported reduction operations: add, mul, min, max.
//
// Pattern before this pass:
// a = while:
//   b = dynamic-update-slice(output, all-reduce(...), loop_induction_variable)
// c = get-tuple-element(a, tuple_index)
//
// Pattern after this pass:
// a = while:
//   b = dynamic-update-slice(output, ..., loop_induction_variable)
// c = all-reduce(get-tuple-element(a, tuple_index))
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

 protected:
  absl::StatusOr<bool> RunImpl(
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
