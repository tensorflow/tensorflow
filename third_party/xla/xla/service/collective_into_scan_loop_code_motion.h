/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_COLLECTIVE_INTO_SCAN_LOOP_CODE_MOTION_H_
#define XLA_SERVICE_COLLECTIVE_INTO_SCAN_LOOP_CODE_MOTION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {

// Attempts to move collective communications performed on the output of `scan`
// loops within the loop.
//
// This transform, combined with the `CollectivePipeliner` and
// `AsyncCollectiveCreator` allows these collectives to be overlapped with the
// computation within the loop.
//
// In pseudo-code, we are trying to convert the following loop:
// ```
// for i in range(len(inputs)):
//   outputs[i], state = some_computation(inputs[i], state)
// all_reduced_outputs = all_reduce(outputs)
// ```
// Into this:
// ```
// for i in range(len(inputs)):
//   output, state = some_computation(inputs[i], state)
//   all_reduced_outputs[i] = all_reduce(output)
// ```
class CollectiveIntoScanLoopCodeMotion : public HloModulePass {
 public:
  explicit CollectiveIntoScanLoopCodeMotion(HloOpcode opcode)
      : opcode_(opcode) {}

  absl::string_view name() const override {
    return "collective-into-scan-loop-code-motion";
  }

  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

 private:
  HloOpcode opcode_;
};

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_INTO_SCAN_LOOP_CODE_MOTION_H_
