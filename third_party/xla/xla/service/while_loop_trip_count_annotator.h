/* Copyright 2019 The OpenXLA Authors.

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

#ifndef XLA_SERVICE_WHILE_LOOP_TRIP_COUNT_ANNOTATOR_H_
#define XLA_SERVICE_WHILE_LOOP_TRIP_COUNT_ANNOTATOR_H_

#include "absl/container/flat_hash_set.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"
#include "xla/statusor.h"

namespace xla {

// Pass that annotates `while` loops with known trip counts.
//
// The annotation is stored as a backend-config on the while loop node.
//
// This pass should run after all passes that might semantically modify a while
// loop, e.g. by unrolling it.  Otherwise, a loop could end up with a
// backend-config that doesn't match its true trip-count.
//
// This pass does some pattern-matching on loop bodies and conditions, so it
// should run after most HLO simplifications and before fusion and layout
// assignment, which make pattern matching much more difficult by e.g.
// introducing `copy` nodes.
class WhileLoopTripCountAnnotator : public HloModulePass {
 public:
  ~WhileLoopTripCountAnnotator() override {}
  absl::string_view name() const override {
    return "while-loop-trip-count-annotator";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_WHILE_LOOP_TRIP_COUNT_ANNOTATOR_H_
