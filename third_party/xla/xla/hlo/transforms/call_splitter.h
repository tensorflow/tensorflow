/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_CALL_SPLITTER_H_
#define XLA_HLO_TRANSFORMS_CALL_SPLITTER_H_

#include <utility>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
#include "xla/util.h"

namespace xla {

// This pass allows splitting a single function call into multiple calls to two
// functions, where the original called computation is split across a specified
// boundary.
//
// For example, given the call
//
// (x) = call(a, b, c), to_apply={(mul(p0, (add(p1, p2)))}
//
// with the boundary predicate "opcode == kMultiply" and the kDown direction, we
// will get:
//
// (t) = call(b, c), to_apply={(add(p0, p1))}
// (y) = call(a, t), to_apply={(mul(p0, p1))}
//
// This also allow splitting functions "vertically" as opposed to "horizontally"
// e.g. for the same predicate, given:
//
// (x, y) = call(a, b, c), to_apply={(add(p0, p1), mul(p1, p2))}
//
// we should get:
//
// (x) = call(a, b), to_apply={(add(p0, p1))}
// (y) = call(b, c), to_apply={(mul(p0, p1))}
// TODO(mkuper): This is not quite ready for production use yet.
class CallSplitter : public HloModulePass {
 public:
  // TODO(mkuper): Do we want a "both" direction that does a 3-way split?
  enum class SplitDirection { kDown, kUp };

  // The `call_predicate` is used to select the calls that should be split. The
  // `boundary_predicate` is used to select the instructions that form the
  // boundary between the two calls.
  explicit CallSplitter(const HloPredicate& call_predicate,
                        const HloPredicate& boundary_predicate,
                        SplitDirection split_direction)
      : call_predicate_(call_predicate),
        boundary_predicate_(boundary_predicate),
        split_direction_(split_direction) {}

  ~CallSplitter() override = default;

  static constexpr absl::string_view kName = "call-splitter";
  absl::string_view name() const override { return kName; }

 protected:
  // Runs the pass on the given module. Returns whether the module was changed.
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;

  // Splits the body of the given call into two computations, according to the
  // given boundary predicate. Returns the two new computations.
  std::pair<HloComputation*, HloComputation*> SplitCallBody(
      HloComputation* body, HloPredicate boundary_predicate);

  absl::flat_hash_map<HloComputation*,
                      std::pair<HloComputation*, HloComputation*>>
      split_call_bodies_;

 protected:
  HloPredicate call_predicate_;
  HloPredicate boundary_predicate_;
  SplitDirection split_direction_;
};
}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_CALL_SPLITTER_H_
