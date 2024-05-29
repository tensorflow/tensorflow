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

#ifndef XLA_SERVICE_HLO_UNSTACKER_H_
#define XLA_SERVICE_HLO_UNSTACKER_H_

#include <stdbool.h>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
// This pass implements unstacking for loop operands. Generally speaking,
// unstacking is the act of breaking a rank n tensor into n smaller n-1 rank
// tensors without changing the semantics of the program. There are different
// patterns that can benefit from unstacking. This pass aims to implement such
// patterns. The patterns implemented are not exhaustive by any means. Lets
// consider a simple example:
// In the pattern below, `I` (the most-major dimension in the stacked tensor),
// is equal to the trip count of the while loop and `i` is the iteration
// variable of the loop. The stacked input is used only as input to a
// shape-covering dynamic-slice (check the definition of a shape-covering
// dynamic-slice: `tensorflow/compiler/xla/service/while_loop_unroller.h`)
//
//   +-while----------------------------------------------------+
//   | param = tuple(..., [I,x1,...,xn]stacked, ...)            |
//   | ...                                                      |
//   | [1,x1,...,xn]slice = ds([I,x1,...,xn]stacked, i, 0, ...) |
//   | ...                                                      |
//   | ops using the slice                                      |
//   | ...                                                      |
//   | ROOT = tuple(..., stacked, ...)                          |
//   +----------------------------------------------------------+
//
// This pattern can be unstacked and rewritten as following:
//
//   +-while-----------------------------------------------------------------+
//   | param = tuple(..., ([1,x1,...,xn], ..., [1,x1,...,xn])unstacked, ...) |
//   | ...                                                                   |
//.  | slice_1 = get_tuple_element(unstacked), index=i                       |
//   | ops using the slice_i                                                 |
//   | ...                                                                   |
//   | ROOT = tuple(..., unstacked, ...)                                     |
//   +-----------------------------------------------------------------------+
//
// where the unstacked input is initialized with the slices outside of the loop:
// unstacked = tuple(slice_1, ..., slice_n)
// To get each slice, the pass introduces a dynamic version of the
// kGetTupleElement instruction using a custom-call. This custom-call is then
// replaced with a normal get-tuple-element during loop unrolling.
//
// Below is a high-level overview of the unstacking algorithm:
// We unstack a module by unstacking inputs to the while loops within the entry
// computation for every index. Given a while loop and a candidate for
// unstacking, the algorithm performs the following two steps:
// 1. The first step is to determine if unstacking is possible by checking if
//  the unstacking of the while operand at the given index can be propagated
//  through the body (and nested bodies if any). Unstacking is possible
//  if a pair of pattern and handler is provided that can identify and handle
//  such pattern that involves all the uses of the stacked operand at the given
//  index.
// 2. Apply the unstacking by executing the changes gathered in the first phase.
class HloUnstacker : public HloModulePass {
 public:
  ~HloUnstacker() override = default;

  explicit HloUnstacker() = default;

  absl::string_view name() const override { return "hlo_unstacker"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_HLO_UNSTACKER_H_
