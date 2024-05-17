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

#ifndef XLA_SERVICE_SCAN_LOOP_ACCUMULATOR_INPUT_UNIFICATION_H_
#define XLA_SERVICE_SCAN_LOOP_ACCUMULATOR_INPUT_UNIFICATION_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_pass_interface.h"

namespace xla {
// This pass looks at the nested loops with accumulator patterns and unifies the
// accumulation buffer with the input. The accumulation pattern usually comes
// from jax.scan function. This transformation is beneficial by removing the
// unnecessary copy of the accumulation buffer in the outer body.
// Below is the pattern that this pass identifies:
//   +-while------------------------------------+
//   | param = tuple(..., prev_acc, ...)        |
//   | ...                                      |
//   | input = gte(param), index=@prev_acc      |
//   | acc = allocate-buffer()                  |
//   | ...                                      |
//   | +-scan----------------------------------+|
//   | | param = tuple(..., acc, input, ...)   ||
//   | | ...                                   ||
//   | | slice = ds(input, i, 0, ...)          ||
//   | | slice' = f(slice, ...)                ||
//   | | acc'  = dus(acc, slice', i, 0, ...)   ||
//   | | ...                                   ||
//   | | ROOT = tuple(..., acc', input, ...)   ||
//   | +---------------------------------------+|
//   | new_acc = gte(scan), index=@acc'         |
//   | copy_acc = copy(new_acc)                 |
//   | ...                                      |
//   | ROOT = tuple(..., copy_acc, ...)         |
//   +------------------------------------------+
//
// To apply the unification we need to find pair of (acc,input). The
// accumulators are found by simply looking for shape-covering write-only
// instructions, in this case acc is written to by dynamic-update-slice that
// covers the entire shape across all the iterations of the scan loop. To find
// the input that corresponds to the accumulator, we follow the accumulated
// output of the scan loop (index @acc') through the outer loop (index
// @prev_acc) and find the index in which it is passed to the scan loop. Below
// is the simplified program after unification:
//
//   +-while------------------------------------+
//   | param = tuple(..., prev_acc, ...)        |
//   | ...                                      |
//   | input = gte(param), index=@prev_acc      |
//   | ...                                      |
//   | +-scan----------------------------------+|
//   | | param = tuple(..., input, ...)        ||
//   | | ...                                   ||
//   | | slice = ds(input, i, 0, ...)          ||
//   | | slice' = f(slice, ...)                ||
//   | | acc'  = dus(input, slice', i, 0, ...) ||
//   | | ...                                   ||
//   | | ROOT = tuple(..., acc', ...)          ||
//   | +---------------------------------------+|
//   | new_acc = gte(scan), index=@acc'         |
//   | ...                                      |
//   | ROOT = tuple(..., new_acc, ...)          |
//   +------------------------------------------+
//
class ScanLoopAccumulatorInputUnification : public HloModulePass {
 public:
  ~ScanLoopAccumulatorInputUnification() override = default;

  explicit ScanLoopAccumulatorInputUnification() = default;

  absl::string_view name() const override {
    return "scan_loop_accumulator_input_unification";
  }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_SCAN_LOOP_ACCUMULATOR_INPUT_UNIFICATION_H_
