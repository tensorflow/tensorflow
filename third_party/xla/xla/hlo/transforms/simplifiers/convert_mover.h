/* Copyright 2022 The OpenXLA Authors.

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

#ifndef XLA_HLO_TRANSFORMS_SIMPLIFIERS_CONVERT_MOVER_H_
#define XLA_HLO_TRANSFORMS_SIMPLIFIERS_CONVERT_MOVER_H_

#include <functional>
#include <utility>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla {

// Moves narrowing conversions up the graph and widening conversions down the
// graph, when we can do so with no effect on numerics. Motivations:
//
//  - It's preferable to spend more of our time in lower precision and less of
//    our time in higher precision.
//
//  - Moving these converts exposes optimization opportunities. For example, in
//    reshape(convert-big-to-small(reshape(convert-small-to-big(x)))), we can
//    commute one of the converts with one of the reshapes.  This leaves us with
//    convert(convert(reshape(reshape))), which can probably be simplified
//    further by algsimp.
class ConvertMover : public HloModulePass {
 public:
  ConvertMover() = default;

  absl::string_view name() const override { return "convert-mover"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla

#endif  // XLA_HLO_TRANSFORMS_SIMPLIFIERS_CONVERT_MOVER_H_
