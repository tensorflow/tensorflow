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

#ifndef XLA_BACKENDS_CPU_TRANSFORMS_EMBEDDED_WHILE_LOOP_UNROLLER_H_
#define XLA_BACKENDS_CPU_TRANSFORMS_EMBEDDED_WHILE_LOOP_UNROLLER_H_

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"

namespace xla::cpu {

// Fully unrolls while loops inside embedded computations (sort comparators,
// reducers, ...), which the CPU nested IR emitter cannot emit as loops. Such
// loops typically come from expanding a scatter in those computations. Loops
// that cannot be unrolled are left unchanged.
class EmbeddedWhileLoopUnroller : public HloModulePass {
 public:
  absl::string_view name() const override {
    return "embedded-while-loop-unroller";
  }

 protected:
  absl::StatusOr<bool> RunImpl(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) override;
};

}  // namespace xla::cpu

#endif  // XLA_BACKENDS_CPU_TRANSFORMS_EMBEDDED_WHILE_LOOP_UNROLLER_H_
