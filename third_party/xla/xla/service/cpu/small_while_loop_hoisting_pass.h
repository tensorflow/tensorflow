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

#ifndef XLA_SERVICE_CPU_SMALL_WHILE_LOOP_HOISTING_PASS_H_
#define XLA_SERVICE_CPU_SMALL_WHILE_LOOP_HOISTING_PASS_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/pass/hlo_pass_interface.h"
namespace xla::cpu {

// Hoists small while loops into a separate function.
// This pass enables the thunk emitter to emit small while loops as a single
// kernel instead of a series of thunks.
class SmallWhileLoopHoistingPass final : public HloModulePass {
 public:
  explicit SmallWhileLoopHoistingPass(int64_t small_buffer_access_size);

  absl::string_view name() const final { return "small-while-loop-hoisting"; }
  using HloPassInterface::Run;
  absl::StatusOr<bool> Run(
      HloModule* module,
      const absl::flat_hash_set<absl::string_view>& execution_threads) final;

 private:
  absl::StatusOr<bool> IsSmall(const HloInstruction* instr);

 private:
  int64_t small_buffer_access_size_;
};
}  // namespace xla::cpu

#endif  // XLA_SERVICE_CPU_SMALL_WHILE_LOOP_HOISTING_PASS_H_
