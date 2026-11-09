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

#include "xla/hlo/ir/hlo_computation_utils.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/hlo/ir/hlo_schedule.h"

namespace xla::hlo_computation_utils {

HloComputation* CloneComputation(HloComputation* computation,
                                 absl::string_view suffix) {
  CHECK_NE(computation, nullptr);
  HloModule* module = computation->parent();
  CHECK_NE(module, nullptr);
  std::string suffix_str(suffix);
  if (!module->has_schedule() ||
      !module->schedule().is_computation_scheduled(computation)) {
    return module->AddEmbeddedComputation(suffix_str.empty()
                                              ? computation->Clone()
                                              : computation->Clone(suffix_str));
  }
  auto [clone, clone_sequence] =
      suffix_str.empty() ? computation->CloneWithSchedule()
                         : computation->CloneWithSchedule(suffix_str);
  HloComputation* clone_ptr = module->AddEmbeddedComputation(std::move(clone));
  module->schedule().set_sequence(clone_ptr, clone_sequence);
  return clone_ptr;
}

}  // namespace xla::hlo_computation_utils
