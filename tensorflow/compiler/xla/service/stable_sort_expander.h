/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_STABLE_SORT_EXPANDER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_STABLE_SORT_EXPANDER_H_

#include "tensorflow/compiler/xla/service/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/op_expander_pass.h"
#include "tensorflow/compiler/xla/statusor.h"

namespace xla {

// HLO pass which expands Sort ops that have the is_stable field set to true
// into equivalent Sort ops which guarantee stable sorting without relying on
// the is_stable field.
class StableSortExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "stable-sort-expander"; }

 private:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;
  StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_STABLE_SORT_EXPANDER_H_
