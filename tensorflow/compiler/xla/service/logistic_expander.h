/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_LOGISTIC_EXPANDER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_LOGISTIC_EXPANDER_H_

#include <utility>

#include "tensorflow/compiler/xla/service/hlo_module.h"
#include "tensorflow/compiler/xla/service/hlo_pass_interface.h"
#include "tensorflow/compiler/xla/service/op_expander_pass.h"

namespace xla {

enum class LogisticExpansionType {
  kTanh,  // Expands as 0.5 + 0.5*tanh(0.5*x)
  kExp,   // Expands as 1.0 / (1.0 + exp(-x))
};

// A pass which performs expansion of the logistic function.
class LogisticExpander : public OpExpanderPass {
 public:
  explicit LogisticExpander(LogisticExpansionType expansion_type)
      : expansion_type_(expansion_type) {}
  ~LogisticExpander() override = default;
  absl::string_view name() const override { return "logistic-expander"; }

 private:
  // Returns `true` if `instruction` should be expanded by this pass.
  bool InstructionMatchesPattern(HloInstruction* instruction) override;
  // Returns a replacement for `instruction`, or nullptr if no replacement is
  // needed (e.g. only the to_apply subcomputation of the instruction was
  // modified).
  StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;
  LogisticExpansionType expansion_type_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_LOGISTIC_EXPANDER_H_
