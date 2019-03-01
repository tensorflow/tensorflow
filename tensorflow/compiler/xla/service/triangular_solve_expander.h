/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_TRIANGULAR_SOLVE_EXPANDER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_TRIANGULAR_SOLVE_EXPANDER_H_

#include "absl/container/flat_hash_map.h"
#include "tensorflow/compiler/xla/service/op_expander_pass.h"

namespace xla {

class TriangularSolveExpander : public OpExpanderPass {
 public:
  absl::string_view name() const override {
    return "triangular_solve_expander";
  }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  StatusOr<HloInstruction*> ExpandInstruction(
      HloInstruction* instruction) override;

 private:
  // Mapping from op signatures to existing computations.
  absl::flat_hash_map<string, HloComputation*> computation_cache_;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_TRIANGULAR_SOLVE_EXPANDER_H_
