/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_OPTIMIZATION_BARRIER_EXPANDER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_OPTIMIZATION_BARRIER_EXPANDER_H_

#include "tensorflow/compiler/xla/service/op_expander_pass.h"

namespace xla {

// This pass removes the opt-barrier operation which is functionally a no-op.
class OptimizationBarrierExpander : public OpExpanderPass {
 public:
  OptimizationBarrierExpander() = default;

  absl::string_view name() const override { return "cse_barrier_expander"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* instruction) override;

  StatusOr<HloInstruction*> ExpandInstruction(HloInstruction* hlo) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_OPTIMIZATION_BARRIER_EXPANDER_H_
