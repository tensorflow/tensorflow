/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_SCATTER_SIMPLIFIER_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_SCATTER_SIMPLIFIER_H_

#include "tensorflow/compiler/xla/service/op_expander_pass.h"

namespace xla {

// This pass rewrites scatter operations into a combination of transposes,
// reshapes and a simpler scatter.
//
// It implements the first two steps of the algorithm decribed in
// ScatterExpander::ExpandInstruction (scatter_expander.cc). Additionally, it
// transposes updates and operands to transform scatter_dims_to_operand_dims
// into the identity mapping. This is different from the algorithm in
// ScatterExpander, which instead applies the mapping in scatter_indices.
//
// The output scatter's attributes will have the following characteristics:
// - scatter_indices is a two-dimensional tensor
// - index_vector_dim is 1
// - inserted_window_dims is []
// - update_window_dims is [0, 1, ...]
// - scatter_dims_to_operand_dims is [0, 1, ...]
//
// The purpose of this pass is to check whether this transformation has any
// performance implications.
class ScatterSimplifier : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "scatter_simplifier"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* inst) override;

  StatusOr<HloInstruction*> ExpandInstruction(HloInstruction* inst) override;
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_SCATTER_SIMPLIFIER_H_
