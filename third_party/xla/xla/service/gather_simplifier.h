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

#ifndef XLA_SERVICE_GATHER_SIMPLIFIER_H_
#define XLA_SERVICE_GATHER_SIMPLIFIER_H_

#include "xla/service/op_expander_pass.h"

namespace xla {

// This pass rewrites gather operations into a combination of transposes,
// reshapes and a simpler gather.
//
// The output gather's attributes will have the following characteristics:
// - start_indices is a two-dimensional tensor
// - index_vector_dim is 1
// - start_index_map is [0, 1, ...]
// - collapsed_slice_dims is []
// - offset_dims is [1, 2, ...]
//
// The purpose of this pass is to check whether this transformation has any
// performance implications.
class GatherSimplifier : public OpExpanderPass {
 public:
  absl::string_view name() const override { return "gather_simplifier"; }

 protected:
  bool InstructionMatchesPattern(HloInstruction* inst) override;

  StatusOr<HloInstruction*> ExpandInstruction(HloInstruction* inst) override;
};

}  // namespace xla

#endif  // XLA_SERVICE_GATHER_SIMPLIFIER_H_
