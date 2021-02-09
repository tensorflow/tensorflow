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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_FUNCTIONAL_OPS_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_FUNCTIONAL_OPS_H_

#include "absl/types/optional.h"
#include "tensorflow/core/common_runtime/inline_function_utils.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {

// Rewrite functional ops into low level primitives:
// - If/While ops lowered into low level control flow primitives: Switch, Merge,
//   Enter, Exit, NextIteration
// - Function calls inlined into the main graph
//
// IMPORTANT: Although SymbolicGradient is a function call, we currently do not
// lower it, because it has been deprecated for a while.
class LowerFunctionalOpsPass : public GraphOptimizationPass {
 public:
  LowerFunctionalOpsPass() = default;
  LowerFunctionalOpsPass(bool keep_lowered_nodes_fetchable)
      : keep_lowered_nodes_fetchable_(keep_lowered_nodes_fetchable) {}

  Status Run(const GraphOptimizationPassOptions& options) override;

  static constexpr const char* const kLowerUsingSwitchMergeAttr =
      LowerFunctionalOpsConstants::kLowerUsingSwitchMergeAttr;
  static constexpr const char* const kLowerAsMultiDeviceFunctionAttr =
      LowerFunctionalOpsConstants::kLowerAsMultiDeviceFunctionAttr;

 private:
  // If defined use the value to control if functional ops must be fetchable
  // after lowering (we add IdentityN in place of all lowered nodes). If not
  // defined, this option will be inferred automatically from the graph (in
  // presence of _Retval or _Arg nodes we do not need to keep nodes fetchable).
  absl::optional<bool> keep_lowered_nodes_fetchable_;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_FUNCTIONAL_OPS_H_
