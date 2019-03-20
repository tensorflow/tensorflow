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
  explicit LowerFunctionalOpsPass(bool lower_function_calls)
      : lower_function_calls_(lower_function_calls) {}

  Status Run(const GraphOptimizationPassOptions& options) override;

  static constexpr const char* const kLowerUsingSwitchMergeAttr =
      "_lower_using_switch_merge";
  static constexpr const char* const kLowerAsMultiDeviceFunctionAttr =
      "_lower_as_multi_device_function";

 private:
  // TODO(ezhulenev): This is only required until Grappler function optimizer is
  // not migrated to use function inlining from common_runtime.
  bool lower_function_calls_ = true;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_LOWER_FUNCTIONAL_OPS_H_
