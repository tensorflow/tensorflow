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

#include "tensorflow/core/common_runtime/lower_function_call_inline_policy.h"

namespace tensorflow {

FunctionCallInlinePolicy GetFunctionCallInlinePolicy(
    bool is_partioned_call, bool has_lower_as_multi_device_function_attr) {
  if (is_partioned_call || has_lower_as_multi_device_function_attr)
    return FunctionCallInlinePolicy::kMultiDevicePlacer;
  return FunctionCallInlinePolicy::kSingleDevicePlacer;
}

FunctionCallInlinePolicy GetFunctionCallInlinePolicy(const Node* n) {
  bool match;
  bool found = TryGetNodeAttr(
      n->attrs(), LowerFunctionalOpsConstants::kLowerAsMultiDeviceFunctionAttr,
      &match);
  return GetFunctionCallInlinePolicy(n->IsPartitionedCall(), found && match);
}

}  // namespace tensorflow
