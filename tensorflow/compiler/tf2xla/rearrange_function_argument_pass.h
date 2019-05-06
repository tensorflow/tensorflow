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

#ifndef TENSORFLOW_COMPILER_TF2XLA_REARRANGE_FUNCTION_ARGUMENT_PASS_H_
#define TENSORFLOW_COMPILER_TF2XLA_REARRANGE_FUNCTION_ARGUMENT_PASS_H_

#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {

// For the function with `func_name`, rewrite any
// StatefulPartitionedCall/If/While node that does not satisfy the rules.
// We will rewrite related FunctionDef to rearrange arguments and return values,
// also adjust node's input/output edges accordingly.
Status RearrangeFunctionArgumentForFunction(
    const string& func_name, const string& new_func_name,
    const protobuf::Map<string, tensorflow::AttrValue>& attrs,
    FunctionLibraryDefinition* fld, FunctionLibraryRuntime* flr,
    std::map<string, absl::optional<string>>* canonicalized_name_to_new_name,
    bool* modified);

// TF/XLA bridge expects FunctionDef to satisfy the following rules:
// 1. DT_RESOURCE arguments are always in the last;
// 2. Do not return DT_RESOURCE as return values.
// But functions defined by Tensorflow might not satisfy them.
// This rewrite pass rewrites the function for TPUCompile/XlaLaunch node
// to follow the rules, using RearrangeFunctionArgumentForFunction() above.
class RearrangeFunctionArgumentPass : public GraphOptimizationPass {
 public:
  Status Run(const GraphOptimizationPassOptions& options) override;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_COMPILER_TF2XLA_REARRANGE_FUNCTION_ARGUMENT_PASS_H_
