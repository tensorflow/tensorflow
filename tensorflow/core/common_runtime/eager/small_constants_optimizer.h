/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_SMALL_CONSTANTS_OPTIMIZER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_SMALL_CONSTANTS_OPTIMIZER_H_

#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/function.pb.h"

namespace tensorflow::small_constants_optimizer {

// Checks whether small constant optimization is enabled for a tf.function.
bool IsSmallConstantOptimizationEnabled(const FunctionDef& fdef);

// Generates new FunctionDefs with the boolean input tensors folded as
// constants into the FunctionDef.
std::vector<FunctionDef> FoldInputTensors(
    const FunctionDef& fdef, const FunctionLibraryDefinition& flib);

// Generates the FunctionDef name for the folded function.
std::string FoldedFunctionName(absl::string_view fname,
                               absl::string_view input_name, bool input_value);

}  // namespace tensorflow::small_constants_optimizer

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EAGER_SMALL_CONSTANTS_OPTIMIZER_H_
