/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_CONSTANT_FOLDING_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_CONSTANT_FOLDING_H_

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/env.h"

// TODO(skyewm): can this be combined with EvaluateConstantTensor?

namespace tensorflow {

// This generator type is used to generate a name for the newly folded node
// based on the node's old name.
using ConstantFoldNameGenerator =
    std::function<string(Graph* graph, string old_name)>;

// Options specific to constant folding optimizations.
struct ConstantFoldingOptions {
  // If "consider" is not a nullptr, then only constant fold a node "n" if
  // consider(n) returns true.
  std::function<bool(const Node*)> consider = nullptr;
  // If shape_map is not a nullptr, it is a map from node n to a
  // vector of the (potentially partially-known) shapes of its
  // outputs.
  const std::unordered_map<string, std::vector<PartialTensorShape>>* shape_map =
      nullptr;  // not owned
  // The maximum size of each constant created during constant folding
  // optimization.
  int64 max_constant_size_in_bytes = 10 * 1024 * 1024;

  // A generator for the name suffix of constant folded nodes. A
  // default id generator that monotonically increases is used if nullptr is
  // passed.
  ConstantFoldNameGenerator generate_new_name = nullptr;
};

// Perform constant folding optimization on "graph".
// Looks for nodes in "graph" that can be completely evaluated statically, i.e.,
// that are only dependent on constants. Evaluates those nodes on a CPU device
// and replaces those nodes with the result of the evaluation.
// "partition_device", if non-null, is the device where all the graph nodes are
// assumed to execute.
// Sets `was_mutated` to true if and only if "graph" has been mutated.
// The status is only set to a non-OK state if an unexpected error is hit
// running the graph.
Status ConstantFold(const ConstantFoldingOptions& opts,
                    FunctionLibraryRuntime* function_library, Env* env,
                    const Device* partition_device, Graph* graph,
                    bool* was_mutated);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_CONSTANT_FOLDING_H_
