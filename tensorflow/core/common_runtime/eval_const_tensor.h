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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_EVAL_CONST_TENSOR_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_EVAL_CONST_TENSOR_H_

#include <cstdint>
#include <optional>

#include "absl/functional/function_ref.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {

class GraphRunner;
class Node;
class OpRegistryInterface;
class ShapeRefiner;
class Tensor;

// Configuration of the graph runner for constant folding.
struct EvaluateConstantTensorRunner {
  // Op registry for temporary graphs. By default, the global registry will
  // be used.
  const OpRegistryInterface* op_registry = nullptr;
  // Version of the graph API to use.
  int32_t graph_def_version = 0;
  // Graph runner for constant folding. By default, a temporary graph runner
  // will be created.
  GraphRunner* graph_runner = nullptr;
};

// Attempts to evaluate an output of the given node. This will only be possible
// if it doesn't depend on any graph inputs (this function is safe to call
// if this isn't the case though).
//
// When the evaluation is successful, the function returns a tensor, otherwise
// it returns std::nullopt.
StatusOr<std::optional<Tensor>> EvaluateConstantTensor(
    // The tensor to be evaluated.
    const Node& node, int node_output,
    // Used to fetch inference contexts for nodes in the graph.
    const ShapeRefiner& refiner,
    // Used to both lookup cached results and request function arguments.
    absl::FunctionRef<std::optional<Tensor>(const Node&, int)> lookup,
    // Configuration of the graph runner. If not set, no attempt to fold a
    // constant subgraph will be made.
    std::optional<EvaluateConstantTensorRunner> runner);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_EVAL_CONST_TENSOR_H_
