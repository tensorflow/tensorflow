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

// Helper functions for tests.

#ifndef TENSORFLOW_COMPILER_JIT_TEST_UTIL_H_
#define TENSORFLOW_COMPILER_JIT_TEST_UTIL_H_

#include <map>
#include <unordered_map>
#include <vector>

#include "tensorflow/compiler/jit/shape_inference.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/partial_tensor_shape.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/public/session_options.h"

namespace tensorflow {

// Tests that the shapes in 'shape_info' for the nodes in `graph` match
// `expected_shapes`. Returns an error if there are nodes in `expected_shapes`
// that do not have shape information. Ignores nodes in `graph` that do not have
// `expected_shapes` entries.
Status ShapeAnnotationsMatch(
    const Graph& graph, const GraphShapeInfo& shape_info,
    std::map<string, std::vector<PartialTensorShape>> expected_shapes);

// A helper object to create GraphOptimizationPassOptions.
struct GraphOptimizationPassWrapper {
  explicit GraphOptimizationPassWrapper() : library(OpRegistry::Global(), {}) {
    session_options.env = Env::Default();
  }

  // Create GraphOptimizationPassOptions with a graph passed in constructor and
  // sensible options.
  GraphOptimizationPassOptions CreateGraphOptimizationPassOptions(
      std::unique_ptr<Graph>* graph) {
    GraphOptimizationPassOptions options;
    options.session_options = &session_options;
    options.flib_def = &library;
    options.graph = graph;
    return options;
  }

  FunctionLibraryDefinition library;
  SessionOptions session_options;
};

}  // namespace tensorflow


#endif  // TENSORFLOW_COMPILER_JIT_TEST_UTIL_H_
