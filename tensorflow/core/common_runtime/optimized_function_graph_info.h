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
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZED_FUNCTION_GRAPH_INFO_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZED_FUNCTION_GRAPH_INFO_H_

#include <memory>
#include <string>
#include <unordered_map>

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/optimized_function_graph.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/statusor.h"

namespace tensorflow {

// Function graph related information after optimizations. This struct can be
// converted to and from
// third_party/tensorflow/core/framework/optimized_function_graph.proto.
struct OptimizedFunctionGraphInfo {
  // enum for distinguishing the origin where the proto is created.
  //
  // kAot: proto is created in ahead-of-time environment, which can be different
  // from the environment where the graph is actually executed.
  //
  // kJit: proto is created in just-in-time execution, which has the same
  // environment as the one the graph is actually executed.
  enum Source { kAot, kJit };

  // Function name.
  string name;
  // Optimized function graph.
  std::unique_ptr<Graph> function_graph;
  // Optimized function library.
  FunctionLibraryDefinition lib_def;
  // Map from original node names to control return names.
  std::unordered_map<string, string> node_name_to_control_ret;
  // Return node types of the function.
  DataTypeVector ret_types;
  // Number of return nodes.
  size_t num_return_nodes;
  // Time spent on running the graph optimization passes for this funciton.
  uint64_t optimization_time_sec;
  // Indicates the source environment where this proto is generated.
  Source source;

  // Converts from the struct to OptimizedFunctionGraph proto.
  static OptimizedFunctionGraph ToProto(const OptimizedFunctionGraphInfo& info);

  // Converts from the proto to struct OptimizedFunctionGraphInfo. Returns error
  // if the conversion fails.
  static StatusOr<OptimizedFunctionGraphInfo> FromProto(
      const OptimizedFunctionGraph& proto);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZED_FUNCTION_GRAPH_INFO_H_
