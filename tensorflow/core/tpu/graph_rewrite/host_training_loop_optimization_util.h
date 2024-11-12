/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_GRAPH_REWRITE_HOST_TRAINING_LOOP_OPTIMIZATION_UTIL_H_
#define TENSORFLOW_CORE_TPU_GRAPH_REWRITE_HOST_TRAINING_LOOP_OPTIMIZATION_UTIL_H_

#include <optional>
#include <string>
#include <unordered_set>
#include <vector>

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/status.h"

namespace tensorflow {
namespace tpu {

struct LoopArgInfo {
  std::string enter_node_name;
  // Exit nodes are optional for loop invariant while loop args.
  std::optional<std::string> exit_node_name;
};

struct HostTrainingLoopInfo {
  // Name and attribute information about the function in which
  // host training loop is included. If host training loop is not
  // inside a function call, then `function_name` and `function_attrs`
  // are nullopt.
  std::optional<std::string> encapsulating_function_name;
  std::optional<AttrValueMap> encapsulating_function_attrs;

  // TPU Compile node as within a host training loop.
  std::string compile_node_name;

  // Name of the while loop in which TPU compile op is located.
  std::string while_loop_name;

  // Name of the node that represents loop condition.
  std::string loop_cond_node_name;

  // Exit and Enter node names for each loop arguments.
  std::vector<LoopArgInfo> loop_arguments;

  std::unordered_set<Node*> loop_nodes;  // NOLINT
};

// Walks through the `graph`, recursively if functional nodes exist, and
// identifies all host training loops. Host training loops are the inner
// most while loops that encapsulates TPUCompileOp node. This would be
// later used/analyzed to introduce host loop specific optimizations such
// as adding sharded weight update.
absl::Status DetectHostTrainingLoop(
    const std::string* current_function_name,
    const AttrValueMap* current_function_attr,
    const FunctionLibraryDefinition* library, Graph* graph,
    FunctionLibraryRuntime* flr,
    std::vector<HostTrainingLoopInfo>* host_training_loops_info);

// Injects VariableReshardOps to before and after TPUExecute op inside
// host training loop body. This effectively applies sharded weight update
// on model weight variables.
absl::Status AddReshardOp(Graph* graph,
                          const HostTrainingLoopInfo& host_loop_info);

}  // namespace tpu
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TPU_GRAPH_REWRITE_HOST_TRAINING_LOOP_OPTIMIZATION_UTIL_H_
