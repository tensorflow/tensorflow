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

#include "tensorflow/compiler/mlir/tf2xla/internal/mlir_bridge_pass_util.h"

#include <functional>
#include <memory>
#include <string>

#include "absl/log/log.h"
#include "absl/status/status.h"
#include "mlir/Support/LogicalResult.h"  // from @llvm-project
#include "tensorflow/compiler/tf2xla/tf2xla_defs.h"
#include "tensorflow/core/common_runtime/function_body.h"
#include "tensorflow/core/common_runtime/function_def_utils.h"
#include "tensorflow/core/graph/graph.h"
#include "tsl/platform/status.h"

namespace tensorflow {

using ::mlir::failure;
using ::mlir::LogicalResult;
using ::mlir::success;

namespace {
LogicalResult HasAttr(
    const Graph& graph, const FunctionLibraryDefinition& function_library,
    const std::function<bool(const Graph& graph)>& predicate) {
  if (predicate(graph)) {
    return success();
  }

  // Check if any reachable functions from the graph has the target attribute.
  GraphDef graph_def;
  graph.ToGraphDef(&graph_def);
  for (const std::string& func_name :
       function_library.ReachableDefinitions(graph_def).ListFunctionNames()) {
    const FunctionDef* func_def = function_library.Find(func_name);
    std::unique_ptr<FunctionBody> func_body;
    absl::Status status = FunctionDefToBodyHelper(
        *func_def, AttrSlice(&func_def->attr()), &function_library, &func_body);
    // This is not expected to happen in practice
    if (!status.ok()) {
      LOG(ERROR) << "Failed to parse " << func_name << ": "
                 << tsl::NullTerminatedMessage(status);
      return failure();
    }
    if (predicate(*func_body->graph)) {
      return success();
    }
  }
  return failure();
}
}  // namespace

bool HasTpuReplicateAttr(const Graph& graph,
                         const FunctionLibraryDefinition& function_library) {
  auto predicate = [](const Graph& graph) {
    for (const Node* node : graph.nodes()) {
      // _tpu_replicate is used in replicated TPU graphs. It will be converted
      // to_replication_info and _xla_compile_device_type in phase 1 pipelines.
      if (node->attrs().FindByString(std::string(kTpuReplicateAttr))) {
        return true;
      }
    }
    return false;
  };
  return HasAttr(graph, function_library, predicate).succeeded();
}

bool HasCompileDeviceTypeAttr(
    const Graph& graph, const FunctionLibraryDefinition& function_library) {
  auto predicate = [](const Graph& graph) {
    for (const Node* node : graph.nodes()) {
      // _xla_compile_device_type is found in CPU/GPU graphs with top-level
      // compilation markers or single-core TPU graphs.
      if (auto attr =
              node->attrs().FindByString(std::string(kCompileDeviceTypeAttr))) {
        return true;
      }
    }
    return false;
  };
  return HasAttr(graph, function_library, predicate).succeeded();
}

bool IsNonReplicatedGraph(const Graph& graph,
                          const FunctionLibraryDefinition& function_library) {
  auto predicate = [](const Graph& graph) {
    const std::string kStatefulPartitionedCallOp = "StatefulPartitionedCall";
    for (const Node* node : graph.nodes()) {
      auto node_op = node->type_string();
      if (node_op == kStatefulPartitionedCallOp) {
        // Functions called by StatefulfulPartitionedCall ops with
        // _XlaMustCompile=true are compiled by XLA.
        auto attr = node->attrs().FindByString(std::string(kMustCompileAttr));
        if (attr != nullptr && attr->b() == true) {
          return true;
        }
      }
    }
    return false;
  };
  return HasAttr(graph, function_library, predicate).succeeded();
}

bool IsSingleCoreTpuGraph(const Graph& graph,
                          const FunctionLibraryDefinition& function_library) {
  auto predicate = [](const Graph& graph) {
    for (const Node* node : graph.nodes()) {
      // _xla_compile_device_type=TPU is found in single-core TPU graphs.
      auto attr =
          node->attrs().FindByString(std::string(kCompileDeviceTypeAttr));
      if (attr && attr->s() == kTpuDevice) {
        return true;
      }
    }
    return false;
  };
  return HasAttr(graph, function_library, predicate).succeeded();
}

}  // namespace tensorflow
