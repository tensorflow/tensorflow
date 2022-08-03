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
#ifndef TENSORFLOW_CORE_TFRT_UTILS_TFRT_GRAPH_EXECUTION_STATE_H_
#define TENSORFLOW_CORE_TFRT_UTILS_TFRT_GRAPH_EXECUTION_STATE_H_

#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "tensorflow/compiler/mlir/tensorflow/translate/mlir_roundtrip_flags.h"
#include "tensorflow/core/common_runtime/graph_execution_state.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/tfrt/fallback/fallback_state.h"

namespace tensorflow {
namespace tfrt_stub {

// This is a TFRT variant of `tensorflow::GraphExecutionState`. It wraps
// `tensorflow::GraphExecutionState` and adds TFRT-specific adjustments.
//
// Responsible for generating an executable `Graph` from the original `GraphDef`
// that specifies the complete graph and from `GraphImportConfig` that specifies
// input/output nodes.
//
// Thread-safe.
class TfrtGraphExecutionState {
 public:
  struct OptimizationResult {
    std::unique_ptr<tensorflow::Graph> graph;
    absl::Duration functionalization_duration;
    absl::Duration grappler_duration;
  };

  struct Options {
    bool run_placer_grappler_on_functions = false;
    bool enable_tfrt_gpu = false;
  };

  // Creates a `GraphExecutionState` given `graph_def` and `fallback_state`.
  static StatusOr<std::unique_ptr<TfrtGraphExecutionState>> Create(
      const Options& options, tensorflow::GraphDef graph_def,
      const FallbackState& fallback_state);

  // Ctor. Do not use directly. Public only for `std::make_unique<>()`.
  TfrtGraphExecutionState(
      const Options& options,
      std::unique_ptr<tensorflow::GraphExecutionState> graph_execution_state,
      const FallbackState& fallback_state,
      absl::flat_hash_set<std::string> functions_to_optimize)
      : options_(options),
        graph_execution_state_(std::move(graph_execution_state)),
        fallback_state_(fallback_state),
        functions_to_optimize_(std::move(functions_to_optimize)) {}

  // Creates an optimized graph by pruning with `graph_import_config` and
  // best-effort Grappler run.
  StatusOr<OptimizationResult> CreateOptimizedGraph(
      tensorflow::GraphImportConfig& graph_import_config);

  // Extends the current graph by `graph`.
  Status Extend(const GraphDef& graph);

  // Return the preprocessed full graph. Note that it does not contain the
  // function library in the original graph.
  const tensorflow::Graph& graph() const {
    absl::MutexLock lock(&graph_execution_state_mu_);
    DCHECK(graph_execution_state_->full_graph());
    return *graph_execution_state_->full_graph();
  }

  // The original graph.
  const GraphDef* original_graph_def() const {
    absl::MutexLock lock(&graph_execution_state_mu_);
    return graph_execution_state_->original_graph_def();
  }

 private:
  // Return the function library in the original graph.
  const FunctionLibraryDefinition& flib_def() const {
    absl::MutexLock lock(&graph_execution_state_mu_);
    return graph_execution_state_->flib_def();
  }

  StatusOr<std::unique_ptr<tensorflow::Graph>> OptimizeGraph(
      const tensorflow::Graph& graph,
      const tensorflow::BuildGraphOptions& build_graph_options);

  Options options_;

  std::unique_ptr<tensorflow::GraphExecutionState> graph_execution_state_
      ABSL_GUARDED_BY(graph_execution_state_mu_);
  // We need this mutex even thought `GraphExecutionState` is thread-safe,
  // because `swap()` is not thread-safe.
  mutable absl::Mutex graph_execution_state_mu_;

  const FallbackState& fallback_state_;
  // Only valid if `options_.run_placer_grappler_on_functions` is true.
  absl::flat_hash_set<std::string> functions_to_optimize_;
};

// Prunes the `graph_def` using the feed/fetch nodes specified in
// `callable_options`. It is a TFRT-specific version that it performs more
// pruning (e.g., prunes the input edges to the feed nodes) than
// `ComputeTransitiveFanin()` so that the graph can be functionalized properly
// later.
Status PruneGraphDef(GraphDef& graph_def,
                     const CallableOptions& callable_options);

// Eliminates ref variables in V1 control flow, which is required for
// functionalization. Current strategy is to insert an identity node between
// each ref node and its ref input and in-place update the ref node to its
// non-ref counterpart.
Status EliminateRefVariablesFromV1ControlFlow(GraphDef& graph_def);

// Removes the "_input_shapes" attribute of functions in the graph.
void RemoveInputShapesInFunctions(tensorflow::GraphDef& graph_def);

// Replaces partitioned calls in the graph that have _XlaMustCompile attribute
// set to true with XlaLaunch op.
// TODO(b/239089915): Clean this up after the logic is implemented in TFXLA
// bridge.
Status BuildXlaLaunchOps(Graph* graph);

}  // namespace tfrt_stub
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_TFRT_UTILS_TFRT_GRAPH_EXECUTION_STATE_H_
