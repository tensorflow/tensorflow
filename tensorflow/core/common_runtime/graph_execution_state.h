/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_EXECUTION_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_EXECUTION_STATE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/build_graph_options.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
struct SessionOptions;

namespace subgraph {
struct RewriteGraphMetadata;
}

struct GraphExecutionStateOptions {
  const DeviceSet* device_set = nullptr;
  const SessionOptions* session_options = nullptr;
  // Unique session identifier. Can be empty.
  string session_handle;
  // A map from node name to device name, representing the unchangeable
  // placement of stateful nodes.
  std::unordered_map<string, string> stateful_placements;
  // Whether to run Placer on the graph.
  bool run_placer = true;

  // Whether to enable tf2xla mlir bridge. The default is true and intends to
  // work for almost all models. Non default values should only applied to
  // selective models.
  bool enable_tf2xla_mlir_bridge = true;
};

// A ClientGraph is simply a sub-graph of the full graph as induced by
// BuildGraphOptions.
struct ClientGraph {
  explicit ClientGraph(std::unique_ptr<FunctionLibraryDefinition> flib,
                       DataTypeVector feed_types, DataTypeVector fetch_types,
                       int64_t collective_graph_key)
      : flib_def(std::move(flib)),
        graph(flib_def.get()),
        feed_types(std::move(feed_types)),
        fetch_types(std::move(fetch_types)),
        collective_graph_key(collective_graph_key) {}
  // Each client-graph gets its own function library since optimization passes
  // post rewrite for execution might want to introduce new functions.
  std::unique_ptr<FunctionLibraryDefinition> flib_def;
  Graph graph;
  DataTypeVector feed_types;
  DataTypeVector fetch_types;
  int64_t collective_graph_key;
};

// GraphExecutionState is responsible for generating an
// executable ClientGraph from the original GraphDef that specifies
// the complete graph and from BuildGraphOptions which specifies
// input/output nodes.
//
// An executable Graph differs from a GraphDef by being Placed,
// meaning that each Node is assigned to a single Device in the
// available set.
//
// When GraphExecutionState is first constructed it instantiates
// a full Graph from the provided GraphDef, and places it, using only
// the static device assignments from the GraphDef.  Nodes without are
// currently placed in a very naive way.  Since stateful Nodes cannot
// be moved after initial placement, it is important that stateful
// Nodes get sensible initial device assignments in the graph
// definition.
//
// Subsequently, GraphExecutionState generates a SimpleClientGraph on
// demand, which is a sub-graph of the latest placement of the full
// Graph.  MasterSession uses such a ClientGraph to execute one or
// more similar client requests.
//
// GraphExecutionState is thread-safe.

class GraphExecutionState {
 public:
  virtual ~GraphExecutionState();

  // Creates a new `GraphExecutionState` for the given
  // `graph_def`, which represents the entire graph for a session.
  static absl::Status MakeForBaseGraph(
      GraphDef&& graph_def, const GraphExecutionStateOptions& options,
      std::unique_ptr<GraphExecutionState>* out_state);

  // Creates a new `GraphExecutionState` and `SimpleClientGraph`
  // for the subgraph of `original_graph_def` defined by
  // `subgraph_options`.
  static absl::Status MakeForPrunedGraph(
      const GraphExecutionState& base_execution_state,
      const GraphExecutionStateOptions& options,
      const BuildGraphOptions& subgraph_options,
      std::unique_ptr<GraphExecutionState>* out_state,
      std::unique_ptr<ClientGraph>* out_client_graph);

  // Creates a new GraphExecutionState representing the
  // concatenation of this graph, and the graph defined by
  // "extension_def". The same name may not be used to define a node
  // in both this graph and "extension_def".
  //
  // If successful, returns OK and the caller takes ownership of "*out".
  // Otherwise returns an error and does not modify "*out".
  //
  // After calling `old_state->Extend()`, `old_state` may no longer be
  // used.
  //
  // NOTE(mrry): This method respects the placement of stateful nodes in
  // in *this, but currently does not transfer any other placement
  // or cost model information to the new graph.
  //
  // Note that using this interface requires setting the value of
  // config.experimental().disable_optimize_for_static_graph() in the state
  // options to `true`, otherwise it will return an error.
  absl::Status Extend(const GraphDef& extension_def,
                      std::unique_ptr<GraphExecutionState>* out) const;

  // Builds a ClientGraph (a sub-graph of the full graph as induced by
  // the Node set specified in "options").  If successful, returns OK
  // and the caller takes the ownership of "*out". Otherwise, returns
  // an error.
  absl::Status BuildGraph(const BuildGraphOptions& options,
                          std::unique_ptr<ClientGraph>* out);

  // Optimize the graph with the node set specified in `options`.
  absl::Status OptimizeGraph(
      const BuildGraphOptions& options, const Graph& graph,
      const FunctionLibraryDefinition* flib_def,
      std::unique_ptr<Graph>* optimized_graph,
      std::unique_ptr<FunctionLibraryDefinition>* optimized_flib);

  // The graph returned by BuildGraph may contain only the pruned
  // graph, whereas some clients may want access to the full graph.
  const Graph* full_graph() { return graph_; }

  // The original graph.
  GraphDef* original_graph_def() { return original_graph_def_.get(); }

  // The original function library of this graph.
  const FunctionLibraryDefinition& flib_def() const { return *flib_def_; }

  // Returns the node with the given name, or null if it does not exist.
  const Node* get_node_by_name(const string& name) const {
    NodeNameToCostIdMap::const_iterator iter =
        node_name_to_cost_id_map_.find(name);
    if (iter != node_name_to_cost_id_map_.end()) {
      return graph_->FindNodeId(iter->second);
    } else {
      return nullptr;
    }
  }

  // Returns the map of stateful placements as a map of
  // node name to placement string.
  std::unordered_map<string, string> GetStatefulPlacements() const {
    return stateful_placements_;
  }

 private:
  GraphExecutionState(std::unique_ptr<GraphDef>&& graph_def,
                      std::unique_ptr<FunctionLibraryDefinition>&& flib_def,
                      const GraphExecutionStateOptions& options);

  absl::Status InitBaseGraph(std::unique_ptr<Graph>&& graph,
                             bool enable_tf2xla_mlir_bridge = true);

  // Map of placed stateful nodes, i.e. nodes for which is_stateful()
  // is true, such as "params" and "queue" nodes.  Once placed these
  // nodes can not be moved to a different device.  Maps node names to
  // device names.
  std::unordered_map<string, string> stateful_placements_;  // Immutable after
                                                            // ctor.
  void SaveStatefulNodes(Graph* graph);
  void RestoreStatefulNodes(Graph* graph);

  // Extract the subset of the graph that needs to be run, adding feed/fetch
  // ops as needed.
  absl::Status PruneGraph(const BuildGraphOptions& options, Graph* graph,
                          subgraph::RewriteGraphMetadata* out_rewrite_metadata);

  // The GraphExecutionState must store a copy of the original GraphDef if
  // either of the following conditions holds:
  //
  // * `session_options_.config.graph_options().place_pruned_graph()` is true.
  // * `session_options_.config.experimental().optimize_for_static_graph()` is
  //   false.
  const std::unique_ptr<GraphDef> original_graph_def_;

  const DeviceSet* device_set_;            // Not owned
  const SessionOptions* session_options_;  // Not owned
  // Unique session identifier. Can be empty.
  string session_handle_;

  // Map from name to Node for the full graph in placed_.
  NodeNameToCostIdMap node_name_to_cost_id_map_;

  // 'flib_def_' is initialized from the initial graph def's library,
  // and may be updated by a graph optimization pass.
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;

  // `rewrite_metadata_` is only set for GraphExecutionState
  // objects created by `MakeForPrunedGraph()`.
  std::unique_ptr<subgraph::RewriteGraphMetadata> rewrite_metadata_;

  // The dataflow graph owned by this object.
  Graph* graph_;

  // Whether to run Placer.
  bool run_placer_;

  GraphExecutionState(const GraphExecutionState&) = delete;
  void operator=(const GraphExecutionState&) = delete;
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_EXECUTION_STATE_H_
