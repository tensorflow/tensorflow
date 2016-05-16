/* Copyright 2016 Google Inc. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLE_GRAPH_EXECUTION_STATE_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLE_GRAPH_EXECUTION_STATE_H_

#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "tensorflow/core/common_runtime/build_graph_options.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/step_stats.pb.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
class SessionOptions;
class StepStats;
class Timeline;

struct SimpleGraphExecutionStateOptions {
  const DeviceSet* device_set = nullptr;
  const SessionOptions* session_options = nullptr;
};

// A SimpleClientGraph is simply a sub-graph of the full graph as induced by
// BuildGraphOptions.
struct SimpleClientGraph {
  Graph graph;
  explicit SimpleClientGraph(const OpRegistryInterface* ops) : graph(ops) {}
  int32 placement_version;
};

// SimpleGraphExecutionState is responsible for generating an
// executable SimpleClientGraph from the original GraphDef that specifies
// the complete graph and from BuildGraphOptions which specifies
// input/output nodes.
//
// An executable Graph differs from a GraphDef by being Placed,
// meaning that each Node is assigned to a single Device in the
// available set.
//
// When SimpleGraphExecutionState is first constructed it instantiates
// a full Graph from the provided GraphDef, and places it, using only
// the static device assignments from the GraphDef.  Nodes without are
// currently placed in a very naive way.  Since stateful Nodes cannot
// be moved after initial placement, it is important that stateful
// Nodes get sensible initial device assignments in the graph
// definition.
//
// Subsequently, SimpleGraphExecutionState generates a SimpleClientGraph on
// demand, which is a sub-graph of the latest placement of the full
// Graph.  MasterSession uses such a SimpleClientGraph to execute one or
// more similar client requests.
//
// SimpleGraphExecutionState is thread-safe.

class SimpleGraphExecutionState {
 public:
  SimpleGraphExecutionState(const OpRegistryInterface* ops,
                            const SimpleGraphExecutionStateOptions& options);

  virtual ~SimpleGraphExecutionState();

  // Initializes the SimpleGraphExecutionState with 'graph_def'.  Can only be
  // called once on an original SimpleGraphExecutionState.  Callee may modify
  // 'graph_def'.
  Status Create(GraphDef* graph_def);

  // Creates a new SimpleGraphExecutionState representing the
  // concatenation of this graph, and the graph defined by
  // "extension_def". The same name may not be used to define a node
  // in both this graph and "extension_def".
  //
  // If successful, returns OK and the caller takes ownership of "*out".
  // Otherwise returns an error and does not modify "*out".
  //
  // NOTE(mrry): This method respects the placement of stateful nodes in
  // in *this, but currently does not transfer any other placement
  // or cost model information to the new graph.
  Status Extend(const GraphDef& extension_def,
                SimpleGraphExecutionState** out) const;

  // Builds a SimpleClientGraph (a sub-graph of the full graph as induced by
  // the Node set specified in "options").  If successful, returns OK
  // and the caller takes the ownership of "*out". Otherwise, returns
  // an error.
  Status BuildGraph(const BuildGraphOptions& options, SimpleClientGraph** out);

  // Returns OK if the named node is found in the placed full graph owned
  // by this execution_state, and sets *out to the NodeDef for that node.
  // It may not exist if name is of a Node added for a particular subgraph
  // execution, e.g. a send, recv or feed node.
  Status GlobalNodeDefByName(const string& name, NodeDef* out);

  // The graph returned by BuildGraph may contain only the pruned
  // graph, whereas some clients may want access to the full graph.
  const Graph* full_graph() {
    mutex_lock l(mu_);
    return graph_;
  }

  // Returns a reference to the current graph_def.  Use must
  // not extend beyond lifetime of SimpleGrahExecutionState object.
  const GraphDef& original_graph_def() { return original_graph_def_; }

  // Returns the map of stateful placements as a map of
  // node name to placement string.
  std::unordered_map<string, string> GetStatefulPlacements() const {
    mutex_lock l(mu_);
    return stateful_placements_;
  }

  // Restores the map of stateful placements as a map of
  // node name to placement string.
  void SetStatefulPlacements(const std::unordered_map<string, string>& sp) {
    mutex_lock l(mu_);
    stateful_placements_ = sp;
  }

 private:
  mutable mutex mu_;

  Status InitBaseGraph(const BuildGraphOptions& options)
      EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Map of placed stateful nodes, i.e. nodes for which is_stateful()
  // is true, such as "params" and "queue" nodes.  Once placed these
  // nodes can not be moved to a different device.  Maps node names to
  // device names.
  std::unordered_map<string, string> stateful_placements_ GUARDED_BY(mu_);
  void SaveStatefulNodes(Graph* graph) EXCLUSIVE_LOCKS_REQUIRED(mu_);
  void RestoreStatefulNodes(Graph* graph) EXCLUSIVE_LOCKS_REQUIRED(mu_);

  const OpRegistryInterface* const ops_;   // Not owned
  GraphDef original_graph_def_;            // Immutable after ctor.
  const DeviceSet* device_set_;            // Not owned
  const SessionOptions* session_options_;  // Not owned

  // The dataflow graph owned by this object.
  Graph* graph_ GUARDED_BY(mu_);

  TF_DISALLOW_COPY_AND_ASSIGN(SimpleGraphExecutionState);
};

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_SIMPLE_GRAPH_EXECUTION_STATE_H_
