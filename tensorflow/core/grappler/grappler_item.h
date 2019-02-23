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

#ifndef TENSORFLOW_CORE_GRAPPLER_GRAPPLER_ITEM_H_
#define TENSORFLOW_CORE_GRAPPLER_GRAPPLER_ITEM_H_

#include <memory>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/variable.pb.h"
#include "tensorflow/core/protobuf/queue_runner.pb.h"

namespace tensorflow {
namespace grappler {

// A TensorFlow model to optimize.
// Models are represented by the combination of a graph, one of more fetch
// nodes, and potentially a set of nodes to feed.
struct GrapplerItem {
  GrapplerItem() = default;
  GrapplerItem(const GrapplerItem& other) = default;
  GrapplerItem(GrapplerItem&& other) = default;
  GrapplerItem& operator=(const GrapplerItem& other) = default;
  GrapplerItem& operator=(GrapplerItem&& other) = default;
  virtual ~GrapplerItem() = default;

  // Create a copy of this GrapplerItem with graph swapped with the argument.
  GrapplerItem WithGraph(GraphDef&& graph) const;

  string id;  // A unique id for this item

  // Inputs
  GraphDef graph;
  std::vector<std::pair<string, Tensor>> feed;
  std::vector<string> fetch;

  // Initialization op(s).
  std::vector<string> init_ops;
  // Expected initialization time in seconds, or 0 if unknown
  int64 expected_init_time = 0;

  // Save/restore ops (if any)
  string save_op;
  string restore_op;
  string save_restore_loc_tensor;

  // Queue runner(s) required to run the queue(s) of this model.
  std::vector<QueueRunnerDef> queue_runners;

  // List of op names to keep in the graph. This includes nodes that are
  // referenced in various collections, and therefore must be preserved to
  // ensure that the optimized metagraph can still be loaded.
  std::vector<string> keep_ops;

  // Return the set of node evaluated during a regular train/inference step.
  std::vector<const NodeDef*> MainOpsFanin() const;
  // Return the set of node run to populate the queues (if any).
  std::vector<const NodeDef*> EnqueueOpsFanin() const;
  // Return the set nodes used by TensorFlow to initialize the graph.
  std::vector<const NodeDef*> InitOpsFanin() const;
  // Return the set of variables accessed during a regular train/inference step.
  std::vector<const NodeDef*> MainVariables() const;
  // Return a set of node names that must be preserved. This includes feed and
  // fetch nodes, keep_ops, init_ops.
  std::unordered_set<string> NodesToPreserve() const;

  struct OptimizationOptions {
    // Is it allowed to add nodes to the graph that do not have registered
    // gradient function.
    bool allow_non_differentiable_rewrites = true;

    // Tensorflow function execution semantics is slightly different from the
    // main Tensorflow graph, and we need to make sure that we do not change it
    // by running Grappler optimizer passes. One main difference is that
    // functions do not prune ops with side-effects and dataset-output ops (see
    // PruneFunctionBody in common_runtime/function.cc).
    bool allow_pruning_stateful_and_dataset_ops = true;

    // If true Grappler will optimize the main graph, and also all functions in
    // the graph function library (function can't be polymorphic, it can't have
    // undefined type parameters in the function signature, or placeholder
    // attributes in the function body).
    bool optimize_function_library = true;
  };

  const std::unordered_set<string>& devices() const;
  // Adds a device to a set of available devices, only if it's a valid fully
  // defined device name. Returns `Status::OK()` if successfully added a device,
  // and an error otherwise.
  Status AddDevice(const string& device);
  // Adds all valid devices from the other Grappler item to the device set.
  Status AddDevices(const GrapplerItem& other);
  // Adds all valid devices from the nodes of the graph to the device set.
  // Returns `Status::OK()` if all device annotations found in a graph are valid
  // fully defined device names, and an error otherwise.
  Status InferDevicesFromGraph();
  // Clears a set of available devices.
  void ClearDevices();

  const OptimizationOptions& optimization_options() const;
  OptimizationOptions& optimization_options();

 private:
  // TODO(ezhulenev) Make GrapplerItem a class and hide all public data members.
  // TODO(ezhulenev): Migrate all unordered collections to absl.

  // A set of fully defined device names that can be used to place the nodes of
  // the `graph`.
  // Example of a fully defined name: "/job:work/replica:1/task:1/device:CPU:0"
  std::unordered_set<string> devices_;

  OptimizationOptions optimization_options_;
};

// Return the transitive fanin of a set of terminal nodes.
std::vector<const NodeDef*> ComputeTransitiveFanin(
    const GraphDef& graph, const std::vector<string>& terminal_nodes);

// Return the transitive fanin of a set of terminal nodes. Sets 'ill_formed' to
// true if one of the node is missing in the graph, or some node inputs don't
// exist.
std::vector<const NodeDef*> ComputeTransitiveFanin(
    const GraphDef& graph, const std::vector<string>& terminal_nodes,
    bool* ill_formed);

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_GRAPPLER_ITEM_H_
