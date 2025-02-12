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

// Classes to maintain a static registry of whole-graph optimization
// passes to be applied by the Session when it initializes a graph.
#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZATION_REGISTRY_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZATION_REGISTRY_H_

#include <functional>
#include <map>
#include <vector>

#include "tensorflow/core/common_runtime/composite_device.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/costmodel.h"
#include "tensorflow/core/graph/graph.h"

namespace tensorflow {
struct SessionOptions;

// All the parameters used by an optimization pass are packaged in
// this struct. They should be enough for the optimization pass to use
// as a key into a state dictionary if it wants to keep state across
// calls.
struct GraphOptimizationPassOptions {
  // Filled in by DirectSession for PRE_PLACEMENT optimizations. Can be empty.
  string session_handle;
  const SessionOptions* session_options = nullptr;
  const CostModel* cost_model = nullptr;

  FunctionLibraryDefinition* flib_def = nullptr;  // Not owned.
  // The DeviceSet contains all the devices known to the system and is
  // filled in for optimizations run by the session master, i.e.,
  // PRE_PLACEMENT, POST_PLACEMENT, and POST_REWRITE_FOR_EXEC. It is
  // nullptr for POST_PARTITIONING optimizations which are run at the
  // workers.
  const DeviceSet* device_set = nullptr;  // Not owned.

  // Maps from a CompositeDevice name to a list of underlying physical
  // devices.
  const std::vector<CompositeDevice*>* composite_devices =
      nullptr;  // Not owned.

  // The graph to optimize, for optimization passes that run before
  // partitioning. Null for post-partitioning passes.
  // An optimization pass may replace *graph with a new graph object.
  std::unique_ptr<Graph>* graph = nullptr;

  // Graphs for each partition, if running post-partitioning. Optimization
  // passes may alter the graphs, but must not add or remove partitions.
  // Null for pre-partitioning passes.
  std::unordered_map<string, std::unique_ptr<Graph>>* partition_graphs =
      nullptr;

  // Indicator of whether or not the graph was derived from a function.
  bool is_function_graph = false;
  // Set when is_function_graph is true. The default device where the function
  // runs. If nullptr, it runs on the local host.
  const Device* default_function_device = nullptr;
  // Set when is_function_graph is true. The function where the graph was
  // derived. `graph` doesn't contain all the information in the function_def,
  // e.g. function attributes.
  const FunctionDef* function_def = nullptr;

  // TODO(b/176491312): Remove this if shape inference on import flag is
  // removed. If True, allows mlir roundtrip to run shape inference on import.
  bool shape_inference_on_tfe_dialect_import = true;

  // A unique filename prefix (using hostname, process ID, thread ID and
  // timestamp) for graph dumps.
  string debug_filename_prefix;

  // Whether to enable tf2xla mlir bridge in compiling SavedModel.
  bool enable_tf2xla_mlir_bridge = true;
};

// Optimization passes are implemented by inheriting from
// GraphOptimizationPass.
class GraphOptimizationPass {
 public:
  virtual ~GraphOptimizationPass() {}
  virtual absl::Status Run(const GraphOptimizationPassOptions& options) = 0;
  void set_name(const string& name) { name_ = name; }
  string name() const { return name_; }

 private:
  // The name of the optimization pass, which is the same as the inherited
  // class name.
  string name_;
};

// The key is a 'phase' number. Phases are executed in increasing
// order. Within each phase the order of passes is undefined.
typedef std::map<int, std::vector<std::unique_ptr<GraphOptimizationPass>>>
    GraphOptimizationPasses;

// A global OptimizationPassRegistry is used to hold all passes.
class OptimizationPassRegistry {
 public:
  // Groups of passes are run at different points in initialization.
  enum Grouping {
    PRE_PLACEMENT,          // after cost model assignment, before placement.
    POST_PLACEMENT,         // after placement.
    POST_REWRITE_FOR_EXEC,  // after re-write using feed/fetch endpoints.
    POST_PARTITIONING,      // after partitioning
  };

  // Add an optimization pass to the registry.
  void Register(Grouping grouping, int phase,
                std::unique_ptr<GraphOptimizationPass> pass);

  const std::map<Grouping, GraphOptimizationPasses>& groups() {
    return groups_;
  }

  // Run all passes in grouping, ordered by phase, with the same
  // options.
  absl::Status RunGrouping(Grouping grouping,
                           const GraphOptimizationPassOptions& options);

  // Returns the global registry of optimization passes.
  static OptimizationPassRegistry* Global();

  // Prints registered optimization passes for debugging.
  void LogGrouping(Grouping grouping, int vlog_level);
  void LogAllGroupings(int vlog_level);

 private:
  std::map<Grouping, GraphOptimizationPasses> groups_;

  const char* GetGroupingName(Grouping grouping) const {
    switch (grouping) {
      case PRE_PLACEMENT:
        return "pre_placement";
      case POST_PLACEMENT:
        return "post_placement";
      case POST_REWRITE_FOR_EXEC:
        return "post_rewrite_for_exec";
      case POST_PARTITIONING:
        return "post_partitioning";
    }
    return "unknown";
  }
};

namespace optimization_registration {

class OptimizationPassRegistration {
 public:
  OptimizationPassRegistration(OptimizationPassRegistry::Grouping grouping,
                               int phase,
                               std::unique_ptr<GraphOptimizationPass> pass,
                               string optimization_pass_name) {
    pass->set_name(optimization_pass_name);
    OptimizationPassRegistry::Global()->Register(grouping, phase,
                                                 std::move(pass));
  }
};

}  // namespace optimization_registration

#define REGISTER_OPTIMIZATION(grouping, phase, optimization) \
  REGISTER_OPTIMIZATION_UNIQ_HELPER(__COUNTER__, grouping, phase, optimization)

#define REGISTER_OPTIMIZATION_UNIQ_HELPER(ctr, grouping, phase, optimization) \
  REGISTER_OPTIMIZATION_UNIQ(ctr, grouping, phase, optimization)

#define REGISTER_OPTIMIZATION_UNIQ(ctr, grouping, phase, optimization)         \
  static ::tensorflow::optimization_registration::OptimizationPassRegistration \
      register_optimization_##ctr(                                             \
          grouping, phase,                                                     \
          ::std::unique_ptr<::tensorflow::GraphOptimizationPass>(              \
              new optimization()),                                             \
          #optimization)

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZATION_REGISTRY_H_
