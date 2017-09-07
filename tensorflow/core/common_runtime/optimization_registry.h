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

  // The graph to optimize, for optimization passes that run before
  // partitioning. Null for post-partitioning passes.
  // An optimization pass may replace *graph with a new graph object.
  std::unique_ptr<Graph>* graph = nullptr;

  // Graphs for each partition, if running post-partitioning. Optimization
  // passes may alter the graphs, but must not add or remove partitions.
  // Null for pre-partitioning passes.
  std::unordered_map<string, std::unique_ptr<Graph>>* partition_graphs =
      nullptr;
};

// Optimization passes are implemented by inheriting from
// GraphOptimizationPass.
class GraphOptimizationPass {
 public:
  virtual ~GraphOptimizationPass() {}
  virtual Status Run(const GraphOptimizationPassOptions& options) = 0;
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

  // Run all passes in grouping, ordered by phase, with the same
  // options.
  Status RunGrouping(Grouping grouping,
                     const GraphOptimizationPassOptions& options);

  // Returns the global registry of optimization passes.
  static OptimizationPassRegistry* Global();

 private:
  std::map<Grouping, GraphOptimizationPasses> groups_;
};

namespace optimization_registration {

class OptimizationPassRegistration {
 public:
  OptimizationPassRegistration(OptimizationPassRegistry::Grouping grouping,
                               int phase,
                               std::unique_ptr<GraphOptimizationPass> pass) {
    OptimizationPassRegistry::Global()->Register(grouping, phase,
                                                 std::move(pass));
  }
};

}  // namespace optimization_registration

#define REGISTER_OPTIMIZATION(grouping, phase, optimization) \
  REGISTER_OPTIMIZATION_UNIQ_HELPER(__COUNTER__, grouping, phase, optimization)

#define REGISTER_OPTIMIZATION_UNIQ_HELPER(ctr, grouping, phase, optimization) \
  REGISTER_OPTIMIZATION_UNIQ(ctr, grouping, phase, optimization)

#define REGISTER_OPTIMIZATION_UNIQ(ctr, grouping, phase, optimization) \
  static optimization_registration::OptimizationPassRegistration       \
      register_optimization_##ctr(                                     \
          grouping, phase,                                             \
          std::unique_ptr<GraphOptimizationPass>(new optimization))

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_OPTIMIZATION_REGISTRY_H_
