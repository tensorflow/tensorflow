/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/protobuf/config.pb.h"

namespace tensorflow {

class GraphOptimizer {
 public:
  using NodePredicate = std::function<bool(const Node*)>;

  struct Options {
    // If not null it maps from nodes in graph to partially-known
    // shapes of their outputs, and may be used, e.g., in the constant folding
    // pass. The use of shape_map implies that the mapping from node name to the
    // vector of partial shapes of its outputs is stable, i.e., no optimization
    // pass may replace a node with a different node of the same name that has a
    // different number of outputs, or outputs with different known shapes.
    // TODO(b/65453533) introduce a unique way to name nodes in a graph.
    std::unordered_map<string, std::vector<PartialTensorShape>>* shape_map =
        nullptr;

    // If not null then only nodes for which cse_consider_fn returns true will
    // be considered for CSE.
    NodePredicate cse_consider_fn = nullptr;

    // If not null then only nodes for which cf_consider_fn returns true will be
    // considered for CF.
    NodePredicate cf_consider_fn = nullptr;

    // If true, multi-device functions will be inlined if
    // opts_.do_function_inlining() is true.
    bool inline_multi_device_functions = false;

    // If true, functions in implementation selection group will be inlined if
    // opts_.do_function_inlining() is true.
    bool inline_impl_selection_group_functions = false;

    // If true all functions will be inlined with a single device function
    // body placer strategy.
    bool inline_with_single_device_body_placer = false;
  };

  explicit GraphOptimizer(const OptimizerOptions& opts);
  ~GraphOptimizer();

  // Applies optimization passes specified in 'opts' to 'graph'.
  // Maybe replace *graph with a new graph object.  'device' is device
  // on which the 'graph' will execute. It's passed to the optimizers
  // so that they can respect constraints if any, that should be
  // respected.
  void Optimize(FunctionLibraryRuntime* runtime, Env* env, const Device* device,
                std::unique_ptr<Graph>* graph,
                const Options& graph_optimizer_options);
  // DEPRECATED: Consider passing a GraphOptimizer::Options object instead.
  void Optimize(
      FunctionLibraryRuntime* runtime, Env* env, const Device* device,
      std::unique_ptr<Graph>* graph,
      const std::unordered_map<string, std::vector<PartialTensorShape>>*
          shape_map,
      const NodePredicate& cse_consider_fn = nullptr,
      const NodePredicate& cf_consider_fn = nullptr,
      bool inline_multi_device_functions = false,
      bool inline_impl_selection_group_functions = false,
      bool inline_with_single_device_body_placer = false);

  const OptimizerOptions& options() { return opts_; }

 private:
  OptimizerOptions opts_;

  TF_DISALLOW_COPY_AND_ASSIGN(GraphOptimizer);
};

// Applies graph rewrite optimization such as inlining, dead code
// removal, etc.
//
// **g is a graph constructed based on the runtime library 'lib'.
// OptimizeGraph mutates **g extensively and replaces '*g' with a
// complete copy. Therefore, the caller should not keep any references
// to nodes *g.
void OptimizeGraph(FunctionLibraryRuntime* lib, std::unique_ptr<Graph>* g,
                   const GraphOptimizer::Options& graph_optimizer_options);
void OptimizeGraph(FunctionLibraryRuntime* lib, std::unique_ptr<Graph>* g);

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_GRAPH_OPTIMIZER_H_
