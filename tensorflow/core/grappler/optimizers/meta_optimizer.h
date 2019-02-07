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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_META_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_META_OPTIMIZER_H_

#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/grappler/grappler_item.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/verifiers/graph_verifier.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/protobuf/config.pb.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"
#include "tensorflow/core/protobuf/verifier_config.pb.h"

namespace tensorflow {
namespace grappler {

// Run the other grappler optimizers based on the specified rewriter config.
class MetaOptimizer : public GraphOptimizer {
 public:
  MetaOptimizer(DeviceBase* cpu_device, const ConfigProto& cfg);
  ~MetaOptimizer() override = default;

  string name() const override { return "meta_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void PrintResult();

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;

 private:
  std::unique_ptr<GraphOptimizer> MakeNewOptimizer(
      const string& optimizer) const;

  // Initialize active optimizers from RewriterConfig toggles.
  Status InitializeOptimizers(
      std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const;
  // Initialize active optimizers from RewriterConfig optimizer names.
  Status InitializeOptimizersByName(
      std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const;
  // Initialize active optimizers from RewriterConfig.custom_optimizers.
  Status InitializeCustomGraphOptimizers(
      const std::set<string>& pre_initialized_optimizers,
      std::vector<std::unique_ptr<GraphOptimizer>>* optimizers) const;
  // Returns the config for a custom graph optimizer. Null if none was found.
  const RewriterConfig::CustomGraphOptimizer* GetCustomGraphOptimizerConfig(
      const string& name) const;

  // Initialiaze active verifiers from the RewriterConfig toggles.
  void InitializeVerifiers(
      std::vector<std::unique_ptr<GraphVerifier>>* inter_optimizer_verifiers,
      std::vector<std::unique_ptr<GraphVerifier>>* post_optimization_verifiers)
      const;

  // Run optimization pass over a single GrapplerItem. Meta optimizer might run
  // multiple such passes: 1) for the main graph 2) for the function library
  Status OptimizeGraph(Cluster* cluster, const GrapplerItem& item,
                       GraphDef* optimized_graph);

  DeviceBase* const cpu_device_;  // may be NULL
  ConfigProto config_proto_;
  RewriterConfig& cfg_;

  struct OptimizerResult {
    string optimizer_name;
    string result;
  };

  struct GraphOptimizationResult {
    explicit GraphOptimizationResult(const string& id) : id(id) {}
    string id;
    std::vector<OptimizerResult> results;
  };

  Status RunOptimizer(GraphOptimizer* optimizer, Cluster* cluster,
                      GrapplerItem* optimized_item, GraphDef* optimized_graph,
                      GraphOptimizationResult* optimization_result);

  std::vector<GraphOptimizationResult> optimization_results_;
};

bool MetaOptimizerEnabled(const ConfigProto& cfg);

// Run the meta optimizer.
//
// If <cpu_device> is non-null, it is the device to be used for executing ops
// during constant folding; if NULL, a new device is created for doing constant
// folding. For performance, it is recommended to pass in an existing cpu_device
// when possible.
Status RunMetaOptimizer(const GrapplerItem& item, const ConfigProto& cfg,
                        DeviceBase* cpu_device, Cluster* cluster,
                        GraphDef* optimized_graph);

// Wrapper around RunMetaOptimizer convenient for optimizing
// function graphs.
//
// Runs grappler optimizations on `g` based on `config_proto`.
// `ret_node_names`: a vector of node names whose outputs are returned,
//    aka fetches. when `g` represent a function, these are _Retval nodes.
// `lib`: function library to use with `g`.
// `device_set`: the set of devices that graph can refer to.
// `cpu_device`: the CPU device.
// `config_proto`: Grapper configuration.
// `grappler_item_id': Grappler item id (e.g. optimized function name).
// `optimization_options`: Grappler optimization constraints that are known only
//    at runtime.
//
// **g is a graph constructed based on the runtime library 'lib'.
// OptimizeGraph mutates **g extensively and replaces '*g' with a
// complete copy. Therefore, the caller should not keep any references
// to nodes *g.
Status OptimizeGraph(
    std::vector<string> ret_node_names, FunctionLibraryDefinition* lib,
    const DeviceSet& device_set, Device* cpu_device,
    const ConfigProto& config_proto, const string& grappler_item_id,
    const GrapplerItem::OptimizationOptions& optimization_options,
    std::unique_ptr<tensorflow::Graph>* g);

}  // namespace grappler
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_META_OPTIMIZER_H_
