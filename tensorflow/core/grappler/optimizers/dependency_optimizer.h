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

#ifndef THIRD_PARTY_TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEPENDENCY_OPTIMIZER_H_
#define THIRD_PARTY_TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEPENDENCY_OPTIMIZER_H_

#include <unordered_set>
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

// Optimize TF computations by removing control dependencies or re-arranging
// them to shorten the critical path for a model step or enable other
// optimizations, such as removing nodes that are effectively noops.
class DependencyOptimizer : public GraphOptimizer {
 public:
  DependencyOptimizer() : opt_level_(RewriterConfig::ON) {}
  explicit DependencyOptimizer(RewriterConfig::Toggle opt_level)
      : opt_level_(opt_level) {}
  ~DependencyOptimizer() override {}

  string name() const override { return "dependency_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;

 private:
  // Returns true if it is safe to convert node to NoOp.
  bool SafeToConvertToNoOp(const NodeDef& node);

  Status OptimizeDependencies(GraphDef* optimized_graph);
  // Tries to simplify the expression that roots at `node` and replaces the uses
  // of `node` to the simplified expression. Returns the name of the simplified
  // tensor (e.g. "split:1") or an empty string if no simplification is
  // performed.
  string TryOptimizeDependencies(NodeDef* node, GraphDef* graph,
                                 std::vector<NodeDef*>* new_nodes);

  bool HasOnlyControlOutputs(const NodeDef* node);

  bool fetch_nodes_known_;
  RewriterConfig::Toggle opt_level_;
  std::unordered_set<string> nodes_to_preserve_;
  std::unique_ptr<NodeMap> node_map_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // THIRD_PARTY_TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_DEPENDENCY_OPTIMIZER_H_
