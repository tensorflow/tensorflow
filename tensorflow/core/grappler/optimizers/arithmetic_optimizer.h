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

#ifndef TENSORFLOW_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_H_
#define TENSORFLOW_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_H_

#include <unordered_set>
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

constexpr char kArithmeticOptimizer[] = "ArithmeticOptimizer";

// Optimize TF computations by reducing the arithmetic complexity required to
// run a model.
class ArithmeticOptimizer : public GraphOptimizer {
 public:
  ArithmeticOptimizer() : opt_level_(RewriterConfig::ON) {}
  explicit ArithmeticOptimizer(RewriterConfig::Toggle opt_level)
      : opt_level_(opt_level) {}
  ~ArithmeticOptimizer() override {}

  string name() const override { return "arithmetic_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;

 private:
  // Returns true is a node with given name and the optimizer prefix already
  // exists.
  bool OptimizedNodeExists(const string& name);

  // Creates a new node in the graph, prefixed with "ArithmeticOptimizer/",
  // updates node_map_, and optionally copies *node_to_copy into the new
  // node, if node_to_copy is not nullptr.
  NodeDef* AddNode(const string& name, const NodeDef* node_to_copy);

  // Returns true if it is safe to dedup node from the graph.
  bool CanDedup(const NodeDef& node) const;

  // Dedup redundant nodes in the graph.
  void DedupComputations();

  // Fix frame dependencies by adding control dependencies from old_input to
  // nodes in new_nodes_for_control_dep, and update frame_map for all nodes in
  // new_nodes.
  void AddFrameControlDeps(const NodeDef* old_node,
                           const std::vector<NodeDef*>& new_nodes,
                           const string& source_for_ctrl_dep,
                           const std::vector<NodeDef*>& sinks_for_control_dep);

  // Runs peep-hole optimizations on `optimized_graph`, e.g., removing inverse
  // transposes.
  Status SimplifyArithmeticOps();
  // Tries to simplify the expression that roots at `node` and replaces the uses
  // of `node` to the simplified expression. Returns the name of the simplified
  // tensor (e.g. "split:1") or an emtpy string if no simplification is
  // performed.
  //
  // `node_map` stores the mapping from node names to NodeDef*, and will be
  // updated according to the rewrite.
  //
  // `new_nodes` will be populated with the new nodes this function creates and
  // updates. The caller can push these nodes into the simplification queue to
  // optimize them further.
  //
  // TODO(jingyue): This interface is not suitable for optimizing nodes with
  // multiple output tensors. We should pass in a tensor name instead of a
  // NodeDef.
  string TrySimplifyAndReplaceUses(const NodeDef* node,
                                   SetVector<NodeDef*>* nodes_to_simplify);

  RewriterConfig::Toggle opt_level_;

  bool fetch_nodes_known_;
  std::unordered_set<string> nodes_to_preserve_;
  std::unique_ptr<NodeMap> node_map_;
  FrameMap frame_map_;
  std::unique_ptr<GraphProperties> graph_properties_;
  GraphDef* optimized_graph_;  // Not owned.
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_GRAPPLER_OPTIMIZERS_ARITHMETIC_OPTIMIZER_H_
