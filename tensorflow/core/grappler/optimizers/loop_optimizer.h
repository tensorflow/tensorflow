/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_LOOP_OPTIMIZER_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_LOOP_OPTIMIZER_H_

#include <unordered_set>
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/grappler/utils/frame.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

constexpr char kLoopOptimizer[] = "LoopOptimizer";

class LoopOptimizer : public GraphOptimizer {
 public:
  LoopOptimizer() : opt_level_(RewriterConfig::ON) {}
  explicit LoopOptimizer(RewriterConfig::Toggle opt_level)
      : opt_level_(opt_level) {}
  ~LoopOptimizer() override {}

  string name() const override { return "loop_optimizer"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* optimized_graph) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimized_graph, double result) override;

 private:
  Status LoopInvariantNodeMotion();
  Status FindInvariantNodes(NodeDef* node);
  Status RevertInvariantNodes();
  Status MoveInvariantNodes(const int fname);
  Status LINMHandleInvariantNode(NodeDef* node, const int num_outputs,
      const int frame_id);
  Status LINMHandleConst(NodeDef* node, const int num_outputs,
      const int frame_id);
  Status LINMHandleInvariantEnter(NodeDef* node, const int num_outputs);

  std::map<NodeDef*, int> invariant_nodes_;
  std::set<int> empty_set_;
  std::map<int, std::set<int>> frame_children_;
  std::map<int, int> frame_parent_;
  std::map<int, const NodeDef*> loop_cond_;
  std::map<int, std::vector<NodeDef*>> invariant_enters_;
  int new_enter_id_;
  RewriterConfig::Toggle opt_level_;

  std::unique_ptr<NodeMap> node_map_;
  FrameMap frame_map_;
  std::unique_ptr<GraphProperties> graph_properties_;
  GraphDef* optimized_graph_;  // Not owned.
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_LOOP_OPTIMIZER_H_
