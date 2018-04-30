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

#ifndef TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONSTANT_FOLDING_H_
#define TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONSTANT_FOLDING_H_

#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/grappler/costs/graph_properties.h"
#include "tensorflow/core/grappler/optimizers/graph_optimizer.h"
#include "tensorflow/core/grappler/utils.h"
#include "tensorflow/core/protobuf/rewriter_config.pb.h"

namespace tensorflow {
namespace grappler {

const char kConstantFoldingConst[] = "ConstantFolding";
const char kConstantFoldingCtrl[] = "ConstantFoldingCtrl";

// Constant folding optimization for a graph.
class ConstantFolding : public GraphOptimizer {
 public:
  static Status CreateNodeDef(const string& name, const TensorValue& tensor,
                              NodeDef* node);
  static string AddControlDependency(const string& input_name, GraphDef* graph,
                                     NodeMap* node_map);

  explicit ConstantFolding(DeviceBase* cpu_device);
  ConstantFolding(RewriterConfig::Toggle opt_level, DeviceBase* cpu_device);

  ~ConstantFolding() override {}

  string name() const override { return "constant folding"; };

  Status Optimize(Cluster* cluster, const GrapplerItem& item,
                  GraphDef* output) override;

  void Feedback(Cluster* cluster, const GrapplerItem& item,
                const GraphDef& optimize_output, double result) override;

 private:
  string OptimizedNodeName(const NodeDef& node, StringPiece suffix) const;
  bool OptimizedNodeExists(const NodeDef& node, StringPiece suffix) const;

  bool IsReallyConstant(const NodeDef& node) const;

  Status MaterializeShapes(const GraphProperties& properties);

  Status MaterializeBroadcastGradientArgs(const NodeDef& node,
                                          const GraphProperties& properties);
  Status MaterializeReductionIndices(NodeDef* node,
                                     const GraphProperties& properties);

  Status MaterializeConstants(const GraphProperties& properties);
  bool IsFoldable(const NodeDef& node) const;

  Status EvaluateNode(const NodeDef& node,
                      const gtl::InlinedVector<TensorValue, 4>& inputs,
                      gtl::InlinedVector<TensorValue, 4>* output) const;

  Status EvaluateOneFoldable(const NodeDef& node,
                             std::vector<NodeDef>* outputs);

  Status FoldNode(NodeDef* node, GraphDef* output_graph);

  bool IsOnes(const NodeDef& node) const;
  bool IsZeros(const NodeDef& node) const;
  void ReplaceOperationWithIdentity(int input_to_forward, NodeDef* node,
                                    GraphDef* graph);
  void ReplaceOperationWithSnapshot(int input_to_forward, NodeDef* node,
                                    GraphDef* graph);
  void ReplaceSubtractionFromZeroByNegation(NodeDef* node, GraphDef* graph);
  Status ReplaceOperationWithConstant(double value, const AttrValue& dtype_attr,
                                      const TensorShapeProto& shape,
                                      NodeDef* node, GraphDef* graph);
  void ReplaceDivisionOfOnesByReciprocal(NodeDef* node, GraphDef* graph);
  Status FoldGraph(GraphDef* output);

  bool IsSimplifiableReduction(const NodeDef& node) const;
  bool IsSimplifiableReshape(const NodeDef& node,
                             const GraphProperties& properties) const;
  Status SimplifyGraph(GraphDef* output, GraphProperties* properties,
                       bool use_shape_info);

  Status RunOptimizationPass(Cluster* cluster, const GrapplerItem& item,
                             GraphDef* output);

  // Points to an externally provided device or to owned_device_;
  RewriterConfig::Toggle opt_level_;
  DeviceBase* cpu_device_;
  std::unique_ptr<DeviceBase> owned_device_;

  std::unique_ptr<ResourceMgr> resource_mgr_;
  GraphDef* graph_;
  std::unique_ptr<NodeMap> node_map_;
  std::unordered_set<string> nodes_to_preserve_;
  std::unordered_set<string> nodes_whitelist_;
  std::unordered_set<string> feed_nodes_;
  bool has_fetch_;
  bool graph_modified_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONSTANT_FOLDING_H_
