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

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/device_base.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/resource_mgr.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
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
  // The size limit will only be considered if the newly created node is greater
  // than original_size (optional).
  static Status CreateNodeDef(const string& name, const TensorValue& tensor,
                              NodeDef* node, size_t original_size = 0);
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
  Status MaterializeConstantValuedNode(NodeDef* node,
                                       const GraphProperties& properties);
  Status MaterializeConstants(const GraphProperties& properties);

  bool IsFoldable(const NodeDef& node) const;

  Status EvaluateNode(const NodeDef& node,
                      const gtl::InlinedVector<TensorValue, 4>& inputs,
                      gtl::InlinedVector<TensorValue, 4>* output) const;

  Status EvaluateOneFoldable(const NodeDef& node, std::vector<NodeDef>* outputs,
                             bool* result_too_large);

  Status FoldMergeNode(NodeDef* node, GraphDef* output_graph);
  Status FoldNode(NodeDef* node, GraphDef* output_graph,
                  bool* result_too_large);

  bool IsOnes(const NodeDef& node) const;
  bool IsZeros(const NodeDef& node) const;
  void ReplaceOperationWithIdentity(int input_to_forward,
                                    const GraphProperties& properties,
                                    NodeDef* node, GraphDef* graph);
  void ReplaceOperationWithSnapshot(int input_to_forward,
                                    const GraphProperties& properties,
                                    NodeDef* node, GraphDef* graph);
  void ReplaceBinaryOperationWithBroadcastTo(int input_to_broadcast,
                                             const GraphProperties& properties,
                                             NodeDef* node, GraphDef* graph);
  void ReplaceSubtractionFromZeroByNegation(NodeDef* node, GraphDef* graph);
  Status ReplaceOperationWithConstant(double value,
                                      const GraphProperties& properties,
                                      const TensorShapeProto& shape,
                                      NodeDef* node, GraphDef* graph);
  void ReplaceDivisionOfOnesByReciprocal(NodeDef* node, GraphDef* graph);
  Status FoldGraph(GraphDef* output,
                   absl::flat_hash_set<string>* nodes_to_not_simplify);

  bool IsSimplifiableReshape(const NodeDef& node,
                             const GraphProperties& properties) const;
  Status SimplifyGraph(bool use_shape_info, GraphDef* optimized_graph,
                       GraphProperties* properties,
                       absl::flat_hash_set<string>* nodes_to_not_simplify);
  Status SimplifyNode(bool use_shape_info, NodeDef* node,
                      GraphDef* optimized_graph, GraphProperties* properties);

  Status RunOptimizationPass(Cluster* cluster, const GrapplerItem& item,
                             GraphDef* output);

  // Applies partial constant folding for Concat which is not commutative.
  // Returns true if the transformation applied successfully.
  bool PartialConcatConstFolding(GraphDef* optimized_graph,
                                 GraphProperties* properties, NodeDef* node);

  // Applies partial constant folding for associative operators AddN and
  // AccumulateNV2. Returns true if the transformation applied successfully.
  bool PartialAssocOpConstFolding(GraphDef* optimized_graph,
                                  GraphProperties* properties, NodeDef* node);

  // Applies partial constant propagation through IdentityN operator.
  // Returns true if the transformation applied successfully.
  bool PartialConstPropThroughIdentityN(NodeDef* node);

  // Pushes down constants on '+' and '*' operators if applicable. Returns true
  // the transformation applied successfully.
  bool ConstantPushDown(GraphDef* optimized_graph, NodeDef* node);

  // Aggregate constants present around a conv operator. Returns true if the
  // transformation was applied successfully.
  bool MulConvPushDown(GraphDef* optimized_graph, NodeDef* node,
                       const GraphProperties& properties);

  // Strength reduces floating point division by a constant Div(x, const) to
  // multiplication by the reciprocal Mul(x, Reciprocal(const)).
  bool ReduceDivToReciprocalMul(GraphDef* optimized_graph, NodeDef* node);

  // Simplifies arithmetic operations with ones or zeros. Returns the status,
  // and updates the success input argument that denotes if any simplification
  // was applied.
  Status SimplifyArithmeticOperations(const GraphProperties& properties,
                                      bool use_shape_info,
                                      GraphDef* optimized_graph, NodeDef* node);

  // Simplifies a Reshape operation to an Identity operation if applicable.
  bool SimplifyReshape(const GraphProperties& properties, bool use_shape_info,
                       NodeDef* node);

  // Returns true if theres a possibility that a Reduce node could be simplified
  // to an Identity/Reshape.
  bool IsReductionCandidateForSimplification(
      const NodeDef& node, const GraphProperties& properties,
      TensorShapeProto* input_tensor_shape,
      TensorShapeProto* output_tensor_shape, bool* is_single_element_op) const;
  // Returns true iff this reduction can be reduced to an identity (i.e if the
  // set of dimensions to reduce along is empty). This happens often in the
  // gradient graphs.
  bool IsReductionSimplifiableToIdentity(
      const NodeDef& node, const TensorShapeProto& input_shape, bool keep_dims,
      const gtl::InlinedVector<TensorValue, 4>& reduction_indices_vector) const;
  // Simplifies a Reduction operation to an Identity/Reshape operation if
  // applicable.
  bool SimplifyReduction(GraphDef* optimized_graph,
                         const GraphProperties& properties, NodeDef* node);

  // Switch(x, x) will always feed false to its false branch and true to
  // its true branch. By rewriting the graph a bit, we can propagate these
  // constants down the two output branches, and just use control dependencies
  // to trigger the selected one at runtime. For example,
  //
  //     +------+
  // x-->|Switch|-->a  (in practice there may be multiple consumers of each
  // x-->|      |-->b   output branch.)
  //     +------+
  //
  // Is rewritten as
  //
  //     +------+
  // x-->|Switch|-->Identity--^>Const(false)-->a
  // x-->|      |-->Identity--^>Const(true)-->b
  //     +------+
  bool SimplifySwitch(GraphDef* optimized_graph, NodeDef* node);

  // Moves constants past Enter node if applicable.
  bool MoveConstantsPastEnter(GraphDef* optimized_graph, NodeDef* node);

  // Simplifies Pack operation if applicable.
  bool SimplifyPack(GraphDef* optimized_graph, NodeDef* node);

  // Simplifies a Squeeze operation to an Identity operation if applicable.
  void SimplifySqueeze(const GraphProperties& properties, bool use_shape_info,
                       GraphDef* optimized_graph, NodeDef* node);

  // Simplifies a Pad operation to an Identity operation if applicable.
  Status SimplifyPad(const GraphProperties& properties, bool use_shape_info,
                     GraphDef* optimized_graph, NodeDef* node);

  // Simplifies a Tile operation to an Identity operation if applicable.
  Status SimplifyTile(const GraphProperties& properties, bool use_shape_info,
                      GraphDef* optimized_graph, NodeDef* node);

  // Simplifies a StridedSlice operation to an Identity operation if applicable.
  Status SimplifyStridedSlice(const GraphProperties& properties,
                              bool use_shape_info, GraphDef* optimized_graph,
                              NodeDef* node);

  // Simplifies a Slice operation to an Identity operation if applicable.
  Status SimplifySlice(const GraphProperties& properties, bool use_shape_info,
                       GraphDef* optimized_graph, NodeDef* node);

  // Removes Reverse op over dimensions with size 1.
  Status RemoveReverse(const GraphProperties& properties, bool use_shape_info,
                       GraphDef* optimized_graph, NodeDef* node);

  // Removes RandomShuffle op if it is scalar or first dimension is of size 1.
  void RemoveRandomShuffle(const GraphProperties& properties,
                           bool use_shape_info, GraphDef* optimized_graph,
                           NodeDef* node);

  // Removes Shuffle or Transpose op over dimensions of size 1.
  Status RemoveShuffleOrTranspose(const GraphProperties& properties,
                                  bool use_shape_info,
                                  GraphDef* optimized_graph, NodeDef* node);

  // Removes Split or SplitV node if possible.
  void RemoveSplitOrSplitV(const GraphProperties& properties,
                           GraphDef* optimized_graph, NodeDef* node);

  bool MergeConcat(const GraphProperties& properties, bool use_shape_info,
                   GraphDef* optimized_graph, NodeDef* node);

  Status AddQuantizedMatMulMinMaxOutConstNodes(NodeDef* node,
                                               GraphDef* optimized_graph);

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
  bool graph_contains_assign_or_inplace_op_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONSTANT_FOLDING_H_
