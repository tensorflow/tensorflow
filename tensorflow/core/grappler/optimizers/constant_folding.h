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

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/types/span.h"
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
extern const int64_t kMaxConstantSize;

// Constant folding optimization for a graph.
class ConstantFolding : public GraphOptimizer {
 public:
  // The size limit will only be considered if the newly created node is greater
  // than original_size (optional).
  static absl::Status CreateNodeDef(const string& name,
                                    const TensorValue& tensor, NodeDef* node,
                                    size_t original_size = 0);
  static string AddControlDependency(const string& input_name, GraphDef* graph,
                                     NodeMap* node_map);

  explicit ConstantFolding(DeviceBase* cpu_device,
                           bool disable_compressed_tensor_optimization = false,
                           bool fold_quantization_emulation = true);
  ConstantFolding(RewriterConfig::Toggle opt_level, DeviceBase* cpu_device,
                  bool disable_compressed_tensor_optimization = false,
                  bool fold_quantization_emulation = true);

  ~ConstantFolding() override {}

  string name() const override { return "constant_folding"; };

  bool UsesFunctionLibrary() const override { return false; }

  absl::Status Optimize(Cluster* cluster, const GrapplerItem& item,
                        GraphDef* output) override;

 private:
  bool ForwardInputs(NodeDef* node, absl::Span<const int> inputs_to_forward);
  string OptimizedNodeName(const NodeDef& node, absl::string_view suffix) const;
  bool OptimizedNodeExists(const NodeDef& node, absl::string_view suffix) const;

  bool IsReallyConstant(const NodeDef& node) const;

  bool GetTensorFromConstNode(const string& node_name_or_input, Tensor* tensor);

  absl::Status MaterializeShapes(const GraphProperties& properties);

  absl::Status MaterializeBroadcastGradientArgs(
      const NodeDef& node, const GraphProperties& properties);
  absl::Status MaterializeReductionIndices(NodeDef* node,
                                           const GraphProperties& properties);
  absl::Status MaterializeConstantValuedNode(NodeDef* node,
                                             const GraphProperties& properties);
  absl::Status MaterializeOutputValues(NodeDef* node,
                                       const GraphProperties& properties);
  absl::Status MaterializeConstants(const GraphProperties& properties);

  bool IsFoldable(const NodeDef& node, const GraphProperties* properties);
  bool IsFoldableUncached(const NodeDef& node,
                          const GraphProperties* properties) const;
  bool MaybeFoldable(const NodeDef& node,
                     const GraphProperties* properties) const;

  absl::Status EvaluateNode(
      const NodeDef& node, const absl::InlinedVector<TensorValue, 4UL>& inputs,
      absl::InlinedVector<TensorValue, 4UL>* output) const;

  absl::Status EvaluateOneFoldable(const NodeDef& node,
                                   std::vector<NodeDef>* outputs,
                                   bool* result_too_large);

  absl::Status FoldMergeNode(NodeDef* node, GraphDef* output_graph);
  absl::Status FoldNode(NodeDef* node, GraphDef* output_graph,
                        bool* result_too_large);

  bool IsOnes(const NodeDef& node) const;
  bool IsZeros(const NodeDef& node) const;
  bool ReplaceOperationWithBroadcastTo(int input_to_broadcast,
                                       const GraphProperties& properties,
                                       NodeDef* node, GraphDef* graph);
  void ReplaceOperationWithIdentity(int input_to_forward,
                                    const GraphProperties& properties,
                                    NodeDef* node, GraphDef* graph);
  void ReplaceOperationWithSnapshot(int input_to_forward,
                                    const GraphProperties& properties,
                                    NodeDef* node, GraphDef* graph);
  void ReplaceOperationWithNoOp(NodeDef* node, GraphProperties* properties,
                                GraphDef* graph);
  void ReplaceBinaryOperationWithBroadcastTo(int input_to_broadcast,
                                             const GraphProperties& properties,
                                             NodeDef* node, GraphDef* graph);
  void ReplaceSubtractionFromZeroByNegation(NodeDef* node, GraphDef* graph);
  absl::Status ReplaceOperationWithConstant(double value,
                                            const GraphProperties& properties,
                                            const TensorShapeProto& shape,
                                            NodeDef* node, GraphDef* graph);

  // Notice: Destroys *value.
  absl::Status ReplaceOperationWithConstantTensor(DataType dtype,
                                                  TensorProto* value,
                                                  NodeDef* node,
                                                  GraphDef* graph);

  void ReplaceDivisionOfOnesByReciprocal(NodeDef* node, GraphDef* graph);
  absl::Status FoldGraph(const GraphProperties& properties, GraphDef* output,
                         absl::flat_hash_set<string>* nodes_to_not_simplify);

  absl::Status IsSimplifiableReshape(const NodeDef& node,
                                     const GraphProperties& properties) const;
  absl::Status SimplifyGraph(
      GraphDef* optimized_graph, GraphProperties* properties,
      absl::flat_hash_set<string>* nodes_to_not_simplify);
  absl::Status SimplifyNode(NodeDef* node, GraphDef* optimized_graph,
                            GraphProperties* properties);

  absl::Status RunOptimizationPass(Cluster* cluster, GrapplerItem* item,
                                   GraphProperties* properties,
                                   GraphDef* optimized_graph);

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

  struct ConstantPushDownContext {
    NodeDef* op_child;
    NodeDef* const_child;
    bool left_child_is_const;
    bool right_child_is_const;
    NodeDef* left_leaf;
    NodeDef* right_leaf;
    bool left_leaf_is_const;
    bool right_leaf_is_const;

    // Shape & type information.
    const std::vector<OpInfo::TensorProperties>* parent_input_props;
    const std::vector<OpInfo::TensorProperties>* op_child_input_props;
  };

  // Populates ctx with pointers to the nodes in expression tree for which
  // constant pushdown optimization is being considered, corresponding to one of
  // the following configurations:
  //
  //               parent                            parent
  //               /    \                            /    \
  //        op_child   const_child            const_child op_child
  //         /     \                                       /     \
  //    left_leaf  right_leaf                        left_leaf  right_leaf
  //
  // Returns true if the expression is possible amenable for optimization.
  // Returns false if must_have_properties is true and input properties for
  // parent and op_child are not known.
  bool PrepareConstantPushDown(const NodeDef& parent,
                               const GraphProperties& properties,
                               bool must_have_properties,
                               ConstantPushDownContext* ctx) const;

  // Pushes down constants on '+', '-', '*', and '/' operators if applicable.
  // Returns true if the transformation applied successfully.
  bool ConstantPushDown(GraphProperties* properties, GraphDef* optimized_graph,
                        NodeDef* node);

  // Pushes down constants on '+' and 'BiasAdd' operators if applicable.
  // Returns true if the graph was modified.
  bool ConstantPushDownBiasAdd(GraphProperties* properties,
                               GraphDef* optimized_graph, NodeDef* node);

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
  absl::Status SimplifyArithmeticOperations(const GraphProperties& properties,
                                            bool use_shape_info,
                                            GraphDef* optimized_graph,
                                            NodeDef* node);

  // Simplifies a Reshape operation to an Identity operation if applicable.
  bool SimplifyReshape(const GraphProperties& properties, bool use_shape_info,
                       NodeDef* node);

  // Returns true iff the node is a reduction and its reduction indices are
  // constant. Sets *indices_is_empty to true if the set of dimensions to reduce
  // along is empty (this happens often in the gradient graphs).
  bool IsReductionWithConstantIndices(const NodeDef& node,
                                      bool* indices_is_empty) const;
  // Returns true if theres a possibility that a Reduce node could be simplified
  // to an Identity/Reshape.
  bool IsReductionCandidateForSimplification(
      const NodeDef& node, const GraphProperties& properties,
      TensorShapeProto* input_tensor_shape,
      TensorShapeProto* output_tensor_shape, bool* is_single_element_op) const;
  // Returns true iff this reduction can be reduced to an identity (i.e if the
  // input dimensions to reduce along are all of size 1 and keep_dims is true).
  bool IsReductionSimplifiableToIdentity(
      const NodeDef& node, const TensorShapeProto& input_shape, bool keep_dims,
      const absl::InlinedVector<TensorValue, 4UL>& reduction_indices_vector)
      const;
  // Changes a reduction into an Identity op, returning true on success.
  bool ReplaceReductionWithIdentity(NodeDef* node) const;

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
  absl::Status SimplifyPad(const GraphProperties& properties,
                           bool use_shape_info, GraphDef* optimized_graph,
                           NodeDef* node);

  // Simplifies a Tile operation to an Identity operation if applicable.
  absl::Status SimplifyTile(const GraphProperties& properties,
                            bool use_shape_info, GraphDef* optimized_graph,
                            NodeDef* node);

  // Simplifies a StridedSlice operation to an Identity operation if applicable.
  absl::Status SimplifyStridedSlice(const GraphProperties& properties,
                                    bool use_shape_info,
                                    GraphDef* optimized_graph, NodeDef* node);

  // Simplifies a Slice operation to an Identity operation if applicable.
  absl::Status SimplifySlice(const GraphProperties& properties,
                             bool use_shape_info, GraphDef* optimized_graph,
                             NodeDef* node);

  // Simplify a Case operation where the output_idx is known.
  bool SimplifyCase(GraphDef* optimized_graph, NodeDef* node);

  // Simplify a Select operation where the predicates are all true or all false.
  bool SimplifySelect(const GraphProperties& properties,
                      GraphDef* optimized_graph, NodeDef* node);

  // Replaces variable updates that are effectively no-ops with NoOp nodes.
  void RemoveRedundantVariableUpdates(GraphProperties* properties,
                                      GraphDef* optimized_graph, NodeDef* node);

  // Removes Reverse op over dimensions with size 1.
  absl::Status RemoveReverse(const GraphProperties& properties,
                             bool use_shape_info, GraphDef* optimized_graph,
                             NodeDef* node);

  // Removes RandomShuffle op if it is scalar or first dimension is of size 1.
  void RemoveRandomShuffle(const GraphProperties& properties,
                           bool use_shape_info, GraphDef* optimized_graph,
                           NodeDef* node);

  // Removes Shuffle or Transpose op over dimensions of size 1.
  absl::Status RemoveShuffleOrTranspose(const GraphProperties& properties,
                                        bool use_shape_info,
                                        GraphDef* optimized_graph,
                                        NodeDef* node);

  // Removes Split or SplitV node if possible.
  void RemoveSplitOrSplitV(const GraphProperties& properties,
                           GraphDef* optimized_graph, NodeDef* node);

  bool GetConcatAxis(const NodeDef& node, int* axis);
  bool MergeConcat(bool use_shape_info, GraphProperties* properties,
                   GraphDef* optimized_graph, NodeDef* node);

  absl::Status AddQuantizedMatMulMinMaxOutConstNodes(NodeDef* node,
                                                     GraphDef* optimized_graph);

  // Points to an externally provided device or to owned_device_;
  RewriterConfig::Toggle opt_level_;
  DeviceBase* cpu_device_;
  std::unique_ptr<DeviceBase> owned_device_;

  std::unique_ptr<ResourceMgr> resource_mgr_;
  GraphDef* graph_;
  std::unique_ptr<NodeMap> node_map_;
  std::unordered_set<string> nodes_to_preserve_;
  // TODO(rmlarsen): Could these be keyed on absl::string_view?
  absl::flat_hash_set<string> nodes_allowlist_;
  absl::flat_hash_set<string> feed_nodes_;
  absl::flat_hash_map<string, bool> maybe_foldable_nodes_;
  bool has_fetch_;
  bool graph_modified_;
  bool graph_contains_assign_or_inplace_op_;
  bool disable_compressed_tensor_optimization_;
  bool fold_quantization_emulation_;
};

}  // end namespace grappler
}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_GRAPPLER_OPTIMIZERS_CONSTANT_FOLDING_H_
