/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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

#ifdef AMD_ZENDNN

#include <algorithm>
#include <functional>
#include <iterator>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/layout_pass_util.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/zen_graph_util.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/util/port.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/util/zen_util.h"

namespace tensorflow {

// This pass implements rewriting of graph to support following scenarios:
// (A) Merging nodes in the graph
// (B) Updating nodes in graph
//
// Example of A : Merging nodes in the graph
// -----------------------------------------
// Currently, we merge Pad + Conv2D together.
// Consider the subgraph below :
//
//        [Const Op]
//                  \
//  [Sub-Graph 1]-->[Pad Op]-->[Conv2D_1]-->[Sub-Graph 2]
//
// As part of fusion, the graph gets transformed to
//
// [Sub-Graph 1]-->[Conv2D_2]-->[Sub-Graph 2]
//
// This fusion is valid provided Conv2D op supports EXPLICIT padding
//
// The padding value from the Pad op is added up to the existing pad value of
// the Conv op and the Pad op is removed.
//
// Only the padding values of the Conv op is updated and the sub-graph linked
// to Pad op is now linked with the Conv op.
//
// Example of B : Rewriting nodes to Zen nodes
// -------------------------------------------
// Consider a Relu node. Current definition of Relu node looks like:
//
//              O = Relu(A)
//
// Relu has 1 input (A), and 1 output (O).
//
// This rewrite pass will generate a new graph node for Relu (new node is
// called ZenRelu) as:
//
//             O = ZenRelu(A)
//
// Rewriting prerequisites:
//  - Rewrite pass requires that op is registered. If the op type is not
//    registered, then any node of this op type will not be rewritten.
//
// Graph rewrite algorithm:
//      Algorithm: Graph Rewrite
//      Input: Graph G, Names of the nodes to rewrite and their new names
//      Output: Modified Graph G' if the nodes are modified, G otherwise.
//      Start:
//        N = TopologicalSort(G)  // N is a set of nodes in toposort order.
//        foreach node n in N
//        do
//          if (ZenOpNodeRewrite(n))  // Can this node be rewritten with Zen op.
//          then
//            E = set of <incoming edge and its src_output slot> of n
//            E' = {}   // a new set of edges for rewritten node
//            foreach <e,s> in E
//            do
//              E' U {<e,s>}  // Copy edges which generate tensors
//            done
//            n' = BuildNewNode(G, new_name, E')
//            MarkRewritten(n')  // Mark the new node as being rewritten.
//          fi
//        done
//
//      Explanation:
//        For graph rewrite, we visit nodes of the input graph in the
//        topological sort order (top-to-bottom fashion). We need this order
//        because while visiting a node we want that all of its input nodes are
//        visited and rewritten if applicable. This is because if we need to
//        rewrite a given node then all of its input nodes need to be fixed (in
//        other words they cannot be deleted later.)
//
class ZenLayoutRewritePass : public GraphOptimizationPass {
 public:
  ZenLayoutRewritePass() {
    // Zen op rewrite information records
    zen_rewrite_db_.push_back({"Conv2D", "_ZenConv2D",
                               CheckValidityForDTypeSupported,
                               UpdateZenOpAttrsConv2D});
    zen_rewrite_db_.push_back({"_FusedConv2D", "_ZenFusedConv2D",
                               CheckValidityFusedConv2D,
                               UpdateZenOpAttrsFusedConv2D});
    zen_rewrite_db_.push_back(
        {"DepthwiseConv2dNative", "_ZenDepthwiseConv2dNative",
         CheckValidityForDTypeSupported, UpdateZenOpAttrsConv2D});
    zen_rewrite_db_.push_back(
        {"_FusedDepthwiseConv2dNative", "_ZenFusedDepthwiseConv2dNative",
         CheckValidityFusedConv2D, UpdateZenOpAttrsFusedConv2D});
    zen_rewrite_db_.push_back({"MatMul", "_ZenMatMul",
                               CheckValidityForDTypeSupported,
                               UpdateZenOpAttrs});
    zen_rewrite_db_.push_back({"_FusedMatMul", "_ZenFusedMatMul",
                               CheckValidityForDTypeSupported,
                               UpdateZenOpAttrs});
    zen_rewrite_db_.push_back({"BatchMatMul", "_ZenBatchMatMul",
                               CheckValidityForDTypeSupported,
                               UpdateZenOpAttrs});
    zen_rewrite_db_.push_back({"BatchMatMulV2", "_ZenBatchMatMulV2",
                               CheckValidityForDTypeSupported,
                               UpdateZenOpAttrs});
    zen_rewrite_db_.push_back({"MaxPool", "_ZenMaxPool",
                               CheckValidityForDTypeSupported,
                               UpdateZenOpAttrs});
    zen_rewrite_db_.push_back({"AvgPool", "_ZenAvgPool",
                               CheckValidityForDTypeSupported,
                               UpdateZenOpAttrs});
    // TF-ZenDNN supports NHWC and blocked format execution. For blocked format,
    // following rewrites are not supported.
    if (!IsBlockedFormatEnabled()) {
      zen_rewrite_db_.push_back({"Softmax", "_ZenSoftmax",
                                 CheckValidityForDTypeSupported,
                                 UpdateZenOpAttrs});
      zen_rewrite_db_.push_back({"ConjugateTranspose", "_ZenConjugateTranspose",
                                 RewriteValid, UpdateZenOpAttrs});
      zen_rewrite_db_.push_back(
          {"Transpose", "_ZenTranspose", RewriteValid, UpdateZenOpAttrs});
      zen_rewrite_db_.push_back({"InvertPermutation", "_ZenInvertPermutation",
                                 RewriteValid, UpdateZenOpAttrs});
      zen_rewrite_db_.push_back({"FusedBatchNorm", "_ZenFusedBatchNorm",
                                 RewriteValid, UpdateZenOpAttrs});
      zen_rewrite_db_.push_back({"FusedBatchNormV2", "_ZenFusedBatchNormV2",
                                 RewriteValid, UpdateZenOpAttrs});
      zen_rewrite_db_.push_back({"FusedBatchNormV3", "_ZenFusedBatchNormV3",
                                 RewriteValid, UpdateZenOpAttrs});
    }
    // TF-ZenDNN currently only supports inference. The graph must not have any
    // of the training ops in tensorflow/core/kernels/training_ops.cc
    tf_training_ops_.push_back("ApplyGradientDescent");
    tf_training_ops_.push_back("ApplyAdadelta");
    tf_training_ops_.push_back("ResourceSparseApplyAdadelta");
    tf_training_ops_.push_back("ApplyProximalGradientDescent");
    tf_training_ops_.push_back("SparseApplyProximalGradientDescent");
    tf_training_ops_.push_back("ApplyAdagrad");
    tf_training_ops_.push_back("ApplyAdagradV2");
    tf_training_ops_.push_back("ApplyProximalAdagrad");
    tf_training_ops_.push_back("SparseApplyAdagrad");
    tf_training_ops_.push_back("SparseApplyAdagradV2");
    tf_training_ops_.push_back("SparseApplyProximalAdagrad");
    tf_training_ops_.push_back("ApplyAdagradDA");
    tf_training_ops_.push_back("SparseApplyAdagradDA");
    tf_training_ops_.push_back("ApplyFtrl");
    tf_training_ops_.push_back("ApplyFtrlV2");
    tf_training_ops_.push_back("SparseApplyFtrl");
    tf_training_ops_.push_back("SparseApplyFtrlV2");
    tf_training_ops_.push_back("ApplyMomentum");
    tf_training_ops_.push_back("ApplyKerasMomentum");
    tf_training_ops_.push_back("ApplyAdam");
    tf_training_ops_.push_back("ApplyAdaMax");
    tf_training_ops_.push_back("ApplyRMSProp");
    tf_training_ops_.push_back("ApplyCenteredRMSProp");
    tf_training_ops_.push_back("ApplyAddSign");
    tf_training_ops_.push_back("ApplyPowerSign");
  }

  // Standard interface to run optimization passes.
  Status Run(const GraphOptimizationPassOptions &options) override;

  // Executes fusion and rewrite passes on the graph. Has an option to dump
  // graph before and after rewrite. Returns true, if and only if the graph
  // mutated, false otherwise.
  bool ZenOpRewritePass(std::unique_ptr<Graph> *g);

  // Replaces TF-Vanilla ops with Zen ops. Returns true if one or more rewrites
  // are successful, false otherwise.
  bool ZenOpUpdate(std::unique_ptr<Graph> *g);

  // Stores Zen op rewrite rules.
  typedef struct {
    string tf_op_name;   // Original name of op of the node in the graph.
    string zen_op_name;  // New name of the op.
    // A function handler to copy attributes from an old node to a new node.
    std::function<bool(const Node *)> check_validity;
    // Returns true if we should rewrite the node.
    std::function<void(const Node *, NodeBuilder *)> update_zen_op_attr;
  } ZenOpRewriteRecord;

 private:
  // Maintain record about nodes to rewrite.
  std::vector<ZenOpRewriteRecord> zen_rewrite_db_;

  // TF training ops list from tensorflow/core/kernels/training_ops.cc
  std::vector<string> tf_training_ops_;

  inline bool HasSubstr(const std::string primary,
                        const std::string sub) const {
    return primary.find(sub) != std::string::npos;
  }

  // Check if the node 'n' has any applicable rewrite rule.
  //
  // @return RewriteInfo* for the applicable rewrite rule.
  const ZenOpRewriteRecord *CheckNodeForZenOpRewrite(const Node *n) const;

  // ZenDNN currently does not support all fusions that grappler performs
  // together with Conv2D and DepthwiseConv2D. We rewrite _FusedConv2D and
  // _FusedDepthwiseConv2dNative only if it includes those we support.
  static bool CheckValidityFusedConv2D(const Node *n) {
    // Return false if the node is not with data type supported by Zen
    // inference. Currently Zen supports inference in float only.
    if (!CheckValidityForDTypeSupported(n)) {
      return false;
    }
    std::vector<string> fused_ops;
    TF_CHECK_OK(GetNodeAttr(n->def(), "fused_ops", &fused_ops));

    return (fused_ops == std::vector<string>{"BiasAdd"} ||
            fused_ops == std::vector<string>{"FusedBatchNorm"} ||
            fused_ops == std::vector<string>{"Relu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Relu"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Relu6"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Add"} ||
            fused_ops == std::vector<string>{"BiasAdd", "Add", "Relu"} ||
            fused_ops == std::vector<string>{"FusedBatchNorm", "Relu"});
  }

  // Currently TF-ZenDNN supports FP32 inference only. Returns, true if node is
  // of float dataype, false otherwise.
  static bool CheckValidityForDTypeSupported(const Node *n) {
    DataType data_type;
    TF_CHECK_OK(GetNodeAttr(n->def(), "T", &data_type));
    return (data_type == DT_FLOAT);
  }

  // Method to provide a 'valid' status for nodes that don't require any check.
  // This method is used in ZenLayoutRewritePass() for creating the record/entry
  // for rewriting native ops with Zen ops.
  static bool RewriteValid(const Node *n) { return true; }

  // Method to find whether the graph has inference ops only. It returns error
  // status if the graph has training ops.
  Status AreAllInferenceOps(std::unique_ptr<Graph> *g);

  // Rewrites input node to a new node specified by its matching rewrite record.
  //
  // Input node may be deleted in case of rewrite. Attempt to use the node
  // after the call can result in undefined behaviors.
  //
  // @input  g - input graph, n - Node to be rewritten,
  //         ri - matching rewrite record,
  //         reorder_flags - flags to populate reorder attributes of Zen op.
  // @return OkStatus(), if the input node is rewritten;
  //         Returns appropriate Status error code otherwise.
  //         Graph is updated in case the input node is rewritten.
  //         Otherwise, it is not updated.
  Status ZenOpNodeRewrite(std::unique_ptr<Graph> *g, Node *orig_node,
                          const ZenOpRewriteRecord *rewrite_record,
                          std::pair<bool, bool> reorder_flags);

  // Functions specific to operators to copy attributes
  // We need operator-specific function to copy attributes because the framework
  // does not provide any generic function for it.
  static void UpdateZenOpAttrs(const Node *orig_node, NodeBuilder *nb);

  static void UpdateZenOpAttrsConv2D(const Node *orig_node, NodeBuilder *nb);

  static void UpdateZenOpAttrsFusedConv2D(const Node *orig_node,
                                          NodeBuilder *nb);

  // This function determines the reorder flags <reorder_before, reorder_after>
  // for each Zen node. Here reordering means converting tensor layout. The
  // 'reorder_before' flag indicates whether the tensors need to be reordered
  // to Zen format before the Zen node. The 'reorder_after' flag indicates
  // whether the tensors need to be reordered back to native nhwc format after
  // the Zen node.
  //
  // @input  nodes - A vector of Zen nodes marked for rewrite to update reorder
  //                 flags.
  // @return An unordered map with nodes as key and value as a pair of reorder
  //         flags.
  std::unordered_map<Node *, std::pair<bool, bool>> GetReorderFlags(
      const std::vector<Node *> &nodes);

  // Update reorder information of all Zen nodes
  //
  // @input g - input graph
  // @return true, if one or more updates are successful; false otherwise.
  bool AddReorderAttrs(std::unique_ptr<Graph> *g);
};

// ZenLayoutRewritePass is executed in phase 0, to make sure it is executed
// before MklLayoutRewritePass (phase 1).
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 0,
                      ZenLayoutRewritePass);

void DeleteNodeAndUpdateLinks(std::unique_ptr<Graph> *, Node *, Node *, int);

const ZenLayoutRewritePass::ZenOpRewriteRecord *
ZenLayoutRewritePass::CheckNodeForZenOpRewrite(const Node *n) const {
  CHECK_NOTNULL(n);  // Crash ok.

  DataType data_type;

  for (auto rewrite_record = zen_rewrite_db_.cbegin();
       rewrite_record != zen_rewrite_db_.cend(); ++rewrite_record) {
    if (n->type_string() == rewrite_record->tf_op_name &&
        rewrite_record->check_validity(n)) {
      TF_CHECK_OK(GetNodeAttr(n->def(), "T", &data_type));
      if (!zen_op_registry::IsZenOpKernelRegistered(rewrite_record->zen_op_name,
                                                    data_type)) {
        // No Zen kernel is registered for op.
        return nullptr;
      }
      return &*rewrite_record;
    }
  }
  return nullptr;
}

// Returns true if 'm' is the last Zen node of the graph, false otherwise.
bool IsLastZenNode(std::unique_ptr<Graph> *g, Node *m) {
  std::vector<Node *> order;
  GetPostOrder(**g, &order);

  for (Node *n : order) {
    if (absl::StrContains((n->type_string()),
                          zen_op_registry::kZenNodePrefix)) {
      return (n == m);
    }
  }
  return false;
}

// Returns the count of incoming data edges to a node.
int IncomingEdgeCount(const Node *n) {
  int count = 0;
  if (n == nullptr) return count;
  for (const Edge *e : n->in_edges()) {
    if (!e->IsControlEdge() && e->src()->type_string() != "Const") {
      count++;
    }
  }
  return count;
}

// Returns the count of outgoing data edges of a node.
int OutgoingEdgeCount(const Node *n) {
  int count = 0;
  if (n == nullptr) return count;
  for (const Edge *e : n->out_edges()) {
    if (!e->IsControlEdge()) {
      count++;
    }
  }
  return count;
}

// Upon pattern match, the Pad op is removed. Conv2D op has been updated
// previously. This pattern (Pad -> Conv2D/FusedConv2D) is observed in ResNet50
// and other ResNet variants.
//
// @input  conv_pattern - a variant of Conv2D pattern.
//         pad_pattern - a Pad op.
// @return True if fusion is performed, false otherwise.
bool ZenFusePadConv(std::unique_ptr<Graph> *g, Node *orig_node,
                    string conv_pattern, string pad_pattern) {
  int source_slot;      // Source output of incoming edge for Pad op.
  string padding = "";  // To check that padding type is set to EXPLICIT.
  // Padding type that should be in Conv2D.
  const string kExplicitPad = "EXPLICIT";

  // If current node is not Pad, return false.
  if (orig_node->type_string() != pad_pattern) {
    return false;
  }
  // Check incoming edges to Pad op (orig_node).
  for (const Edge *n : orig_node->in_edges()) {
    if (n->IsControlEdge() || n->src()->type_string() == "Const") {
      continue;
    }
    // Store source output of incoming edge for Pad op.
    source_slot = n->src_output();
    // Check outgoing edges from Pad op (orig_node).
    for (const Edge *e : orig_node->out_edges()) {
      // Check for 2nd pattern (Conv2D).
      if (!e->IsControlEdge() && e->dst()->type_string() == conv_pattern) {
        TF_CHECK_OK(GetNodeAttr((e->dst())->def(), "padding", &padding));
        // If padding type is not EXPLICIT, fusion of Pad op cannot be performed
        if (padding != kExplicitPad) {
          return false;
        }
        // Remove Pad node as it's Fused with Conv2D (FusedPadConv2D).
        DeleteNodeAndUpdateLinks(g, orig_node, n->src(), source_slot);
        return true;
      }  // end of if condition to check conv_pattern among non-control edges
    }    // end of for loop for out edges
  }      // end of for loop for in edges
  // return false as Pad removal is not performed
  return false;
}

// Delete node 'm' and update the incoming and outgoing links of it.
//
// @input  g - input graph.
// @input  m - node to be deleted.
// @input  source_node - previous node of 'm'.
// @input  source_output_slot - source output of node 'm'.
// @return None.
void DeleteNodeAndUpdateLinks(std::unique_ptr<Graph> *g, Node *m,
                              Node *source_node, int source_output_slot) {
  std::unordered_set<Node *> unique_node;

  // Handle outgoing control edges.
  for (const Edge *e : m->out_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(source_node);
      if (result.second) {
        (*g)->AddControlEdge(source_node, e->dst(), true);
      }
    } else {
      auto result = (*g)->AddEdge(source_node, source_output_slot, e->dst(),
                                  e->dst_input());
      DCHECK_NE(result, nullptr);
    }
  }
  unique_node.clear();

  // Handle incoming control edges.
  for (const Edge *e : m->in_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->src());
      if (result.second) {
        (*g)->AddControlEdge(e->src(), source_node, true);
      }
    }
  }
  unique_node.clear();
  (*g)->RemoveNode(m);
}

// Remove the successor of Zen node if it matches with 'pattern'.
//
// @input  g - input graph.
// @input  orig_node - Source Zen node.
// @input  pattern - Pattern to check in the successor nodes of 'orig_node'.
// @return True, if the pattern is found in successor nodes of 'orig_node' and
//         delete the successor node (otherwise false).
bool ZenOpRemoveSuccessor(std::unique_ptr<Graph> *g, const Node *orig_node,
                          string pattern) {
  if (OutgoingEdgeCount(orig_node) != 1) {
    return false;
  }

  for (const Edge *e : orig_node->out_edges()) {
    if (!e->IsControlEdge() && e->dst()->type_string() == pattern &&
        IncomingEdgeCount(e->dst()) == 1) {
      DeleteNodeAndUpdateLinks(g, e->dst(), e->src(), e->src_output());
      return true;
    }
  }
  return false;
}

// Fuse Conv2D-Bias-ReLU
bool FuseConv2DBiasRelu(std::unique_ptr<Graph> *g, const Node *orig_node,
                        string pattern) {
  return ZenOpRemoveSuccessor(g, orig_node, pattern);
}

void ZenLayoutRewritePass::UpdateZenOpAttrs(const Node *orig_node,
                                            NodeBuilder *nb) {
  string name;
  AttrSlice attr_list(orig_node->def());

  for (auto iter = attr_list.begin(); iter != attr_list.end(); ++iter) {
    name = iter->first;
    // Skip the following attributes because they are handled elsewhere.
    if (name == "reorder_before" || name == "reorder_after" ||
        name == "is_eager" || name == "in_links" || name == "out_links" ||
        name == "reset") {
      continue;
    }

    nb->Attr(name, iter->second);
  }
}

// Used internally in UpdateZenOpAttrsConv2D and UpdateZenOpAttrsFusedConv2D to
// update 'padding' attribute according to PadConv2D fusion.
//
// @input  padding - 'padding' attribute of 'orig_node'.
//         orig_node - Node with which Pad op needs to be fused.
//         explicit_paddings - a vector of padding values for each dimension.
// @return True if fusion can take place, false otherwise.
//
bool UpdateAttributePadConv2D(string padding, const Node *orig_node,
                              std::vector<int32> &explicit_paddings) {
  // Part of PadConv2D fusion
  // If padding is VALID and the current FusedConv2D op is preceded by Pad op,
  // then we are updating the padding attribute to EXPLICIT and setting
  // explicit_paddings attribute.
  // If padding is EXPLICIT and the pattern Pad op -> FusedConv2D op exists,
  // then we are updating the explicit_paddings attribute only.
  const string kValidPad = "VALID";
  const string kExplicitPad = "EXPLICIT";
  const string kPadPattern = "Pad";

  // Temporary fix for num_host_args argument of _FusedConv2D node.
  if (orig_node->type_string() == "_FusedConv2D") {
    string data_format;
    string filter_format;
    int num_host_args = 0;
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "filter_format", &filter_format));
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "num_host_args", &num_host_args));

    if ((data_format != "NCHW" && data_format != "NHWC") ||
        (filter_format != "HWIO" && filter_format != "OIHW") ||
        (num_host_args != 0)) {
      // Not supporting num_host_args for _FusedConv2D and Pad match.
      VLOG(1) << "ZenLayoutRewritePass::" << orig_node->name()
              << " can be match with pad but currently" << orig_node->name()
              << " only supported without host args";
      return false;
    }
  }

  // If padding is not VALID or EXPLICIT, fusion cannot be performed.
  if (padding != kValidPad && padding != kExplicitPad) {
    return false;
  }

  // Check incoming edges to origin node (FusedConv2D).
  for (const Edge *m : orig_node->in_edges()) {
    // Skip if previous node is Const.
    if (m->src()->type_string() == "Const") {
      continue;
    }
    // If previous node is kPadPattern, pattern (Pad op -> FusedConv2D op) has
    // been found.
    if (m->src()->type_string() == kPadPattern) {
      // Get original explicit padding values if padding = EXPLICIT.
      std::vector<int32> explicit_paddings_orig = {};
      if (padding == kExplicitPad) {
        TF_CHECK_OK(GetNodeAttr(orig_node->def(), "explicit_paddings",
                                &explicit_paddings_orig));
      }
      // 'input' will hold the const op before Pad op.
      Node *input = nullptr;
      // Index 0 has the input data and Index 1 has the padding values (which is
      // needed).
      TF_CHECK_OK((m->src())->input_node(1, &input));
      // Check if input is constant
      if (input->IsConstant()) {
        Tensor explicit_padding_tensor;
        // value attribute has the Tensor with explicit padding values.
        TF_CHECK_OK(
            GetNodeAttr((input)->def(), "value", &explicit_padding_tensor));
        // Number of elements in explicit_padding_tensor (should be 8).
        int num_elements = explicit_padding_tensor.NumElements();
        // 'padding_1d_tensor' is an Eigen Tensor with datatype int32.
        auto padding_1d_tensor = explicit_padding_tensor.flat<int32>();
        // For dimension i (starting from 0), the padding values
        // will be at 2*i and 2*i + 1
        for (int index_pad = 0; index_pad < num_elements; index_pad++) {
          if (padding == kValidPad) {
            explicit_paddings.insert(explicit_paddings.begin() + index_pad,
                                     padding_1d_tensor(index_pad));
          } else if (padding == kExplicitPad) {
            explicit_paddings.insert(explicit_paddings.begin() + index_pad,
                                     padding_1d_tensor(index_pad) +
                                         explicit_paddings_orig.at(index_pad));
          }
        }  // end of for loop for padding values.
        // PadConv2D fusion can be performed.
        return true;
      }  // end of if condition to check constant op.
    }    // end of if condition for Pad op.
  }      // end of for loop for input edges for FusedConv2D op.
  return false;
}

// Copies the attributes from Conv2D op to ZenConv2D op. 'padding' and
// 'explicit_paddings' attributes are updated accordingly to PadConv2D fusion.
void ZenLayoutRewritePass::UpdateZenOpAttrsConv2D(const Node *orig_node,
                                                  NodeBuilder *nb) {
  DataType T;
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  std::vector<int32> explicit_paddings = {};

  // Get attributes from TF op node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));

  // 'padding_update' determines if padding attributes needs to be modified.
  bool padding_update = false;
  // PadConv2D fusion can be done for VALID and EXPLICIT padding.
  if (padding != "SAME") {
    // Check if PadConv2D fusion can be done and get the padding values.
    padding_update =
        UpdateAttributePadConv2D(padding, orig_node, explicit_paddings);
  }
  // Update Zen op with attributes from TF op.
  nb->Attr("T", T);
  nb->Attr("strides", strides);
  // Update 'padding' attribute for PadConv2D fusion.
  if (padding_update == true) {
    nb->Attr("padding", "EXPLICIT");                   // Updates padding type.
    nb->Attr("explicit_paddings", explicit_paddings);  // sets padding values.
  } else {
    // 'padding' attribute for condition when fusion is not performed.
    nb->Attr("padding", padding);
    // If 'padding' is EXPLICIT, then 'explicit_paddings' attribute needs to be
    // set.
    if (padding == "EXPLICIT") {
      std::vector<int32> explicit_paddings_tmp = {};
      TF_CHECK_OK(GetNodeAttr(orig_node->def(), "explicit_paddings",
                              &explicit_paddings_tmp));
      nb->Attr("explicit_paddings", explicit_paddings_tmp);
    }
  }
  nb->Attr("data_format", data_format);
  nb->Attr("dilations", dilations);
}

// Copies the attributes from FusedConv2D op to ZenFusedConv2D op. 'padding' and
// 'explicit_paddings' attributes are updated accordingly to PadFusedConv2D
// fusion.
void ZenLayoutRewritePass::UpdateZenOpAttrsFusedConv2D(const Node *orig_node,
                                                       NodeBuilder *nb) {
  DataType T;
  int num_args;
  float epsilon;
  string data_format;
  string padding;
  std::vector<int32> strides;
  std::vector<int32> dilations;
  std::vector<int32> explicit_paddings = {};

  // Get attributes from TF op node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "num_args", &num_args));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "dilations", &dilations));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "epsilon", &epsilon));

  // 'padding_update' determines if padding attributes needs to be modified.
  bool padding_update = false;
  // PadFusedConv2D fusion can be done for VALID and EXPLICIT padding.
  if (padding != "SAME") {
    // Check if PadFusedConv2D fusion can be done and get the padding values.
    padding_update =
        UpdateAttributePadConv2D(padding, orig_node, explicit_paddings);
  }
  // Update Zen op with attributes from TF op.
  nb->Attr("T", T);
  nb->Attr("num_args", num_args);
  nb->Attr("strides", strides);
  // Update padding attribute for PadConv2D fusion.
  if (padding_update == true) {
    nb->Attr("padding", "EXPLICIT");                   // Updates padding type.
    nb->Attr("explicit_paddings", explicit_paddings);  // sets padding values.
  } else {
    // 'padding' attribute for condition when fusion is not performed.
    nb->Attr("padding", padding);
    // If 'padding' is EXPLICIT, then 'explicit_paddings' attribute needs to be
    // set.
    if (padding == "EXPLICIT") {
      std::vector<int32> explicit_paddings_tmp = {};
      TF_CHECK_OK(GetNodeAttr(orig_node->def(), "explicit_paddings",
                              &explicit_paddings_tmp));
      nb->Attr("explicit_paddings", explicit_paddings_tmp);
    }
  }
  nb->Attr("data_format", data_format);
  nb->Attr("dilations", dilations);
  nb->Attr("epsilon", epsilon);
}

static void FillInputs(const Node *n,
                       gtl::InlinedVector<Node *, 4> *control_edges,
                       gtl::InlinedVector<std::pair<Node *, int>, 4> *in) {
  control_edges->clear();
  for (const Edge *e : n->in_edges()) {
    if (e->IsControlEdge()) {
      control_edges->push_back(e->src());
    } else {
      (*in)[e->dst_input()] = std::make_pair(e->src(), e->src_output());
    }
  }
  std::sort(control_edges->begin(), control_edges->end());
}

Status ZenLayoutRewritePass::ZenOpNodeRewrite(
    std::unique_ptr<Graph> *g, Node *orig_node,
    const ZenOpRewriteRecord *rewrite_record,
    std::pair<bool, bool> reorder_flags) {
  DCHECK_NE(rewrite_record, nullptr);
  DCHECK_NE(orig_node, nullptr);

  Node *new_node = nullptr;
  std::vector<string> fused_ops = {};
  int num_data_inputs = 0;
  for (const Edge *e : orig_node->in_edges()) {
    if (!e->IsControlEdge()) {
      num_data_inputs++;
    }
  }

  gtl::InlinedVector<Node *, 4> control_edges;
  gtl::InlinedVector<std::pair<Node *, int>, 4> inputs(num_data_inputs);
  FillInputs(orig_node, &control_edges, &inputs);

  NodeBuilder nb(orig_node->name().c_str(),
                 rewrite_record->zen_op_name.c_str());

  nb.Device(orig_node->def().device());
  TF_RETURN_IF_ERROR(zendnn::CopyInputs(orig_node, inputs, &nb));
  rewrite_record->update_zen_op_attr(const_cast<const Node *>(orig_node), &nb);

  nb.Attr("reorder_before", reorder_flags.first);
  nb.Attr("reorder_after", reorder_flags.second);
  nb.Attr("in_links", IncomingEdgeCount(orig_node));
  nb.Attr("out_links", OutgoingEdgeCount(orig_node));
  nb.Attr("reset", IsLastZenNode(g, orig_node));

  // Add/Update Fused Op Attribute.
  if (orig_node->type_string() == "_ZenFusedConv2D" ||
      orig_node->type_string() == "_FusedConv2D" ||
      orig_node->type_string() == "_FusedDepthwiseConv2dNative" ||
      orig_node->type_string() == "_ZenFusedDepthwiseConv2dNative") {
    TF_CHECK_OK(GetNodeAttr(orig_node->def(), "fused_ops", &fused_ops));
    if (FuseConv2DBiasRelu(g, orig_node, "Relu")) {
      if (fused_ops.size() == 1) {
        fused_ops.push_back("Relu");
      }
    }
    if (FuseConv2DBiasRelu(g, orig_node, "Relu6")) {
      if (fused_ops.size() == 1) {
        fused_ops.push_back("Relu6");
      }
    }
    nb.Attr("fused_ops", fused_ops);
  }
  TF_RETURN_IF_ERROR(nb.Finalize(&**g, &new_node));

  std::unordered_set<Node *> unique_node;
  for (const Edge *e : orig_node->in_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->src());
      if (result.second) {
        (*g)->AddControlEdge(e->src(), new_node, true);
      }
    }
  }
  unique_node.clear();

  for (const Edge *e : orig_node->out_edges()) {
    if (e->IsControlEdge()) {
      auto result = unique_node.insert(e->dst());
      if (result.second) {
        (*g)->AddControlEdge(new_node, e->dst(), true);
      }
    } else {
      auto result =
          (*g)->AddEdge(new_node, e->src_output(), e->dst(), e->dst_input());
      DCHECK_NE(result, nullptr);
    }
  }
  new_node->set_assigned_device_name(orig_node->assigned_device_name());
  (*g)->RemoveNode(orig_node);

  return OkStatus();
}

std::unordered_map<Node *, std::pair<bool, bool>>
ZenLayoutRewritePass::GetReorderFlags(const std::vector<Node *> &nodes) {
  // Map from node to [reorder_before, reorder_after]
  std::unordered_map<Node *, std::pair<bool, bool>> reorder_flags;
  bool first_reorder_completed = false;  // assuming only one input

  for (Node *n : nodes) {
    // When setting reorder_before, we check if the input ops are read ops
    // typically to avoid considering read ops from filter weights as they are
    // reordered anyway in the Zen op. However, for the first op, there will be
    // two read ops, one from weights, and one from input data. To handle this
    // special case, this bool variable is used.
    bool reorder_before, reorder_after;
    reorder_before = reorder_after = false;

    for (const Edge *e : n->out_edges()) {
      Node *dst = e->dst();
      if (!dst->IsOp() || e->IsControlEdge()) {
        continue;
      }

      auto it = std::find(nodes.begin(), nodes.end(), dst);
      if (it == nodes.end()) {
        VLOG(1) << "ZenLayoutRewritePass::GetReorderFlags: At " << n->name()
                << " " << n->type_string() << ", non-Zen output - "
                << dst->name() << " " << dst->type_string();
        // Did not find the next node. This means that the next node is not a
        // Zen node, thus, we must reorder.
        reorder_after = true;
        // Exit the loop as remaining edges will not change this flag.
        break;
      }
    }

    for (const Edge *e : n->in_edges()) {
      Node *src = e->src();
      if (!src->IsOp() || e->IsControlEdge() ||
          HasSubstr(src->type_string(), "Const")) {
        continue;
      }

      if (HasSubstr(src->type_string(), "_Arg")) {
        // Found a placeholder op.
        VLOG(1) << "ZenLayoutRewritePass::GetReorderFlags: At " << n->name()
                << " " << n->type_string() << ", a placeholder op "
                << src->name() << " " << src->type_string();
        // In this case, we don't need to worry about a read op from data.
        first_reorder_completed = true;
        reorder_before = true;
        break;
      }

      // Ignore read ops coming from weights.
      if (HasSubstr(src->name(), "read")) {
        // Found read op, check if it is the first.
        if (!first_reorder_completed) {
          VLOG(1) << "ZenLayoutRewritePass::GetReorderFlags: At " << n->name()
                  << " " << n->type_string() << ", encountered first read op "
                  << src->name() << " " << src->type_string();
          first_reorder_completed = true;
          reorder_before = true;
          break;
        }
        continue;
      }

      auto it = std::find(nodes.begin(), nodes.end(), src);
      if (it == nodes.end()) {
        VLOG(1) << "ZenLayoutRewritePass::GetReorderFlags: At " << n->name()
                << " " << n->type_string() << ", non-Zen input - "
                << src->name() << " " << src->type_string();
        // Did not find the previous node. This means that the previous node is
        // not a Zen node, thus, we must reorder.
        reorder_before = true;
        // Exit the loop since remaining edges will not change this flag.
        break;
      }
    }

    std::pair<bool, bool> n_flags(reorder_before, reorder_after);
    reorder_flags[n] = n_flags;
  }

  // Handle the case of branches separately.
  // Case 1
  for (Node *n : nodes) {
    // Let A and B be Zen nodes, and X be a non-Zen node.
    // rb - reorder_before, ra - reorder_after
    // Handle first case of branching:
    //       A (rb=True, ra)
    //     /   \
    //    X     B(rb, ra=False)
    if (reorder_flags[n].second == false) {
      for (const Edge *e : n->out_edges()) {
        Node *dst = e->dst();
        auto it = std::find(nodes.begin(), nodes.end(), dst);
        if (it != nodes.end() && reorder_flags[dst].first) {
          // Found Zen node.
          reorder_flags[n].second = true;
          break;
        }
      }
    }
    // Reorder flags set to true cannot be altered.
  }

  // Case 2
  for (Node *n : nodes) {
    // Let A and B be Zen nodes, and X be a non-Zen node.
    // rb - reorder_before, ra - reorder_after
    // Handle second case of branching:
    //    B(rb=False, ra)   X
    //                  \  /
    //                   A(rb,ra=True)
    if (reorder_flags[n].first == false) {
      for (const Edge *e : n->in_edges()) {
        Node *src = e->src();
        auto it = std::find(nodes.begin(), nodes.end(), src);
        if (it != nodes.end() && reorder_flags[src].second) {
          // Found Zen node.
          reorder_flags[n].first = true;
          break;
        }
      }
    }
    // Reorder flags set to true cannot be altered.
  }

  // Case 3
  for (Node *n : nodes) {
    // Let A be a Zen nodes, and B and X be a Zen/Non Zen node.
    // rb - reorder_before, ra - reorder_after
    // Handle third case of branching:
    //    B(rb, ra=True)    X (set ra=True) if one of the siblings has ra=True
    //                  \  /
    //                   A
    if (reorder_flags[n].second == false) {
      for (const Edge *e : n->out_edges()) {
        Node *dst = e->dst();
        for (const Edge *f : dst->in_edges()) {
          Node *src = f->src();
          auto it = std::find(nodes.begin(), nodes.end(), src);
          if (it != nodes.end() && src != n && reorder_flags[src].second) {
            // Found a sibling with reorder_after set to True.
            reorder_flags[n].second = true;
            break;
          }
        }
      }
    }
    // Reorder flags set to true cannot be altered.
  }

  return reorder_flags;
}

bool ZenLayoutRewritePass::AddReorderAttrs(std::unique_ptr<Graph> *g) {
  bool result = false;
  CHECK_NOTNULL(g);  // Crash ok.

  std::vector<Node *> order;
  GetReversePostOrder(**g, &order);
  std::vector<Node *> zen_nodes;

  for (Node *n : order) {
    std::string op_name = n->type_string();
    bool is_eager;

    // NOTE: Every Zen op must have the prefix "_Zen".
    auto found = op_name.find(zen_op_registry::kZenNodePrefix);
    if (found != std::string::npos) {
      // Found a Zen op.
      TF_CHECK_OK(GetNodeAttr(n->def(), "is_eager", &is_eager));
      if (is_eager == false) {
        zen_nodes.push_back(n);
      }
    }
  }

  std::unordered_map<Node *, std::pair<bool, bool>> reorder_flags =
      GetReorderFlags(zen_nodes);

  for (Node *n : zen_nodes) {
    std::string node_name = n->name();
    std::string op_name = n->type_string();
    std::pair<bool, bool> n_reorder = reorder_flags[n];

    ZenOpRewriteRecord rewrite_record;
    for (auto it = zen_rewrite_db_.begin(); it < zen_rewrite_db_.end(); it++) {
      if (op_name == it->zen_op_name) {
        rewrite_record = *it;
        break;
      }
    }

    // Rewrite op with a copy containing the new reorder flags.
    if (ZenOpNodeRewrite(g, n, &rewrite_record, n_reorder) == OkStatus()) {
      VLOG(1) << "ZenLayoutRewritePass::AddReorderAttrs: Node " << node_name
              << " " << op_name << " updated reorders to " << n_reorder.first
              << " " << n_reorder.second;
      result = true;
    }
  }

  return result;
}

bool ZenLayoutRewritePass::ZenOpUpdate(std::unique_ptr<Graph> *g) {
  bool result = false;
  std::vector<Node *> order;
  GetReversePostOrder(**g, &order);
  for (Node *n : order) {
    if (!n->IsOp() || !zendnn::CanOpRunOnCPUDevice(n)) {
      continue;
    }

    const ZenOpRewriteRecord *rewrite_record = nullptr;
    if ((rewrite_record = CheckNodeForZenOpRewrite(n)) != nullptr) {
      string node_name = n->name();
      string op_name = n->type_string();
      std::pair<bool, bool> n_reorder(true, true);
      if (ZenOpNodeRewrite(g, n, rewrite_record, n_reorder) == OkStatus()) {
        VLOG(1) << "ZenLayoutRewritePass::ZenOpUpdate: Node " << op_name
                << " rewritten with ZenOp " << rewrite_record->zen_op_name;
        result = true;
      } else {
        // Rewriting the node with Zen op failed. Hence the existing node will
        // be there with graph for inference.
        VLOG(1) << "ZenLayoutRewritePass::ZenOpUpdate: Failed to rewrite node "
                << node_name << " with Zen op " << op_name;
      }
    }
  }
  return result;
}

// Method to find whether the graph has inference ops only. It returns error
// status if the graph has training ops.
Status ZenLayoutRewritePass::AreAllInferenceOps(std::unique_ptr<Graph> *g) {
  std::vector<Node *> order;
  GetReversePostOrder(**g, &order);
  for (Node *n : order) {
    if (!n->IsOp()) {
      continue;
    }
    for (auto op = tf_training_ops_.cbegin(); op != tf_training_ops_.cend();
         ++op) {
      if (n->type_string().find(*op) != string::npos) {
        return Status(absl::StatusCode::kUnimplemented,
                      "Training operation found! Currently TF-ZenDNN "
                      "does not support training. Set environment "
                      "variable TF_ENABLE_ZENDNN_OPTS to '0' for training.");
      }
    }
  }
  return OkStatus();
}

bool ZenLayoutRewritePass::ZenOpRewritePass(std::unique_ptr<Graph> *g) {
  bool result = false;
  CHECK_NOTNULL(g);  // Crash ok.

  // Before we proceed further for Zen Op rewrites first the graph shall be
  // checked for inference ops only as TF-ZenDNN currently does not support
  // training, it supports inference only.
  TF_CHECK_OK(AreAllInferenceOps(g));

  std::vector<Node *> order;

  DumpGraph("\nBefore ZenRewritePass:\n", &**g);

  // Two passes of Graph optimization:

  // First pass implements Basic Fusion Eg. Conv2D-Bias-Relu.
  result = ZenOpUpdate(g);
  if (!result) {
    VLOG(1) << "ZenLayoutRewritePass::ZenOpRewritePass: No opportunity for Zen "
            << "op conversion found in first graph optimization pass.";
  }
  // Second Pass - Enable Fused Optimizations
  // Enable Advanced Graph Optimizations
  GetReversePostOrder(**g, &order);
  for (Node *n : order) {
    if (!n->IsOp() || !zendnn::CanOpRunOnCPUDevice(n)) {
      continue;
    }
    // Fused Optimizations

    // Check and perform Pad fusion with FusedConv2D/Conv2D (Removes Pad op and
    // expects n to be Pad op).
    if (ZenFusePadConv(g, n, "_ZenFusedConv2D", "Pad")) {
      VLOG(1) << "ZenLayoutRewritePass::ZenOpRewritePass: "
              << "FusedConvPad Successful";
    } else if (ZenFusePadConv(g, n, "_ZenConv2D", "Pad")) {
      VLOG(1) << "ZenLayoutRewritePass::ZenOpRewritePass: ConvPad Successful";
    } else {
      VLOG(1) << "ZenLayoutRewritePass::ZenOpRewritePass: No opportunity for "
              << "Conv-Pad fusion found in second graph optimization pass.";
    }
  }
  // Third Pass to implement optimizations over Zen ops.
  result = ZenOpUpdate(g);
  if (!result) {
    VLOG(1) << "ZenLayoutRewritePass::ZenOpRewritePass: No opportunity for Zen "
            << "op conversion found in third graph optimization pass.";
  }

  result = AddReorderAttrs(g);
  if (!result) {
    VLOG(1) << "ZenLayoutRewritePass::ZenOpRewritePass: No reorder attributes "
            << "were updated.";
  }
  DumpGraph("\nAfter ZenRewritePass:\n", &**g);
  return result;
}

Status ZenLayoutRewritePass::Run(const GraphOptimizationPassOptions &options) {
  if (!IsZenDnnEnabled()) {
    VLOG(2) << "TF-ZENDNN: ZenDNN Inference is disabled! ";
    return OkStatus();
  }

  if (options.graph == nullptr && options.partition_graphs == nullptr) {
    return OkStatus();
  }

  if (options.graph != nullptr) {
    std::unique_ptr<Graph> *graph = std::move(options.graph);
    ZenOpRewritePass(graph);
    options.graph->reset(graph->release());
  } else {
    for (auto &g : *options.partition_graphs) {
      std::unique_ptr<Graph> *graph = std::move(&g.second);
      ZenOpRewritePass(graph);
      (&g.second)->reset(graph->release());
    }
  }

  return OkStatus();
}

}  // namespace tensorflow

#endif  // AMD_ZENDNN
