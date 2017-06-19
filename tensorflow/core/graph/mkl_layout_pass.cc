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

#ifdef INTEL_MKL

#include <algorithm>
#include <functional>
#include <memory>
#include <queue>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>
#include <vector>
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/util/tensor_format.h"

#include "tensorflow/core/graph/mkl_layout_pass.h"
#include "tensorflow/core/util/mkl_util.h"

namespace tensorflow {

// This pass implements rewriting of graph to support following scenarios:
// (A) Merging nodes in the graph
// (B) Rewriting a node in the graph to a new node
//     Rewrite happens under following 2 scenarios:
//     1) Propagating Mkl layout as an additional output tensor
//        (we will loosely call a tensor that carries Mkl layout as Mkl tensor
//         henceforth.) from every Mkl supported NN layer.
//     2) Context-based rewrite: This is needed in order to optimize
//        gradient ops of Conv2D+AddBias. Gradient op of both the Conv2D and
//        MatMul is BiasAddGrad, and we need to rewrite BiasAddGrad into
//        Conv2D-specific BiasAddGrad, and MatMul-specific BiasAddGrad.
//        This is context-specific optimization, where the context is the
//        forward operator that the BiasAddGrad corresponds to.
//
// Example of A : Merging nodes in the graph
// -----------------------------------------
// Currently, we merge Conv2D+AddBias together. Consider Conv2D and BiasAdd as:
//
//           O = Conv2D(A, B)
//           P = BiasAdd(O, C)
//
// We merge them into Conv2DWithBias as:
//           P = _MklConv2DWithBias(A, A_m, B, B_m, C, C_m)
//
// The meaning of A_m, B_m and C_m is explained in B.1.
//
// Merge rules:
//  - The merge for Conv2D and BiasAdd happens when the output of Conv2D _only_
//    goes to BiasAdd.
//  - Also, the intersection of attributes of both the nodes must have same
//    values.
//  - Both the nodes must have been assigned to same device (if any).
//
// Example of B.1 : Rewriting nodes to Mkl nodes
// ---------------------------------------------
// Consider a Relu node. Current definition of Relu node looks like:
//
//           O = Relu(A)
//
// Relu has 1 input (A), and 1 output (O).
//
// This rewrite pass will generate a new graph node for Relu (new node is
// called MklRelu) as:
//
//          O, O_m = MklRelu(A, A_m)
//
// MklRelu has 2 inputs (A and A_m) and 2 outputs (O and O_m). Here input A is
// same as input A of Relu; output O is same as output O of Relu. O_m is the
// additional output tensor that will be set by MklRelu, and it represents
// Mkl tensor corresponding to O -- in other words, O_m is some kind of
// metadata for O. A_m is additional input of Relu, and it represents metadata
// for A - as O_m is metadata for O, A_m is metadata for A. MklRelu receives
// this metadata from previous node in the graph.
//
// When a previous node in the graph is an Mkl node, A_m will represent a valid
// Mkl tensor. But when a previous node is not an Mkl node, A_m will represent
// a dummy Mkl tensor.
//
// Rewriting rules:
//  - Selection of a node for rewriting happens by registering the op type of
//    the node with the rewriting pass. If the op type is not registered, then
//    all nodes of this op type will not be rewritten.
//  - Number of inputs after rewriting:
//      Since for every input Tensorflow tensor, the rewritten node gets Mkl
//      tensor(s), rewritten node gets 2*N inputs, where N is the number of
//      inputs for the original node.
//  - Number of outputs after rewriting:
//      Since for every output Tensorflow tensor, the rewritten node generates
//      Mkl tensor(s), the rewritten node generates 2*N outputs, where N is the
//      number of outputs of the original node.
//  - Ordering of Tensorflow tensors and Mkl tensors:
//      Since every rewritten node generates twice the number of inputs and
//      outputs, one could imagine various orderings among Tensorflow tensors
//      and Mkl tensors. E.g., assume an op 'Conv2D' that takes (A, B) as
//      inputs, then the new op '_MklConv2D' can take inputs A, B, A_m and B_m
//      in A, A_m, B, B_m order or it can also take them in A, B, A_m, B_m
//      order. Among N inputs one can get N! permutations.
//
//      So the question is: which order do we follow? We support 2 types of
//      orderings: (1) interleaved, and (2) contiguous. Interleaved ordering
//      follows an intuitive order where an Mkl tensor follows the
//      corresponding Tensorflow tensor immediately. In the context of the
//      above example, it will be: A, A_m, B, B_m. Note that the ordering rule
//      applies to both the inputs and outputs. Contiguous ordering means
//      all the Tensorflow tensors are contiguous followed by all the Mkl
//      tensors. We use contiguous ordering as default.
//
// Graph rewrite algorithm:
//      Algorithm: Graph Rewrite
//      Input: Graph G, Names of the nodes to rewrite and their new names
//      Output: Modified Graph G' if the nodes are modified, G otherwise.
//      Start:
//        N = Topological_Sort(G) // N is a set of nodes in toposort order.
//        foreach node n in N
//        do
//          if (Is_MKL_Op(n))  // Can this node accept an Mkl layout as input.
//          then
//            E = set of <incoming edge and its src_output slot> of n
//            E' = {}   // a new set of edges for rewritten node
//            foreach <e,s> in E
//            do
//              E' U {<e,s>}  // First copy edge which generates Tensorflow
//                            // tensor as it is
//              m = Source node of edge e
//              if Is_Rewritten(m)  // Did we rewrite this node in this pass?
//              then
//                E' U {<m,s+1>}    // If yes, then m will generate an Mkl
//                                  // tensor as an additional output.
//              else
//                d = Generate_Dummy_Mkl_Tensor()  // If not, generate a dummy
//                                                 // Mkl tensor.
//                E' U {<d,0>}  // The dummy Mkl tensor has only 1 output slot.
//              fi
//            done
//            n' = Build_New_Node(G,new_name,E')
//            Mark_Rewritten(n')  // Mark the new node as being rewritten.
//          fi
//        done
//
//      Explanation:
//        For graph rewrite, we visit nodes of the input graph in the
//        topological sort order. With this ordering, we visit nodes in the
//        top-to-bottom fashion. We need this order because while visiting a
//        node we want that all of its input nodes are visited and rewritten if
//        applicable. This is because if we need to rewrite a given node
//        then all of its input nodes need to be fixed (in other words they
//        cannot be deleted later.)
//
//        While visiting a node, we first check if the op type of the node is
//        an Mkl op. If it is, then we rewrite that node after constructing
//        new inputs to the node. If the op type of the node is not Mkl op,
//        then we do not rewrite that node.
//
// Handling workspace propagation for certain ops:
//
//        Certain backward ops in MKL (MaxPool, LRN and BatchNorm) require
//        passing of a workspace from their respective forward ops. Workspace
//        tensors provide memory for storing results of intermediate operations
//        which are helpful in backward propagation. TensorFlow does not have
//        a notion of a workspace and as a result does not allow producing
//        additional outputs from these forward ops. For these ops, we need
//        to add 2 extra edges between forward ops and their corresponding
//        backward ops - the first extra edge carries a workspace tensor and
//        the second one carries an Mkl tensor for the workspace tensor.
//
//        Example:
//
//        Typical graph for MaxPool and its gradient looks like:
//
//        A = MaxPool(T)
//        B = MaxPoolGrad(X, A, Y)
//
//        We will transform this graph to propagate the workspace as:
//        (with the contiguous ordering)
//
//        A, W, A_m, W_m = MklMaxPool(T, T_m)
//        B, B_m = MklMaxPoolGrad(X, A, Y, W, X_m, A_m, Y_m, W_m)
//
//        Here W is the workspace tensor. Transformed tensor names with the
//        suffix _m are Mkl tensors, and this transformation has been done
//        using the algorithm discussed earlier. The transformation for
//        workspace propagation only adds extra outputs (W, W_m) for a forward
//        op and connects them to the corresponding backward ops.
//
//        Terms:
//
//        Forward op name = name of the op in the forward pass
//          where a workspace tensor originates (MaxPool in this example)
//        Backward op name = name of the op in the backward pass that receives
//          a workspace tensor from the forward op (MaxPoolGrad in the example)
//        Slot = Position of the output or input slot that will be
//               used by the workspace tensor (1 for MklMaxPool as W is the 2nd
//               output of MaxPool (0 is 1st); 3 for MklMaxPoolGrad)
//
//        Question:
//
//        How do we associate a backward op to a forward op? There can be more
//        than one op with the exact same name.
//
//        In this example, we associate MaxPoolGrad with MaxPool. But there
//        could be more than one MaxPool ops. To solve this problem, we look
//        for _direct_ edge between a forward op and a backward op (tensor A is
//        flowing along this edge in the example).
//
//        How do we transform forward and backward ops when there is no direct
//        edge between them? In such a case, we generate dummy tensors for
//        workspace tensors. For the example, transformation of MaxPool will
//        be exactly same as it would be when there is a direct edge between
//        the forward and the backward op --- it is just that MaxPool won't
//        generate any workspace tensor. For MaxPoolGrad, the transformation
//        will also be same, but instead of connecting W and W_m with the
//        outputs of MaxPool, we will produce dummy tensors for them, and we
//        will set workspace_enabled attribute to false.
//
// Example of B.2 : Context-based node rewrite
// -------------------------------------------
// Consider BiasAddGrad op as:
//
//           O = _MklConv2D(A, B, C, A_m, B_m, C_m)
//           P = BiasAddGrad(O)
//
// Then we rewrite it as:
//
//           P = Conv2DWithBiasBackpropBias(O, O_m)
//
// 'Distance' between input of BiasAddGrad and _MklConv2D in terms of hops is
// the context matching depth. If _MklConv2DWithBias is not within the context
// matching depth, then we do not rewrite BiasAddGrad.

// How many hops do we search for matching node in the backward dataflow graph?
// We use maxhop of 10 based on empirical observations. Also, these are
// maxhops in backward data-flow graph. Since input of forward nodes (Conv2D)
// directly goes to backward nodes, we do not expect the hop-distance
// would be more than few nodes.
static size_t kNodeMergeContextMaxDepth = 10;

class MklLayoutRewritePass : public GraphOptimizationPass {
 public:
  MklLayoutRewritePass() {
    // NOTE: names are alphabetically sorted.
    csinfo_.avg_pool = "AvgPool";
    csinfo_.avg_pool_grad = "AvgPoolGrad";
    csinfo_.bias_add = "BiasAdd";
    csinfo_.bias_add_grad = "BiasAddGrad";
    csinfo_.concat = "Concat";
    csinfo_.concatv2 = "ConcatV2";
    csinfo_.conv2d = "Conv2D";
    csinfo_.conv2d_grad_input = "Conv2DBackpropInput";
    csinfo_.conv2d_grad_filter = "Conv2DBackpropFilter";
    csinfo_.fused_batch_norm = "FusedBatchNorm";
    csinfo_.fused_batch_norm_grad = "FusedBatchNormGrad";
    csinfo_.identity = "Identity";
    csinfo_.lrn = "LRN";
    csinfo_.lrn_grad = "LRNGrad";
    csinfo_.matmul = "MatMul";
    csinfo_.max_pool = "MaxPool";
    csinfo_.max_pool_grad = "MaxPoolGrad";
    csinfo_.mkl_conv2d = "_MklConv2D";
    csinfo_.mkl_conv2d_with_bias = "_MklConv2DWithBias";
    csinfo_.mkl_conv2d_with_bias_backprop_bias =
                                   "_MklConv2DWithBiasBackpropBias";
    csinfo_.relu                  = "Relu";
    csinfo_.relu_grad             = "ReluGrad";
    csinfo_.reshape               = "Reshape";
    csinfo_.split                 = "Split";

    // NOTE: names are alphabetically sorted.
    rinfo_.push_back({csinfo_.avg_pool,
                      GetMklOpName(csinfo_.avg_pool),
                      CopyAttrsPooling, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.avg_pool_grad,
                      GetMklOpName(csinfo_.avg_pool_grad),
                      CopyAttrsPooling, AlwaysRewrite, nullptr});
    // BiasAddGrad gets written into Conv2DWithBiasBackpropBias depending
    // on if context contains Conv2D.
    rinfo_.push_back({csinfo_.bias_add_grad,
                      csinfo_.mkl_conv2d_with_bias_backprop_bias,
                      CopyAttrsBiasAddGrad, ContextMatchRewrite,
                      &biasaddgrad_conv2dwithbias_context_});
    // BiasAddGrad gets written into BiasAddGrad depending on if context
    // contains MatMul.
    rinfo_.push_back({csinfo_.bias_add_grad, csinfo_.matmul,
                      CopyAttrsBiasAddGrad, ContextMatchRewrite,
                      &biasaddgrad_matmul_context_});
    rinfo_.push_back({csinfo_.concat,
                      GetMklOpName(csinfo_.concat),
                      CopyAttrsConcat, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.concatv2,
                      GetMklOpName(csinfo_.concatv2),
                      CopyAttrsConcatV2, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.conv2d,
                      GetMklOpName(csinfo_.conv2d),
                      CopyAttrsConv2D, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.conv2d_grad_filter,
                      GetMklOpName(csinfo_.conv2d_grad_filter),
                      CopyAttrsConv2D, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.conv2d_grad_input,
                      GetMklOpName(csinfo_.conv2d_grad_input),
                      CopyAttrsConv2D, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.fused_batch_norm,
                      GetMklOpName(csinfo_.fused_batch_norm),
                      CopyAttrsFusedBatchNorm, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.fused_batch_norm_grad,
                      GetMklOpName(csinfo_.fused_batch_norm_grad),
                      CopyAttrsFusedBatchNorm, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.identity,
                      GetMklOpName(csinfo_.identity),
                      CopyAttrsIdentity, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.lrn,
                      GetMklOpName(csinfo_.lrn),
                      CopyAttrsLRN, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.lrn_grad,
                      GetMklOpName(csinfo_.lrn_grad),
                      CopyAttrsLRN, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.max_pool,
                      GetMklOpName(csinfo_.max_pool),
                      CopyAttrsPooling, NonDepthBatchWisePoolRewrite, nullptr});
    rinfo_.push_back({csinfo_.max_pool_grad,
                      GetMklOpName(csinfo_.max_pool_grad),
                      CopyAttrsPooling, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.relu,
                      GetMklOpName(csinfo_.relu),
                      CopyAttrsRelu, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.relu_grad,
                      GetMklOpName(csinfo_.relu_grad),
                      CopyAttrsRelu, AlwaysRewrite, nullptr});
    rinfo_.push_back({csinfo_.reshape,
                      GetMklOpName(csinfo_.reshape),
                      CopyAttrsReshape, AlwaysRewrite, nullptr});

    // Add info about which ops to add workspace edge to and the slots.
    wsinfo_.push_back({csinfo_.lrn, csinfo_.lrn_grad, 0, 2, 1, 3});
    wsinfo_.push_back({csinfo_.max_pool, csinfo_.max_pool_grad, 0, 1, 1, 3});

    // Add a rule for merging nodes
    minfo_.push_back({csinfo_.mkl_conv2d, csinfo_.bias_add, 0,
                      csinfo_.mkl_conv2d_with_bias});

    // We use maxhop of 10 based on empirical observations. Also, these are
    // maxhops in backward data-flow graph. Since input of forward nodes
    // (Conv2D) directly goes to backward nodes, we do not expect the
    // hop-distance would be more than few nodes.
    biasaddgrad_matmul_context_ = {csinfo_.bias_add_grad, csinfo_.matmul,
                                   kNodeMergeContextMaxDepth};

    biasaddgrad_conv2dwithbias_context_ = {csinfo_.bias_add_grad,
                                   csinfo_.mkl_conv2d_with_bias,
                                   kNodeMergeContextMaxDepth};

    cinfo_.push_back(&biasaddgrad_matmul_context_);
    cinfo_.push_back(&biasaddgrad_conv2dwithbias_context_);
  }

  // Standard interface to run pass
  Status Run(const GraphOptimizationPassOptions& options);

  // Helper function which does most of heavy lifting for rewriting
  // Mkl nodes to propagate Mkl tensor as additional output
  //
  // Extracts common functionality between Run public interface and
  // test interface.
  //
  // @return true, if and only if graph is mutated; false otherwise.
  bool RunPass(std::unique_ptr<Graph>* g);

  /// Structure to specify the context information used in a node rewrite rule
  typedef struct {
    string node;     // Name of the node to be rewritten
    string fwd;      // Name of the node in the forward pass that this node
                     // corresponds to
    size_t max_hop;  // Maximum number of hops the fwd is located
                     // from this node. If the fwd is farther than max_hop
                     // then we do not rewrite the node.
  } ContextInfo;

  /// Structure to specify the name of an original node, its new name after
  /// rewrite, the number of inputs to the original node, the function to
  /// be used to copy attributes for the op, and the rule (if any) which
  /// must hold for rewriting the node
  typedef struct {
    string name;      // Original name of op of the node in the graph
    string new_name;  // New name of the op of the node in the graph
    // A function handler to copy attributes from an old node to a new node.
    std::function<void(const Node*, NodeBuilder*)> copy_attrs;
    // A rule under which to rewrite this node
    std::function<bool(const Node*, const ContextInfo* c)> rewrite_rule;
    // ContextInfo, if any, to be used for rewrite
    ContextInfo* context;
  } RewriteInfo;

  /// Structure to specify a forward op, a backward op, and the slot numbers
  /// in the forward and backward ops where we will add a workspace edge.
  typedef struct {
    string fwd_op;    // Name of a forward op in the graph
    string bwd_op;    // Name of a backward op in the graph
    int fwd_slot;     // Output slot in the forward op node where actual
                      // output tensor resides
    int bwd_slot;     // Input slot in the backward op node where actual
                      // input tensor resides
    int ws_fwd_slot;  // Output slot in the forward op node where workspace
                      // edge is added
    int ws_bwd_slot;  // Input slot in the backward op node where workspace
                      // edge is added
  } WorkSpaceInfo;

  /// Structure to specify information used in node merge
  typedef struct {
    string pred;      // Predecessor node string
    string succ;      // Successor node string
    int op;           // The operand no the predecessor node corresponds
                      // to the successor node
    string new_node;  // Name of the node after merge
  } MergeInfo;

  /// Structure to store all constant strings
  /// NOTE: names are alphabetically sorted.
  struct {
    string avg_pool;
    string avg_pool_grad;
    string bias_add;
    string bias_add_grad;
    string concat;
    string concatv2;
    string conv2d;
    string conv2d_grad_input;
    string conv2d_grad_filter;
    string fused_batch_norm;
    string fused_batch_norm_grad;
    string identity;
    string lrn;
    string lrn_grad;
    string matmul;
    string max_pool;
    string max_pool_grad;
    string mkl_conv2d;
    string mkl_conv2d_with_bias;
    string mkl_conv2d_with_bias_backprop_bias;
    string relu;
    string relu_grad;
    string reshape;
    string split;
  } csinfo_;

 private:
  /// Maintain info about nodes to rewrite
  std::vector<RewriteInfo> rinfo_;

  /// Maintain info about nodes to add workspace edge
  std::vector<WorkSpaceInfo> wsinfo_;

  /// Maintain info about nodes to be merged
  std::vector<MergeInfo> minfo_;

  /// Maintain info about nodes to rewrite
  static std::vector<ContextInfo*> cinfo_;

  /// Context variables used in referencing rules
  static ContextInfo biasaddgrad_matmul_context_;
  static ContextInfo biasaddgrad_conv2dwithbias_context_;

  /// Hash table to maintain nodes visited in the graph.
  std::unordered_set<const Node*> visited_nodes_;

 private:
  // Check if we rewrote node 'n'
  //
  // If we rewrote the node, then the rewritten node will produce
  // Mkl tensor as output. If we did not rewrite the node, then
  // we need to insert dummy Mkl node on the input side.
  //
  // Returns true if node is rewritten, false otherwise.
  inline bool IsRewrittenNode(Node* n) const {
    return visited_nodes_.find(n) != visited_nodes_.end();
  }

  // Mark the node as rewritten
  inline void MarkRewrittenNode(Node* n) { visited_nodes_.insert(n); }

  // Clear all visited nodes
  inline void UnMarkRewrittenNodes() { visited_nodes_.clear(); }

  // Is OpDef::ArgDef a list type? It could be N * T or list(type).
  // Refer to opdef.proto for details of list type.
  inline bool ArgIsList(const OpDef::ArgDef& arg) const {
    return !arg.type_list_attr().empty() || !arg.number_attr().empty();
  }

  // Get length of a list in 'n' if 'arg' is of list type. Refer to
  // description of ArgIsList for definition of list type.
  inline int GetTensorListLength(const OpDef::ArgDef& arg, Node* n) {
    CHECK_EQ(ArgIsList(arg), true);
    int N = 0;
    const string attr_name = !arg.type_list_attr().empty()
                                 ? arg.type_list_attr()
                                 : arg.number_attr();
    if (!arg.type_list_attr().empty()) {
      std::vector<DataType> value;
      TF_CHECK_OK(GetNodeAttr(n->def(), attr_name, &value));
      N = value.size();
    } else {
      TF_CHECK_OK(GetNodeAttr(n->def(), attr_name, &N));
    }
    return N;
  }

  // Get the name of Mkl op from original TensorFlow op
  // We prefix 'Mkl' to the original op to get Mkl op.
  // TODO(nhasabni) We should move this to mkl_util.h.
  inline string GetMklOpName(const string& name) const {
    // Prefix that we add to Tensorflow op name to construct Mkl op name.
    const char* const kMklOpPrefix = "_Mkl";
    return string(kMklOpPrefix) + name;
  }

  // Can op represented by node 'n' run on DEVICE_CPU?
  // Op can run on CPU with MKL if the runtime assigned device or the
  // user requested device contains device CPU, or both are empty.
  bool CanOpRunOnCPUDevice(const Node* n) {
    bool result = true;
    string reason;

    // Substring that should be checked for in device name for CPU device.
    const char* const kCPUDeviceSubStr = "cpu";

    // If Op has been specifically assigned to a non-CPU device, then No.
    if (!n->assigned_device_name().empty() &&
        !StringPiece(n->assigned_device_name()).contains(kCPUDeviceSubStr)) {
      result = false;
      reason = "Op has been assigned a runtime device that is not CPU.";
    }

    // If user has specifically assigned this op to a non-CPU device, then No.
    if (!n->def().device().empty() &&
        !StringPiece(n->def().device()).contains(kCPUDeviceSubStr)) {
      result = false;
      reason = "User has assigned a device that is not CPU.";
    }

    if (result == false) {
      VLOG(1) << "MklLayoutRewritePass: Skipping rewriting of the node "
              << n->type_string() << ", reason: " << reason;
    }

    // Otherwise Yes.
    return result;
  }

  // Return a node that can be merged with input node 'n'
  //
  // @return pointer to the node if we can find such a
  // node. Otherwise, it returns nullptr.
  Node* CheckForNodeMerge(const Node* n) const;

  // Merge predecessor node with its successor.
  // Currently, we merge Conv2D with BiasAdd only.
  //
  // Input nodes succ and pred may be deleted if the call to
  // this function is successful. Attempt to use the pointers
  // after the call to function may result in undefined behaviors.
  //
  // @input g - input graph, succ - successor node, pred - predecessor node
  // @return Status::OK(), if merging is successful and supported.
  //         Returns appropriate Status error code otherwise.
  //         Graph is updated in case nodes are merged. Otherwise, it is
  //         not updated.
  Status MergeNode(std::unique_ptr<Graph>* g, Node* succ, Node* pred);

  // Check if the node 'n' has any applicable rewrite rule
  // We check for 2 scenarios for rewrite.
  //
  // @return RewriteInfo* for the applicable rewrite rule
  const RewriteInfo* CheckForNodeRewrite(const Node* n) const;

  // Default rewrite rule to be used in scenario 1 for rewrite.
  // @return - true (since we want to always rewrite)
  static bool AlwaysRewrite(const Node* n, const ContextInfo* c = nullptr) {
    return true;
  }

  // Check if we are performing pooling on depth or batch. If it is, then we
  // do not rewrite MaxPool node to Mkl version.
  // @return - true (if it is not a depth/batch wise pooling case);
  //           false otherwise.
  static bool NonDepthBatchWisePoolRewrite(const Node* n,
                                           const ContextInfo* c) {
    CHECK_NOTNULL(n);

    string data_format_str;
    TensorFormat data_format;
    std::vector<int32> ksize, strides;
    CHECK_EQ(GetNodeAttr(n->def(), "ksize", &ksize).ok(), true);
    CHECK_EQ(GetNodeAttr(n->def(), "strides", &strides).ok(), true);
    CHECK_EQ(GetNodeAttr(n->def(), "data_format", &data_format_str).ok(),
             true);
    CHECK_EQ(FormatFromString(data_format_str, &data_format), true);

    // Condition that specifies non-batch-wise and non-depth-wise pooling.
    if (GetTensorDim(ksize,   data_format, 'N') == 1 &&
        GetTensorDim(strides, data_format, 'N') == 1 &&
        GetTensorDim(ksize,   data_format, 'C') == 1 &&
        GetTensorDim(strides, data_format, 'C') == 1) {
      return true;
    }

    return false;
  }

  // Rewrite rule that uses context-information for matching,
  // used in scenario 2.
  //
  // @input - Node 'n' for which to search for matching context
  // @input - The context 'c' under which to rewrite
  // @return - true if we can rewrite node under context 'c';
  //           false otherwise.
  static bool ContextMatchRewrite(const Node* n, const ContextInfo* c);

  // Helper function that searches the matching contextinfo for the node.
  // Implements depth-first search in the data dependence graph for the
  // gradient op in the backward direction.
  //
  // @input n - Node (gradient op) whose contextinfo is to be searched,
  //        fwd_node - pointer to node from the forward pass that this node
  //        belongs to. fwd_node cannot be NULL.
  // @return Matching contextinfo in case a match is found; null otherwise.
  //         Also updates *fwd_node with pointer to forward node that this
  //         context matches.
  static const ContextInfo* SearchMatchingContext(const Node* n,
                                                  const Node** fwd_node);

  // Rewrites input node to a new node specified by its matching rewrite info.
  //
  // Method first searches matching rewrite info for input node and then
  // uses that info to rewrite.
  //
  // Input node may be deleted in case of rewrite. Attempt to use the node
  // after the call can result in undefined behaviors.
  //
  // @input  g - input graph, n - Node to be rewritten,
  //         ri - matching rewriteinfo
  // @return Status::OK(), if the input node is rewritten;
  //         Returns appropriate Status error code otherwise.
  //         Graph is updated in case the input node is rewritten.
  //         Otherwise, it is not updated.
  Status RewriteNode(std::unique_ptr<Graph>* g, Node* n, const RewriteInfo* ri);

  // Get nodes that will feed a list of TF tensors to the new
  // node that we are constructing.
  //
  // @input g - input graph,
  // @input inputs - inputs to old node that we are using for constructing
  //                 new inputs,
  // @input input_idx - the index in the 'inputs' vector pointing to the
  //                    current input that we have processed so far
  // @output input_idx - index will be incremented by the number of nodes
  //                     from 'inputs' that are processed
  // @input list_length - The expected length of list of TF tensors
  // @output output_nodes - the list of new nodes creating TF tensors
  //
  // @return None
  void GetNodesProducingTFTensorList(
      const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs,
      int* input_idx, int list_length,
      std::vector<NodeBuilder::NodeOut>* output_nodes);

  // Get nodes that will feed a list of Mkl tensors to the new
  // node that we are constructing.
  //
  // @input g - input graph,
  // @input orig_node - Original node that we are rewriting
  // @input inputs - inputs to old node that we are using for constructing
  //                 new inputs,
  // @input input_idx - the index in the 'inputs' vector pointing to the
  //                    current input that we have processed so far
  // @output input_idx - index will be incremented by the number of nodes
  //                     from 'inputs' that are processed
  // @input list_length - The expected length of list of Mkl tensors
  // @output output_nodes - the list of new nodes creating Mkl tensors
  //
  // @return None
  void GetNodesProducingMklTensorList(std::unique_ptr<Graph>* g,
    Node* orig_node, const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs,
    int* input_idx, int list_length,
    std::vector<NodeBuilder::NodeOut>* output_nodes);

  // Get a node that will feed an Mkl tensor to the new
  // node that we are constructing. The output node could be (1) 'n'
  // if it is Mkl layer, or (2) a dummy node producing dummy Mkl tensor
  // if 'n' is not an Mkl layer.
  //
  // @input g - input graph,
  // @input orig_node - Original node that we are rewriting,
  // @input n - Node based on which we are creating Mkl node,
  // @input n_output_slot - the output slot of node 'n'
  //            which is feeding to the node that we are constructing
  // @output mkl_node - the new node that will feed Mkl tensor
  // @output mkl_node_output_slot - the slot number of mkl_node that
  //                                will feed the tensor
  // @return None
  void GetNodeProducingMklTensor(std::unique_ptr<Graph>* g, Node* orig_node,
    Node* n, int n_output_slot, Node** mkl_node, int* mkl_node_output_slot);

  // Setup new inputs using old inputs 'inputs' for the rewritten node in 'nb'
  // in graph 'g'. Original node is input in 'old_node'. Inputs to 'nb' are
  // set up in contiguous fashion. 'workspace_tensors' carry graph nodes
  // producing workspace edges if 'are_workspace_tensors_available' is true.
  // Otherwise, 'workspace_tensors' is empty vector.
  //
  // For details, refer to 'Ordering of inputs after rewriting' section in the
  // documentation above.
  //
  // Returns Status::OK() if setting up inputs is successful, otherwise
  // returns appropriate status code.
  int SetUpContiguousInputs(
      std::unique_ptr<Graph>* g,
      const gtl::InlinedVector<std::pair<Node*, int>, 4>& old_node_inputs,
      NodeBuilder* nb, Node* old_node,
      std::vector<NodeBuilder::NodeOut>* workspace_tensors,
      bool are_workspace_tensors_available);

  // Setup new inputs using old inputs 'inputs' for the rewritten node in 'nb'
  // in graph 'g'. Original node is input in 'orig_node'.
  //
  // For details, refer to 'Ordering of Tensorflow tensors and Mkl tensors'
  // section in the documentation above.
  //
  // Returns Status::OK() if setting up inputs is successful, otherwise
  // returns appropriate status code.
  Status SetUpInputs(std::unique_ptr<Graph>* g,
                     const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs,
                     NodeBuilder* nb, Node* orig_node);

  // Add workspace edge on the input or output side of Node 'orig_node' by using
  // NodeBuilder 'nb' for the new node provided. If 'orig_node' does not dictate
  // adding workspace edge then do not add it. Workspace Tensorflow and Mkl
  // tensors, if they need to be added, will be set into these tensors.
  // If we set workspace tensors, then are_ws_tensors_added should be true.
  void AddWorkSpaceEdgeIfNeeded(std::unique_ptr<Graph>* g, Node* orig_node,
                                NodeBuilder* nb,
                                std::vector<NodeBuilder::NodeOut>* ws_tensors,
                                bool* are_ws_tensors_added);

  // Functions specific to operators to copy attributes
  // We need operator-specific function to copy attributes because the framework
  // does not provide any generic function for it.
  // NOTE: names are alphabetically sorted.
  static void CopyAttrsBiasAddGrad(const Node* orig_node, NodeBuilder* nb);
  static void CopyAttrsConcat(const Node* orig_node, NodeBuilder* nb);
  static void CopyAttrsConcatV2(const Node* orig_node, NodeBuilder* nb);
  static void CopyAttrsConv2D(const Node* orig_node, NodeBuilder* nb);
  static void CopyAttrsFusedBatchNorm(const Node* orig_node, NodeBuilder* nb);
  static void CopyAttrsIdentity(const Node* orig_node, NodeBuilder* nb);
  static void CopyAttrsLRN(const Node* orig_node, NodeBuilder* nb);
  static void CopyAttrsPooling(const Node* orig_node, NodeBuilder* nb);
  static void CopyAttrsRelu(const Node* orig_node, NodeBuilder* nb);
  static void CopyAttrsReshape(const Node* orig_node, NodeBuilder* nb);
  static void CopyAttrsSplit(const Node* orig_node, NodeBuilder* nb);

  // Generate a graph node in graph 'g' representing a dummy Mkl tensor node,
  // using node for original node 'orig_node' and return it in '*out'.
  // TODO(nhasabni) We should move this to mkl_util.h
  void GetDummyMklTensorNode(std::unique_ptr<Graph>* g, Node** out,
                             Node* orig_node);
  void GetDummyWorkspaceTensorNode(std::unique_ptr<Graph>* g, Node** out,
                                   Node* orig_node);
};

MklLayoutRewritePass::ContextInfo
  MklLayoutRewritePass::biasaddgrad_conv2dwithbias_context_;
MklLayoutRewritePass::ContextInfo
  MklLayoutRewritePass::biasaddgrad_matmul_context_;
std::vector<MklLayoutRewritePass::ContextInfo*> MklLayoutRewritePass::cinfo_;

// We register Mkl rewrite pass for phase 1 in post partitioning group.
// We register it here so that we get a complete picture of all users of Mkl
// nodes. Do not change the ordering of the Mkl passes.
const OptimizationPassRegistry::Grouping kMklLayoutRewritePassGroup =
    OptimizationPassRegistry::POST_PARTITIONING;
REGISTER_OPTIMIZATION(kMklLayoutRewritePassGroup, 1, MklLayoutRewritePass);

//////////////////////////////////////////////////////////////////////////
//           Helper functions for creating new node
//////////////////////////////////////////////////////////////////////////

static void FillInputs(const Node* n,
                       gtl::InlinedVector<Node*, 4>* control_edges,
                       gtl::InlinedVector<std::pair<Node*, int>, 4>* in) {
  control_edges->clear();
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      control_edges->push_back(e->src());
    } else {
      (*in)[e->dst_input()] = std::make_pair(e->src(), e->src_output());
    }
  }
  std::sort(control_edges->begin(), control_edges->end());
  if (n->op_def().is_commutative()) {
    // For commutative inputs, we sort the input by the input Node*
    // to get a canonical ordering (so that add(a,b) and add(b, a) will
    // hash to the same value if is_commutative is true for 'add').
    std::sort(in->begin(), in->end());
  }
}

void MklLayoutRewritePass::GetNodesProducingTFTensorList(
    const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs, int* input_idx,
    int list_length, std::vector<NodeBuilder::NodeOut>* output_nodes) {
  CHECK_LT(*input_idx, inputs.size());
  CHECK_GT(list_length, 0);
  CHECK_NOTNULL(output_nodes);
  output_nodes->reserve(list_length);

  while (list_length != 0) {
    CHECK_GT(list_length, 0);
    CHECK_LT(*input_idx, inputs.size());
    Node* n = inputs[*input_idx].first;
    int slot = inputs[*input_idx].second;
    // If input node 'n' is just producing a single tensor at
    // output slot 'slot' then we just add that single node.
    output_nodes->push_back(NodeBuilder::NodeOut(n, slot));
    (*input_idx)++;
    list_length--;
  }
}

// TODO(nhasabni) We should move this to mkl_util.h.
void MklLayoutRewritePass::GetDummyMklTensorNode(std::unique_ptr<Graph>* g,
                                                 Node** out, Node* orig_node) {
  // We use a tensor of shape {8} and value 0,0,0,0,0,0,0,0 to represent
  // dummy Mkl tensor. 8 = 2*size_t.
  const DataType dt = DataTypeToEnum<uint8>::v();
  TensorProto proto;
  proto.set_dtype(dt);
  uint8 zero[8] = {0, 0, 0, 0, 0, 0, 0, 0};
  proto.set_tensor_content(const_cast<const void*>(static_cast<void*>(&zero)),
                           8);
  TensorShape dummy_shape({8});
  dummy_shape.AsProto(proto.mutable_tensor_shape());
  TF_CHECK_OK(NodeBuilder((*g)->NewName("DMT"), "Const")
               .Attr("value", proto)
               .Attr("dtype", dt)
               .Device(orig_node->def().device())  // We place this node on
                                                   // the same device as the
                                                   // device of the original
                                                   // node.
               .Finalize(&**g, out));

  // If number of inputs to the original node is > 0, then we add
  // control dependency between 1st input (index 0) of the original node and
  // the dummy Mkl node. This is needed because control-flow ops such as Enter,
  // Merge, etc, require frame_name of the dummy Mkl node to be same as the
  // rewritten node. Adding control edge between 1st input of the original node
  // and the dummy Mkl node ensures that the dummy node is in the same frame
  // as the original node. Choosing 1st input is not necessary - any input of
  // the original node is fine because all the inputs of a node are always in
  // the same frame.
  if (orig_node->num_inputs() > 0) {
    Node* orig_input0 = nullptr;
    TF_CHECK_OK(orig_node->input_node(0,
                                      const_cast<const Node**>(&orig_input0)));
    CHECK_NOTNULL((*g)->AddControlEdge(orig_input0, *out));
  }

  (*out)->set_assigned_device_name(orig_node->assigned_device_name());
}

void MklLayoutRewritePass::GetNodesProducingMklTensorList(
    std::unique_ptr<Graph>* g,
    Node* orig_node,
    const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs,
    int* input_idx, int list_length,
    std::vector<NodeBuilder::NodeOut>* output_nodes) {
  CHECK_LT(*input_idx, inputs.size());
  CHECK_GT(list_length, 0);
  CHECK_NOTNULL(output_nodes);
  output_nodes->reserve(list_length);

  while (list_length != 0) {
    CHECK_GT(list_length, 0);
    CHECK_LT(*input_idx, inputs.size());
    Node* n = inputs[*input_idx].first;
    int slot = inputs[*input_idx].second;
    // If 'n' is producing a single tensor, then create a single Mkl tensor
    // node.
    Node* mkl_node = nullptr;
    int mkl_node_output_slot = 0;
    GetNodeProducingMklTensor(g, orig_node, n, slot, &mkl_node,
                              &mkl_node_output_slot);
    output_nodes->push_back(NodeBuilder::NodeOut(mkl_node,
                                                mkl_node_output_slot));
    (*input_idx)++;
    list_length--;
  }
}

// Get an input node that will feed Mkl tensor to the new
// node that we are constructing. An input node could be (1) 'n'
// if it is Mkl layer, or (2) a dummy node producing dummy Mkl tensor
// if 'n' is not an Mkl layer.
void MklLayoutRewritePass::GetNodeProducingMklTensor(std::unique_ptr<Graph>* g,
    Node* orig_node, Node* n,
    int n_output_slot, Node** mkl_node, int* mkl_node_output_slot) {
  CHECK_NOTNULL(n);
  CHECK_NOTNULL(mkl_node);
  CHECK_NOTNULL(mkl_node_output_slot);
  if (IsRewrittenNode(n)) {
    // If we have visited this node and rewritten it, then it will generate
    // an edge that will receive Mkl tensor from a node.
    // First, let's assert that this op is Mkl layer.
    DataType T;
    TF_CHECK_OK(GetNodeAttr(n->def(), "T", &T));
    // If this op has been rewritten, then its name must have been same as
    // Mkl op.
    CHECK_EQ(mkl_op_registry::IsMklOp(n->type_string(), T), true);
    // output slot number for Mkl tensor would be N+slot number of TensorFlow
    // tensor, where N is total number of TensorFlow tensors.
    *mkl_node = n;
    *mkl_node_output_slot =
        GetTensorMetaDataIndex(n_output_slot, n->num_outputs());
  } else {
    // If we have not visited the node and rewritten it, then we need
    // to create a dummy node that will feed a dummy Mkl tensor to this node.
    // DummyMklTensor node has no input and generates only 1 output
    // (dummy Mkl tensor) as output slot number 0.
    GetDummyMklTensorNode(g, mkl_node, orig_node);
    CHECK_NOTNULL(*mkl_node);
    *mkl_node_output_slot = 0;
  }
}

int MklLayoutRewritePass::SetUpContiguousInputs(
    std::unique_ptr<Graph>* g,
    const gtl::InlinedVector<std::pair<Node*, int>, 4>& old_node_inputs,
    NodeBuilder* nb, Node* old_node,
    std::vector<NodeBuilder::NodeOut>* workspace_tensors,
    bool are_workspace_tensors_available) {
  CHECK_NOTNULL(workspace_tensors);
  CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);

  // Number of input slots to original op
  // Input slots are represented by .Input() calls in REGISTER_OP.
  int old_node_input_slots = old_node->op_def().input_arg_size();
  // Actual number of inputs can be greater than or equal to number
  // of Input slots because inputs of type list could be unfolded.
  CHECK_GE(old_node_inputs.size(), old_node_input_slots);
  int nn_slot_idx = 0;  // slot index for inputs of new node

  // Let's copy all inputs (TF tensors) of original node to new node.
  int iidx = 0;
  for (int on_slot_idx = 0; on_slot_idx < old_node_input_slots; on_slot_idx++) {
    // An input slot could be a single tensor or a list. We need
    // to handle this case accordingly.
    CHECK_LT(iidx, old_node_inputs.size());
    const OpDef::ArgDef& arg = old_node->op_def().input_arg(on_slot_idx);
    if (ArgIsList(arg)) {
      std::vector<NodeBuilder::NodeOut> new_node_inputs;
      int N = GetTensorListLength(arg, old_node);
      GetNodesProducingTFTensorList(old_node_inputs, &iidx, N,
                                    &new_node_inputs);
      nb->Input(new_node_inputs);
      nn_slot_idx++;
    } else {
      nb->Input(old_node_inputs[iidx].first, old_node_inputs[iidx].second);
      iidx++;
      nn_slot_idx++;
    }
  }

  // If workspace tensors are available for this op and we are using
  // contiguous ordering then we need to add Tensorflow tensor for
  // workspace here because Tensorflow tensor for workspace is the
  // last tensor in the list of Tensorflow tensors.
  if (are_workspace_tensors_available) {
    CHECK_EQ(workspace_tensors->size(), 2);
    // Tensorflow tensor
    nb->Input((*workspace_tensors)[0].node, (*workspace_tensors)[0].index);
    nn_slot_idx++;
  }

  // Let's now setup all Mkl inputs to new node.
  // Number of Mkl inputs must be same as number of TF inputs.
  iidx = 0;
  for (int on_slot_idx = 0; on_slot_idx < old_node_input_slots; on_slot_idx++) {
    // An input slot could be a single tensor or a list. We need
    // to handle this case accordingly.
    CHECK_LT(iidx, old_node_inputs.size());
    const OpDef::ArgDef& arg = old_node->op_def().input_arg(on_slot_idx);
    if (ArgIsList(arg)) {
      std::vector<NodeBuilder::NodeOut> new_node_inputs;
      int N = GetTensorListLength(arg, old_node);
      GetNodesProducingMklTensorList(g, old_node, old_node_inputs, &iidx,
                                     N, &new_node_inputs);
      nb->Input(new_node_inputs);
      nn_slot_idx++;
    } else {
      Node* mkl_node = nullptr;
      int mkl_node_output_slot = 0;
      GetNodeProducingMklTensor(g, old_node, old_node_inputs[iidx].first,
                                old_node_inputs[iidx].second,
                                &mkl_node, &mkl_node_output_slot);
      nb->Input(mkl_node, mkl_node_output_slot);
      iidx++;
      nn_slot_idx++;
    }
  }

  // If workspace tensors are available for this op and we are using
  // contiguous ordering then we need to add Mkl tensor for
  // workspace here because Mkl tensor for workspace is the
  // last tensor in the list of Mkl tensors.
  if (are_workspace_tensors_available) {
    CHECK_EQ(workspace_tensors->size(), 2);
    // Mkl tensor
    nb->Input((*workspace_tensors)[1].node, (*workspace_tensors)[1].index);
    nn_slot_idx++;
  }

  return nn_slot_idx;
}

Status MklLayoutRewritePass::SetUpInputs(
    std::unique_ptr<Graph>* g,
    const gtl::InlinedVector<std::pair<Node*, int>, 4>& old_node_inputs,
    NodeBuilder* nb, Node* old_node) {
  // Let's check if we need to add workspace tensors for this node.
  // We add workspace edge only for MaxPool, LRN and BatchNorm.
  std::vector<NodeBuilder::NodeOut> workspace_tensors;
  bool are_workspace_tensors_available = false;
  AddWorkSpaceEdgeIfNeeded(g, old_node, nb, &workspace_tensors,
                           &are_workspace_tensors_available);

  int new_node_input_slots = 0;
  if (kTensorOrdering == MklTfTensorOrdering::TENSORS_INTERLEAVED) {
    // TODO(nhasabni): implement this function just for same of completion.
    // We do not use interleaved ordering right now.
    return Status(
        error::Code::UNIMPLEMENTED,
        "Interleaved ordering of tensors is currently not supported.");
  } else {
    CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
    new_node_input_slots = SetUpContiguousInputs(
        g, old_node_inputs, nb, old_node, &workspace_tensors,
        are_workspace_tensors_available);
  }

  // Sanity check
  int old_node_input_slots = old_node->op_def().input_arg_size();
  if (!are_workspace_tensors_available) {
    // If we are not adding workspace tensors for this op, then the total
    // number of input slots to the new node _must_ be 2 times the number
    // of input slots to the original node: N original Tensorflow tensors and
    // N for Mkl tensors corresponding to each Tensorflow tensors.
    CHECK_EQ(new_node_input_slots, old_node_input_slots * 2);
  } else {
    // If we are adding workspace tensors for this op, then the total
    // The total number of input slots to new node _must_ be 2 times the number
    // of input slots to the original node: N original Tensorflow tensors and
    // N for Mkl tensors corresponding to each Tensorflow tensors plus 2
    // (for workspace Tensorflow tensor and workspace Mkl tensor).
    CHECK_EQ(new_node_input_slots, old_node_input_slots * 2 + 2);
  }

  return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
//           Helper functions related to workspace pass
//////////////////////////////////////////////////////////////////////////

// TODO(nhasabni) We should move this to mkl_util.h.
void MklLayoutRewritePass::GetDummyWorkspaceTensorNode(
    std::unique_ptr<Graph>* g, Node** out, Node* orig_node) {
  // We use a tensor of shape {1} and value 0 to represent
  // dummy float tensor. We need this as a dummy workspace tensor.
  // Workspace tensor has type float.
  const DataType dt = DataTypeToEnum<float>::v();
  TensorProto proto;
  proto.set_dtype(dt);
  float zero[1] = {0};
  proto.set_tensor_content(const_cast<const void*>(static_cast<void*>(&zero)),
                           4);
  TensorShape dummy_shape({1});
  dummy_shape.AsProto(proto.mutable_tensor_shape());
  TF_CHECK_OK(NodeBuilder((*g)->NewName("DMT"), "Const")
                .Attr("value", proto)
                .Attr("dtype", dt)
                .Device(orig_node->def().device())  // We place this node on
                                                    // same the device as the
                                                    // device of the original
                                                    // node.
                .Finalize(&**g, out));

  // If number of inputs to the original node is > 0, then we add
  // control dependency between 1st input (index 0) of the original node and
  // the dummy Mkl node. This is needed because control-flow ops such as Enter,
  // Merge, etc, require frame_name of the dummy Mkl node to be same as the
  // rewritten node. Adding control edge between 1st input of the original node
  // and the dummy Mkl node ensures that the dummy node is in the same frame
  // as the original node. Choosing 1st input is not necessary - any input of
  // the original node is fine because all the inputs of a node are always in
  // the same frame.
  if (orig_node->num_inputs() > 0) {
    Node* orig_input0 = nullptr;
    TF_CHECK_OK(orig_node->input_node(0,
                                      const_cast<const Node**>(&orig_input0)));
    CHECK_NOTNULL((*g)->AddControlEdge(orig_input0, *out));
  }

  (*out)->set_assigned_device_name(orig_node->assigned_device_name());
}

void MklLayoutRewritePass::AddWorkSpaceEdgeIfNeeded(
    std::unique_ptr<Graph>* g, Node* orig_node, NodeBuilder* nb,
    std::vector<NodeBuilder::NodeOut>* ws_tensors, bool* are_ws_tensors_added) {
  bool workspace_edge_added = false;  // Default initializer
  CHECK_NOTNULL(are_ws_tensors_added);
  *are_ws_tensors_added = false;  // Default initializer

  DataType T;
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  for (auto ws : wsinfo_) {
    if (orig_node->type_string() == ws.fwd_op &&
        mkl_op_registry::IsMklOp(GetMklOpName(orig_node->type_string()), T)) {
      // If this op is a fwd op, then we need to check if there is an
      // edge from this node's fwd_slot to bwdop's bwd_slot. If there is
      // an edge, then we just add an attribute on this node for setting
      // workspace_passed to true. We don't add actual workspace edge
      // in this node. Actual workspace edge gets added in the backward
      // op for this node.
      for (const Edge* e : orig_node->out_edges()) {
        if (e->src_output() == ws.fwd_slot &&
            e->dst()->type_string() == ws.bwd_op &&
            e->dst_input() == ws.bwd_slot) {
          nb->Attr("workspace_enabled", true);
          VLOG(1) << "MklLayoutRewritePass: workspace_enabled for "
                  << orig_node->type_string();
          workspace_edge_added = true;
          // We found the edge that we were looking for, so break.
          break;
        }
      }

      if (!workspace_edge_added) {
        // If we are here, then we did not find backward operator for this
        // node.
        nb->Attr("workspace_enabled", false);
      }
    } else if (orig_node->type_string() == ws.bwd_op &&
               mkl_op_registry::IsMklOp(GetMklOpName(orig_node->type_string()),
                                        T)) {
      // If this op is a bwd op, then we need to add workspace edge and
      // it's Mkl tensor edge between its corresponding fwd op and this
      // op. Corresponding fwd op is specified in 'fwd_op' field of
      // workspace info. fwd_slot and bwd_slot in workspace info specify
      // an edge between which slots connect forward and backward op.
      // Once all these criteria match, we add a workspace edge between
      // ws_fwd_slot and ws_bwd_slot. Its corresponding Mkl tensor is
      // determined by interleaved/contiguous ordering. Function
      // DataIndexToMetaDataIndex tells us the location of Mkl tensor
      // from the location of the Tensorflow tensor.
      for (const Edge* e : orig_node->in_edges()) {
        if (e->src_output() == ws.fwd_slot &&
            // We would have rewritten the forward op, so we need to use
            // GetMklOpName call to get its Mkl name.
            e->src()->type_string() == GetMklOpName(ws.fwd_op) &&
            e->dst_input() == ws.bwd_slot) {
          nb->Attr("workspace_enabled", true);
          CHECK_NOTNULL(ws_tensors);
          // Add workspace edge between fwd op and bwd op.
          ws_tensors->push_back(NodeBuilder::NodeOut(e->src(), ws.ws_fwd_slot));
          // Add Mkl tensor edge for workspace edge between fwd op and bwd op.
          ws_tensors->push_back(NodeBuilder::NodeOut(
              e->src(), DataIndexToMetaDataIndex(ws.ws_fwd_slot,
                                                 e->src()->num_outputs())));
          *are_ws_tensors_added = true;
          // In terms of input ordering, we add these calls to add Input
          // here because workspace edge (and its Mkl tensor) is the last
          // edge in the fwdop and bwdop. So all inputs before workspace
          // tensor have been added by SetUpInputs function.
          VLOG(1) << "MklLayoutRewritePass: workspace_enabled for "
                  << orig_node->type_string();
          workspace_edge_added = true;
          // We found the edge that we were looking for, so break.
          break;
        }
      }

      // If we are here means we did not find fwd op that feeds to this
      // bwd op. So in this case, we need to generate dummy tensors for
      // workspace input and Mkl tensor for workspace, and set
      // workspace_enabled to false.
      if (!workspace_edge_added) {
        nb->Attr("workspace_enabled", false);
        Node* dmt_ws = nullptr;      // Dummy tensor for workspace
        Node* dmt_mkl_ws = nullptr;  // Dummy Mkl tensor for workspace
        GetDummyWorkspaceTensorNode(g, &dmt_ws, orig_node);
        GetDummyMklTensorNode(g, &dmt_mkl_ws, orig_node);
        CHECK_NOTNULL(dmt_ws);
        CHECK_NOTNULL(dmt_mkl_ws);
        CHECK_NOTNULL(ws_tensors);
        // We add dummy tensor as workspace tensor.
        ws_tensors->push_back(NodeBuilder::NodeOut(dmt_ws, 0));
        // We add dummy tensor as Mkl tensor for workspace tensor.
        ws_tensors->push_back(NodeBuilder::NodeOut(dmt_mkl_ws, 0));
        *are_ws_tensors_added = true;
        VLOG(1) << "MklLayoutRewritePass: dummy workspace_enabled for "
                << orig_node->type_string();
      }
    } else {
      // If this node does not match any workspace info, then we do not
      // do anything special for workspace propagation for it.
    }
  }
}

//////////////////////////////////////////////////////////////////////////
// Op-specific functions to copy attributes from old node to new node
//////////////////////////////////////////////////////////////////////////

void MklLayoutRewritePass::CopyAttrsConv2D(const Node* orig_node,
                                           NodeBuilder* nb) {
  DataType T;
  string data_format;
  string padding;
  std::vector<int32> strides;
  bool use_cudnn_on_gpu;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));
  TF_CHECK_OK(
      GetNodeAttr(orig_node->def(), "use_cudnn_on_gpu", &use_cudnn_on_gpu));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("strides", strides);
  nb->Attr("padding", padding);
  nb->Attr("data_format", data_format);
  nb->Attr("use_cudnn_on_gpu", use_cudnn_on_gpu);
}

void MklLayoutRewritePass::CopyAttrsBiasAddGrad(const Node* orig_node,
                                                NodeBuilder* nb) {
  DataType T;
  string data_format;
  std::vector<int32> strides;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("strides", strides);
  nb->Attr("data_format", data_format);
}

void MklLayoutRewritePass::CopyAttrsIdentity(const Node* orig_node,
                                             NodeBuilder* nb) {
  DataType T;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  // Add attributes to new node.
  nb->Attr("T", T);
}

void MklLayoutRewritePass::CopyAttrsLRN(const Node* orig_node,
                                        NodeBuilder* nb) {
  DataType T;
  int depth_radius;
  float bias;
  float alpha;
  float beta;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "depth_radius", &depth_radius));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "bias", &bias));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "alpha", &alpha));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "beta", &beta));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("depth_radius", depth_radius);
  nb->Attr("bias", bias);
  nb->Attr("alpha", alpha);
  nb->Attr("beta", beta);
}

void MklLayoutRewritePass::CopyAttrsPooling(const Node* orig_node,
                                            NodeBuilder* nb) {
  DataType T;
  string data_format;
  string padding;
  std::vector<int32> ksize, strides;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "ksize", &ksize));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("ksize", ksize);
  nb->Attr("strides", strides);
  nb->Attr("padding", padding);
  nb->Attr("data_format", data_format);
}

void MklLayoutRewritePass::CopyAttrsRelu(const Node* orig_node,
                                         NodeBuilder* nb) {
  DataType T;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));

  // Add attributes to new node.
  nb->Attr("T", T);
}

void MklLayoutRewritePass::CopyAttrsReshape(const Node* orig_node,
                                           NodeBuilder* nb) {
  DataType T;
  DataType Tshape;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tshape", &Tshape));
  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("Tshape", Tshape);
}

void MklLayoutRewritePass::CopyAttrsSplit(const Node* orig_node,
                                          NodeBuilder* nb) {
  DataType T;
  string data_format;
  int num_split;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "num_split", &num_split));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("num_split", num_split);
  nb->Attr("data_format", data_format);
}

void MklLayoutRewritePass::CopyAttrsConcat(const Node* orig_node,
                                           NodeBuilder* nb) {
  DataType T;
  int N;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "N", &N));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("N", N);
}

void MklLayoutRewritePass::CopyAttrsConcatV2(const Node* orig_node,
                                             NodeBuilder* nb) {
  DataType T;
  int N;
  DataType tidx;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "N", &N));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "Tidx", &tidx));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("N", N);
  nb->Attr("Tidx", tidx);
}

void MklLayoutRewritePass::CopyAttrsFusedBatchNorm(const Node* orig_node,
                                                   NodeBuilder* nb) {
  DataType T;
  float epsilon;
  string data_format;
  bool is_training;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "epsilon", &epsilon));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "data_format", &data_format));
  TF_CHECK_OK(GetNodeAttr(orig_node->def(), "is_training", &is_training));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("epsilon", epsilon);
  nb->Attr("data_format", data_format);
  nb->Attr("is_training", is_training);
}

//////////////////////////////////////////////////////////////////////////
//           Helper functions related to node merge pass
//////////////////////////////////////////////////////////////////////////

Node* MklLayoutRewritePass::CheckForNodeMerge(const Node* a) const {
  // TODO(nhasabni) Add check for type of node similar to CheckForNodeRewrite
  // once we support BiasAddGrad as Mkl layer.

  // Search for all matching mergeinfo.
  // We allow more than one match for extensibility.
  std::vector<const MergeInfo*> matching_mi;
  for (auto mi = minfo_.cbegin(); mi != minfo_.cend(); ++mi) {
    if (a->type_string() == mi->succ) {
      matching_mi.push_back(&*mi);
    }
  }

  for (const MergeInfo* mi : matching_mi) {
    const int N_in = a->num_inputs();
    if (mi->op >= N_in) {
      continue;
    }

    // Get the control edges and input of node
    gtl::InlinedVector<Node*, 4> a_control_edges;
    gtl::InlinedVector<std::pair<Node*, int>, 4> a_in(N_in);
    FillInputs(a, &a_control_edges, &a_in);

    // Get operand op of the operator
    Node* b = nullptr;
    b = a_in[mi->op].first;
    if (b == nullptr || (b->type_string() != mi->pred)) {
      // NOTE: Should the first check be assert?
      continue;
    }

    const int B_in = b->num_inputs();
    gtl::InlinedVector<Node*, 4> b_control_edges;
    gtl::InlinedVector<std::pair<Node*, int>, 4> b_in(B_in);
    FillInputs(b, &b_control_edges, &b_in);

    // Shouldn't merge if a and b have different control edges.
    if (a_control_edges != b_control_edges) {
      continue;
    } else {
      // We found a match.
      return b;
    }
  }

  return nullptr;
}

Status MklLayoutRewritePass::MergeNode(std::unique_ptr<Graph>* g, Node* succ,
                                       Node* pred) {
  CHECK_NOTNULL(succ);
  CHECK_NOTNULL(pred);

  if (succ->type_string() == csinfo_.bias_add &&
      pred->type_string() == csinfo_.mkl_conv2d) {
    // 1. Get all attributes from input nodes.
    DataType T_pred, T_succ;
    string padding;
    std::vector<int32> strides;
    string data_format_pred, data_format_succ;
    bool use_cudnn_on_gnu;
    TF_CHECK_OK(GetNodeAttr(pred->def(), "T", &T_pred));
    TF_CHECK_OK(GetNodeAttr(succ->def(), "T", &T_succ));
    TF_CHECK_OK(GetNodeAttr(pred->def(), "padding", &padding));
    TF_CHECK_OK(GetNodeAttr(pred->def(), "strides", &strides));
    TF_CHECK_OK(GetNodeAttr(pred->def(), "data_format", &data_format_pred));
    TF_CHECK_OK(GetNodeAttr(succ->def(), "data_format", &data_format_succ));
    TF_CHECK_OK(
        GetNodeAttr(pred->def(), "use_cudnn_on_gpu", &use_cudnn_on_gnu));
    // We check to ensure that data formats of both succ and pred are same.
    // We expect them to be same, so we can enforce this as assert.
    // But assert can be too strict, so we enforce this as a check.
    // If the check fails, then we do not merge two nodes.
    // We also do same check for devices.
    if (data_format_pred != data_format_succ || T_pred != T_succ ||
        pred->assigned_device_name() != succ->assigned_device_name() ||
        pred->def().device() != succ->def().device()) {
      return Status(error::Code::INVALID_ARGUMENT,
                    "data_format or T attribute or devices of Conv2D and "
                    "BiasAdd do not match. Will skip node merge optimization");
    }

    const int succ_num = succ->num_inputs();
    gtl::InlinedVector<Node*, 4> succ_control_edges;
    gtl::InlinedVector<std::pair<Node*, int>, 4> succ_in(succ_num);
    FillInputs(succ, &succ_control_edges, &succ_in);

    const int pred_num = pred->num_inputs();
    gtl::InlinedVector<Node*, 4> pred_control_edges;
    gtl::InlinedVector<std::pair<Node*, int>, 4> pred_in(pred_num);
    FillInputs(pred, &pred_control_edges, &pred_in);

    // We need to ensure that there is only 1 edge between Conv2D and AddBias.
    // Otherwise, merging is semantically incorrect.
    if (pred->out_edges().size() != 1) {
      return Status(error::Code::INVALID_ARGUMENT,
                    "Conv2D has multiple outputs."
                    "Will skip node merge optimization");
    }

    for (const Edge* e : pred->out_edges()) {
      if (e->dst() != succ) {
        return Status(error::Code::INVALID_ARGUMENT,
                      "Conv2D does not feed to BiasAdd."
                      "Will skip node merge optimization");
      }
    }

    // 2. Get inputs from both the nodes.
    // Find the 2 inputs from the conv and the bias from the add Bias.
    // Get operand 0, 1 of conv2D and their Mkl tensors.
    CHECK_EQ(pred->in_edges().size(), 4);  // _MklConv2D must have 4 inputs.
    // Get operand 1 of add_bias
    // BiasAdd must have 2 inputs: Conv, bias
    CHECK_EQ(succ->in_edges().size(), 2);
    Node* oper3_mkl = nullptr;  // Mkl tensor corresponding to oper3
    int oper3_mkl_slot = 0;     // For dummy MKL tensor node, output slot is 0.
    GetDummyMklTensorNode(g, &oper3_mkl, pred);  // Get dummy Mkl tensor node
    // as BiasAdd does not have Mkl tensor as input.
    CHECK_NOTNULL(oper3_mkl);

    // We will use the node name of BiasAdd as the name of new node
    // Build new node. We use same name as original node, but change the op
    // name.
    NodeBuilder nb(succ->name(), csinfo_.mkl_conv2d_with_bias);
    if (kTensorOrdering == MklTfTensorOrdering::TENSORS_INTERLEAVED) {
      nb.Input(pred_in[0].first, pred_in[0].second);  // In1 of Conv2D
      // pred_in[1] will be Mkl tensor for In1 if we follow interleaved
      // ordering, and it will be 2nd Tensorflow tensor for Conv2D if
      // we follow contiguous ordering.
      nb.Input(pred_in[1].first, pred_in[1].second);  // Mkl for In1
      nb.Input(pred_in[2].first, pred_in[2].second);  // In2 of Conv2D
      nb.Input(pred_in[3].first, pred_in[3].second);  // Mkl for In2
      nb.Input(succ_in[1].first, succ_in[1].second);  // In2 of BiasAdd
      nb.Input(oper3_mkl, oper3_mkl_slot);            // Mkl for In2 of BiasAdd
    } else {
      CHECK_EQ(kTensorOrdering, MklTfTensorOrdering::TENSORS_CONTIGUOUS);
      nb.Input(pred_in[0].first, pred_in[0].second);  // In1 of Conv2D
      // pred_in[1] will be Mkl tensor for In1 if we follow interleaved
      // ordering, and it will be 2nd Tensorflow tensor for Conv2D if
      // we follow contiguous ordering.
      nb.Input(pred_in[1].first, pred_in[1].second);  // In2 of Conv2D
      nb.Input(succ_in[1].first, succ_in[1].second);  // In2 of BiasAdd
      nb.Input(pred_in[2].first, pred_in[2].second);  // Mkl for In1 of Conv2D
      nb.Input(pred_in[3].first, pred_in[3].second);  // Mkl for In2 of Conv2D
      nb.Input(oper3_mkl, oper3_mkl_slot);            // Mkl for In2 of BiasAdd
    }

    // Copy attributes from Conv2D to Conv2DWithBias.
    CopyAttrsConv2D(const_cast<const Node*>(pred), &nb);

    // Copy the device assigned to old node to new node.
    nb.Device(succ->def().device());

    // Create node.
    Node* new_node;
    nb.Finalize(&**g, &new_node);
    CHECK_NOTNULL(new_node);

    // Set the Mkl layer label for this op.
    new_node->AddAttr("_kernel", mkl_op_registry::kMklOpLabel);

    // Incoming data edges from 'pred' node and 'succ' node to new 'new_node'
    // node are already copied in BuildNode. We handle control edges now.
    for (const Edge* e : pred->in_edges()) {
      if (e->IsControlEdge()) {
        CHECK_NOTNULL((*g)->AddControlEdge(e->src(), new_node));
      }
    }
    for (const Edge* e : succ->in_edges()) {
      if (e->IsControlEdge()) {
        CHECK_NOTNULL((*g)->AddControlEdge(e->src(), new_node));
      }
    }

    // Incoming edges are fixed, we will fix the outgoing edges now.
    // First, we will fix outgoing control edges from 'pred' node.
    // We don't need to handle outgoing data edges from 'pred' node
    // because pred has only 1 output going to succ node (we enforced
    // this check for merge already).
    for (const Edge* e : pred->out_edges()) {
      if (e->IsControlEdge()) {
        CHECK_NOTNULL((*g)->AddControlEdge(new_node, e->dst()));
      }
    }

    // Second, we will fix outgoing control and data edges from 'succ' node.
    for (const Edge* e : succ->out_edges()) {
      if (e->IsControlEdge()) {
        CHECK_NOTNULL((*g)->AddControlEdge(new_node, e->dst()));
      } else {
        CHECK_NOTNULL((*g)->AddEdge(new_node, e->src_output(), e->dst(),
                                  e->dst_input()));
      }
    }

    // Copy device assigned to old node to new node.
    // It's ok to use pred or succ as we have enforced a check that
    // both have same device assigned.
    new_node->set_assigned_device_name(pred->assigned_device_name());

    VLOG(1) << "MklLayoutRewritePass: Merged old node:" << pred->DebugString()
            << ", and node: " << succ->DebugString()
            << ", into node:" << new_node->DebugString();

    (*g)->RemoveNode(succ);
    (*g)->RemoveNode(pred);
    MarkRewrittenNode(new_node);

    return Status::OK();
  }

  return Status(error::Code::UNIMPLEMENTED,
                "Unimplemented case for node merge optimization.");
}

//////////////////////////////////////////////////////////////////////////
//           Helper functions for node rewrite
//////////////////////////////////////////////////////////////////////////

Status MklLayoutRewritePass::RewriteNode(std::unique_ptr<Graph>* g,
                                         Node* orig_node,
                                         const RewriteInfo* ri) {
  CHECK_NOTNULL(ri);
  CHECK_NOTNULL(orig_node);

  VLOG(1) << "MklLayoutRewritePass: Original node:" << orig_node->DebugString();

  // Check if this is scenario 2 (context-based rewrite).
  // Get the matching ContextInfo if it is.
  const Node* fwd_node = nullptr;
  const ContextInfo* ci = nullptr;
  bool is_context_based_rewrite = false;
  if ((ci = SearchMatchingContext(orig_node, &fwd_node)) != nullptr) {
    CHECK_NOTNULL(fwd_node);
    is_context_based_rewrite = true;

    // Sanity checks for context-based rewrite (if any)
    if (orig_node->type_string() == csinfo_.bias_add_grad &&
        ri->new_name == csinfo_.mkl_conv2d_with_bias_backprop_bias) {
      DataType orig_T, ctx_T;
      string orig_data_format, ctx_data_format;
      TF_CHECK_OK(GetNodeAttr(orig_node->def(), "T", &orig_T));
      TF_CHECK_OK(
          GetNodeAttr(orig_node->def(), "data_format", &orig_data_format));
      TF_CHECK_OK(GetNodeAttr(fwd_node->def(), "T", &ctx_T));
      TF_CHECK_OK(
          GetNodeAttr(fwd_node->def(), "data_format", &ctx_data_format));

      if (orig_data_format != ctx_data_format || orig_T != ctx_T ||
          orig_node->assigned_device_name() !=
              fwd_node->assigned_device_name() ||
          orig_node->def().device() != fwd_node->def().device()) {
        return Status(
            error::Code::INVALID_ARGUMENT,
            "data_format or T attribute or devices of BiasAddGrad and "
            "Conv2D do not match. Will skip node rewrite optimization");
      }
    } else if (orig_node->type_string() == csinfo_.bias_add_grad &&
               ri->new_name == csinfo_.matmul) {
      // When BiasAddGrad has MatMul in context, we do not do any rewrite
      // and leave BiasAddGrad as it is. But we check for this condition
      // when we check for node rewrite rule. So we should not even come
      // here for MatMul. So we will fail now.
        return Status(
            error::Code::INVALID_ARGUMENT,
            "No rewrite is required for BiasAddGrad for MatMul context.");
    }
  }

  // Get all inputs.
  const int num_inputs = orig_node->in_edges().size();
  gtl::InlinedVector<Node*, 4> control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> inputs(num_inputs);
  FillInputs(orig_node, &control_edges, &inputs);

  // Build new node. We use same name as original node, but change the op name.
  NodeBuilder nb(orig_node->name().c_str(), ri->new_name.c_str());
  // Copy user-specified device assigned to original node to new node.
  nb.Device(orig_node->def().device());
  // Set up new inputs to the rewritten node.
  Status s = SetUpInputs(g, inputs, &nb, orig_node);
  if (s != Status::OK()) {
    return s;
  }

  // Copy attributes from original node to new node (for scenario 1).
  // For context-based rewrite, we use context to copy the attributes.
  if (is_context_based_rewrite) {
    if (orig_node->type_string() == csinfo_.bias_add_grad &&
        ri->new_name == csinfo_.mkl_conv2d_with_bias_backprop_bias) {
      CHECK_NOTNULL(fwd_node);
      ri->copy_attrs(fwd_node, &nb);
    } else {
      return Status(error::Code::UNIMPLEMENTED,
                    "Unimplemented case for node rewrite optimization.");
    }
  } else {
    ri->copy_attrs(const_cast<const Node*>(orig_node), &nb);
  }
  // Set the Mkl layer label for this op.
  nb.Attr("_kernel", mkl_op_registry::kMklOpLabel);

  // Finalize graph and get new node.
  Node* new_node = nullptr;
  TF_CHECK_OK(nb.Finalize(&**g, &new_node));
  CHECK_NOTNULL(new_node);

  // Incoming data edges from 'orig_node' node to new 'new_node' node are
  // already copied in BuildNode. We need to handle control edges now.
  for (const Edge* e : orig_node->in_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL((*g)->AddControlEdge(e->src(), new_node));
    }
  }

  // Copy outgoing edges from 'orig_node' node to new
  // 'new_node' node, since the output also follows same ordering among
  // Tensorflow tensors and Mkl tensors. We need to connect Tensorflow
  // tensors appropriately. Specifically, nth output of the original node
  // will become 2*nth output of the Mkl node for the interleaved ordering
  // of the tensors. For the contiguous ordering of the tensors, it will be n.
  // GetTensorDataIndex provides this mapping function.
  for (const Edge* e : orig_node->out_edges()) {
    if (e->IsControlEdge()) {
      CHECK_NOTNULL((*g)->AddControlEdge(new_node, e->dst()));
    } else {
      CHECK_NOTNULL((*g)->AddEdge(new_node, GetTensorDataIndex(e->src_output(),
                            e->src()->num_outputs()),
                    e->dst(), e->dst_input()));
    }
  }

  // Copy the runtime device assigned from original code to new node.
  new_node->set_assigned_device_name(orig_node->assigned_device_name());

  // Delete original node and mark new node as rewritten.
  (*g)->RemoveNode(orig_node);
  MarkRewrittenNode(new_node);

  VLOG(1) << "MklLayoutRewritePass: New node:" << new_node->DebugString();
  return Status::OK();
}

const MklLayoutRewritePass::ContextInfo*
MklLayoutRewritePass::SearchMatchingContext(const Node* n,
                                            const Node** fwd_node) {
  CHECK_NOTNULL(n);
  CHECK_NOTNULL(fwd_node);
  *fwd_node = nullptr;

  // Search for matching contextinfo based on node name.
  // There could be more than one matching contextinfos.
  bool is_matching_cinfo_found = false;
  std::vector<const ContextInfo*> mci;
  for (auto ci = cinfo_.cbegin(); ci != cinfo_.cend(); ++ci) {
    if (n->type_string() == (*ci)->node) {
      mci.push_back(*ci);
      is_matching_cinfo_found = true;
    }
  }
  // If no matching contextinfo is found, return immediately.
  if (!is_matching_cinfo_found) {
    return nullptr;
  }

  VLOG(1) << "MklLayoutRewritePass: Searching graph for: " << n->type_string()
          << " in backwards.";

  // Now we will check for forward op name for context info in data
  // flow graph. Get the max hops we should search for the fwd node.
  // We are now going to search (breadth-first) backwards in data
  // dependence graph (for up to max hops) from n for the node
  // specified in fwd.
  // queue to maintain nodes to be visited and depth info for
  // breadth-first search
  std::queue<std::pair<const Node*, int>> nqueue;
  const Node* curr_node = n;
  size_t curr_depth = 0;
  nqueue.push(std::make_pair(curr_node, curr_depth));

  while (curr_depth < kNodeMergeContextMaxDepth && !nqueue.empty()) {
    std::pair<const Node*, int> curr_pair = nqueue.front();
    nqueue.pop();

    std::set<const Node*> visited_nodes;
    curr_node = curr_pair.first;
    curr_depth = curr_pair.second;
    CHECK_NOTNULL(curr_node);

    VLOG(1) << "MklLayoutRewritePass: Visiting node: "
            << curr_node->type_string() << " at depth: " << curr_depth
            << " for node: " << n->type_string();

    // If we find a match, we return immediately.
    for (const ContextInfo* ci : mci) {
      if (curr_node->type_string() == ci->fwd) {
        *fwd_node = curr_node;
        return ci;
      }
    }

    // Else we explore backward edges from current node.
    // Add the source nodes of all incoming edges of the node to the queue.
    for (const Edge* e : curr_node->in_edges()) {
      // We do not visit already visited node.
      if (visited_nodes.find(e->src()) == visited_nodes.end()) {
        // Depth of these nodes is 1 more than the depth of current node.
        nqueue.push(std::make_pair(e->src(), curr_depth + 1));
        visited_nodes.insert(e->src());
      }
    }
  } /* while */

  return nullptr;
}

bool MklLayoutRewritePass::ContextMatchRewrite(const Node* n,
                                               const ContextInfo* c) {
  const Node* fwd_node = nullptr;
  return SearchMatchingContext(n, &fwd_node) == c;
}

const MklLayoutRewritePass::RewriteInfo*
MklLayoutRewritePass::CheckForNodeRewrite(const Node* n) const {
  CHECK_NOTNULL(n);

  // First check if node along with its type is supported by MKL layer.
  // We do not want to rewrite an op into Mkl op if types are not supported.
  // E.g., MklRelu does not support INT32. So we cannot rewrite Relu to
  // MklRelu if type is INT32.
  DataType T;
  if (!GetNodeAttr(n->def(), "T", &T).ok()) {
    return nullptr;
  }

  // BiasAddGrad is not an Mkl layer, so we make an exception for it.
  if (n->type_string() != csinfo_.bias_add_grad) {
    if (!mkl_op_registry::IsMklOp(GetMklOpName(n->type_string()), T)) {
      return nullptr;
    }
  }

  // We support 2 types of node rewrites:
  // 1. Rewriting BiasAddGrad depending on its MklConv2DWithBias context.
  // 2. Rewriting an op to Mkl op always
  // We return true if any of these 2 conditions is met.

  // Find matching RewriteInfo and then check that rewrite rule applies.
  for (auto ri = rinfo_.cbegin(); ri != rinfo_.cend(); ++ri) {
    if (n->type_string().compare(ri->name) == 0 &&
        ri->rewrite_rule(n, ri->context)) {
      // If we are rewriting BiasAddGrad into BiasAddGrad for MatMul context,
      // then we just return directly.
      if (n->type_string() == csinfo_.bias_add_grad &&
          ri->context->fwd == csinfo_.matmul &&
          ri->new_name == csinfo_.bias_add_grad) {
        return nullptr;
      }
      return &*ri;
    }
  }

  // Else return not found.
  return nullptr;
}

///////////////////////////////////////////////////////////////////////////////
//              Run function for the pass
///////////////////////////////////////////////////////////////////////////////

bool MklLayoutRewritePass::RunPass(std::unique_ptr<Graph>* g) {
  bool result = false;
  CHECK_NOTNULL(g);

  DumpGraph("Before running MklLayoutRewritePass", &**g);

  std::vector<Node*> order;
  GetReversePostOrder(**g, &order);  // This will give us topological sort.

  for (Node* n : order) {
    // If node is not an op or it cannot run on CPU device, then skip.
    if (!n->IsOp() || !CanOpRunOnCPUDevice(n)) {
      continue;
    }

    const RewriteInfo* ri = nullptr;
    Node* predn = nullptr;
    // We will first search if node is to be rewritten
    if ((ri = CheckForNodeRewrite(n)) != nullptr) {
      string node_name = n->name();
      string op_name = n->type_string();

      VLOG(1) << "MklLayoutRewritePass: Scheduled node " << node_name
              << " with op " << op_name << " for rewrite using"
              << " layout optimization.";

      if (RewriteNode(g, n, ri) == Status::OK()) {
        VLOG(1) << "MklLayoutRewritePass: rewrote node " << node_name
                << " with op " << op_name << " for Mkl layout optimization.";
        result = true;
      }
    } else if ((predn = CheckForNodeMerge(n)) != nullptr) {
      // Otherwise, we will check if the node is to be merged.
      string n1_name = n->name();
      string n2_name = predn->name();

      VLOG(1) << "MklLayoutRewritePass: Scheduled nodes " << n1_name << " and "
              << n2_name << " for merging";

      if (MergeNode(g, n, predn) == Status::OK()) {
        VLOG(1) << "MklLayoutRewritePass: Merged nodes " << n1_name << " and "
                << n2_name;
        result = true;
      }
    }
  }

  DumpGraph("After running MklLayoutRewritePass", &**g);

  // Clear marked nodes as the same graph pass may be used multiple times.
  UnMarkRewrittenNodes();

  return result;
}

bool RunMklLayoutRewritePass(std::unique_ptr<Graph>* g) {
  return MklLayoutRewritePass().RunPass(g);
}

Status MklLayoutRewritePass::Run(
  const GraphOptimizationPassOptions& options) {
  if (options.graph == nullptr && options.partition_graphs == nullptr) {
    return Status::OK();
  }

  auto process_graph = [&](std::unique_ptr<Graph>* g) {
    // Get the ownership of a graph
    std::unique_ptr<Graph>* ng = std::move(g);
    RunPass(ng);
    // Return the ownership of a graph back
    g->reset(ng->release());
  };

  if (kMklLayoutRewritePassGroup !=
      OptimizationPassRegistry::POST_PARTITIONING) {
    // For any pre-partitioning phase, a graph is stored in options.graph.
    process_graph(options.graph);
  } else {
    // For post partitioning phase, graphs are stored in
    // options.partition_graphs.
    for (auto& pg : *options.partition_graphs) {
      process_graph(&pg.second);
    }
  }

  return Status::OK();
}

}  // namespace tensorflow

#endif
