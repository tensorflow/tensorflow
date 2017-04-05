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
//     2) Context-based rewrite: This is neded in order to optimize
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
//           P = MklConv2DWithBias(A, A_m, B, B_m, C, C_m)
//
// Meaning of A_m, B_m and C_m is explained in B.1.
//
// Merge rules:
//  - Merge for Conv2D and BiasAdd happens only when output of Conv2D _only_
//    goes to BiasAdd.
//  - Also, the intersection of attributes of both the nodes must have same
//    values.
//  - Both the nodes must have been assigned to same device (if any).
//
// Example of B.1 : Rewriting nodes to Mkl nodes
// ---------------------------------------------
// Consider Relu layer. Current definition of Relu layer looks like:
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
// MklRelu has 2 inputs (A and A_m) and 2 outputs (O and O_m). Here A input is
// same as A input of Relu; O output is same as O output of Relu. O_m is the
// additional output tensor that will be set by MklRelu, and it represents
// Mkl tensor corresponding to O -- in other words, O_m is some kind of
// metadata for O. A_m is additional input of Relu, and it represents metadata
// for A - as O_m is metadata for O, A_m is metadata for A. MklRelu receives
// this metadata from previous layer (in the graph).
//
// When previous layer in the graph is Mkl layer, A_m will represent a valid
// Mkl tensor. But when previous Mkl layer is not an Mkl layer, then A_m
// represents a dummy Mkl tensor.
//
// Rewriting rules:
//  - Selection of an op for rewriting happens by registering an op with this
//     pass. If an op is not registered, then it is not rewritten.
//  - Number of inputs after rewriting:
//      Since for every input Tensorflow tensor, the rewritten layer gets Mkl
//      tensor, rewritten op gets 2*N inputs, where N is the number of inputs
//      for original op.
//  - Number of outputs after rewriting:
//      Since for every output Tensorflow tensor, the rewritten layer generates
//      Mkl tensor, rewritten op generates 2*N outputs, where N is the number
//      of outputs of original op.
//  - Ordering of Tensorflow tensors and Mkl tensors:
//      Since every op generates twice the number of inputs and outputs, one
//      could imagine different ordering among Tensorflow tensors and Mkl
//      tensors. E.g., let's assume an op 'Conv2D' takes (A, B) as input, then
//      new op 'MklConv2D' can take (A, A_m, B, B_m) as input or it can also
//      take (A, B, A_m, B_m) as input. Among N inputs one can get N!
//      permutations.
//
//      So the question is: which one do we follow? Currently, we follow an
//      intuitive order where Mkl tensor follows a corresponding Tensorflow
//      tensor immediately. In the context of above example, it will be: (A,
//      A_m, B, B_m). We follow same ordering rule for output tensors.
//
// NOTE: Current rewriting approach rewrites an op to Mkl op without any
//      conditions. But in the future, it may be possible to consider
//      conditions such as input shapes and sizes to rewrite an op.
//
// Graph rewrite algorithm:
//      Algorithm: Graph Rewrite
//      Input: Graph G, Names of nodes to rewrite and their new nodes
//      Output: Modified Graph G' if nodes are modified, G otherwise.
//      Start:
//        N = Topological_Sort(G) // N is set of nodes in toposort order.
//        foreach node n in N
//        do
//          if (Is_MKL_Layer(n))  // Can this layer accept Mkl layout as input.
//          then
//            E = set of <incoming edge and its src_output slot> of n
//            E' = {}   // new set of edges for rewritten node
//            foreach <e,s> in E
//            do
//              E' U {<e,s>}  // First copy edge which generates Tensorflow
//                            // tensor as it is
//              m = Source node of edge e
//              if Is_Rewritten(m)  // Did we rewrite this node in this pass?
//              then
//                E' U {<m,s+1>}    // If yes, then m will generate Mkl tensor
//                                  // as output.
//              else
//                d = Generate_Dummy_Mkl_Tensor()  // If not, generate dummy
//                                                 // Mkl tensor.
//                E' U {<d,0>}   // Dummy Mkl tensor has only 1 output slot.
//              fi
//            done
//            n' = Build_New_Node(G,new_name,E')
//            Mark_Rewritten(n')  // Mark new node as being rewritten.
//          fi
//        done
//
//      Explanation:
//        For graph rewrite, we visit nodes of the graph in the topological
//        sort order. With this ordering, we visit nodes in top-to-bottom
//        fashion. We need this order because while visiting a node we want
//        all of its input nodes (parents) visited (and rewritten if
//        applicable). This is because if we need to rewrite a current node
//        then all of its input nodes need to be fixed (in other words they
//        cannot be removed later.)
//
//        While visiting each node, we first check if it is Mkl layer. If
//        it is, then we rewrite that node after constructing new inputs to
//        the node. If it is not Mkl layer, then we do not rewrite the node.
//
// Handling workspace propagation for certain ops:
//
//        Certain backward ops in MKL (MaxPool, LRN and BatchNorm) require
//        passing of workspace from their corresponding forward ops. But
//        TensorFlow does not have a notion of workspace and as a result
//        does not allow producing additional outputs from these forward ops.
//        For these ops, we need to add an additional edge between forward
//        ops and their corresponding backward ops, and this edge carries
//        workspace tensor value and another edge carries Mkl tensor for
//        workspace tensor.
//
//        Example:
//
//        Typical graph for MaxPool and its gradient looks like:
//
//        A = MaxPool(T)
//        B = MaxPoolGrad(X, A, Y)
//
//        We will transform this graph to propagate workspace as:
//
//        A, A_m, W, W_m = MklMaxPool(T, T_m)
//        B, B_m = MklMaxPoolGrad(X, X_m, A, A_m, Y, Y_m, W, W_m)
//
//        Here W is the workspace tensor. Transformed tensors with name
//        suffix _m are Mkl tensors and this transformation has been done
//        using the algorithm discussed earlier. The transformation for
//        workspace only adds extra outputs (W, W_m) for forward op and
//        connects them to corresponding backward ops.
//
//        Terms:
//
//        Forward op name = name of the op in the forward pass
//          where workspace originates (MaxPool in this example)
//        Backward op name = name of the op in the backward pass that receives
//          workspace from forward op (MaxPoolGrad in the example)
//        Slot = Number of the output or input slot that will be
//               used by the workspace (2 for MklMaxPool as W is 3rd
//               output of MaxPool (0 is 1st); 6 for MklMaxPoolGrad)
//
//        Question:
//
//        How do we associate backward op to forward op? There can be more
//        than one op with exact same name.
//
//        In this example we associate MaxPoolGrad with MaxPool. But there
//        could be more than one MaxPool ops. To solve this problem, we look
//        for _direct_ edge between forward op and backward op (tensor A is
//        flowing along this edge in the example.)
//
//        How do we transform forward and backward op when there is no direct
//        edge between them? In such case, we generate dummy tensors as
//        workspace tensors. For the example, transformation of MaxPool will
//        be exactly same --- it is just that MaxPool won't generate any
//        workspace tensor. For MaxPoolGrad, transformation will also be same,
//        but instead of connecting W and W_m with outputs of MaxPool, we will
//        produce dummy tensors for them, and we will set workspace_enabled
//        attribute to false.
//
// Example of B.2 : Context-based node rewrite
// -------------------------------------------
// Consider BiasAddGrad op as:
//
//           O = MklConv2D(A, A_m, B, B_m, C, C_m)
//           P = BiasAddGrad(O)
//
// Then we rewrite is as:
//
//           P = Conv2DWithBiasBackpropBias(O, O_m)
//
// 'Distance' between input of BiasAddGrad and MklConv2D in terms of hops is
// the context matching depth. If MklConv2DWithBias is not within the context
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
    csinfo_.conv2d = "Conv2D";
    csinfo_.mklconv2d = "MklConv2D";
    csinfo_.mklconv2dwithbias = "MklConv2DWithBias";
    csinfo_.mklconv2dwithbiasbackpropbias = "MklConv2DWithBiasBackpropBias";
    csinfo_.biasadd = "BiasAdd";
    csinfo_.matmul = "MatMul";
    csinfo_.biasaddgrad = "BiasAddGrad";
    csinfo_.relu = "Relu";
    csinfo_.relugrad = "ReluGrad";
    csinfo_.maxpool = "MaxPool";
    csinfo_.maxpoolgrad = "MaxPoolGrad";
    csinfo_.avgpool = "AvgPool";
    csinfo_.avgpoolgrad = "AvgPoolGrad";
    csinfo_.conv2dgradinput = "Conv2DBackpropInput";
    csinfo_.conv2dgradfilter = "Conv2DBackpropFilter";

    rinfo_.push_back(
        {csinfo_.conv2d, csinfo_.mklconv2d, 2, CopyAttrsConv2D, AlwaysRewrite});
    rinfo_.push_back({csinfo_.conv2dgradfilter,
                      GetMklOpName(csinfo_.conv2dgradfilter), 3,
                      CopyAttrsConv2D, AlwaysRewrite});
    rinfo_.push_back({csinfo_.conv2dgradinput,
                      GetMklOpName(csinfo_.conv2dgradinput), 3, CopyAttrsConv2D,
                      AlwaysRewrite});
    rinfo_.push_back({csinfo_.relu, GetMklOpName(csinfo_.relu), 1,
                      CopyAttrsRelu, AlwaysRewrite});
    rinfo_.push_back({csinfo_.maxpool, GetMklOpName(csinfo_.maxpool), 1,
                      CopyAttrsPooling, AlwaysRewrite});
    rinfo_.push_back({csinfo_.maxpoolgrad, GetMklOpName(csinfo_.maxpoolgrad), 3,
                      CopyAttrsPooling, AlwaysRewrite});
    rinfo_.push_back({csinfo_.avgpool, GetMklOpName(csinfo_.avgpool), 1,
                      CopyAttrsPooling, AlwaysRewrite});
    rinfo_.push_back({csinfo_.avgpoolgrad, GetMklOpName(csinfo_.avgpoolgrad), 2,
                      CopyAttrsPooling, AlwaysRewrite});

    // Add info about which ops to add workspace edge to and the slots.
    wsinfo_.push_back({csinfo_.maxpool, csinfo_.maxpoolgrad, 0, 1, 2, 6});

    // Add a rule for merging nodes
    minfo_.push_back(
        {csinfo_.mklconv2d, csinfo_.biasadd, 0, csinfo_.mklconv2dwithbias});

    // We use maxhop of 10 based on empirical observations. Also, these are
    // maxhops in backward data-flow graph. Since input of forward nodes
    // (Conv2D) directly goes to backward nodes, we do not expect the
    // hop-distance would be more than few nodes.
    cinfo_.push_back({csinfo_.biasaddgrad, csinfo_.mklconv2dwithbias,
                      kNodeMergeContextMaxDepth});
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

 private:
  /// Structure to specify name of original op, its new name after rewrite,
  /// the number of inputs to the original op, and the function to be used
  /// to copy attributes for the op
  typedef struct {
    string name;     // Original name of the op in the graph
    string newname;  // New name of op in the graph
    int numins;      // Number of inputs to the original op
    // Function handler to copy attributes from old node to new node.
    std::function<void(const Node*, NodeBuilder*)> copyattrs;
    std::function<bool(const Node*)> rewriterule;  // Rule under which to
                                                   // rewrite this node.
  } RewriteInfo;

  /// Structure to specify forward op, backward op, and the slot numbers
  /// in forward and backward op where we will add workspace edge.
  typedef struct {
    string fwdop;   // Name of the forward op in the graph
    string bwdop;   // Name of the backward op in the graph
    int fwdslot;    // Output slot in the forward op node where actual
                    // output tensor resides
    int bwdslot;    // Input slot in the backward op node where actual
                    // input tensor resides
    int wsfwdslot;  // Output slot in the forward op node where workspace
                    // edge is added
    int wsbwdslot;  // Input slot in the backward op node where workspace
                    // edge is added
  } WorkSpaceInfo;

  /// Structure to specify information used in node merge
  typedef struct {
    string pred;     // Predecessor node string
    string succ;     // Successor node string
    int op;          // What operand no the predecessor node corresponds
                     // to successor node?
    string newnode;  // Name of the node after merge
  } MergeInfo;

  /// Structure to specify the context information used in node rewrite rule
  typedef struct {
    string node;    // Name of the node to be rewritten
    string fwd;     // Node name in forward pass that this node
                    // corresponds to
    size_t maxhop;  // Maximum number of hops the fwd is located
                    // from this node. If fwd is farther than maxhop
                    // then we do not rewrite the node.
  } ContextInfo;

  /// Structure to store all constant strings
  struct {
    string relu;
    string relugrad;
    // Conv ops
    string conv2d;
    string mklconv2d;
    string conv2dgradinput;
    string conv2dgradfilter;
    string mklconv2dwithbias;
    string mklconv2dwithbiasbackpropbias;
    // Pooling ops
    string maxpool;
    string maxpoolgrad;
    string avgpool;
    string avgpoolgrad;
    // Others
    string biasadd;
    string matmul;
    string biasaddgrad;
  } csinfo_;

  /// Maintain info about nodes to rewrite
  std::vector<RewriteInfo> rinfo_;

  /// Maintain info about nodes to add workspace edge
  std::vector<WorkSpaceInfo> wsinfo_;

  /// Maintain info  to be merged
  std::vector<MergeInfo> minfo_;

  /// Maintain info about nodes to rewrite
  static std::vector<ContextInfo> cinfo_;

  /// Hash table to maintain nodes visited in the graph.
  std::unordered_set<const Node*> visited_nodes_;

 private:
  // Predicate to check if we rewrote node 'n'
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

  // Get the name of Mkl op from original TensorFlow op
  // We prefix 'Mkl' to the original op to get Mkl op.
  // TODO(nhasabni) We should move this to mkl_util.h.
  inline string GetMklOpName(const string& name) const {
    // Prefix that we add to Tensorflow op name to construct Mkl op name.
    const char* const kMklOpPrefix = "Mkl";
    return string(kMklOpPrefix) + name;
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
  // after the call to function may result is undefined behaviors.
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
  static bool AlwaysRewrite(const Node* n) { return true; }
  // Rewrite rule that uses context-information for matching
  // used in scenario 2.
  //
  // @input - Node 'n' for which to search for matching context
  // @return - true if matching context is found; false otherwise.
  static bool ContextMatchRewrite(const Node* n);

  // Helper function that searches the matching contextinfo for the node.
  // Implements depth-first search in the data dependence graph for the
  // gradient op in the backward direction.
  //
  // @input n - Node (gradient op) whose contextinfo is to be searched,
  //        fwdn - pointer to node from the forward pass that this node
  //        belongs to. fwdn cannot be NULL.
  // @return Matching contextinfo in case a match is found; null otherwise.
  //         Also updates *fwdn with pointer to forward node that this context
  //         matches.
  static const ContextInfo* SearchMatchingContext(const Node* n,
                                                  const Node** fwdn);

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

  // Setup new inputs using old inputs 'inputs' for the rewritten node in 'nb'
  // in graph 'g'. Original node is input in 'orign'.
  //
  // For details, refer to 'Number of inputs after rewriting' section in the
  // documentation above.
  //
  // Returns Status::OK() if setting up inputs is successful, otherwise
  // returns appropriate status code.
  Status SetUpInputs(std::unique_ptr<Graph>* g,
                     const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs,
                     NodeBuilder* nb, Node* orign);

  // Add workspace edge on the input or output side of Node 'orign' by using
  // NodeBuilder 'nb' for the new node provided. If 'orign' does not dictate
  // adding workspace edge then do not add it.
  void AddWorkSpaceEdgeIfNeeded(std::unique_ptr<Graph>* g, Node* orign,
                                NodeBuilder* nb);

  // Functions specific to operators to copy attributes
  // We need operator-specific function to copy attributes because the framework
  // does not provide any generic function for it.
  static void CopyAttrsConv2D(const Node* orign, NodeBuilder* nb);
  static void CopyAttrsBiasAddGrad(const Node* orign, NodeBuilder* nb);
  static void CopyAttrsPooling(const Node* orign, NodeBuilder* nb);
  static void CopyAttrsRelu(const Node* orign, NodeBuilder* nb);

  // Generate a graph node in graph 'g' representing a dummy Mkl tensor node,
  // using node for original node 'orign' and return it in '*out'.
  // TODO(nhasabni) We should move this to mkl_util.h
  void GetDummyMklTensorNode(std::unique_ptr<Graph>* g, Node** out,
                             Node* orign);
  void GetDummyWorkspaceTensorNode(std::unique_ptr<Graph>* g, Node** out,
                                   Node* orign);
};

std::vector<MklLayoutRewritePass::ContextInfo> MklLayoutRewritePass::cinfo_;

// We register Mkl rewrite pass for phase 1 in pre-placement group.
// Do not change the ordering of the Mkl passes.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1,
                      MklLayoutRewritePass);

//////////////////////////////////////////////////////////////////////////
//           Helper functions for creating new node
//////////////////////////////////////////////////////////////////////////

static void FillInputs(const Node* n,
                       gtl::InlinedVector<Node*, 4>* control_edges,
                       gtl::InlinedVector<std::pair<Node*, int>, 4>* in) {
  DCHECK_EQ(in->size(), n->num_inputs());
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

// TODO(nhasabni) We should move this to mkl_util.h.
void MklLayoutRewritePass::GetDummyMklTensorNode(std::unique_ptr<Graph>* g,
                                                 Node** out, Node* orign) {
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
  TF_CHECK_OK(
      NodeBuilder((*g)->NewName("DMT"), "Const")
          .Attr("value", proto)
          .Attr("dtype", dt)
          .Device(orign->def().device())  // We place this node on same
                                          // device as device of original
                                          // node.
          .Finalize(&**g, out));
  (*out)->set_assigned_device_name(orign->assigned_device_name());
}

Status MklLayoutRewritePass::SetUpInputs(
    std::unique_ptr<Graph>* g,
    const gtl::InlinedVector<std::pair<Node*, int>, 4>& inputs, NodeBuilder* nb,
    Node* orign) {
  std::vector<NodeBuilder::NodeOut> new_inputs;

  // 1. Let's setup inputs for the new node.
  for (int i = 0; i < inputs.size(); i++) {
    Node* n = inputs[i].first;
    // First let's copy original TF tensor input as it is.
    new_inputs.push_back(NodeBuilder::NodeOut(n, inputs[i].second));

    // Second, let's add edge to propagate Mkl tensors from input Mkl layers,
    // or generate a dummy Mkl tensor representing not-mkl-tensor case.
    if (IsRewrittenNode(n)) {
      // If we have visited this node and rewritten it, then it will generate
      // an edge that will receive Mkl tensor from a node.
      // First, let's assert that this op is Mkl layer.
      DataType T;
      TF_CHECK_OK(GetNodeAttr(n->def(), "T", &T));
      // If this op has been rewritten, then its name must have been same as
      // Mkl op.
      CHECK_EQ(mkl_layer_registry::IsMklLayer(n->type_string(), T), true);
      // src slot number for Mkl tensor would be the one next to TF tensor
      // slot number.
      new_inputs.push_back(NodeBuilder::NodeOut(n, inputs[i].second + 1));
    } else {
      // If we have not visited the node and rewritten it, then we need
      // to create a dummy node that will feed a non-Mkl tensor to this node.
      // DummyMklTensor node has no input and generates only 1 output
      // (dummy Mkl tensor) as output slot number 0.
      Node* dmt = nullptr;
      GetDummyMklTensorNode(g, &dmt, orign);
      CHECK_NOTNULL(dmt);
      new_inputs.push_back(NodeBuilder::NodeOut(dmt, 0));
    }
  }

  // The total number of inputs to new node _must_ be 2 times the number
  // of inputs to the original node: N original Tensorflow tensors and
  // N for Mkl tensors corresponding to each Tensorflow tensors.
  CHECK_EQ(new_inputs.size(), inputs.size() * 2);

  // 2. Let's add the new inputs.
  for (auto ni : new_inputs) {
    nb->Input(ni.node, ni.index);
  }

  return Status::OK();
}

//////////////////////////////////////////////////////////////////////////
//           Helper functions related to workspace pass
//////////////////////////////////////////////////////////////////////////

// TODO(nhasabni) We should move this to mkl_util.h.
void MklLayoutRewritePass::GetDummyWorkspaceTensorNode(
    std::unique_ptr<Graph>* g, Node** out, Node* orign) {
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
  TF_CHECK_OK(
      NodeBuilder((*g)->NewName("DMT"), "Const")
          .Attr("value", proto)
          .Attr("dtype", dt)
          .Device(orign->def().device())  // We place this node on same
                                          // device as device of original
                                          // node.
          .Finalize(&**g, out));
  (*out)->set_assigned_device_name(orign->assigned_device_name());
}

void MklLayoutRewritePass::AddWorkSpaceEdgeIfNeeded(std::unique_ptr<Graph>* g,
                                                    Node* orign,
                                                    NodeBuilder* nb) {
  bool workspace_edge_added = false;
  DataType T;
  TF_CHECK_OK(GetNodeAttr(orign->def(), "T", &T));
  for (auto ws : wsinfo_) {
    if (orign->type_string() == ws.fwdop &&
        mkl_layer_registry::IsMklLayer(GetMklOpName(orign->type_string()), T)) {
      // If this op is a fwd op, then we need to check if there is an
      // edge from this node's fwdslot to bwdop's bwdslot. If there is
      // an edge, then we just add an attribute on this node for setting
      // workspace_passed to true. We don't add actual workspace edge
      // in this node. Actual workspace edge gets added in the backward
      // op for this node.
      for (const Edge* e : orign->out_edges()) {
        if (e->src_output() == ws.fwdslot &&
            e->dst()->type_string() == ws.bwdop &&
            e->dst_input() == ws.bwdslot) {
          nb->Attr("workspace_enabled", true);
          VLOG(1) << "MklLayoutRewritePass: workspace_enabled for "
                  << orign->type_string();
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
    } else if (orign->type_string() == ws.bwdop &&
               mkl_layer_registry::IsMklLayer(
                   GetMklOpName(orign->type_string()), T)) {
      // If this op is a bwd op, then we need to add workspace edge and
      // it's Mkl tensor edge between its corresponding fwd op and this
      // op. Corresponding fwd op is specified in 'fwdop' field of
      // workspace info. fwdslot and bwdslot in workspace info specify
      // an edge between which slots connect forward and backward op.
      // Once all these criteria match, we add a workspace edge between
      // wsfwdslot and wsbwdslot. It's corresponding Mkl tensor is added
      // in wsfwdslot+1 and wsbwdslot+1.
      for (const Edge* e : orign->in_edges()) {
        if (e->src_output() == ws.fwdslot &&
            // We would have rewritten the forward op, so we need to use
            // GetMklOpName call to get its Mkl name.
            e->src()->type_string() == GetMklOpName(ws.fwdop) &&
            e->dst_input() == ws.bwdslot) {
          nb->Attr("workspace_enabled", true);
          // Add workspace edge between fwd op and bwd op.
          nb->Input(e->src(), ws.wsfwdslot);
          // Add Mkl tensor edge for workspace edge between fwd op and bwd op.
          nb->Input(e->src(), ws.wsfwdslot + 1);
          // In terms of input ordering, we add these calls to add Input
          // here because workspace edge (and its Mkl tensor) is the last
          // edge in the fwdop and bwdop. So all inputs before workspace
          // tensor have been added by SetUpInputs function.
          VLOG(1) << "MklLayoutRewritePass: workspace_enabled for "
                  << orign->type_string();
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
        GetDummyWorkspaceTensorNode(g, &dmt_ws, orign);
        GetDummyMklTensorNode(g, &dmt_mkl_ws, orign);
        CHECK_NOTNULL(dmt_ws);
        CHECK_NOTNULL(dmt_mkl_ws);
        nb->Input(dmt_ws, 0);      // We add dummy tensor as workspace tensor.
        nb->Input(dmt_mkl_ws, 0);  // We add dummy tensor as Mkl
                                   // tensor for workspace tensor.
        VLOG(1) << "MklLayoutRewritePass: dummy workspace_enabled for "
                << orign->type_string();
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

void MklLayoutRewritePass::CopyAttrsConv2D(const Node* orign, NodeBuilder* nb) {
  DataType T;
  string data_format;
  string padding;
  std::vector<int32> strides;
  bool use_cudnn_on_gpu;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orign->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orign->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orign->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orign->def(), "data_format", &data_format));
  TF_CHECK_OK(GetNodeAttr(orign->def(), "use_cudnn_on_gpu", &use_cudnn_on_gpu));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("strides", strides);
  nb->Attr("padding", padding);
  nb->Attr("data_format", data_format);
  nb->Attr("use_cudnn_on_gpu", use_cudnn_on_gpu);
}

void MklLayoutRewritePass::CopyAttrsBiasAddGrad(const Node* orign,
                                                NodeBuilder* nb) {
  DataType T;
  string data_format;
  std::vector<int32> strides;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orign->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orign->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orign->def(), "data_format", &data_format));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("strides", strides);
  nb->Attr("data_format", data_format);
}

void MklLayoutRewritePass::CopyAttrsPooling(const Node* orign,
                                            NodeBuilder* nb) {
  DataType T;
  string data_format;
  string padding;
  std::vector<int32> ksize, strides;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orign->def(), "T", &T));
  TF_CHECK_OK(GetNodeAttr(orign->def(), "ksize", &ksize));
  TF_CHECK_OK(GetNodeAttr(orign->def(), "strides", &strides));
  TF_CHECK_OK(GetNodeAttr(orign->def(), "padding", &padding));
  TF_CHECK_OK(GetNodeAttr(orign->def(), "data_format", &data_format));

  // Add attributes to new node.
  nb->Attr("T", T);
  nb->Attr("ksize", ksize);
  nb->Attr("strides", strides);
  nb->Attr("padding", padding);
  nb->Attr("data_format", data_format);
}

void MklLayoutRewritePass::CopyAttrsRelu(const Node* orign, NodeBuilder* nb) {
  DataType T;

  // Get all attributes from old node.
  TF_CHECK_OK(GetNodeAttr(orign->def(), "T", &T));

  // Add attributes to new node.
  nb->Attr("T", T);
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

    gtl::InlinedVector<Node*, 4> b_control_edges;
    gtl::InlinedVector<std::pair<Node*, int>, 4> b_in(N_in);
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

  if (succ->type_string() == csinfo_.biasadd &&
      pred->type_string() == csinfo_.mklconv2d) {
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
    CHECK_EQ(pred->in_edges().size(), 4);  // MklConv2D must have 4 inputs.
    // Get operand 1 of add_bias
    // BiasAdd must have 2 inputs: Conv, bias
    CHECK_EQ(succ->in_edges().size(), 2);
    Node* oper3_mkl = nullptr;  // Mkl tensor corresponding to oper3
    int oper3_mkl_slot = 0;     // For dummy MKL tensor node, output slot is 0.
    GetDummyMklTensorNode(g, &oper3_mkl, succ);  // Get dummy Mkl tensor node
    // as BiasAdd does not have Mkl tensor as input.
    CHECK_NOTNULL(oper3_mkl);

    // We will use the node name of BiasAdd as the name of new node
    // Build new node. We use same name as original node, but change the op
    // name.
    NodeBuilder nb(succ->name(), csinfo_.mklconv2dwithbias);
    nb.Input(pred_in[0].first, pred_in[0].second);  // In1 of Conv2D
    nb.Input(pred_in[1].first, pred_in[1].second);  // Mkl for In1
    nb.Input(pred_in[2].first, pred_in[2].second);  // In2 of Conv2D
    nb.Input(pred_in[3].first, pred_in[3].second);  // Mkl for In2
    nb.Input(succ_in[1].first, succ_in[1].second);  // In2 of BiasAdd
    nb.Input(oper3_mkl, oper3_mkl_slot);            // Mkl for In2 of BiasAdd

    // Copy attributes from Conv2D to Conv2DWithBias.
    CopyAttrsConv2D(const_cast<const Node*>(pred), &nb);

    // Copy the device assigned to old node to new node.
    nb.Device(succ->def().device());

    // Create node.
    Node* newn;
    nb.Finalize(&**g, &newn);
    CHECK_NOTNULL(newn);

    // Set the Mkl layer label for this op.
    newn->AddAttr("_kernel", mkl_layer_registry::kMklLayerLabel);

    // Incoming edges are fixed, we will fix the outgoing edges now.
    for (const Edge* e : succ->out_edges()) {
      (*g)->AddEdge(newn, e->src_output(), e->dst(), e->dst_input());
    }

    // Copy device assigned to old node to new node.
    // It's ok to use pred or succ as we have enforced a check that
    // both have same device assigned.
    newn->set_assigned_device_name(pred->assigned_device_name());

    VLOG(1) << "MklLayoutRewritePass: Merged old node:" << pred->DebugString()
            << ", and node: " << succ->DebugString()
            << ", into node:" << newn->DebugString();

    (*g)->RemoveNode(succ);
    (*g)->RemoveNode(pred);
    MarkRewrittenNode(newn);

    return Status::OK();
  }

  return Status(error::Code::UNIMPLEMENTED,
                "Unimplemented case for node merge optimization.");
}

//////////////////////////////////////////////////////////////////////////
//           Helper functions for node rewrite
//////////////////////////////////////////////////////////////////////////

Status MklLayoutRewritePass::RewriteNode(std::unique_ptr<Graph>* g, Node* orign,
                                         const RewriteInfo* ri) {
  CHECK_NOTNULL(ri);
  CHECK_NOTNULL(orign);

  VLOG(1) << "MklLayoutRewritePass: Original node:" << orign->DebugString();

  // Check if this is scenario 2 (context-based rewrite).
  // Get the matching ContextInfo if it is.
  const Node* fwdn = nullptr;
  const ContextInfo* ci = nullptr;
  bool is_context_based_rewrite = false;
  if ((ci = SearchMatchingContext(orign, &fwdn)) != nullptr) {
    CHECK_NOTNULL(fwdn);
    is_context_based_rewrite = true;

    // Sanity checks for context-based rewrite (if any)
    if (orign->type_string() == csinfo_.biasaddgrad &&
        ri->newname == csinfo_.mklconv2dwithbiasbackpropbias) {
      DataType orig_T, ctx_T;
      string orig_data_format, ctx_data_format;
      TF_CHECK_OK(GetNodeAttr(orign->def(), "T", &orig_T));
      TF_CHECK_OK(GetNodeAttr(orign->def(), "data_format", &orig_data_format));
      TF_CHECK_OK(GetNodeAttr(fwdn->def(), "T", &ctx_T));
      TF_CHECK_OK(GetNodeAttr(fwdn->def(), "data_format", &ctx_data_format));

      if (orig_data_format != ctx_data_format || orig_T != ctx_T ||
          orign->assigned_device_name() != fwdn->assigned_device_name() ||
          orign->def().device() != fwdn->def().device()) {
        return Status(
            error::Code::INVALID_ARGUMENT,
            "data_format or T attribute or devices of BiasAddGrad and "
            "Conv2D do not match. Will skip node rewrite optimization");
      }
    }
  }

  // Get all inputs.
  const int num = orign->num_inputs();
  CHECK_EQ(num, ri->numins);
  gtl::InlinedVector<Node*, 4> control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> inputs(num);
  FillInputs(orign, &control_edges, &inputs);

  // Build new node. We use same name as original node, but change the op name.
  NodeBuilder nb(orign->name().c_str(), ri->newname.c_str());
  // Copy user-specified device assigned to original node to new node.
  nb.Device(orign->def().device());
  // Set up new inputs to the rewritten node.
  Status s = SetUpInputs(g, inputs, &nb, orign);
  if (s != Status::OK()) {
    return s;
  }

  // Copy attributes from original node to new node (for scenario 1).
  // For context-based rewrite, we use context to copy the attributes.
  if (is_context_based_rewrite) {
    if (orign->type_string() == csinfo_.biasaddgrad &&
        ri->newname == csinfo_.mklconv2dwithbiasbackpropbias) {
      CHECK_NOTNULL(fwdn);
      ri->copyattrs(fwdn, &nb);
    } else {
      return Status(error::Code::UNIMPLEMENTED,
                    "Unimplemented case for node rewrite optimization.");
    }
  } else {
    ri->copyattrs(const_cast<const Node*>(orign), &nb);
  }
  // Set the Mkl layer label for this op.
  nb.Attr("_kernel", mkl_layer_registry::kMklLayerLabel);

  // Add workspace edge to this node if needed.
  // We add workspace edge only for MaxPool, LRN and BatchNorm.
  AddWorkSpaceEdgeIfNeeded(g, orign, &nb);

  // Finalize graph and get new node.
  Node* newn = nullptr;
  TF_CHECK_OK(nb.Finalize(&**g, &newn));
  CHECK_NOTNULL(newn);

  // Incoming edges from 'orign' node to new 'newn' node are already copied
  // in BuildNode. Copy outgoing edges from 'orign' node to new 'newn' node.
  // Since the output also follows same ordering among Tensorflow tensors and
  // Mkl tensors. We need to connect Tensorflow tensors appropriately.
  // Specifically, nth output of original node will become 2*nth output of
  // Mkl node. GetTensorDataIndex provides this mapping function.
  for (const Edge* e : orign->out_edges()) {
    // We need to handle control-edges by using their original slot number.
    // Generally, -1 is reserved for control slot.
    if (e->src_output() < 0) {
      (*g)->AddEdge(newn, e->src_output(), e->dst(), e->dst_input());
    } else {
      (*g)->AddEdge(newn, GetTensorDataIndex(e->src_output()), e->dst(),
                    e->dst_input());
    }
  }

  // Copy the runtime device assigned from original code to new node.
  newn->set_assigned_device_name(orign->assigned_device_name());

  // Delete original node and mark new node as rewritten.
  (*g)->RemoveNode(orign);
  MarkRewrittenNode(newn);

  VLOG(1) << "MklLayoutRewritePass: New node:" << newn->DebugString();
  return Status::OK();
}

const MklLayoutRewritePass::ContextInfo*
MklLayoutRewritePass::SearchMatchingContext(const Node* n, const Node** fwdn) {
  CHECK_NOTNULL(n);
  CHECK_NOTNULL(fwdn);
  *fwdn = nullptr;

  // Search for matching contextinfo based on node name.
  // There could be more than one matching contextinfos.
  bool is_matching_cinfo_found = false;
  std::vector<const ContextInfo*> mci;
  for (auto ci = cinfo_.cbegin(); ci != cinfo_.cend(); ++ci) {
    if (n->type_string() == ci->node) {
      mci.push_back(&*ci);
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
        *fwdn = curr_node;
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

bool MklLayoutRewritePass::ContextMatchRewrite(const Node* n) {
  const Node* fwdn = nullptr;
  return SearchMatchingContext(n, &fwdn) != nullptr;
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
  if (!mkl_layer_registry::IsMklLayer(GetMklOpName(n->type_string()), T)) {
    return nullptr;
  }

  // We support 2 types of node rewrites:
  // 1. Rewriting BiasAddGrad depending on its context.
  // 2. Rewriting an op to Mkl op always
  // We return true if any of these 2 conditions is met.

  // Find matching RewriteInfo and then check that rewrite rule applies.
  for (auto ri = rinfo_.cbegin(); ri != rinfo_.cend(); ++ri) {
    if (n->type_string().compare(ri->name) == 0 && ri->rewriterule(n)) {
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
    if (!n->IsOp()) {
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

Status MklLayoutRewritePass::Run(const GraphOptimizationPassOptions& options) {
  if (options.graph == nullptr) {
    return Status::OK();
  }

  // Get the ownership of graph
  std::unique_ptr<Graph>* g = std::move(options.graph);

  RunPass(g);

  // Return the ownership of graph back
  options.graph->reset(g->release());

  return Status::OK();
}

}  // namespace tensorflow

#endif
