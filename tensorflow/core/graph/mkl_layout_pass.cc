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

#include <functional>
#include <memory>
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

// This pass implements rewriting of graph for propagating Mkl
// layout as an additional output tensor (we will loosely call a
// tensor that carries Mkl layout as Mkl tensor henceforth.)
// from every Mkl supported NN layer.
//
// As a example, consider Relu layer. Current definition of Relu
// layer looks like:
//
//           O = Relu(A)
//
// Relu has 1 input (A), and 1 output (O).
//
// This rewrite pass will generate a new graph node for Relu
// (new node is called MklRelu) as:
//
//          O, O_m = MklRelu(A, A_m)
//
// MklRelu has 2 inputs (A and A_m) and 2 outputs (O and O_m).
// Here A input is same as A input of Relu; O output is same
// as O output of Relu. O_m is the additional output tensor
// that will be set by MklRelu, and it represents Mkl tensor
// corresponding to O -- in other words, O_m is some kind of
// metadata for O. A_m is additional input of Relu, and it
// represents metadata for A - as O_m is metadata for O, A_m
// is metadata for A. MklRelu receives this metadata from
// previous layer (in the graph).
//
// When previous layer in the graph is Mkl layer, A_m will
// represent a valid Mkl tensor. But when previous Mkl layer
// is not an Mkl layer, then A_m represents a dummy Mkl tensor.
//
// Rewriting rules:
//   - Selection of an op for rewriting happens by registering
//     an op with this pass. If an op is not registered, then
//     it is not rewritten.
//  - Number of inputs after rewriting:
//      Since for every input Tensorflow tensor, the rewritten
//      layer gets Mkl tensor, rewritten op gets 2*N inputs,
//      where N is the number of inputs for original op.
//  - Number of outputs after rewriting:
//      Since for every output Tensorflow tensor, the rewritten
//      layer generates Mkl tensor, rewritten op generates 2*N
//      outputs, where N is the number of outputs of original op.
//  - Ordering of Tensorflow tensors and Mkl tensors:
//      Since every op generates twice the number of inputs and
//      outputs, one could imagine different ordering among
//      Tensorflow tensors and Mkl tensors. E.g., let's assume
//      an op 'Conv2D' takes (A, B) as input, then new op
//      'MklConv2D' can take (A, A_m, B, B_m) as input or it
//      can also take (A, B, A_m, B_m) as input. Among N inputs
//      one can get N! permutations.
//
//      So the question is: which one do we follow? Currently,
//      we follow an intuitive order where Mkl tensor follows a
//      corresponding Tensorflow tensor immediately. In the
//      context of above example, it will be: (A, A_m, B, B_m).
//      We follow same ordering rule for output tensors.
//
// NOTE: Current rewriting approach rewrites an op to Mkl op without
//      any conditions. But in the future, it may be possible to
//      consider conditions such as input shapes and sizes to rewrite
//      an op.
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
class MklLayoutRewritePass : public GraphOptimizationPass {
 public:
  MklLayoutRewritePass() {
    csinfo_.conv2d = "Conv2D";

    ninfo_.push_back(
        {csinfo_.conv2d, GetMklOpName(csinfo_.conv2d), 2, CopyAttrsConv2D});
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
    std::function<void(Node*, NodeBuilder*)>
        copyattrs;  // Function handler
                    // to copy attributes from old node to new node.
  } NodesInfo;

  /// Structure to store all constant strings
  struct {
    string relu;
    string relugrad;
    string conv2d;
  } csinfo_;

  /// Maintain info about nodes to rewrite
  std::vector<NodesInfo> ninfo_;

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

  // Get the name of Mkl op from original TensorFlow op
  // We prefix 'Mkl' to the original op to get Mkl op.
  // TODO(nhasabni) We should move this to mkl_util.h.
  inline string GetMklOpName(const string& name) const {
    // Prefix that we add to Tensorflow op name to construct Mkl op name.
    const char* const kMklOpPrefix = "Mkl";
    return string(kMklOpPrefix) + name;
  }

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

  // Rewrite Node 'n' in graph 'g' with rewrite information specified in 'ni'
  // Returns Status::OK() if node rewrite is successful, otherwise returns
  // appropriate error status
  Status RewriteNode(std::unique_ptr<Graph>* g, Node* n, const NodesInfo& ni);

  // Functions specific to operators to copy attributes
  // We need operator-specific function to copy attributes because the framework
  // does not provide any generic function for it.
  static void CopyAttrsConv2D(Node* orign, NodeBuilder* nb);

  // Generate a graph node in graph 'g' representing a dummy Mkl tensor node,
  // using node for original node 'orign' and return it in '*out'.
  // TODO(nhasabni) We should move this to mkl_util.h
  void GetDummyMklTensorNode(std::unique_ptr<Graph>* g, Node** out,
                             Node* orign);
};

// We register Mkl rewrite pass for phase 1 in pre-placement group.
// Do not change the ordering of the Mkl passes.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1,
                      MklLayoutRewritePass);

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

//////////////////////////////////////////////////////////////////////////

// Macros to build new node with different number of inputs.
// We need this way because we need to specify all the inputs when
// building a node. Comment at core/graph/node_builder.h, line 85-86.

#define SETUP_INPUTS1(nb, op1)      \
  do {                              \
    nb->Input(op1.node, op1.index); \
  } while (0)

#define SETUP_INPUTS2(nb, op1, op2) \
  do {                              \
    nb->Input(op1.node, op1.index); \
    nb->Input(op2.node, op2.index); \
  } while (0)

#define SETUP_INPUTS3(nb, op1, op2, op3) \
  do {                                   \
    nb->Input(op1.node, op1.index);      \
    nb->Input(op2.node, op2.index);      \
    nb->Input(op3.node, op3.index);      \
  } while (0)

#define SETUP_INPUTS4(nb, op1, op2, op3, op4) \
  do {                                        \
    nb->Input(op1.node, op1.index);           \
    nb->Input(op2.node, op2.index);           \
    nb->Input(op3.node, op3.index);           \
    nb->Input(op4.node, op4.index);           \
  } while (0)

#define SETUP_INPUTS5(nb, op1, op2, op3, op4, op5) \
  do {                                             \
    nb->Input(op1.node, op1.index);                \
    nb->Input(op2.node, op2.index);                \
    nb->Input(op3.node, op3.index);                \
    nb->Input(op4.node, op4.index);                \
    nb->Input(op5.node, op5.index);                \
  } while (0)

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
      CHECK_EQ(mkl_layer_registry::IsMklLayer(n->type_string()), true);
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

  // 2. Let's build the node with new inputs.
  switch (new_inputs.size()) {
    case 0:  // We don't need to do anything for no input as we have
             // already built node.
      break;
    case 1:
      SETUP_INPUTS1(nb, new_inputs[0]);
      break;
    case 2:
      SETUP_INPUTS2(nb, new_inputs[0], new_inputs[1]);
      break;
    case 3:
      SETUP_INPUTS3(nb, new_inputs[0], new_inputs[1], new_inputs[2]);
      break;
    case 4:
      SETUP_INPUTS4(nb, new_inputs[0], new_inputs[1], new_inputs[2],
                    new_inputs[3]);
      break;
    case 5:
      SETUP_INPUTS5(nb, new_inputs[0], new_inputs[1], new_inputs[2],
                    new_inputs[3], new_inputs[4]);
      break;
    default: {
      return Status(error::Code::UNIMPLEMENTED,
                    "Could not create node with given number of inputs");
    }
  }

  return Status::OK();
}

void MklLayoutRewritePass::CopyAttrsConv2D(Node* orign, NodeBuilder* nb) {
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

Status MklLayoutRewritePass::RewriteNode(std::unique_ptr<Graph>* g, Node* orign,
                                         const NodesInfo& ni) {
  VLOG(1) << "MKLLayoutRewritePass: Original node:" << orign->DebugString();

  // Get all inputs.
  const int num = orign->num_inputs();
  CHECK_EQ(num, ni.numins);
  gtl::InlinedVector<Node*, 4> control_edges;
  gtl::InlinedVector<std::pair<Node*, int>, 4> inputs(num);
  FillInputs(orign, &control_edges, &inputs);

  // Build new node. We use same name as original node, but change the op name.
  NodeBuilder nb(orign->name().c_str(), ni.newname.c_str());
  // Copy user-specified device assigned to original node to new node.
  nb.Device(orign->def().device());
  // Set up new inputs to the rewritten node.
  Status s = SetUpInputs(g, inputs, &nb, orign);
  if (s != Status::OK()) {
    return s;
  }
  // Copy attributes from original node to new node.
  ni.copyattrs(orign, &nb);
  // Set the Mkl layer label for this op.
  nb.Attr("_kernel", mkl_layer_registry::kMklLayerLabel);
  Node* newn = nullptr;

  // Finalize graph and get new node.
  TF_CHECK_OK(nb.Finalize(&**g, &newn));
  CHECK_NOTNULL(newn);

  // Incoming edges from 'orign' node to new 'newn' node are already copied
  // in BuildNode. Copy outgoing edges from 'orign' node to new 'newn' node.
  for (const Edge* e : orign->out_edges()) {
    (*g)->AddEdge(newn, e->src_output(), e->dst(), e->dst_input());
  }

  // Copy the runtime device assigned from original code to new node.
  newn->set_assigned_device_name(orign->assigned_device_name());

  // Delete original node and mark new node as rewritten.
  (*g)->RemoveNode(orign);
  MarkRewrittenNode(newn);

  VLOG(1) << "MKLLayoutRewritePass: New node:" << newn->DebugString();
  return Status::OK();
}

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

    for (const NodesInfo& ni : ninfo_) {
      DataType dtype = DT_INVALID;
      // An op needs to have data type (T) attribute and its corresponding
      // Mkl op name must be supported.
      if (GetNodeAttr(n->def(), "T", &dtype) == Status::OK() &&
          mkl_layer_registry::IsMklLayer(GetMklOpName(n->type_string())) &&
          n->type_string().compare(ni.name) == 0) {
        string node_name = n->name();
        string op_name = n->type_string();

        VLOG(1) << "MKLLayoutRewritePass: Scheduled node " << node_name
                << " with op " << op_name << " for rewrite using"
                << " layout optimization.";

        if (RewriteNode(g, n, ni) == Status::OK()) {
          VLOG(1) << "MKLLayoutRewritePass: Successfully rewrote node "
                  << node_name << " with op " << op_name
                  << " for Mkl layout optimization.";
          result = true;
          break;  // We found matching nodesinfo so no need to search next.
        }
      }
    }
  }

  DumpGraph("After running MklLayoutRewritePass", &**g);

  return result;
}

///////////////////////////////////////////////////////////////////////////////
//              Run function for the pass
///////////////////////////////////////////////////////////////////////////////

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
