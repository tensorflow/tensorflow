/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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
// This module implements node merging optimization on the graph.
// We process the nodes in the graph in reverse postorder
// (i.e. inputs before their downstream dependencies).
//
#include <set>
#include <vector>
#include <queue>
#include <utility>

#include "tensorflow/core/graph/mkl_optimizer_merge.h"

#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/common_runtime/function.h"

namespace tensorflow {

// How many hops do we search for matching node in the backward dataflow graph?
// We use maxhop of 10 based on empirical observations. Also, these are
// maxhops in backward data-flow graph. Since input of forward nodes (Conv2D)
// directly goes to backward nodes, we do not expect the hop-distance
// would be more than few nodes.
static size_t kNodeMergeContextMaxDepth = 10;

// This optimization pass performs two tasks: merge
// nodes in the forward pass, and rewrite the gradient ops
// corresponding to merged forward ops.
//
// Merging nodes in the graph: Currently, it merges Conv2D+AddBias together.
//
// Rewriting nodes in the graph: This is neded in order to optimize
// gradient ops of Conv2D+AddBias. Gradient op of both the Conv2D and
// MatMul is BiasAddGrad, and we need to rewrite BiasAddGrad into
// Conv2D-specific BiasAddGrad, and MatMul-specific BiasAddGrad.
// This is context-specific optimization, where the context is the
// forward operator that the BiasAddGrad corresponds to.
class NodeMergeRewritePass : public GraphOptimizationPass {
 public:
  NodeMergeRewritePass() {
    csinfo_.conv2d                     = "Conv2D";
    csinfo_.conv2dwithbias             = "Conv2DWithBias";
    csinfo_.conv2dwithbiasbackpropbias = "Conv2DWithBiasBackpropBias";
    csinfo_.biasadd                    = "BiasAdd";
    csinfo_.matmul                     = "MatMul";
    csinfo_.biasaddgrad                = "BiasAddGrad";

    minfo_.push_back({csinfo_.conv2d, csinfo_.biasadd, 0,
                      csinfo_.conv2dwithbias});

    // We use maxhop of 10 based on emperical observations. Also, these are
    // maxhops in backward data-flow graph. Since input of forward nodes
    // (Conv2D) directly goes to backward nodes, we do not expect the
    // hop-distance would be more than few nodes.
    rinfo_.push_back({csinfo_.biasaddgrad, csinfo_.conv2dwithbiasbackpropbias,
                  {csinfo_.conv2dwithbias, kNodeMergeContextMaxDepth}});
    rinfo_.push_back({csinfo_.biasaddgrad, csinfo_.conv2dwithbiasbackpropbias,
                  {csinfo_.conv2d, kNodeMergeContextMaxDepth}});
    // For now, we are rewriting BiasAddGrad to BiasAddGrad for MatMul. This is
    // because we do not have a separate Op for MatMulwithBias.
    rinfo_.push_back({csinfo_.biasaddgrad, csinfo_.biasaddgrad,
                      {csinfo_.matmul, kNodeMergeContextMaxDepth}});
  }

  // Standard interface to run optimization pass
  Status Run(const GraphOptimizationPassOptions& options);

  // Helper function which does most of heavy lifting for node merge
  //
  // Extracts common functionality between Run public interface and
  // test interface.
  //
  // @return true, if and only if graph is mutated; false otherwise.
  bool RunPass(std::unique_ptr<Graph>* g);

 private:
  /// Structure to specify information used in node merge
  typedef struct {
    string pred;  // Predecessor node string
    string succ;  // Successor node string
    int    op;    // What operand no the predecessor node corresponds
                  // to successor node?
    string newnode;  // Name of the node after merge
  } MergeInfo;

  /// Structure to specify information used in node rewrite
  typedef struct {
    string node;  // Name of the node to be rewritten
    string rewrite;  // New name of the node after rewrite
    typedef struct {
        string fwd;  // Node name in forward pass that this node
                       // corresponds to
        size_t maxhop;  // Maximum number of hops the mfwd_ is located
                         // from this node. If mfwd_ is farther than mmaxhop_
                         // then we do not rewrite the node.
    } ContextInfo;
    ContextInfo cinfo;  // Context for rewrite
  } RewriteInfo;

  /// Structure to store all constant strings
  typedef struct {
    string conv2d;
    string conv2dwithbias;
    string conv2dwithbiasbackpropbias;
    string biasadd;
    string matmul;
    string biasaddgrad;
  } ConstStringInfo;

  ConstStringInfo csinfo_;
  std::vector<MergeInfo> minfo_;
  std::vector<RewriteInfo> rinfo_;

 private:
  // Return a node that can be merged with input node
  //
  // @return pointer to the node if we can find such a
  // node. Otherwise, it returns nullptr.
  Node* FindNodeForMerge(const Node* a) const;

  // Merge predecessor node with its successor.
  // Currently, we merge Conv2D with AddBias only.
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

  // Is input node (n) a candidate for rewrite?
  //
  // @return true, if it can be rewritten; false, otherwise.
  bool IsApplicableRewriteNode(const Node* n) const;

  // Rewrites input node to a new node specified by its matching rewrite info.
  //
  // Method first searches matching rewrite info for input node and then
  // uses that info to rewrite.
  //
  // Input node may be deleted in case of rewrite. Attempt to use the node
  // after the call can result in undefined behaviors.
  //
  // @input  g - input graph, n - Node to be rewritten
  // @return Status::OK(), if the input node is rewritten;
  //         Returns appropriate Status error code otherwise.
  //         Graph is updated in case the input node is rewritten.
  //         Otherwise, it is not updated.
  Status RewriteNode(std::unique_ptr<Graph>* g, Node* n);

  // Helper function that searches the matching rewriteinfo for the node.
  // Implements depth-first search in the data dependence graph for the
  // gradient op in backward direction.
  //
  // @input n - Node (gradient op) whose rewriteinfo is to be searched,
  //        fwdn - pointer to node from the forward pass that this node
  //        belongs to
  // @return Matching rewriteinfo in case a match is found; null otherwise.
  const RewriteInfo* FindMatchingRewriteInfo(const Node* n,
                                             const Node** fwdn) const;
};

/// We register merge optimizer for phase 1 and MKLToTF insertion for phase 2.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 1,
                      NodeMergeRewritePass);

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

Node* NodeMergeRewritePass::FindNodeForMerge(const Node* a) const {
  // Search for all matching mergeinfo.
  // We allow more than one match for extensibility.
  std::vector<const MergeInfo*> matching_mi;
  for (auto mi = minfo_.cbegin(); mi != minfo_.cend(); ++mi) {
    if (a->type_string() == mi->succ) {
      matching_mi.push_back(&*mi);
    }
  }

  VLOG(1) << "FindNodeForMerge: " << a->type_string();

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
    Node *b = nullptr;
    b = a_in[mi->op].first;
    if (b == nullptr || (b->type_string() != mi->pred)) {
      // NOTE: Should the first check be assert?
      continue;
    }

    VLOG(1) << "     FindNode: " << b->type_string();

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

Status NodeMergeRewritePass::MergeNode(std::unique_ptr<Graph>* g,
                                     Node* succ, Node* pred) {
  CHECK_NOTNULL(succ);
  CHECK_NOTNULL(pred);

  if (succ->type_string() == csinfo_.biasadd &&
      pred->type_string() == csinfo_.conv2d) {
    // 1. Get all attributes from input nodes.
    DataType T_pred, T_succ;
    string padding;
    std::vector<int32> strides;
    string data_format_pred, data_format_succ;
    bool use_cudnn_on_gnu;
    int groups = 1;
    TF_CHECK_OK(GetNodeAttr(pred->def(), "T", &T_pred));
    TF_CHECK_OK(GetNodeAttr(succ->def(), "T", &T_succ));
    TF_CHECK_OK(GetNodeAttr(pred->def(), "padding", &padding));
    TF_CHECK_OK(GetNodeAttr(pred->def(), "strides", &strides));
    TF_CHECK_OK(GetNodeAttr(pred->def(), "data_format", &data_format_pred));
    TF_CHECK_OK(GetNodeAttr(succ->def(), "data_format", &data_format_succ));
    TF_CHECK_OK(GetNodeAttr(pred->def(), "use_cudnn_on_gpu",
                            &use_cudnn_on_gnu));
    // Groups attribute may not be there on the input node. So we do not
    // check for error in GetNodeAttr call.
    GetNodeAttr(pred->def(), "groups", &groups);
    // We check to ensure that data formats of both succ and pred are same.
    // We expect them to be same, so we can enforce this as assert.
    // But assert can be too strict, so we enforce this as a check.
    // If the check fails, then we do not merge two nodes.
    if (data_format_pred != data_format_succ ||
        T_pred != T_succ) {
      return Status(error::Code::INVALID_ARGUMENT,
                    "data_format or T attribute of Conv2D and BiasAdd"
                    "do not match. Will skip node merge optimization");
    }

    // 2. Get inputs from both the nodes.
    // Find the 2 inputs from the conv and the bias from the add Bias.
    Node* oper1 = nullptr;
    Node* oper2 = nullptr;
    Node* oper3 = nullptr;

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

    for (const Edge *e : pred->out_edges()) {
      if (e->dst() != succ) {
        return Status(error::Code::INVALID_ARGUMENT,
                    "Conv2D does not feed to BiasAdd."
                    "Will skip node merge optimization");
      }
    }

    // Get operand 0, 1 of conv2D
    oper1 = pred_in[0].first;
    oper2 = pred_in[1].first;
    // Get operand 1 of add_bias
    oper3 = succ_in[1].first;

    Node* ret;
    // We will use the node name of BiasAdd as the name of new node
    TF_CHECK_OK(NodeBuilder(succ->name(), csinfo_.conv2dwithbias)
                  .Input(oper1)
                  .Input(oper2)
                  .Input(oper3)
                  .Attr("T", T_pred)
                  .Attr("strides", strides)
                  .Attr("padding", padding)
                  .Attr("data_format", data_format_pred)
                  .Attr("use_cudnn_on_gpu", use_cudnn_on_gnu)
                  .Attr("groups", groups)
                  .Finalize(&**g, &ret));
    CHECK_NOTNULL(ret);

    // Incoming edges are fixed, we will fix the outgoing edges now.
    for (const Edge* e : succ->out_edges()) {
      (*g)->AddEdge(ret, e->src_output(), e->dst(), e->dst_input());
    }

    (*g)->RemoveNode(succ);
    (*g)->RemoveNode(pred);

    return Status::OK();
  }

  return Status(error::Code::UNIMPLEMENTED,
                "Unimplemented case for node merge optimization.");
}

Status NodeMergeRewritePass::RewriteNode(std::unique_ptr<Graph>* g, Node *n) {
  CHECK_NOTNULL(n);

  // Get the matching rewriteinfo for the node
  const Node* fwdn = nullptr;
  const RewriteInfo* ri = FindMatchingRewriteInfo(n, &fwdn);
  if (ri == nullptr || fwdn == nullptr) {
    VLOG(1) << "Rewriteinfo not found for: " << n->type_string();
    return Status(error::Code::INVALID_ARGUMENT,
                  "Rewrite info not found for the node."
                  "Will skip node rewrite optimization");
  }

  VLOG(1) << "Rewrite called for: " << n->type_string();

  if (n->type_string() == csinfo_.biasaddgrad &&
      ri->node         == csinfo_.biasaddgrad &&
      (ri->rewrite     == csinfo_.conv2dwithbiasbackpropbias ||
       ri->rewrite     == csinfo_.biasaddgrad)) {
    DataType T; string data_format;
    TF_CHECK_OK(GetNodeAttr(n->def(), "T", &T));
    TF_CHECK_OK(GetNodeAttr(n->def(), "data_format", &data_format));

    int n_num = n->num_inputs();  // this must be 1.
    CHECK_EQ(n_num, 1);

    gtl::InlinedVector<Node*, 4> n_control_edges;
    gtl::InlinedVector<std::pair<Node*, int>, 4> n_in(n_num);
    FillInputs(n, &n_control_edges, &n_in);

    Node *ret = nullptr, *op = n_in[0].first;

    if (ri->rewrite == csinfo_.conv2dwithbiasbackpropbias) {
      // Get strides info from Conv2D (node in the forward pass that this
      // node corresponds to).
      std::vector<int32> strides;
      TF_CHECK_OK(GetNodeAttr(fwdn->def(), "strides", &strides));

      // We use same name as original node name as there may be fetchoutputs
      // associated with it.
      TF_CHECK_OK(NodeBuilder(n->name(), ri->rewrite)
                    .Input(op)
                    .Attr("T", T)
                    .Attr("data_format", data_format)
                    .Attr("strides", strides)
                    .Finalize(&**g, &ret));
    } else {
      CHECK_EQ(ri->rewrite, csinfo_.biasaddgrad);
      TF_CHECK_OK(NodeBuilder(n->name(), ri->rewrite)
                    .Input(op)
                    .Attr("T", T)
                    .Attr("data_format", data_format)
                    .Finalize(&**g, &ret));
    }

    CHECK_NOTNULL(ret);

    // Incoming edges are fixed, we will fix the outgoing edges now.
    for (const Edge* e : n->out_edges()) {
      (*g)->AddEdge(ret, e->src_output(), e->dst(), e->dst_input());
    }

    VLOG(1) << "Rewrite node: " << n->type_string() << " successful";
    (*g)->RemoveNode(n);

    return Status::OK();
  }

  return Status(error::Code::UNIMPLEMENTED,
                "Unimplemented case for node rewrite optimization.");
}

const NodeMergeRewritePass::RewriteInfo*
NodeMergeRewritePass::FindMatchingRewriteInfo(const Node* n,
                                              const Node** fwdn) const {
  CHECK_NOTNULL(n);
  CHECK_NOTNULL(fwdn);
  *fwdn = nullptr;

  // Search for matching rewriteinfo based on node name.
  // There could be more than one matching rewriteinfos.
  std::vector<const RewriteInfo*> matching_ri;
  for (auto ri = rinfo_.cbegin(); ri != rinfo_.cend(); ++ri) {
    if (n->type_string() == ri->node) {
      matching_ri.push_back(&*ri);
    }
  }

  VLOG(1) << "Searching graph for: " << n->type_string() << " in backwards.";

  // Now we will check for forward op name for rewrite info in data
  // flow graph. Get the max hops we should search for the fwd node
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
    curr_node  = curr_pair.first;
    curr_depth = curr_pair.second;
    CHECK_NOTNULL(curr_node);

    VLOG(1) << "Visiting node: " << curr_node->type_string()
            << " at depth: " << curr_depth
            << " for node: " << n->type_string();

    // If we find a match, we return immediately with the matching rewrite
    // info.
    for (const RewriteInfo* ri : matching_ri) {
      if (curr_node->type_string() == ri->cinfo.fwd) {
        *fwdn = curr_node;
        return ri;
      }
    }

    // Else we explore backward edges from current node.
    // Add the source nodes of all incoming edges of the node to the queue.
    for (const Edge* e : curr_node->in_edges()) {
      // We do not visit already visited node.
      if (visited_nodes.find(e->src()) == visited_nodes.end()) {
         // Depth of these nodes is 1 more than the depth of current node.
         nqueue.push(std::make_pair(e->src(), curr_depth+1));
         visited_nodes.insert(e->src());
      }
    }
  } /* while */

  return nullptr;
}

bool NodeMergeRewritePass::IsApplicableRewriteNode(const Node *n) const {
  CHECK_NOTNULL(n);

  // Search for matching rewriteinfo
  // Even if we find one match, we return true.
  bool match_found = false;
  for (const RewriteInfo &ri : rinfo_) {
    if (n->type_string() == ri.node) {
      match_found = true;
      break;
    }
  }

  return match_found;
}

bool NodeMergeRewritePass::RunPass(std::unique_ptr<Graph>* g) {
  bool result = false;
  CHECK_NOTNULL(g);

  DumpGraph("Before OptimizeMerge", &**g);

  std::vector<Node*> order;
  GetReversePostOrder(**g, &order);
  std::vector<std::pair<Node*, Node*>> nodes_to_be_merged;
  std::vector<Node*> nodes_to_be_rewritten;

  VLOG(1) << "Running NodeMerge Optimization";

  for (Node* n : order) {
    if (!n->IsOp()) continue;
    Node* n1 = nullptr;
    if ((n1 = FindNodeForMerge(n)) != nullptr) {
      VLOG(1) << "Scheduled nodes " << n->name() << " and "
              << n1->name() << " for merging";
      nodes_to_be_merged.push_back(std::make_pair(n, n1));
    } else if (IsApplicableRewriteNode(n)) {
      VLOG(1) << "Scheduled node " << n->name() << " for rewrite";
      nodes_to_be_rewritten.push_back(n);
    }
  }

  for (std::pair < Node*, Node* > i : nodes_to_be_merged) {
    // Even if MergeNode merges single pair of nodes, we
    // need to return true.
    string n1_name = i.first->name();
    string n2_name = i.second->name();
    if (MergeNode(g, i.first, i.second) == Status::OK()) {
      VLOG(1) << "Merged nodes " << n1_name << " and " << n2_name;
      result = true;
    }
  }

  DumpGraph("After OptimizeMerge(nodemerge)", &**g);

  for (Node* i : nodes_to_be_rewritten) {
    string name = i->name();
    if (RewriteNode(g, i) == Status::OK()) {
      VLOG(1) << "Rewrite node: " << name << " successful.";
      result = true;
    }
  }

  DumpGraph("After OptimizeMerge(noderewrite)", &**g);

  return result;
}

bool OptimizeNodeMerge(std::unique_ptr<Graph>* g) {
  return NodeMergeRewritePass().RunPass(g);
}

Status NodeMergeRewritePass::Run(const GraphOptimizationPassOptions& options) {
  // Currently checking only for two cases - Conv2D+Bias and Matmul+Bias.
  // It is possible to extend it to other operators in future.
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
