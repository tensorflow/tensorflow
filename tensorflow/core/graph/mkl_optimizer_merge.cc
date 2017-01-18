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
#include <algorithm>
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

///////////////////////////////////////////////////////////////////////////////
//              Functions related to node merging
///////////////////////////////////////////////////////////////////////////////

/*
  Return a node that can be merged with Node a

  Returns pointer to the node if we can find such a
  node. Otherwise, it returns NULL.
*/
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

  for (const MergeInfo *mi : matching_mi) {
    const int N_in = a->num_inputs();
    if (mi->op >= N_in) {
      // NOTE: This should be again an assert. But we skip such case.
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


/*
  Merge predecessor node with its successor.
  Currently, we support merging Conv2D with AddBias, and
  MatMul with Add node.

  Returns true, if merging is successful and supported.
  Returns false otherwise.
*/
bool NodeMergeRewritePass::MergeNode(std::unique_ptr<Graph>* g,
                                     Node* succ, Node* pred) {
  if (succ == NULL || pred == NULL)
    return false;

  bool result = false;

  if (succ->type_string() == csinfo_.biasadd &&
      pred->type_string() == csinfo_.conv2d) {
    // 1. Get all attributes from input nodes.
    DataType T_pred, T_succ;
    TF_CHECK_OK(GetNodeAttr(pred->def(), "T", &T_pred));
    TF_CHECK_OK(GetNodeAttr(succ->def(), "T", &T_succ));
    if (T_pred != T_succ) {
      return false;
    }

    string padding;
    TF_CHECK_OK(GetNodeAttr(pred->def(), "padding", &padding));

    std::vector<int32> strides;
    TF_CHECK_OK(GetNodeAttr(pred->def(), "strides", &strides));

    // We check to ensure that data formats of both succ and pred are same.
    // We expect them to be same, so we can enforce this as assert.
    // But assert can be too strict, so we enforce this as a check.
    // If the check fails, then we do not merge two nodes.
    string data_format_pred, data_format_succ;
    TF_CHECK_OK(GetNodeAttr(pred->def(), "data_format", &data_format_pred));
    TF_CHECK_OK(GetNodeAttr(succ->def(), "data_format", &data_format_succ));
    if (data_format_pred != data_format_succ)
    return false;

    bool use_cudnn_on_gnu;
    TF_CHECK_OK(GetNodeAttr(pred->def(), "use_cudnn_on_gpu",
                            &use_cudnn_on_gnu));

    // Groups attribute may not be there on the input node. So we do not
    // check for error in GetNodeAttr call.
    int groups = 1;
    GetNodeAttr(pred->def(), "groups", &groups);

    // 2. Get inputs from both the nodes.
    // Find the 2 inputs from the conv and the bias from the add Bias.
    Node *oper1 = NULL, *oper2 = NULL, *oper3 = NULL;

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
      return false;
    }

    for (const Edge *e : pred->out_edges()) {
      if (e->dst() != succ) {
        return false;
      }
    }

    // Get operand 0, 1 of conv2D
    oper1 = pred_in[0].first;
    oper2 = pred_in[1].first;
    // Get operand 1 of add_bias
    oper3 = succ_in[1].first;

    Node* ret;
    TF_CHECK_OK(NodeBuilder((*g)->NewName("n"), csinfo_.conv2dwithbias)
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

    for (const Edge* e : succ->out_edges()) {
      (*g)->AddEdge(ret, e->src_output(), e->dst(), e->dst_input());
    }

    (*g)->RemoveNode(succ);
    (*g)->RemoveNode(pred);

    result = true;
  } else if (succ->type_string() == csinfo_.add &&
             pred->type_string() == csinfo_.matmul) {
    // 1. Get all attributes from input nodes.
    DataType T_pred, T_succ;
    bool transpose_a, transpose_b;

    // Check for type (T).
    TF_CHECK_OK(GetNodeAttr(pred->def(), "T", &T_pred));
    TF_CHECK_OK(GetNodeAttr(succ->def(), "T", &T_succ));
    if (T_pred != T_succ) {
      return false;
    }

    TF_CHECK_OK(GetNodeAttr(pred->def(), "transpose_a", &transpose_a));
    TF_CHECK_OK(GetNodeAttr(pred->def(), "transpose_b", &transpose_b));

    // NOTE: what about succ_is_sparse and pred_is_sparse attributes of
    // matmul? Those are not represented in matmulmkl.

    // 2. Get inputs from both the nodes.
    const int succ_num = succ->num_inputs(); /* this must be 2. */
    gtl::InlinedVector<Node*, 4> succ_control_edges;
    gtl::InlinedVector<std::pair<Node*, int>, 4> succ_in(succ_num);
    FillInputs(succ, &succ_control_edges, &succ_in);

    const int pred_num = pred->num_inputs(); /* this must be 2. */
    gtl::InlinedVector<Node*, 4> pred_control_edges;
    gtl::InlinedVector<std::pair<Node*, int>, 4> pred_in(pred_num);
    FillInputs(pred, &pred_control_edges, &pred_in);

    // MatMul may have more than 1 successor. In such case, we merge
    // matmul and add only if successor of matmul is shape (2 outgoing
    // edges from MatMul).
    bool shape_node_found = false;
    if (pred->out_edges().size() == 2) {
      for (const Edge *e : pred->out_edges()) {
        if (e->dst() != succ && e->dst()->type_string() != "Shape")
          return false;
      }
      shape_node_found = true;
    } else if (pred->out_edges().size() == 1) {
      // Or we may have a case that there is only 1 edge between MatMul
      // and Add. Otherwise, merging is semantically incorrect.
      // No-op here.
    } else {
      // For any other case, we do not merge.
      return false;
    }

    // Find the inputs from add and matmul.
    Node *mm_op1 = NULL, *mm_op2 = NULL, *bias = NULL;
    mm_op1 = pred_in[0].first;  // 1st operand of matmul
    mm_op2 = pred_in[1].first;  // 2nd operand of matmul
    bias   = succ_in[1].first;  // bias is 2nd operand of Add

    if (mm_op1 == NULL || mm_op2 == NULL || bias == NULL) {
      result = false;
    } else {
      Node* ret;
      TF_CHECK_OK(NodeBuilder((*g)->NewName("n"), csinfo_.matmulmkl)
        .Input(mm_op1)
        .Input(mm_op2)
        .Input(bias)
        .Attr("T", T_pred)
        .Attr("transpose_a", transpose_a)
        .Attr("transpose_b", transpose_b)
        .Finalize(&**g, &ret));

      CHECK_NOTNULL(ret);

      for (const Edge* e : succ->out_edges()) {
        (*g)->AddEdge(ret, e->src_output(), e->dst(), e->dst_input());
      }

      // This is to add edge from matmulmkl to shape, corresponding to
      // the edge from matmul to shape.
      if (shape_node_found == true) {
        for (const Edge* e : pred->out_edges()) {
          (*g)->AddEdge(ret, e->src_output(), e->dst(), e->dst_input());
        }
      }

      (*g)->RemoveNode(succ);
      (*g)->RemoveNode(pred);

      result = true;
    }
  }

  return result;
}

///////////////////////////////////////////////////////////////////////////////
//              Functions related to node rewriting
///////////////////////////////////////////////////////////////////////////////

bool NodeMergeRewritePass::RewriteNode(std::unique_ptr<Graph>* g, Node *n) {
  if (n == nullptr)
    return false;

  // Get the matching rewriteinfo for the node
  const Node *fwdn = nullptr;
  const RewriteInfo* ri = FindMatchingRewriteInfo(n, &fwdn);
  if (ri == nullptr || fwdn == nullptr) {
    VLOG(1) << "Rewriteinfo not found for: " << n->type_string();
    return false;
  }

  VLOG(1) << "Rewrite called for: " << n->type_string();

  if (n->type_string() == csinfo_.biasaddgrad &&
      ri->node       == csinfo_.biasaddgrad    &&
      (ri->rewrite   == csinfo_.conv2dwithbiasbackpropbias ||
       ri->rewrite   == csinfo_.biasaddgrad)) {
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
    } else if (ri->rewrite == csinfo_.biasaddgrad) {
      TF_CHECK_OK(NodeBuilder(n->name(), ri->rewrite)
       .Input(op)
       .Attr("T", T)
       .Attr("data_format", data_format)
       .Finalize(&**g, &ret));
    } else {
      return false;
    }

    CHECK_NOTNULL(ret);

    // Incoming edges are fixed, we will fix the outgoing edges now.
    for (const Edge* e : n->out_edges()) {
      (*g)->AddEdge(ret, e->src_output(), e->dst(), e->dst_input());
    }

    VLOG(1) << "Rewrite node: " << n->type_string() << " successful";
    (*g)->RemoveNode(n);
    return true;
  } else {
    VLOG(1) << "Unsupported node: " << n->type_string() << " for rewrite";
  }
  return false;
}

const NodeMergeRewritePass::RewriteInfo*
NodeMergeRewritePass::FindMatchingRewriteInfo(const Node *n,
                                              const Node **fwdn) const {
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
  // flow graph.
  // Get the max hops we should search for the fwd node
  // We are now going to search (breadth-first) backwards in data
  // dependence graph (for up to max hops) from n for the node
  // specified in fwd.
  // queue to maintain nodes to be visited and depth info for
  // breadth-first search
  std::queue<std::pair<const Node *, int>> nqueue;
  const Node *curr_node = n;
  int curr_depth = 0;
  nqueue.push(std::make_pair(curr_node, curr_depth));

  while (curr_depth < NODEMERGE_CONTEXT_MAXDEPTH && !nqueue.empty()) {
    std::pair<const Node *, int> curr_pair = nqueue.front();
    nqueue.pop();

    std::set<const Node*> visited_nodes;
    curr_node  = curr_pair.first;
    curr_depth = curr_pair.second;
    DCHECK_NE(curr_node, nullptr);

    VLOG(1) << "Visiting node: " << curr_node->type_string()
            << " at depth: " << curr_depth
            << " for node: " << n->type_string();

    // If we find a match, we return immediately with the matching rewrite
    // info.
    for (const RewriteInfo *ri : matching_ri) {
      if (curr_node->type_string() == ri->cinfo.fwd) {
        *fwdn = curr_node;
        return ri;
      }
    }

    // Else we explore backward edges from current node.
    // Add the source nodes of all incoming edges of the node to the queue.
    for (const Edge *e : curr_node->in_edges()) {
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

bool NodeMergeRewritePass::DoNodeMerge(std::unique_ptr<Graph>* g) {
  bool result = false;
  CHECK_NOTNULL(g);

  DumpGraph("Before OptimizeMerge", &**g);

  std::vector<Node*> order;
  GetReversePostOrder(**g, &order);
  std::vector<std::pair<Node*, Node*>> to_be_merged;
  std::vector<Node*> to_be_rewritten;

  VLOG(1) << "Running NodeMerge Optimization";

  for (Node* n : order) {
    if (!n->IsOp()) continue;
    Node *n1 = nullptr;
    if ((n1 = FindNodeForMerge(n)) != nullptr) {
      VLOG(1) << "Scheduled nodes " << n->name() << " and "
              << n1->name() << " for merging";
      to_be_merged.push_back(std::make_pair(n, n1));
    } else if (IsApplicableRewriteNode(n)) {
      VLOG(1) << "Scheduled node " << n->name() << " for rewrite";
      to_be_rewritten.push_back(n);
    }
  }

  for (std::pair < Node*, Node* > i : to_be_merged) {
    // Even if MergeNode merges single pair of nodes, we
    // need to return true.
    string n1name = i.first->name();
    string n2name = i.second->name();
    if (MergeNode(g, i.first, i.second)) {
      VLOG(1) << "Merged nodes " << n1name << " and " << n2name;
      result = true;
    }
  }

  DumpGraph("After OptimizeMerge(nodemerge)", &**g);

  for (Node* i : to_be_rewritten) {
    string name = i->name();
    if (RewriteNode(g, i)) {
      VLOG(1) << "Rewrite node: " << name << " successful.";
      result = true;
    }
  }

  DumpGraph("After OptimizeMerge(noderewrite)", &**g);

  return result;
}

///////////////////////////////////////////////////////////////////////////////
//              Run function for the pass
///////////////////////////////////////////////////////////////////////////////

bool OptimizeNodeMerge(std::unique_ptr<Graph>* g) {
  // Get the ownership of the graph.
  NodeMergeRewritePass *pass = new NodeMergeRewritePass();
  bool result = pass->DoNodeMerge(g);
  // Return the ownership of graph back
  // g->reset(pass->GetGraph()->release());
  delete pass;
  return result;
}

Status NodeMergeRewritePass::Run(const GraphOptimizationPassOptions& options) {
  // Currently checking only for two cases - Conv2D+Bias and Matmul+Bias.
  // It is possible to extend it to other operators in future.
  if (options.graph == nullptr) return Status::OK();

  // Get the ownership of graph
  std::unique_ptr<Graph>* g = std::move(options.graph);

  bool result = DoNodeMerge(g);

  // Return the ownership of graph back
  options.graph->reset(g->release());

  return Status::OK();
}

}  // namespace tensorflow

#endif
