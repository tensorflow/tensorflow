/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include <deque>
#include <vector>

#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/executor.h"
#include "tensorflow/core/common_runtime/graph_optimizer.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/gradients.h"
#include "tensorflow/core/graph/graph_constructor.h"
#include "tensorflow/core/graph/optimizer_cse.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {

// TODO(andydavis) Remove some of the code duplicated between this module
// and that in 'common_runtime/function.cc'.
// A few string constant used throughout this module.
static const char* const kArgOp = "_Arg";
static const char* const kRetOp = "_Retval";
static const char* const kGradientOp = "SymbolicGradient";
static const char* const kNodeLabel = "Func";

// Represents the index-th output of a node.
struct Endpoint {
  Node* node;
  int index;

  // Returns the string name represents this endpoint.
  string name() const {
    if (index == 0) {
      return node->name();
    } else {
      return strings::StrCat(node->name(), ":", index);
    }
  }

  DataType dtype() const { return node->output_type(index); }
};

struct EndpointHash {
  uint64 operator()(const Endpoint& x) const {
    return Hash64(reinterpret_cast<const char*>(&x.node), sizeof(Node*),
                  x.index);
  }
};

struct EndpointEq {
  bool operator()(const Endpoint& x, const Endpoint& y) const {
    return (x.node == y.node) && (x.index == y.index);
  }
};

static Node* AddZerosLike(Graph* g, Endpoint input) {
  DCHECK_LT(0, input.dtype());
  DCHECK_LT(input.dtype(), DT_FLOAT_REF);
  NodeDef ndef;
  ndef.set_name(g->NewName(kNodeLabel));
  ndef.set_op("ZerosLike");
  ndef.add_input(input.name());
  AddNodeAttr("T", input.dtype(), &ndef);
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  g->AddEdge(input.node, input.index, ret, 0);
  return ret;
}

static Node* AddSymGrad(Graph* g, Node* n, gtl::ArraySlice<Endpoint> grads) {
  const int num_x = n->num_inputs();
  const int num_y = n->num_outputs();
  CHECK_EQ(num_y, grads.size());

  NodeDef ndef;
  ndef.set_name(g->NewName(kNodeLabel));
  ndef.set_op(kGradientOp);

  // The gradient node should have num_x + num_y inputs.
  std::vector<Endpoint> n_inputs(num_x);
  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) continue;
    n_inputs[e->dst_input()] = {e->src(), e->src_output()};
  }
  DataTypeVector in_types;
  for (const Endpoint& ep : n_inputs) {
    ndef.add_input(ep.name());
    in_types.push_back(ep.dtype());
  }
  for (const Endpoint& ep : grads) {
    ndef.add_input(ep.name());
    in_types.push_back(ep.dtype());
  }
  CHECK_EQ(ndef.input_size(), num_x + num_y);

  AddNodeAttr("Tin", in_types, &ndef);

  // The gradient node's outputs have the same types as the node 'n's
  // inputs.
  AddNodeAttr("Tout", n->input_types(), &ndef);
  NameAttrList func;
  func.set_name(n->type_string());
  *(func.mutable_attr()) = n->def().attr();
  AddNodeAttr("f", func, &ndef);
  Status s;
  Node* ret = g->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  return ret;
}

class SymbolicGradientBuilder {
 public:
  SymbolicGradientBuilder(gtl::ArraySlice<Node*> y_nodes,
                          gtl::ArraySlice<Node*> x_nodes,
                          gtl::ArraySlice<Node*> y_grad_nodes,
                          std::vector<GradNodeOutput>* x_grad_nodes,
                          Graph* graph);

  Status Compute();

 private:
  gtl::ArraySlice<Node*> y_nodes_;
  gtl::ArraySlice<Node*> x_nodes_;
  gtl::ArraySlice<Node*> y_grad_nodes_;
  std::vector<GradNodeOutput>* x_grad_nodes_;
  Graph* graph_;  // Not owned.

  // A vector of output endpoints which represents backpropagated
  // gradients
  typedef std::vector<Endpoint> BackpropedGradients;

  // backprops_ is a map from an output endpoint to its accumulated
  // gradients.  When an output endpoint has accumulated all its
  // gradients, we add a node which sums them up.
  std::unordered_map<Endpoint, BackpropedGradients, EndpointHash, EndpointEq>
      backprops_;

  // pending[i] is count-down counter for i-th node's expected
  // backprops.  When pending[i] becomes zero, we collected all
  // backprop gradients for all output endpoint of the ith-node.
  std::vector<int> pending_;

  // 'ready' keeps track of nodes that have been completely
  // backpropped. Initially, for every output y of the function f, we
  // add dy as an input of the gradient function.
  std::deque<Node*> ready_;

  // The set of nodes at which to stop backprop (and populate 'x_grad_nodes_').
  std::unordered_map<int, int> stop_nodes_;

  // Initialize pending_ and ready_.
  void InitBackprop();

  // In the original function body, there is a forward edge from 'src'
  // to 'dst', when the backprop algorithm constructs the node
  // 'dst_grad' which computes the gradient, we need to propagate it
  // to 'src'.
  void BackpropAlongEdge(const Endpoint& dst_grad, const Endpoint& src);
  void BackpropZerosAlongEdge(const Endpoint& src);

  Endpoint SumGradients(const Endpoint& src);

  TF_DISALLOW_COPY_AND_ASSIGN(SymbolicGradientBuilder);
};

SymbolicGradientBuilder::SymbolicGradientBuilder(
    gtl::ArraySlice<Node*> y_nodes,
    gtl::ArraySlice<Node*> x_nodes,
    gtl::ArraySlice<Node*> y_grad_nodes,
    std::vector<GradNodeOutput>* x_grad_nodes,
    Graph* graph) : y_nodes_(y_nodes), x_nodes_(x_nodes),
                    y_grad_nodes_(y_grad_nodes), x_grad_nodes_(x_grad_nodes),
                    graph_(graph)  {
  CHECK_EQ(y_nodes_.size(), y_grad_nodes.size());
  x_grad_nodes_->clear();
  x_grad_nodes_->resize(x_nodes_.size());
  stop_nodes_.reserve(x_nodes_.size());
  for (int i = 0; i < x_nodes_.size(); ++i) {
    stop_nodes_.insert(std::make_pair(x_nodes_[i]->id(), i));
  }
}

void SymbolicGradientBuilder::BackpropAlongEdge(const Endpoint& dst_grad,
                                                const Endpoint& src) {
  CHECK_NOTNULL(src.node);
  auto iter = backprops_.find(src);
  if (iter != backprops_.end()) {
    auto* grads = &iter->second;
    grads->push_back(dst_grad);
    if (--pending_[src.node->id()] == 0) {
      ready_.push_back(src.node);
    }
  }
}

void SymbolicGradientBuilder::BackpropZerosAlongEdge(const Endpoint& src) {
  CHECK_NOTNULL(src.node);
  auto iter = backprops_.find(src);
  if (iter != backprops_.end()) {
    if (--pending_[src.node->id()] == 0) {
      ready_.push_back(src.node);
    }
  }
}

void SymbolicGradientBuilder::InitBackprop() {
  pending_.resize(graph_->num_node_ids(), 0);
  {
    backprops_.clear();
    std::unordered_set<Node*> visited;
    std::deque<Node*> queue;
    for (Node* n : x_nodes_) {
      queue.push_back(n);
      visited.insert(n);
    }

    // Going forward to figure out which endpoints need backprop-ed.
    // A node's endpoints need to be backprop-ed only if one of the
    // arg node can reach the node via data edges.
    while (!queue.empty()) {
      Node* n = queue.front();
      queue.pop_front();
      for (int i = 0; i < n->num_outputs(); ++i) {
        backprops_[{n, i}].clear();
      }
      int num_expected_backprops = 0;
      for (const Edge* e : n->out_edges()) {
        if (e->IsControlEdge()) continue;
        ++num_expected_backprops;
        if (visited.find(e->dst()) == visited.end()) {
          queue.push_back(e->dst());
          visited.insert(e->dst());
        }
      }
      pending_[n->id()] = num_expected_backprops;
    }
  }

  {
    const int num_y = y_grad_nodes_.size();
    for (int i = 0; i < num_y; ++i) {
      Node* y = y_nodes_[i];
      Node* dy = y_grad_nodes_[i];
      for (const Edge* e : y->in_edges()) {
        if (e->IsControlEdge()) continue;
        BackpropAlongEdge({dy, e->dst_input()}, {e->src(), e->src_output()});
      }
    }
  }
  CHECK(!ready_.empty());
}

Endpoint SymbolicGradientBuilder::SumGradients(const Endpoint& src) {
  const DataType dtype = src.dtype();
  auto iter = backprops_.find(src);
  CHECK(iter != backprops_.end());
  const auto& grads = iter->second;
  if (grads.empty()) {
    // Nothing propagated back. The best we can come up is zeros.
    Node* zero_like = AddZerosLike(graph_, src);
    return {zero_like, 0};
  }
  if (grads.size() == 1) {
    // Just one backprop edge.
    return grads[0];
  }
  // Otherwise, adds backprop-ed gradients.
  NodeDef ndef;
  ndef.set_name(graph_->NewName(kNodeLabel));
  ndef.set_op("AddN");  // N-way Add
  for (const Endpoint& ep : grads) {
    ndef.add_input(ep.name());
  }
  AddNodeAttr("N", static_cast<int64>(grads.size()), &ndef);
  AddNodeAttr("T", dtype, &ndef);
  Status s;
  Node* add = graph_->AddNode(ndef, &s);
  TF_CHECK_OK(s);
  for (size_t i = 0; i < grads.size(); ++i) {
    const Endpoint& ep = grads[i];
    graph_->AddEdge(ep.node, ep.index, add, i);
  }
  return {add, 0};
}

static bool IsPrimitiveOpWithNoGrad(const string& func) {
  gradient::Creator creator;
  Status s = gradient::GetOpGradientCreator(func, &creator);
  return s.ok() && (creator == nullptr);
}

Status SymbolicGradientBuilder::Compute() {
  // Initialize backprops.
  InitBackprop();

  // Backward propagation.
  gtl::InlinedVector<Endpoint, 8> dy;
  while (!ready_.empty()) {
    // n has collected all gradients.
    Node* n = ready_.front();
    ready_.pop_front();

    // "n" has num_x inputs and num_y outputs.
    const int num_x = n->num_inputs();
    const int num_y = n->num_outputs();

    auto iter = stop_nodes_.find(n->id());
    if (iter != stop_nodes_.end()) {
      // Stop backprop and add gradient sum to 'x_grad_nodes'.
      // TODO(andydavis) Support stop nodes with more than one output.
      CHECK_EQ(1, num_y);
      Endpoint grad = SumGradients({n, 0});
      (*x_grad_nodes_)[iter->second] = {grad.node, grad.index};
      continue;
    }

    // dy[i] is the sum of i-th output's backpropped gradients.
    dy.clear();
    dy.resize(num_y, {nullptr, 0});
    for (int i = 0; i < num_y; ++i) {
      dy[i] = SumGradients({n, i});
    }

    if (IsPrimitiveOpWithNoGrad(n->type_string())) {
      // No grad defined for this op: Backprop zeros along the in edges.
      for (const Edge* e : n->in_edges()) {
        if (e->IsControlEdge()) continue;
        BackpropZerosAlongEdge({e->src(), e->src_output()});
      }
      continue;
    }

    // Adds a gradient node with num_x + num_y inputs and num_x
    // outputs.
    Node* grad = AddSymGrad(graph_, n, dy);
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      graph_->AddEdge(e->src(), e->src_output(), grad, e->dst_input());
    }
    for (int i = 0; i < num_y; ++i) {
      graph_->AddEdge(dy[i].node, dy[i].index, grad, num_x + i);
    }

    // Backprops along the in edges.
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      BackpropAlongEdge({grad, e->dst_input()}, {e->src(), e->src_output()});
    }
  }

  return Status::OK();
}

Status AddSymbolicGradients(gtl::ArraySlice<Node*> y_nodes,
                            gtl::ArraySlice<Node*> x_nodes,
                            gtl::ArraySlice<Node*> y_grad_nodes,
                            std::vector<GradNodeOutput>* x_grad_nodes,
                            Graph* graph) {
  SymbolicGradientBuilder builder(y_nodes, x_nodes, y_grad_nodes, x_grad_nodes,
                                  graph);
  return builder.Compute();
}

}  // end namespace tensorflow
