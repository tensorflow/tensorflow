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

#include "tensorflow/cc/framework/gradients.h"

#include <deque>
#include <vector>

#include "tensorflow/cc/framework/grad_op_registry.h"
#include "tensorflow/cc/framework/while_gradients.h"
#include "tensorflow/cc/ops/standard_ops.h"
#include "tensorflow/core/common_runtime/graph_constructor.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/while_context.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/platform/macros.h"

namespace tensorflow {
namespace {

struct OutputHash {
  uint64 operator()(const Output& x) const {
    return x.hash();
  }
};

struct OutputEq {
  bool operator()(const Output& x, const Output& y) const {
    return (x.node() == y.node()) && (x.index() == y.index());
  }
};

class SymbolicGradientBuilder {
 public:
  SymbolicGradientBuilder(const Scope& scope,
                          const ops::GradOpRegistry* registry,
                          const std::vector<Output>& outputs,
                          const std::vector<Output>& inputs,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs);

  Status AddGradients();

  static Output NoGradient() { return Output(nullptr, -1); }

 private:
  Status Initialize();

  // For each forward edge from `src` to `dst` in the initial/forward graph:
  // propagates gradients `dst_grad` backwards along the edge from `src`
  // to `dst` in the graph. This will add `dst_grad` to the list of pending
  // gradients for the node associated with `src`.
  Status BackpropAlongEdge(const Output& dst_grad, const Output& src);

  // Adds a node to the graph (returned in `grad`) that sums the in-bound
  // gradients to `src` (if there are more than one).
  Status SumGradients(const Output& src, Output* grad);

  // Returns true if `opname` is registered in `registry_` with no gradient
  // function, false otherwise.
  bool IsPrimitiveOpWithNoGrad(const string& opname);

  // Call the gradient function for `op`, storing the result in `grad_outputs`.
  Status CallGradFunction(const Operation& op,
                          const std::vector<Output>& grad_inputs,
                          std::vector<Output>* grad_outputs);

  // Returns a list mapping whether each node in the graph is reachable
  // from outputs_. Keyed by node id.
  std::vector<bool> GetReachableNodes();

  // Creates the gradient subgraph for a while loop (or just stores
  // `summed_grads` if not all incoming gradients are available yet). All exit
  // nodes (which are the first nodes of a loop encountered in the backwards
  // pass) are passed to this function rather than processed normally.
  // `summed_grads` is the sum of `exit_node`s gradients.
  Status ProcessWhileLoop(Node* exit_node, const Output& summed_grads);

  // Gets the set of node ids at which to stop backprop. These are all elements
  // of `outputs_` that do not get transitively consumed by other `outputs_`.
  // Used to identify nodes at which to stop backprop.
  std::unordered_set<int> GetStopBackpropNodes(
      const std::vector<bool>& reachable_nodes,
      const std::unordered_set<int>& output_nodes) const;

  const Scope& scope_;
  const ops::GradOpRegistry* registry_;
  const std::vector<Output>& outputs_;
  const std::vector<Output>& inputs_;
  const std::vector<Output>& grad_inputs_;
  std::vector<Output>* grad_outputs_;

  // A vector of output endpoints which represents backpropagated gradients.
  typedef std::vector<Output> BackproppedGradients;

  // backprops_ is a map from a node output to its accumulated
  // gradients.  When a node output has accumulated all its
  // gradients, we add a node which sums them up.
  std::unordered_map<Output, BackproppedGradients, OutputHash, OutputEq>
      backprops_;

  // pending[i] is count-down counter for i-th node's expected
  // backprops.  When pending[i] becomes zero, we collected all
  // backprop gradients for all outputs of the ith-node.
  std::vector<int> pending_;

  // `ready` keeps track of nodes that have been completely
  // backpropped. Initially, for every output in `outputs_`, we add initial
  // gradients from `grad_inputs_`.
  std::deque<Node*> ready_;

  // The set of node ids in `inputs_`. Used to identify nodes at backprop
  // frontier. Maps from Output -> index into `grad_outputs_`.
  std::unordered_map<Output, int, OutputHash, OutputEq> input_nodes_;

  // For each while loop in the graph, collects the summed gradients for each of
  // the loop's exit nodes. Note that unlike backprops_, this map contains the
  // output of SumGradients(), not the input (i.e. each exit node may have
  // multiple incoming gradients, but we only store the combined Output here).
  std::map<WhileContext*, std::map<Node*, Output>> while_backprops_;

  TF_DISALLOW_COPY_AND_ASSIGN(SymbolicGradientBuilder);
};

SymbolicGradientBuilder::SymbolicGradientBuilder(
    const Scope& scope, const ops::GradOpRegistry* registry,
    const std::vector<Output>& outputs, const std::vector<Output>& inputs,
    const std::vector<Output>& grad_inputs, std::vector<Output>* grad_outputs)
    : scope_(scope),
      registry_(registry),
      outputs_(outputs),
      inputs_(inputs),
      grad_inputs_(grad_inputs),
      grad_outputs_(grad_outputs) {}

Status SymbolicGradientBuilder::BackpropAlongEdge(const Output& dst_grad,
                                                  const Output& src) {
  if (src.node() == nullptr) {
    return errors::Internal("Attempted to backprop along an invalid edge.");
  }
  auto iter = backprops_.find(src);
  if (iter != backprops_.end()) {
    auto* grads = &iter->second;
    grads->push_back(dst_grad);
    if (--pending_[src.node()->id()] == 0) {
      ready_.push_back(src.node());
    }
  }
  return Status::OK();
}

std::vector<bool> SymbolicGradientBuilder::GetReachableNodes() {
  std::vector<bool> reachable_nodes(scope_.graph()->num_node_ids(), false);
  std::deque<Node*> queue;
  for (const Output& out : outputs_) {
    if (!reachable_nodes[out.node()->id()]) {
      queue.push_back(out.node());
      reachable_nodes[out.node()->id()] = true;
    }
  }

  while (!queue.empty()) {
    Node* n = queue.front();
    queue.pop_front();
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      if (!reachable_nodes[e->src()->id()]) {
        queue.push_back(e->src());
        reachable_nodes[e->src()->id()] = true;
      }
    }
  }
  return reachable_nodes;
}

std::unordered_set<int> SymbolicGradientBuilder::GetStopBackpropNodes(
    const std::vector<bool>& reachable_nodes,
    const std::unordered_set<int>& output_nodes) const {
  // Output nodes that get transitively consumed by other `outputs_` are stored
  // in `internal_outputs`.
  std::unordered_set<int> internal_outputs;
  std::unordered_set<Node*> visited;
  // Initialize `queue` for BFS traversal. Nodes in `queue` hold upcoming nodes
  // along with the last Node in `output_` encountered along that path. If no
  // `output_` node was encountered, pair.second will be nullptr.
  std::deque<std::pair<Node*, Node*>> queue;
  for (const Output& nout : inputs_) {
    auto const& pair = visited.insert(nout.node());
    if (pair.second) {
      queue.push_back(std::make_pair(nout.node(), static_cast<Node*>(nullptr)));
    }
  }
  // BFS from nodes in 'inputs_' along out edges for the entire graph. Internal
  // output nodes are recorded during the traversal. All nodes that are output
  // nodes but not internal output nodes are considered the frontier of the
  // output nodes, and thus our stop backprop nodes.
  while (!queue.empty()) {
    std::pair<Node*, Node*> p = queue.front();
    Node* n = p.first;
    queue.pop_front();
    for (const Edge* e : n->out_edges()) {
      // If a node is not reachable from outputs_, we can stop.
      if (e->IsControlEdge() || !reachable_nodes[e->dst()->id()]) continue;

      auto const& pair = visited.insert(e->dst());
      if (pair.second) {
        int node_id = e->dst()->id();
        Node* last_output_node = p.second;
        if (output_nodes.find(node_id) != output_nodes.end()) {
          // We reached an output node.
          if (last_output_node != nullptr) {
            // If we had already found an output node on this path so we mark
            // it as an internal output.
            internal_outputs.insert(last_output_node->id());
          }
          // Mark this newly found output node to insert in the queue.
          last_output_node = e->dst();
        }
        queue.push_back(std::make_pair(e->dst(), last_output_node));
      }
    }
  }
  // Finally, we set stop_backprop_nodes to all output_nodes that aren't also
  // internal_outputs.
  std::unordered_set<int> stop_backprop_nodes;
  for (int output_node : output_nodes) {
    if (internal_outputs.find(output_node) == internal_outputs.end()) {
      stop_backprop_nodes.insert(output_node);
    }
  }
  return stop_backprop_nodes;
}

Status SymbolicGradientBuilder::Initialize() {
  if (outputs_.size() != grad_inputs_.size()) {
    return errors::InvalidArgument(
        "Must specify a gradient input for each output.");
  }
  std::vector<bool> reachable_nodes = GetReachableNodes();
  for (const Output& input : inputs_) {
    if (!reachable_nodes[input.node()->id()]) {
      return errors::InvalidArgument(
          "Cannot compute the partial derivative for node '",
          input.node()->name(),
          "' as it's unreachable from the output node(s).");
    }
  }
  grad_outputs_->clear();
  grad_outputs_->resize(inputs_.size());

  std::unordered_set<int> output_nodes;
  output_nodes.reserve(outputs_.size());
  for (size_t i = 0; i < outputs_.size(); ++i) {
    output_nodes.insert(outputs_[i].node()->id());
  }

  std::unordered_set<int> stop_backprop_nodes =
      GetStopBackpropNodes(reachable_nodes, output_nodes);

  // Populate `input_nodes_` from Outputs in `inputs_`.
  input_nodes_.reserve(inputs_.size());
  for (size_t i = 0; i < inputs_.size(); ++i) {
    input_nodes_.insert({inputs_[i], i});
  }

  // TODO(andydavis) Consider a more efficient data structure for `pending_` to
  // handle computing gradients over small subgraphs from a very large graph.
  pending_.resize(scope_.graph()->num_node_ids(), 0);
  {
    backprops_.clear();
    std::unordered_set<Node*> visited;
    std::deque<Node*> queue;
    for (const Output& nout : inputs_) {
      auto const& pair = visited.insert(nout.node());
      if (pair.second) {
        queue.push_back(nout.node());
      }
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
      if (stop_backprop_nodes.find(n->id()) == stop_backprop_nodes.end()) {
        // Internal node: continue BFS along connected outputs.
        for (const Edge* e : n->out_edges()) {
          // If a node is not reachable from outputs_,
          // we don't expect it to receive a backpropagated gradient.
          // It will not be counted in num_expected_backprops.
          if (e->IsControlEdge() || !reachable_nodes[e->dst()->id()]) continue;
          auto const& pair = visited.insert(e->dst());
          if (pair.second) {
            queue.push_back(e->dst());
          }
          ++num_expected_backprops;
        }
      }
      if (output_nodes.find(n->id()) != output_nodes.end()) {
        // Output node: update `num_expected_backprops` for each Output in
        // `outputs_` that references `n`.
        for (const Output& output : outputs_) {
          if (output.node() == n) {
            ++num_expected_backprops;
          }
        }
      }
      pending_[n->id()] = num_expected_backprops;
    }
  }

  {
    // Initialize backprop with `grad_inputs_`.
    const size_t num_dy = grad_inputs_.size();
    for (size_t i = 0; i < num_dy; ++i) {
      TF_RETURN_IF_ERROR(BackpropAlongEdge(grad_inputs_[i], outputs_[i]));
    }
  }
  return Status::OK();
}

Status SymbolicGradientBuilder::SumGradients(const Output& src, Output* grad) {
  auto iter = backprops_.find(src);
  if (iter == backprops_.end()) {
    return errors::Internal(
        "Unable to find backprop list for node.id ", src.node()->name());
  }
  const auto& grads = iter->second;
  // Filter any backpropped 'NoGradient' Outputs from 'grads' (if needed).
  // Return any valid backpropped gradients that remain after filtering,
  // or 'NoGradient' otherwise.
  std::vector<Output> grads_to_keep;
  for (const Output& o : grads) {
    if (o == NoGradient()) continue;
    grads_to_keep.push_back(o);
  }

  if (grads_to_keep.empty()) {
    // Nothing propagated back. Return 'NoGradient'.
    *grad = NoGradient();
  } else if (grads_to_keep.size() == 1) {
    // Just one backprop edge.
    *grad = grads_to_keep[0];
  } else {
    // Otherwise, adds backprop-ed gradients.
    // TODO(andydavis) Use a better accumulator here.
    *grad = ops::AddN(scope_, grads_to_keep);
  }

  return Status::OK();
}

bool SymbolicGradientBuilder::IsPrimitiveOpWithNoGrad(const string& opname) {
  ops::GradFunc grad_fn;
  Status s = registry_->Lookup(opname, &grad_fn);
  return s.ok() && (grad_fn == nullptr);
}

Status SymbolicGradientBuilder::CallGradFunction(
    const Operation& op,
    const std::vector<Output>& grad_inputs,
    std::vector<Output>* grad_outputs) {
  ops::GradFunc grad_fn;
  TF_RETURN_IF_ERROR(registry_->Lookup(op.node()->type_string(), &grad_fn));
  TF_RETURN_IF_ERROR(grad_fn(scope_, op, grad_inputs, grad_outputs));
  TF_RETURN_IF_ERROR(scope_.status());
  return Status::OK();
}

Status SymbolicGradientBuilder::ProcessWhileLoop(Node* exit_node,
                                                 const Output& summed_grads) {
  // TODO(skyewm): detect second-order gradient and return bad status
  // TODO(skyewm): handle (or at least detect) nested while loops

  // TODO(skyewm): handle NoGradient in while loop
  if (summed_grads == NoGradient()) {
    return errors::Unimplemented(
        "Missing gradient into while loop not yet implemented");
  }

  DCHECK(exit_node->IsExit());
  WhileContext* while_ctx = exit_node->while_ctx();
  DCHECK(while_ctx != nullptr);

  // Record 'summed_grads' as the backprop input associated with 'exit_node'
  std::map<Node*, Output>& backprops = while_backprops_[while_ctx];
  DCHECK(backprops.find(exit_node) == backprops.end());
  backprops[exit_node] = summed_grads;

  // Wait until we have all exit nodes' backprops collected before processing
  // the while loop.
  // TODO(skyewm): what if not all the exit nodes are reachable?
  if (backprops.size() < while_ctx->exit_nodes().size()) return Status::OK();

  // We've seen all the exit nodes for this loop and have collected all the
  // backprops. Create the gradient graph for the while loop.
  Scope while_scope =
      scope_.NewSubScope(strings::StrCat(while_ctx->frame_name(), "_grad"));
  std::vector<Output> dy;
  for (Node* n : while_ctx->exit_nodes()) dy.push_back(backprops[n]);
  std::vector<Output> dx;
  TF_RETURN_IF_ERROR(AddWhileLoopGradient(while_ctx, while_scope, dy, &dx));

  // Backprop along the in edges to the while loop (i.e. the inputs to the enter
  // nodes)
  DCHECK_EQ(dx.size(), while_ctx->enter_nodes().size());
  for (int i = 0, end = dx.size(); i < end; ++i) {
    Node* enter_node = while_ctx->enter_nodes()[i];
    for (const Edge* e : enter_node->in_edges()) {
      if (e->IsControlEdge()) continue;
      TF_RETURN_IF_ERROR(BackpropAlongEdge(dx[i], {e->src(), e->src_output()}));
    }
  }
  return Status::OK();
}

Status SymbolicGradientBuilder::AddGradients() {
  // Initialize backprops.
  TF_RETURN_IF_ERROR(Initialize());

  // Backward propagation.
  std::vector<Output> dy;
  while (!ready_.empty()) {
    // n has collected all gradients.
    Node* n = ready_.front();
    ready_.pop_front();

    // dy[i] is the sum of i-th output's backpropped gradients.
    const int num_y = n->num_outputs();
    dy.clear();
    dy.resize(num_y, {nullptr, 0});
    std::vector<int> no_grad_dy_indices;
    for (int i = 0; i < num_y; ++i) {
      TF_RETURN_IF_ERROR(SumGradients({n, i}, &dy[i]));
      if (dy[i] == NoGradient()) {
        no_grad_dy_indices.push_back(i);
      }
      auto iter = input_nodes_.find({n, i});
      if (iter != input_nodes_.end()) {
        // Return gradients for Output in 'grad_outputs_'.
        (*grad_outputs_)[iter->second] = dy[i];
      }
    }

    // Stop backprop if none of the inputs to `n` are in `backprops_'.
    bool stop_node = true;
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      if (backprops_.find({e->src(), e->src_output()}) != backprops_.end()) {
        stop_node = false;
        break;
      }
    }

    if (stop_node) {
      continue;
    }

    // Special case: if we find an exit node, process the associated while loop.
    // Note that ProcessWhileLoop() calls BackpropAlongEdge() if necessary
    // (which updates ready_), and we skip all the regular processing below
    // after calling it.
    if (n->IsExit()) {
      DCHECK_EQ(dy.size(), 1);
      TF_RETURN_IF_ERROR(ProcessWhileLoop(n, dy[0]));
      continue;
    }
    // All loop-specific control flow ops should have been handled above
    DCHECK(!n->IsEnter() && !n->IsNextIteration()) << n->DebugString();

    const int num_no_grad = no_grad_dy_indices.size();
    if (IsPrimitiveOpWithNoGrad(n->type_string()) || num_no_grad == num_y) {
      // No grad defined for this op, or all outputs returned 'NoGradient':
      // Backprop 'NoGradient' along the in edges.
      for (const Edge* e : n->in_edges()) {
        if (e->IsControlEdge()) continue;
        TF_RETURN_IF_ERROR(
            BackpropAlongEdge(NoGradient(), {e->src(), e->src_output()}));
      }
      continue;
    }

    if (num_no_grad > 0 && num_no_grad < num_y) {
      // The outputs of 'n' returned a mixture of valid gradients and
      // 'NoGradient'. Therefore, we need to add 'ZerosLike' nodes for each
      // 'NoGradient' output before we call the gradient function for 'n'.
      // TODO(andydavis) If static shapes are known, replace 'ZerosLike' with
      // zero-filled Constant node of appropriate shape.
      for (const int dy_index : no_grad_dy_indices) {
        dy[dy_index] = ops::ZerosLike(scope_, Output(n, dy_index));
      }
    }

    // TODO(andydavis) Add option to encapsulate grad function in
    // SymbolicGradientOp (as opposed to inlining into the graph).
    std::vector<Output> dx;
    TF_RETURN_IF_ERROR(CallGradFunction(Operation(n), dy, &dx));

    // Backprop along the in edges.
    // TODO(andydavis) Find cleaner way to map each grad output returned by
    // gradient function to the src node/output to which it should be
    // backpropped. Maybe grad functions can return a vector of Output pairs to
    // make this association explicit.
    for (const Edge* e : n->in_edges()) {
      if (e->IsControlEdge()) continue;
      size_t dx_index = e->dst_input();
      if (dx_index >= dx.size()) {
        return errors::Internal(
            "Invalid gradient output index: ", dx_index, " size: ", dx.size());
      }
      TF_RETURN_IF_ERROR(
          BackpropAlongEdge(dx[dx_index], {e->src(), e->src_output()}));
    }
  }

  // Check if any input nodes still have pending gradients and have not been
  // processed yet. This happens if not all outputs of a node are in 'inputs_'.
  std::unordered_map<Node*, int> requested_grads;
  for (const Output& nout : inputs_) {
    if (pending_[nout.node()->id()] > 0) {
      DCHECK_GT(nout.node()->num_outputs(), 1);
      int idx = input_nodes_[nout];
      DCHECK(((*grad_outputs_)[idx].node() == nullptr));
      TF_RETURN_IF_ERROR(SumGradients(nout, &(*grad_outputs_)[idx]));
      ++requested_grads[nout.node()];
    }
  }
  for (const auto& p : requested_grads) {
    int num_requested_inputs = p.first->num_outputs() - pending_[p.first->id()];
    CHECK_EQ(num_requested_inputs, p.second);
  }
  return Status::OK();
}

}  // namespace

Status AddSymbolicGradients(const Scope& scope,
                            const std::vector<Output>& outputs,
                            const std::vector<Output>& inputs,
                            const std::vector<Output>& grad_inputs,
                            std::vector<Output>* grad_outputs) {
  SymbolicGradientBuilder builder(scope, ops::GradOpRegistry::Global(), outputs,
                                  inputs, grad_inputs, grad_outputs);
  return builder.AddGradients();
}

Status AddSymbolicGradients(const Scope& scope,
                            const std::vector<Output>& outputs,
                            const std::vector<Output>& inputs,
                            std::vector<Output>* grad_outputs) {
  std::vector<Output> grad_inputs;
  grad_inputs.reserve(outputs.size());
  for (const Output& output : outputs) {
    grad_inputs.emplace_back(ops::OnesLike(scope, output));
  }
  return AddSymbolicGradients(scope, outputs, inputs, grad_inputs,
                              grad_outputs);
}

Output NoGradient() { return SymbolicGradientBuilder::NoGradient(); }

}  // end namespace tensorflow
