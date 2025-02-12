/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/jit/clone_constants_for_better_clustering.h"

#include <string>

#include "absl/algorithm/container.h"
#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "xla/status_macros.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

using tsl::StatusOr;

class CloneConstantsForBetterClusteringPassImpl {
 public:
  explicit CloneConstantsForBetterClusteringPassImpl(Graph* graph)
      : graph_(graph), unique_name_counter_(0) {}
  absl::Status Run();

 private:
  absl::Status CloneSmallConstantInputs(
      const absl::flat_hash_set<string>& name_set, Node* n);
  string GenerateUniqueName(const absl::flat_hash_set<string>& name_set,
                            absl::string_view prefix);
  absl::StatusOr<Node*> CloneNode(const absl::flat_hash_set<string>& name_set,
                                  Node* n);

  Graph* graph_;
  int unique_name_counter_;
};

string CloneConstantsForBetterClusteringPassImpl::GenerateUniqueName(
    const absl::flat_hash_set<string>& name_set, absl::string_view prefix) {
  string candidate;
  do {
    candidate = absl::StrCat(prefix, "/clone_", unique_name_counter_++);
  } while (name_set.contains(candidate));
  return candidate;
}

absl::StatusOr<Node*> CloneConstantsForBetterClusteringPassImpl::CloneNode(
    const absl::flat_hash_set<string>& name_set, Node* n) {
  NodeDef new_in_def = n->def();
  new_in_def.clear_input();
  new_in_def.set_name(GenerateUniqueName(name_set, new_in_def.name()));
  TF_ASSIGN_OR_RETURN(Node * new_in, graph_->AddNode(new_in_def));

  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      graph_->AddControlEdge(e->src(), new_in);
    } else {
      graph_->AddEdge(e->src(), e->src_output(), new_in, e->dst_input());
    }
  }

  new_in->set_assigned_device_name(n->assigned_device_name());
  return new_in;
}

namespace {
absl::StatusOr<bool> IsConstantSmall(Node* n) {
  const TensorProto* proto = nullptr;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "value", &proto));

  int64_t total_elements = 1;
  for (const auto& dim : proto->tensor_shape().dim()) {
    if (dim.size() < 0) {
      return errors::Internal("Unknown dimension size in constant tensor ",
                              n->name());
    }
    total_elements *= dim.size();
  }

  // TODO(sanjoy): It may make sense to combine this threshold with XLA's "large
  // constant" threshold, if there is one.
  const int kSmallTensorThreshold = 16;
  return total_elements < kSmallTensorThreshold;
}

// We only clone small constants since we want to avoid increasing memory
// pressure on GPUs.
absl::StatusOr<bool> IsSmallConstant(Node* n) {
  if (!n->IsConstant()) {
    return false;
  }

  return IsConstantSmall(n);
}

bool IsInPlaceOp(absl::string_view op_name) {
  return op_name == "InplaceUpdate" || op_name == "InplaceAdd" ||
         op_name == "InplaceSub";
}
}  // namespace

absl::Status
CloneConstantsForBetterClusteringPassImpl::CloneSmallConstantInputs(
    const absl::flat_hash_set<string>& name_set, Node* n) {
  std::vector<const Edge*> in_edges;
  // Get the edges and sort them so we clone in a deterministic order.
  absl::c_copy(n->in_edges(), std::back_inserter(in_edges));
  absl::c_stable_sort(in_edges, [](const Edge* e1, const Edge* e2) {
    return e1->id() < e2->id();
  });
  for (const Edge* e : in_edges) {
    Node* input = e->src();
    TF_ASSIGN_OR_RETURN(bool is_small_constant, IsSmallConstant(input));
    if (is_small_constant && input->out_edges().size() != 1) {
      VLOG(2) << "Cloning small constant " << input->name();
      TF_ASSIGN_OR_RETURN(Node* const input_cloned, CloneNode(name_set, input));
      if (e->IsControlEdge()) {
        graph_->AddControlEdge(input_cloned, e->dst());
      } else {
        int dst_input = e->dst_input();
        TF_RET_CHECK(e->src_output() == 0)
            << "expected constant to have exactly one non-control output, but "
               "found output index = "
            << e->src_output();
        graph_->RemoveEdge(e);
        graph_->AddEdge(input_cloned, 0, n, dst_input);
      }
    }
  }
  return absl::OkStatus();
}

absl::Status CloneConstantsForBetterClusteringPassImpl::Run() {
  absl::flat_hash_set<string> name_set;
  absl::c_transform(graph_->nodes(), std::inserter(name_set, name_set.begin()),
                    [](Node* n) { return n->name(); });
  std::vector<Node*> nodes;
  for (Node* n : graph_->nodes()) {
    // We rely on the immutability of Tensors to safely clone Const operations.
    // However, "in place" ops do not respect the immutability of Tensors so we
    // avoid this transformation when such ops are present in the graph.
    //
    // In-place operations are problematic because they break the semantic
    // illusion that tensorflow::Tensor instances are immutable.  For instance
    // if we have the following graph:
    //
    // digraph {
    //   SRC -> Const
    //   SRC -> I
    //   SRC -> V
    //   Const -> Identity
    //   Const -> InplaceAdd [label="x"]
    //   I -> InplaceAdd [label="i"]
    //   V -> InplaceAdd [label="v"]
    //   InplaceAdd -> Identity [style=dotted]
    // }
    //
    // then the value produced by `Identity` is Const+I*V since InplaceAdd
    // modifies the tensor in place.  However, if we clone `Const` and turn the
    // graph into:
    //
    // digraph {
    //   SRC -> "Const/clone_1"
    //   SRC -> "Const/clone_2"
    //   SRC -> I
    //   SRC -> V
    //   "Const/clone_1" -> Identity
    //   "Const/clone_2" -> InplaceAdd [label="x"]
    //   I -> InplaceAdd [label="i"]
    //   V -> InplaceAdd [label="v"]
    //   InplaceAdd -> Identity [style=dotted]
    // }
    //
    // then `Identity` no longer produces Const+I*V because the InplaceAdd
    // operation only modifies Const/clone_2 in place.

    if (IsInPlaceOp(n->type_string())) {
      return absl::OkStatus();
    }
    nodes.push_back(n);
  }

  // Iterate over a copy of the nodes to avoid iterating over g->nodes() while
  // creating more nodes.
  for (Node* n : nodes) {
    TF_RETURN_IF_ERROR(CloneSmallConstantInputs(name_set, n));
  }
  return absl::OkStatus();
}

absl::Status CloneConstantsForBetterClusteringPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (GetGlobalJitLevelForGraph(options) == OptimizerOptions::OFF) {
    return absl::OkStatus();
  }

  Graph* g = options.graph->get();

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("before_clone_constants_for_better_clustering", *g);
  }

  TF_RETURN_IF_ERROR(CloneConstantsForBetterClusteringPassImpl{g}.Run());

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("after_clone_constants_for_better_clustering", *g);
  }

  return absl::OkStatus();
}

}  // namespace tensorflow
