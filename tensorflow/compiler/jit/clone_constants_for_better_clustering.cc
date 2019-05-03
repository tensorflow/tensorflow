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

#include "tensorflow/compiler/jit/xla_cluster_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/framework/tensor.pb.h"

namespace tensorflow {

using se::port::StatusOr;

string CloneConstantsForBetterClusteringPass::GenerateUniqueName(
    const absl::flat_hash_set<string>& name_set, absl::string_view prefix) {
  string candidate;
  do {
    candidate = absl::StrCat(prefix, "/clone_", unique_name_counter_++);
  } while (name_set.contains(candidate));
  return candidate;
}

StatusOr<Node*> CloneConstantsForBetterClusteringPass::CloneNode(
    Graph* g, const absl::flat_hash_set<string>& name_set, Node* n) {
  NodeDef new_in_def = n->def();
  new_in_def.clear_input();
  new_in_def.set_name(GenerateUniqueName(name_set, new_in_def.name()));
  Status s;
  Node* new_in = g->AddNode(new_in_def, &s);
  TF_RETURN_IF_ERROR(s);

  for (const Edge* e : n->in_edges()) {
    if (e->IsControlEdge()) {
      g->AddControlEdge(e->src(), new_in);
    } else {
      g->AddEdge(e->src(), e->src_output(), new_in, e->dst_input());
    }
  }

  new_in->set_assigned_device_name(n->assigned_device_name());
  return new_in;
}

namespace {
// We only clone host constants for now since we want to avoid increasing memory
// pressure on GPUs.
StatusOr<bool> IsSmallHostConstant(Node* n) {
  if (!n->IsConstant()) {
    return false;
  }

  DeviceNameUtils::ParsedName parsed;
  TF_RET_CHECK(
      DeviceNameUtils::ParseFullName(n->assigned_device_name(), &parsed));
  if (parsed.type != DEVICE_CPU) {
    return false;
  }

  const TensorProto* proto = nullptr;
  TF_RETURN_IF_ERROR(GetNodeAttr(n->def(), "value", &proto));

  // TODO(sanjoy): It may make sense to combine this threshold with XLA's "large
  // constant" threshold, if there is one.
  const int kSmallTensorThreshold = 16;
  int64 total_elements = 1;
  for (const auto& dim : proto->tensor_shape().dim()) {
    if (dim.size() < 0) {
      return errors::Internal("Unknown dimension size in constant tensor ",
                              n->name());
    }
    total_elements *= dim.size();
  }
  return total_elements < kSmallTensorThreshold;
}

bool IsInPlaceOp(absl::string_view op_name) {
  return op_name == "InplaceUpdate" || op_name == "InplaceAdd" ||
         op_name == "InplaceSub";
}
}  // namespace

Status CloneConstantsForBetterClusteringPass::CloneSmallHostConstantInputs(
    Graph* g, const absl::flat_hash_set<string>& name_set, Node* n) {
  std::vector<const Edge*> in_edges;
  absl::c_copy(n->in_edges(), std::back_inserter(in_edges));
  for (const Edge* e : in_edges) {
    Node* input = e->src();
    TF_ASSIGN_OR_RETURN(bool is_small_host_constant,
                        IsSmallHostConstant(input));
    if (is_small_host_constant && input->out_edges().size() != 1) {
      VLOG(2) << "Cloning small host constant " << input->name();
      TF_ASSIGN_OR_RETURN(Node* const input_cloned,
                          CloneNode(g, name_set, input));
      if (e->IsControlEdge()) {
        g->AddControlEdge(input_cloned, e->dst());
      } else {
        int dst_input = e->dst_input();
        TF_RET_CHECK(e->src_output() == 0)
            << "expected constant to have exactly one non-control output, but "
               "found output index = "
            << e->src_output();
        g->RemoveEdge(e);
        g->AddEdge(input_cloned, 0, n, dst_input);
      }
    }
  }
  return Status::OK();
}

Status CloneConstantsForBetterClusteringPass::Run(
    const GraphOptimizationPassOptions& options) {
  if (GetGlobalJitLevelForGraph(options) == OptimizerOptions::OFF) {
    return Status::OK();
  }

  Graph* g = options.graph->get();
  absl::flat_hash_set<string> name_set;
  absl::c_transform(g->nodes(), std::inserter(name_set, name_set.begin()),
                    [](Node* n) { return n->name(); });
  std::vector<Node*> nodes;
  for (Node* n : g->nodes()) {
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
      return Status::OK();
    }
    nodes.push_back(n);
  }

  // Iterate over a copy of the nodes to avoid iterating over g->nodes() while
  // creating more nodes.
  for (Node* n : nodes) {
    TF_RETURN_IF_ERROR(CloneSmallHostConstantInputs(g, name_set, n));
  }
  return Status::OK();
}

}  // namespace tensorflow
