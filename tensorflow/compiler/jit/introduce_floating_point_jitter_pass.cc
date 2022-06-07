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

#include "tensorflow/compiler/jit/introduce_floating_point_jitter_pass.h"
#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "tensorflow/cc/framework/scope_internal.h"
#include "tensorflow/cc/ops/const_op.h"
#include "tensorflow/cc/ops/math_ops.h"
#include "tensorflow/compiler/jit/flags.h"
#include "tensorflow/core/graph/tensor_id.h"

namespace tensorflow {
namespace {
std::vector<std::pair<Node*, std::vector<int>>> GetNodesToModify(
    const Graph& g, absl::Span<const string> tensor_names) {
  absl::flat_hash_map<string, Node*> name_to_node;
  for (Node* n : g.op_nodes()) {
    name_to_node[n->name()] = n;
  }

  absl::flat_hash_map<Node*, std::vector<int>> nodes_to_modify_map;

  for (const string& tensor_name : tensor_names) {
    TensorId tensor_id = ParseTensorName(tensor_name);
    auto it = name_to_node.find(tensor_id.node());
    DCHECK(it != name_to_node.end());
    nodes_to_modify_map[it->second].push_back(tensor_id.index());
  }

  std::vector<std::pair<Node*, std::vector<int>>> nodes_to_modify;
  absl::c_copy(nodes_to_modify_map, std::back_inserter(nodes_to_modify));

  absl::c_sort(nodes_to_modify,
               [](const std::pair<Node*, std::vector<int>>& a,
                  const std::pair<Node*, std::vector<int>>& b) {
                 return a.first->id() < b.first->id();
               });

  for (auto& p : nodes_to_modify) {
    absl::c_sort(p.second);
    p.second.erase(std::unique(p.second.begin(), p.second.end()),
                   p.second.end());
  }

  return nodes_to_modify;
}

Status IntroduceJitterToTensor(
    Graph* g, Node* n, int oidx, float jitter_amount,
    absl::flat_hash_map<std::pair<DataType, Node*>, Output>*
        node_to_jitter_constant) {
  std::vector<const Edge*> edges_to_update;
  absl::c_copy_if(n->out_edges(), std::back_inserter(edges_to_update),
                  [&](const Edge* e) { return e->src_output() == oidx; });

  if (edges_to_update.empty()) {
    VLOG(1) << "No users for " << TensorId(n->name(), oidx).ToString();
    return OkStatus();
  }

  VLOG(1) << "Updating " << edges_to_update.size() << " users for  "
          << TensorId(n->name(), oidx).ToString();

  Status status;
  Scope s = NewInternalScope(g, &status, /*refiner=*/nullptr)
                .NewSubScope(absl::StrCat(n->name(), "/jitter"));

  Output node_out(n, oidx);
  Output jitter_constant;
  DataType dtype = n->output_type(oidx);
  auto it = node_to_jitter_constant->find({dtype, n});
  if (it == node_to_jitter_constant->end()) {
    Tensor constant_tensor;
    if (dtype == DT_FLOAT) {
      constant_tensor = Tensor(static_cast<float>(jitter_amount));
    } else if (dtype == DT_HALF) {
      constant_tensor = Tensor(Eigen::half(jitter_amount));
    } else {
      return errors::Unimplemented("Only float and half are supported");
    }

    jitter_constant =
        ops::Const(s.WithOpName("jitter_amount"), constant_tensor);
    (*node_to_jitter_constant)[{dtype, n}] = jitter_constant;
  } else {
    jitter_constant = it->second;
  }

  Output jittered_output =
      ops::Add(s.NewSubScope(absl::StrCat(oidx)).WithOpName("jittered_output"),
               jitter_constant, node_out);

  TF_RETURN_IF_ERROR(status);

  for (const Edge* e : edges_to_update) {
    VLOG(3) << "Updating " << e->dst()->name();
    TF_RETURN_IF_ERROR(
        g->UpdateEdge(jittered_output.node(), 0, e->dst(), e->dst_input()));
  }

  // Add a control edge to make sure that the two inputs to jittered_output are
  // from the same frame.
  g->AddControlEdge(n, jitter_constant.node());

  return OkStatus();
}
}  // namespace

Status IntroduceFloatingPointJitter(Graph* graph,
                                    absl::Span<string const> tensor_names,
                                    float jitter_amount) {
  if (tensor_names.empty()) {
    VLOG(3) << "Nothing to do";
    return OkStatus();
  }

  std::vector<std::pair<Node*, std::vector<int>>> nodes_to_modify =
      GetNodesToModify(*graph, tensor_names);

  absl::flat_hash_map<std::pair<DataType, Node*>, Output>
      node_to_jitter_constant;
  for (const auto& p : nodes_to_modify) {
    for (int oidx : p.second) {
      TF_RETURN_IF_ERROR(IntroduceJitterToTensor(
          graph, p.first, oidx, jitter_amount, &node_to_jitter_constant));
    }
  }

  return OkStatus();
}

Status IntroduceFloatingPointJitterPass::Run(
    const GraphOptimizationPassOptions& options) {
  const IntroduceFloatingPointJitterPassFlags& flags =
      GetIntroduceFloatingPointJitterPassFlags();

  return IntroduceFloatingPointJitter(options.graph->get(), flags.tensor_names,
                                      flags.jitter_amount);
}
}  // namespace tensorflow
