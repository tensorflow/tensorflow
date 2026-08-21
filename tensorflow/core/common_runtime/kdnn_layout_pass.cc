// Copyright 2026 The TensorFlow Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifdef KDNN_ENABLED  // Enable via --define=enable_kdnn=true in Bazel
// Note: KDNN is mutually exclusive with oneDNN/MKL. Do not enable both.

#include "tensorflow/core/common_runtime/kdnn_layout_pass.h"

#include <algorithm>
#include <functional>
#include <memory>
#include <queue>
#include <set>
#include <stack>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

#include "absl/base/call_once.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/optimization_registry.h"
#include "tensorflow/core/common_runtime/process_util.h"
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/graph/algorithm.h"
#include "tensorflow/core/graph/node_builder.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/util/tensor_format.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/util/port.h"

namespace tensorflow {

// Static instance.
KdnnConstStringsInfo kdnn_csinfo_ = {"Sigmoid"};

KdnnLayoutRewritePass::KdnnLayoutRewritePass()
    : num_intra_threads_(port::MaxParallelism()) {
  // Add rewrite info for KDNN operations
  // Only Sigmoid for now
  rinfo_.push_back({kdnn_csinfo_.sigmoid, "_KdnnSigmoid",
                    CopyAttrsAll, SigmoidRewrite, "KDNN Sigmoid"});
  // Add more as needed...
}

bool KdnnLayoutRewritePass::RunPass(std::unique_ptr<Graph>* g) {
  DCHECK(g);
  bool result = false;

  DumpGraph("Before running KdnnLayoutRewritePass", &**g);

  std::vector<Node*> order;
  GetReversePostOrder(**g, &order);
  for (Node* n : order) {
    if (!n->IsOp()) {
      continue;
    }

    for (const auto& ri : rinfo_) {
      if (n->type_string() != ri.op_name) {
        continue;
      }
      if (!ri.should_rewrite(n)) {
        continue;
      }

      VLOG(1) << "KdnnLayoutRewritePass: Rewriting node " << n->name()
              << " (" << n->type_string() << ") to " << ri.new_op_name;

      // Build new node with new op name and same inputs.
      NodeBuilder nb(n->name(), ri.new_op_name);
      nb.Device(n->def().device());

      std::vector<const Edge*> in_edges;
      for (const Edge* in_edge : n->in_edges()) {
        if (!in_edge->IsControlEdge()) {
          in_edges.push_back(in_edge);
        }
      }
      std::sort(in_edges.begin(), in_edges.end(),
                [](const Edge* a, const Edge* b) {
                  return a->dst_input() < b->dst_input();
                });
      for (const Edge* in_edge : in_edges) {
        nb.Input(in_edge->src(), in_edge->src_output());
      }

      ri.copy_attrs(n, &nb, false);

      Node* new_node = nullptr;
      Status s = nb.Finalize(&**g, &new_node);
      if (!s.ok()) {
        VLOG(1) << "KdnnLayoutRewritePass: Failed to finalize node "
                << n->name() << ": " << s;
        continue;
      }

      new_node->set_assigned_device_name(n->assigned_device_name());

      std::unordered_set<Node*> unique_nodes;
      for (const Edge* in_edge : n->in_edges()) {
        if (in_edge->IsControlEdge()) {
          auto insert_result = unique_nodes.insert(in_edge->src());
          if (insert_result.second) {
            (*g)->AddControlEdge(in_edge->src(), new_node, true);
          }
        }
      }
      unique_nodes.clear();

      for (const Edge* out_edge : n->out_edges()) {
        if (out_edge->IsControlEdge()) {
          auto insert_result = unique_nodes.insert(out_edge->dst());
          if (insert_result.second) {
            (*g)->AddControlEdge(new_node, out_edge->dst(), true);
          }
        } else {
          const Edge* new_edge =
              (*g)->AddEdge(new_node, out_edge->src_output(), out_edge->dst(),
                            out_edge->dst_input());
          DCHECK(new_edge);
        }
      }

      (*g)->RemoveNode(n);
      result = true;
      break;
    }
  }

  DumpGraph("After running KdnnLayoutRewritePass", &**g);

  return result;
}

Status KdnnLayoutRewritePass::Run(const GraphOptimizationPassOptions& options) {
  if (options.graph == nullptr && options.partition_graphs == nullptr) {
    return OkStatus();
  }

  if (options.session_options != nullptr) {
    num_intra_threads_ =
        options.session_options->config.intra_op_parallelism_threads();
  }

  auto process_graph = [&](std::unique_ptr<Graph>* g) {
    RunPass(g);
  };

  if (options.partition_graphs != nullptr) {
    for (auto& pg : *options.partition_graphs) {
      process_graph(&pg.second);
    }
  } else {
    process_graph(options.graph);
  }

  return OkStatus();
}

void KdnnLayoutRewritePass::CopyAttrsAll(const Node* old_node,
                                         NodeBuilder* nb,
                                         bool change_format) {
  AttrSlice attr_list(old_node->def());
  for (auto iter = attr_list.begin(); iter != attr_list.end(); ++iter) {
    const std::string& name = iter->first;
    const AttrValue& attr = iter->second;
    nb->Attr(name, attr);
  }
}

bool KdnnLayoutRewritePass::SigmoidRewrite(const Node* n) {
  // Check if input and output data types are float
  if (IsKDNNEnabled() && !n->input_types().empty() &&
      n->input_types()[0] == DT_FLOAT &&
      !n->output_types().empty() && n->output_types()[0] == DT_FLOAT) {
    return true;
  }
  return false;
}


// Register the pass
REGISTER_OPTIMIZATION(OptimizationPassRegistry::POST_PARTITIONING, 0,
                      KdnnLayoutRewritePass);

}  // namespace tensorflow

#endif  // KDNN_ENABLED
