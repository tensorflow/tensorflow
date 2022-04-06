/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/common_runtime/forward_type_inference.h"

#include <functional>
#include <queue>
#include <string>
#include <string_view>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

int MAX_VISITS_PER_NODE = 2;

bool all_inputs_closed(const Node& n, const absl::flat_hash_set<int>& closed) {
  for (const auto& e : n.in_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    if (!closed.contains(e->src()->id())) {
      return false;
    }
  }
  return true;
}

}  // namespace

Status ForwardTypeInferencePass::Run(
    const GraphOptimizationPassOptions& options) {
  VLOG(1) << "ForwardTypeInferencePass::Run";

  DCHECK(options.graph != nullptr);
  Graph* g = options.graph->get();
  DCHECK(g != nullptr);
  FunctionLibraryDefinition* flib_def = options.flib_def;
  DCHECK(flib_def != nullptr);

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("forward_type_inference_before", *g, flib_def);
  }

  for (Node* n : g->nodes()) {
    // TODO(mdan): Needed?
    n->UpdateProperties();
  }

  static FullTypeDef* no_type = new FullTypeDef();

  auto process_node = [&flib_def](Node* n, bool& updated) {
    VLOG(3) << "  processing " << n->name();
    VLOG(4) << "\n  node: " << n->def().DebugString()
            << "\n  op def: " << n->op_def().DebugString();
    const OpRegistrationData* reg;
    TF_RETURN_IF_ERROR(flib_def->LookUp(n->op_def().name(), &reg));

    if (reg->fwd_type_fn == nullptr) {
      VLOG(4) << "  " << n->name() << " no type inference function";
      return Status::OK();
    }

    std::vector<std::reference_wrapper<const FullTypeDef>> input_types;
    for (const auto& in_edge : n->in_edges()) {
      if (in_edge->IsControlEdge()) {
        continue;
      }
      input_types.push_back(*no_type);
    }
    for (const auto& in_edge : n->in_edges()) {
      if (in_edge->IsControlEdge()) {
        continue;
      }
      VLOG(5) << "  in edge: " << in_edge->DebugString();
      NodeDef* ndef = in_edge->src()->mutable_def();
      if (ndef->has_experimental_type()) {
        const auto& t = ndef->experimental_type();
        if (t.type_id() != TFT_UNSET) {
          DCHECK(t.type_id() == TFT_PRODUCT) << ndef->DebugString();
          DCHECK(t.args_size() > in_edge->src_output()) << ndef->DebugString();
          input_types.at(in_edge->dst_input()) = t.args(in_edge->src_output());
        }
      }
    }

    // TODO(b/224775462): Populate with types from function references.
    TypeRefMap type_vars;

    const auto& infer_ret = reg->fwd_type_fn(input_types, type_vars);
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        infer_ret.status(), "while inferring type of node '", n->name(), "'");
    const auto& infer_type = *infer_ret;

    if (infer_type.type_id() == TFT_UNSET) {
      VLOG(3) << "  " << n->name() << " no new type information";
      return Status::OK();
    }

    if (!n->def().has_experimental_type() ||
        !full_type::IsEqual(n->def().experimental_type(), infer_type)) {
      *(n->mutable_def()->mutable_experimental_type()) = infer_type;
      updated = true;
      VLOG(3) << "  " << n->name() << " updated";
    } else {
      VLOG(3) << "  " << n->name() << " same type after inference";
    }

    return Status::OK();
  };

  std::list<int> queue;
  absl::flat_hash_set<int> in_queue;
  absl::flat_hash_map<int, int> visit_count;
  // Open nodes. A node is open if it has never been visited.
  absl::flat_hash_set<int> open;
  // Closed nodes. A closed node will never be visited again.
  absl::flat_hash_set<int> closed;

  // Upper bound. Worst-case is a cycle in which no nodes have type info,
  // case in which there will be max_passes iterations, each visiting one node.
  int max_passes = g->num_nodes();

  int visits = 0;

  // Start with niladic nodes. If none exist, a random one will be selected at
  // the end of first iteration.
  for (Node* n : g->nodes()) {
    const int nid = n->id();
    bool niladic = true;
    for (const auto& e : n->in_edges()) {
      if (!e->IsControlEdge()) {
        niladic = false;
        break;
      }
    }
    if (niladic) {
      queue.emplace_back(nid);
      in_queue.emplace(nid);
    }
    open.emplace(nid);
    visit_count.emplace(nid, 0);
  }

  for (int i = 0; i < max_passes; i++) {
    VLOG(2) << "Iteration " << i << ", " << queue.size() << " nodes in queue";

    while (!queue.empty()) {
      int nid = queue.front();
      Node* n = g->FindNodeId(nid);
      bool updated = false;
      VLOG(3) << "  visiting " << n->name();
      visits++;
      visit_count[nid]++;

      TF_RETURN_IF_ERROR(process_node(n, updated));
      VLOG(4) << "  done " << n->def().DebugString();

      queue.pop_front();
      in_queue.erase(nid);
      open.erase(nid);

      if (all_inputs_closed(*n, closed)) {
        VLOG(3) << "  closing " << n->name();
        closed.emplace(nid);
      }

      for (const auto& out_edge : n->out_edges()) {
        if (out_edge->IsControlEdge()) {
          continue;
        }
        Node* c = out_edge->dst();
        int cid = c->id();
        // Update the graph to fixed point, with iterations limited
        // by MAX_VISITS_PER_NODE.
        if (closed.contains(cid) || in_queue.contains(cid) ||
            visit_count.at(cid) >= MAX_VISITS_PER_NODE) {
          continue;
        }
        if (all_inputs_closed(*c, closed) || updated) {
          queue.emplace_back(cid);
          in_queue.emplace(cid);
        }
      }
    }

    VLOG(2) << "Done iteration " << i << ", " << closed.size()
            << " nodes closed";

    if (open.empty()) {
      VLOG(1) << "Finished after " << i << " iterations; done " << closed.size()
              << " of " << g->num_nodes() << " nodes in " << visits
              << " visits";
      break;
    } else {
      queue.emplace_back(*(open.begin()));
    }
  }

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("forward_type_inference_after", *g, flib_def);
  }

  return Status::OK();
}

Status WeakForwardTypeInferencePass::Run(
    const GraphOptimizationPassOptions& options) {
  ForwardTypeInferencePass pass;
  const auto& pass_status = pass.Run(options);
  if (!pass_status.ok()) {
    LOG_FIRST_N(WARNING, 1)
        << "Type inference failed. This indicates an "
           "invalid graph that escaped type checking. Error message: "
        << pass_status.ToString();
  }
  return Status::OK();
}

// Note: This needs to run last because Placer needs it.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 99999,
                      WeakForwardTypeInferencePass);

}  // namespace tensorflow
