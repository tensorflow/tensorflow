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

#include "tensorflow/core/common_runtime/type_inference.h"

#include <functional>
#include <list>
#include <queue>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/core/framework/full_type.pb.h"
#include "tensorflow/core/framework/full_type_util.h"
#include "tensorflow/core/framework/op_def_builder.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/util/dump_graph.h"

namespace tensorflow {

namespace {

int MAX_VISITS_PER_NODE = 3;

typedef absl::flat_hash_map<int, std::reference_wrapper<TypeInferenceFn const>>
    ForwardInferMap;
typedef absl::flat_hash_map<
    int, std::pair<int, std::reference_wrapper<TypeInferenceFn const>>>
    ReverseInferMap;

bool all_sources_closed(const Node& n, const absl::flat_hash_set<int>& closed,
                        const ForwardInferMap& forward,
                        const ReverseInferMap& reverse) {
  for (const auto& e : n.out_edges()) {
    if (e->IsControlEdge()) {
      continue;
    }
    int dst_id = e->dst()->id();
    if (reverse.contains(dst_id) && !closed.contains(dst_id)) {
      return false;
    }
  }
  if (forward.contains(n.id())) {
    for (const auto& e : n.in_edges()) {
      if (e->IsControlEdge()) {
        continue;
      }
      if (!closed.contains(e->src()->id())) {
        return false;
      }
    }
  }
  return true;
}

std::vector<std::reference_wrapper<const FullTypeDef>> input_types(
    const Node& n) {
  static FullTypeDef* no_type = new FullTypeDef();

  std::vector<std::reference_wrapper<const FullTypeDef>> input_types;
  for (const auto& in_edge : n.in_edges()) {
    if (in_edge->IsControlEdge()) {
      continue;
    }
    input_types.push_back(*no_type);
  }
  for (const auto& in_edge : n.in_edges()) {
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
  return input_types;
}

Status update_inferred_type(Node* target, const FullTypeDef& t, bool& updated) {
  if (t.type_id() == TFT_UNSET) {
    VLOG(3) << "  " << target->name() << " no inferred type";
    return OkStatus();
  }

  if (target->def().has_experimental_type()) {
    const auto existing = target->def().experimental_type();
    if (full_type::IsSubtype(existing, t)) {
      VLOG(3) << "  " << target->name() << " no new type info";
      return OkStatus();
    } else if (!full_type::IsSubtype(t, existing)) {
      // The only allowable type mismatches are those which would further
      // specialize the existing type.
      return Status(
          error::INVALID_ARGUMENT,
          absl::StrCat("type mismatch for node '", target->name(),
                       "': expected a subtype of:\n", existing.DebugString(),
                       "\n  got:\n", t.DebugString(), "\n  "));
    }
  }

  *(target->mutable_def()->mutable_experimental_type()) = t;
  updated = true;
  VLOG(3) << "  " << target->name() << " updated";
  return OkStatus();
}

StatusOr<FullTypeDef> run_inference(const string& fn_name,
                                    const TypeRefVector& in_types) {
  // TODO(b/224776031): Things remaining to implement:
  //  * look up function by name
  //  * execute pass on its graph
  //  * get retnode types
  //  * return them here
  return OkStatus();
}

}  // namespace

Status TypeInferencePass::Run(
    const GraphOptimizationPassOptions& options) {
  VLOG(1) << "TypeInferencePass::Run";

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

  // Cache type inference functions, to avoid repeated flib_def lookups.
  ForwardInferMap forward;
  ReverseInferMap reverse;
  for (Node* n : g->nodes()) {
    VLOG(4) << "\n  node: " << n->def().DebugString()
            << "\n  op def: " << n->op_def().DebugString();
    const OpRegistrationData* reg;
    TF_RETURN_IF_ERROR(flib_def->LookUp(n->op_def().name(), &reg));
    if (reg->fwd_type_fn != nullptr) {
      forward.emplace(n->id(), reg->fwd_type_fn);
    }
    if (reg->rev_type_fn != nullptr) {
      reverse.emplace(n->id(), std::make_pair(reg->rev_type_input,
                                              std::cref(reg->rev_type_fn)));
    }
  }

  auto infer_forward = [&forward](Node* n, bool& updated) {
    if (!forward.contains(n->id())) {
      return OkStatus();
    }
    VLOG(4) << "  " << n->name() << " has forward function";

    auto in_types = input_types(*n);
    const auto& infer_ret = forward.at(n->id())(in_types, run_inference);

    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        infer_ret.status(),
        absl::StrCat("while inferring type of node '", n->name(), "'"));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        update_inferred_type(n, *infer_ret, updated),
        "while updating its output type.");

    return OkStatus();
  };

  auto infer_reverse = [&reverse](Node* n, bool& updated) {
    if (!reverse.contains(n->id())) {
      return OkStatus();
    }
    VLOG(4) << "  " << n->name() << " has reverse function";

    auto in_types = input_types(*n);
    auto rev_idx_and_fn = reverse.at(n->id());
    const auto& infer_ret = rev_idx_and_fn.second(in_types, run_inference);

    const Edge* e;
    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        n->input_edge(rev_idx_and_fn.first, &e),
        absl::StrCat("while querying input ", rev_idx_and_fn.first, " of '",
                     n->name(), "'"));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        infer_ret.status(),
        absl::StrCat("while inferring type of node '", e->src()->name(),
                     "' via '", n->name(), "'"));

    TF_RETURN_WITH_CONTEXT_IF_ERROR(
        update_inferred_type(e->src(), *infer_ret, updated),
        absl::StrCat("while updating its output type inferred from '",
                     n->name(), ","));

    return OkStatus();
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
      VLOG(3) << "  visiting " << n->name();
      visits++;
      visit_count[nid]++;
      DCHECK(!closed.contains(nid));

      bool updated = false;
      TF_RETURN_IF_ERROR(infer_forward(n, updated));
      TF_RETURN_IF_ERROR(infer_reverse(n, updated));

      VLOG(4) << "  done " << n->def().DebugString();

      queue.pop_front();
      in_queue.erase(nid);
      open.erase(nid);

      // Update the graph to fixed point, with iterations limited
      // by MAX_VISITS_PER_NODE.
      if (visit_count.at(nid) >= MAX_VISITS_PER_NODE) {
        VLOG(3) << "  closing " << n->name() << " - visit limit reached";
        closed.emplace(nid);
      } else if (all_sources_closed(*n, closed, forward, reverse)) {
        VLOG(3) << "  closing " << n->name() << " - all sources closed";
        closed.emplace(nid);
      }

      for (const auto& out_edge : n->out_edges()) {
        if (out_edge->IsControlEdge()) {
          continue;
        }
        Node* c = out_edge->dst();
        int cid = c->id();
        if (closed.contains(cid) || in_queue.contains(cid)) {
          continue;
        }
        if (updated || all_sources_closed(*c, closed, forward, reverse)) {
          queue.emplace_back(cid);
          in_queue.emplace(cid);
        }
      }
      if (updated && reverse.contains(nid)) {
        const Edge* e;
        TF_RETURN_IF_ERROR(n->input_edge(reverse.at(nid).first, &e));
        Node* c = e->src();
        int cid = c->id();
        if (!closed.contains(cid) && !in_queue.contains(cid)) {
          queue.emplace_back(cid);
          in_queue.emplace(cid);
        }
      }
    }

    VLOG(2) << "Done iteration " << i << ", " << closed.size()
            << " nodes closed";

    if (open.empty()) {
      VLOG(1) << "Finished after " << i + 1 << " iterations; done "
              << closed.size() << " of " << g->num_nodes() << " nodes in "
              << visits << " visits";
      break;
    } else {
      queue.emplace_back(*(open.begin()));
    }
  }

  if (VLOG_IS_ON(1)) {
    DumpGraphToFile("forward_type_inference_after", *g, flib_def);
  }

  return OkStatus();
}

Status WeakTypeInferencePass::Run(
    const GraphOptimizationPassOptions& options) {
  TypeInferencePass pass;
  const auto& pass_status = pass.Run(options);
  if (!pass_status.ok()) {
    LOG_FIRST_N(WARNING, 1)
        << "Type inference failed. This indicates an "
           "invalid graph that escaped type checking. Error message: "
        << pass_status.ToString();
  }
  return OkStatus();
}

// Note: This needs to run last because Placer needs it.
REGISTER_OPTIMIZATION(OptimizationPassRegistry::PRE_PLACEMENT, 99999,
                      WeakTypeInferencePass);

}  // namespace tensorflow
