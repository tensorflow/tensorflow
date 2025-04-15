/* Copyright 2017 The OpenXLA Authors.

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

#include "xla/service/call_graph.h"

#include <deque>
#include <memory>
#include <queue>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/map_util.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"

namespace xla {

using absl::StrAppendFormat;
using absl::StrCat;

std::string CallContextToString(CallContext context) {
  switch (context) {
    case CallContext::kNone:
      return "kNone";
    case CallContext::kControlFlow:
      return "kControlFlow";
    case CallContext::kEmbedded:
      return "kEmbedded";
    case CallContext::kBoth:
      return "kBoth";
  }
}

std::ostream& operator<<(std::ostream& out, const CallContext& context) {
  out << CallContextToString(context);
  return out;
}

CallContext GetInstructionCallContext(HloOpcode opcode) {
  switch (opcode) {
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
    case HloOpcode::kWhile:
    case HloOpcode::kAsyncStart:
    case HloOpcode::kAsyncUpdate:
    case HloOpcode::kAsyncDone:
      return CallContext::kControlFlow;
    case HloOpcode::kAllReduce:
    case HloOpcode::kReduceScatter:
    case HloOpcode::kAllReduceStart:
    case HloOpcode::kMap:
    case HloOpcode::kReduce:
    case HloOpcode::kReduceWindow:
    case HloOpcode::kScatter:
    case HloOpcode::kSelectAndScatter:
    case HloOpcode::kSort:
    case HloOpcode::kTopK:
    case HloOpcode::kFusion:
    case HloOpcode::kCustomCall:
      return CallContext::kEmbedded;
    default:
      return CallContext::kNone;
  }
}

std::string CallSite::ToString() const {
  return StrCat(
      instruction()->name(), " calls in context ",
      CallContextToString(context()), ": ",
      absl::StrJoin(called_computations(), ", ",
                    [](std::string* out, const HloComputation* computation) {
                      absl::StrAppend(out, computation->name());
                    }));
}

CallGraphNode::CallGraphNode(HloComputation* computation)
    : computation_(computation) {}

const CallSite* CallGraphNode::GetCallSite(
    const HloInstruction* instruction) const {
  auto it = callsite_instructions_.find(instruction);
  if (it == callsite_instructions_.end()) {
    return nullptr;
  }
  return &callsites_[it->second];
}

absl::string_view CallGraphNode::ToString() const {
  return computation_->name();
}

void CallGraphNode::AddCallerCallSite(const CallSite& caller_callsite) {
  caller_callsites_.push_back(caller_callsite);
  HloComputation* caller = caller_callsite.instruction()->parent();
  if (!ContainsKey(caller_set_, caller)) {
    callers_.push_back(caller);
    caller_set_.insert(caller);
  }
}

void CallGraphNode::AddCallSiteForInstruction(
    HloInstruction* instruction,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  CHECK_EQ(instruction->parent(), computation());
  const CallContext context = GetInstructionCallContext(instruction->opcode());
  if (!instruction->called_computations().empty()) {
    CHECK(context == CallContext::kControlFlow ||
          context == CallContext::kEmbedded);
    callsite_instructions_.insert({instruction, callsites_.size()});
    callsites_.push_back(
        CallSite(instruction, instruction->called_computations(), context));
    // Update callee computations to include any new computations called by this
    // instruction.
    for (auto* callee : callsites_.back().called_computations()) {
      if (HloInstruction::IsThreadIncluded(callee->execution_thread(),
                                           execution_threads) &&
          !ContainsKey(callee_set_, callee)) {
        callees_.push_back(callee);
        callee_set_.insert(callee);
      }
    }
  }
}

CallGraph::CallGraph(
    const HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads)
    : module_(module), execution_threads_(execution_threads) {}

const CallGraphNode& CallGraph::GetNode(
    const HloComputation* computation) const {
  DCHECK(node_indices_.contains(computation));
  return nodes_[node_indices_.find(computation)->second];
}

CallGraphNode& CallGraph::GetNode(const HloComputation* computation) {
  DCHECK(node_indices_.contains(computation));
  return nodes_[node_indices_.find(computation)->second];
}

bool CallGraph::DominatesHelper(
    const HloComputation* a, const HloComputation* b,
    absl::flat_hash_set<const HloComputation*>* visited) const {
  if (a == b || ContainsKey(*visited, b)) {
    // The call graph is guaranteed to be acyclic so any previously visited node
    // we encounter was already determined to be dominated.
    return true;
  }

  const CallGraphNode& b_node = GetNode(b);
  if (b_node.callers().empty()) {
    // We reached a root node without hitting 'a'. 'a' does not dominate 'b'.
    return false;
  }

  // Walk up the callers of 'b' until we hit 'a' or a root node (no callers).
  visited->insert(b);
  for (const HloComputation* b_caller : b_node.callers()) {
    if (!DominatesHelper(a, b_caller, visited)) {
      return false;
    }
  }
  return true;
}

bool CallGraph::Dominates(const HloComputation* a,
                          const HloComputation* b) const {
  absl::flat_hash_set<const HloComputation*> visited;
  return DominatesHelper(a, b, &visited);
}

bool CallGraph::CanReach(const HloComputation* a,
                         const HloComputation* b) const {
  if (a == b) {
    return true;
  }

  const CallGraphNode& b_node = GetNode(b);
  for (const HloComputation* b_caller : b_node.callers()) {
    if (CanReach(a, b_caller)) {
      return true;
    }
  }
  return false;
}

namespace {
// Returns the call context of a computation which is called from contexts 'a'
// and 'b'.
CallContext UnionContexts(CallContext a, CallContext b) {
  if (a == CallContext::kNone) {
    return b;
  } else if (b == CallContext::kNone) {
    return a;
  } else if (a == b) {
    return a;
  } else {
    // Contexts are different and neither is kNone, i.e. one is kControlFlow and
    // the other is kEmbedded.
    return CallContext::kBoth;
  }
}

}  // namespace

void CallGraph::SetCallContexts() {
  std::queue<CallGraphNode*> worklist;

  // Initialize worklist with all roots of the call graph (computations without
  // callers).
  for (const HloComputation* computation :
       module_->computations(execution_threads_)) {
    CallGraphNode& node = GetNode(computation);
    if (node.callers().empty()) {
      node.set_context(CallContext::kControlFlow);
      worklist.push(&node);
    }
  }

  while (!worklist.empty()) {
    CallGraphNode* node = worklist.front();
    worklist.pop();

    for (const CallSite& callsite : node->callsites()) {
      for (const HloComputation* callee : callsite.called_computations()) {
        if (!HloInstruction::IsThreadIncluded(callee->execution_thread(),
                                              execution_threads_)) {
          continue;
        }
        CallGraphNode& callee_node = GetNode(callee);

        // Update context of callee computation based on the callsite and its
        // current context.
        CallContext context_to_add;
        if (callsite.context() == CallContext::kEmbedded) {
          context_to_add = CallContext::kEmbedded;
        } else {
          CHECK_EQ(callsite.context(), CallContext::kControlFlow);
          context_to_add = node->context();
        }
        CallContext new_context =
            UnionContexts(context_to_add, callee_node.context());

        if (new_context != callee_node.context()) {
          // Context of computation has been changed so add node to worklist.
          callee_node.set_context(new_context);
          worklist.push(&callee_node);
        }
      }
    }
  }

  // No node should have a kNone calling context.
  for (const HloComputation* computation :
       module_->computations(execution_threads_)) {
    CHECK_NE(GetNode(computation).context(), CallContext::kNone);
  }
}

void CallGraph::SetNodeDepths() {
  std::queue<CallGraphNode*> worklist;

  // Initialize node depths to -1.
  for (CallGraphNode& node : nodes_) {
    node.set_depth(-1);
  }

  // Initialize worklist with all roots of the call graph (computations without
  // callers).
  for (const HloComputation* computation :
       module_->computations(execution_threads_)) {
    CallGraphNode& node = GetNode(computation);
    if (node.callers().empty()) {
      node.set_depth(0);
      worklist.push(&node);
    }
  }

  while (!worklist.empty()) {
    CallGraphNode* node = worklist.front();
    worklist.pop();
    for (const HloComputation* callee : node->callees()) {
      CallGraphNode& callee_node = GetNode(callee);
      if (callee_node.depth() < node->depth() + 1) {
        callee_node.set_depth(node->depth() + 1);
        worklist.push(&callee_node);
      }
    }
  }

  for (CallGraphNode& node : nodes_) {
    CHECK_NE(node.depth(), -1);
  }
}

/* static */
std::unique_ptr<CallGraph> CallGraph::Build(
    const HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  // Constructor for CallGraph is private so std::make_unique can't be used.
  auto call_graph =
      absl::WrapUnique<CallGraph>(new CallGraph(module, execution_threads));

  VLOG(3) << "Building call graph for:";
  XLA_VLOG_LINES(3, module->ToString());

  // Construct nodes of the call graph and populate the callsites.
  for (HloComputation* computation : module->computations(execution_threads)) {
    auto it_added = call_graph->node_indices_.insert(
        {computation, call_graph->nodes_.size()});
    // All computations should be unique, so the computation should not already
    // exist in the map.
    CHECK(it_added.second);
    call_graph->nodes_.emplace_back(computation);

    // Add all callsites in this computation.
    for (HloInstruction* instruction : computation->instructions()) {
      call_graph->nodes_.back().AddCallSiteForInstruction(instruction,
                                                          execution_threads);
    }
  }

  // Add caller callsites to each node.
  for (const HloComputation* computation :
       module->computations(execution_threads)) {
    for (const CallSite& callsite :
         call_graph->GetNode(computation).callsites()) {
      for (auto* callee : callsite.called_computations()) {
        if (!HloInstruction::IsThreadIncluded(callee->execution_thread(),
                                              execution_threads)) {
          continue;
        }
        // Add caller callsites.
        call_graph->GetNode(callee).AddCallerCallSite(callsite);
      }
    }
  }

  call_graph->SetCallContexts();
  call_graph->SetNodeDepths();

  XLA_VLOG_LINES(2, call_graph->ToString());

  return call_graph;
}

absl::Status CallGraph::VisitNodesInternal(
    VisitorFunction visitor_func, const CallGraphNode& node,
    absl::flat_hash_set<const CallGraphNode*>* visited) const {
  auto pair = visited->insert(&node);
  if (!pair.second) {
    // Node was not inserted. Node has already been visited.
    return absl::OkStatus();
  }

  for (const HloComputation* computation : node.callees()) {
    TF_RETURN_IF_ERROR(
        VisitNodesInternal(visitor_func, GetNode(computation), visited));
  }

  return visitor_func(node);
}

absl::Status CallGraph::VisitNodes(VisitorFunction visitor_func,
                                   bool visit_unreachable_nodes) const {
  absl::flat_hash_set<const CallGraphNode*> visited;
  if (visit_unreachable_nodes) {
    // Traverse from all roots in the call graph.
    for (const CallGraphNode& node : nodes()) {
      if (node.callers().empty()) {
        TF_RETURN_IF_ERROR(VisitNodesInternal(visitor_func, node, &visited));
      }
    }
  } else {
    // Traverse only from the entry computation.
    TF_RETURN_IF_ERROR(VisitNodesInternal(
        visitor_func, GetNode(module_->entry_computation()), &visited));
  }

  return absl::OkStatus();
}

bool CallGraph::IsFlattened() const {
  for (const CallGraphNode& node : nodes_) {
    if (node.context() == CallContext::kBoth) {
      return false;
    }
    if (node.context() == CallContext::kControlFlow &&
        !node.computation()->IsAsyncComputation() &&
        node.caller_callsites().size() > 1) {
      return false;
    }
  }
  return true;
}

std::vector<HloInstruction*> CallGraph::GetComputationCallers(
    const HloComputation* c) const {
  std::vector<HloInstruction*> callers;
  for (const auto& callsite : GetNode(c).caller_callsites()) {
    callers.push_back(callsite.instruction());
  }
  return callers;
}

std::pair<HloInstruction*, HloInstruction*>
CallGraph::NearestAncestorsInSameComputation(HloInstruction* a,
                                             HloInstruction* b) const {
  // Lambda which returns the next instruction in the callee->caller chain in
  // the call graph. This is the unique instruction which calls the computation
  // containing 'instruction'. If more than one instruction calls the
  // computation containing 'instruction' or no instructions call the
  // computation then nullptr is returned.
  auto next_caller = [this](HloInstruction* instruction) -> HloInstruction* {
    const CallGraphNode& node = GetNode(instruction->parent());
    if (node.caller_callsites().size() != 1) {
      if (instruction->parent()->IsAsyncComputation()) {
        return node.caller_callsites()[0].instruction();
      }
      return nullptr;
    }
    return node.caller_callsites()[0].instruction();
  };

  // Iterate through the callee->caller chains and find the earliest common
  // element.
  HloInstruction* a_ancestor = a;
  HloInstruction* b_ancestor = b;
  int a_depth = GetNode(a->parent()).depth();
  int b_depth = GetNode(b->parent()).depth();

  // Advance a_ancestor (b_ancestor) up the call chain until the call depth of
  // a_ancestor or b_ancestor are the same. Necessarily each call to next_caller
  // reduces the depth by exactly one.
  if (a_depth > b_depth) {
    for (int i = 0; i < a_depth - b_depth; ++i) {
      a_ancestor = next_caller(a_ancestor);
      if (a_ancestor == nullptr) {
        return {nullptr, nullptr};
      }
    }
  } else if (b_depth > a_depth) {
    for (int i = 0; i < b_depth - a_depth; ++i) {
      b_ancestor = next_caller(b_ancestor);
      if (b_ancestor == nullptr) {
        return {nullptr, nullptr};
      }
    }
  }

  while ((a_ancestor != nullptr) && (b_ancestor != nullptr)) {
    if (a_ancestor->parent() == b_ancestor->parent()) {
      return {a_ancestor, b_ancestor};
    }

    a_ancestor = next_caller(a_ancestor);
    b_ancestor = next_caller(b_ancestor);
  }
  return {nullptr, nullptr};
}

template <typename T>
absl::flat_hash_set<const T*> CallGraph::NearestCommonAncestorsHelper(
    std::vector<const T*>& starting_nodes) {
  // Check if T is either HloInstruction or HloComputation.
  CHECK(
      (std::is_same_v<T, HloInstruction> || std::is_same_v<T, HloComputation>));

  if (starting_nodes.empty()) {
    return absl::flat_hash_set<const T*>();
  }
  if (starting_nodes.size() == 1) {
    return absl::flat_hash_set<const T*>({starting_nodes[0]});
  }

  // There could be multiple nearest common ancestors in a DAG.
  absl::flat_hash_set<const T*> nearest_common_ancestors;

  // Initialize `visited_ancestors` for each provided nodes.
  std::vector<absl::flat_hash_set<const T*>> visited_ancestors;
  visited_ancestors.reserve(starting_nodes.size());
  for (int idx = 0; idx < starting_nodes.size(); ++idx) {
    visited_ancestors.push_back(
        absl::flat_hash_set<const T*>({starting_nodes[idx]}));
  }

  // Initialize BFS queue for each provided nodes.
  std::vector<std::deque<const T*>> bfs_queues;
  bfs_queues.reserve(starting_nodes.size());
  for (int idx = 0; idx < starting_nodes.size(); ++idx) {
    bfs_queues.push_back(std::deque<const T*>({starting_nodes[idx]}));
  }

  // Lambda to check if the BFS has finished (i.e., all queues in `bfs_queues`
  // are empty).
  auto is_bfs_finished = [&bfs_queues]() -> bool {
    return absl::c_all_of(
        bfs_queues, [](std::deque<const T*> queue) { return queue.empty(); });
  };

  // Lambda to check if there are common nodes in all the
  // `visited_ancestors`. Save results in `nearest_common_ancestors`. Return
  // true if they are found, otherwise return false.
  auto find_common_nodes = [&visited_ancestors,
                            &nearest_common_ancestors]() -> bool {
    absl::flat_hash_set<const T*> common_nodes(visited_ancestors[0]);
    for (int idx = 1; idx < visited_ancestors.size(); ++idx) {
      absl::erase_if(common_nodes, [&](auto k) {
        return !visited_ancestors[idx].contains(k);
      });
    }
    nearest_common_ancestors = common_nodes;
    return !nearest_common_ancestors.empty();
  };

  // BFS body.
  // For each BFS step, we check if there is a common node in all the visited
  // ancestors (`find_common_nodes()`), and if yes, that common node is the
  // nearest ancestor we are looking for. Otherwise, we conduct BFS from each
  // bfs_queue, and update `bfs_queues` and `visited_ancestors` accordingly.
  while (!is_bfs_finished() && !find_common_nodes()) {
    for (int idx = 0; idx < bfs_queues.size(); ++idx) {
      auto cur_queue = bfs_queues[idx];
      std::deque<const T*> next_queue;
      auto& visited_ancestor = visited_ancestors[idx];

      while (!cur_queue.empty()) {
        const T* node = cur_queue.back();
        cur_queue.pop_back();

        // Identify ancestor of node.
        std::vector<T*> ancestors_to_visit;
        if constexpr (std::is_same_v<T, HloInstruction>) {
          // For instruction, the ancestors are its users and
          // control_successors.
          ancestors_to_visit = node->users();
          ancestors_to_visit.insert(ancestors_to_visit.end(),
                                    node->control_successors().begin(),
                                    node->control_successors().end());
        } else if constexpr (std::is_same_v<T, HloComputation>) {
          // For computation, the ancestors are its caller computations.
          for (auto caller_instruction : GetComputationCallers(node)) {
            ancestors_to_visit.push_back(caller_instruction->parent());
          }
        }

        for (auto ancestor : ancestors_to_visit) {
          if (!visited_ancestor.contains(ancestor)) {
            next_queue.push_back(ancestor);
            visited_ancestor.insert(ancestor);
          }
        }
      }

      bfs_queues[idx] = next_queue;
    }
  }

  CHECK(!nearest_common_ancestors.empty())
      << "At least one nearest_common_ancestor";

  // If one of the computed nearest common ancestors is inside
  // `starting_nodes`, we would only return the ones that are inside
  // `starting_nodes`.
  if (absl::c_any_of(starting_nodes, [&nearest_common_ancestors](const T* nca) {
        return nearest_common_ancestors.contains(nca);
      })) {
    absl::erase_if(nearest_common_ancestors, [&starting_nodes](const T* nca) {
      return std::find(starting_nodes.begin(), starting_nodes.end(), nca) ==
             starting_nodes.end();
    });
  }

  return nearest_common_ancestors;
}

absl::flat_hash_set<const HloComputation*>
CallGraph::NearestCommonAncestorComputations(
    std::vector<const HloComputation*> computations) {
  return NearestCommonAncestorsHelper<HloComputation>(computations);
}

absl::flat_hash_set<const HloInstruction*>
CallGraph::NearestCommonAncestorInstructions(
    std::vector<const HloInstruction*> instructions) {
  if (instructions.empty()) {
    return absl::flat_hash_set<const HloInstruction*>();
  }

  // Check if all the instructions belong to the same computation.
  auto computation = instructions[0]->parent();
  CHECK(absl::c_all_of(instructions, [&computation](
                                         const HloInstruction* instruction) {
    return instruction->parent() == computation;
  })) << "All provided instructions should be in the same computation";

  return NearestCommonAncestorsHelper<HloInstruction>(instructions);
}

std::string CallGraph::ToString() const {
  std::string out;
  StrAppendFormat(&out, "Call graph for module %s:\n", module_->name());
  for (const CallGraphNode& node : nodes()) {
    StrAppendFormat(&out, "Computation %s:\n", node.computation()->name());
    StrAppendFormat(&out, "  calls:\n");
    for (const HloComputation* callee : node.callees()) {
      StrAppendFormat(&out, "    %s\n", callee->name());
    }
    StrAppendFormat(&out, "  called by:\n");
    for (const HloComputation* caller : node.callers()) {
      StrAppendFormat(&out, "    %s\n", caller->name());
    }
    StrAppendFormat(&out, "  callsites:\n");
    for (const CallSite& callsite : node.callsites()) {
      StrAppendFormat(&out, "    %s\n", callsite.ToString());
    }
  }
  return out;
}

}  // namespace xla
