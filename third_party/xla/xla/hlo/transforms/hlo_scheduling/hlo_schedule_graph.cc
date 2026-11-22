/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/transforms/hlo_scheduling/hlo_schedule_graph.h"

#include <algorithm>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_instruction_utils.h"
#include "xla/hlo/ir/hlo_instructions.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/transforms/hlo_scheduling/async_tracker.h"
#include "xla/hlo/transforms/hlo_scheduling/latency_estimator.h"
#include "xla/map_util.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/side_effect_util.h"
#include "xla/status_macros.h"

namespace xla {
namespace {

bool IsCustomCallWithForceEarlyAttribute(const HloInstruction* instr) {
  auto attr = instr->get_frontend_attribute("scheduler_hint");
  return instr->opcode() == HloOpcode::kCustomCall && attr.has_value() &&
         attr.value() == "force_early";
}

int GetForceDelayPriority(const HloInstruction* instr) {
  auto attr = instr->get_frontend_attribute("scheduler_delay_priority");
  if (attr.has_value()) {
    int out;
    CHECK(absl::SimpleAtoi(attr.value(), &out))
        << "Failed to parse scheduler_delay_priority attribute: "
        << attr.value();
    return out;
  }
  return 0;
}

bool HasForceDelayAsyncAttribute(const HloInstruction* instr) {
  auto attr = instr->get_frontend_attribute("scheduler_hint");
  return attr.has_value() && attr.value() == "force_delay_async";
}

}  // namespace

const HloInstruction* FindStart(const HloInstruction* done) {
  if (done == nullptr) {
    return nullptr;
  }
  if (done->IsAsyncDone()) {
    const HloInstruction* start =
        hlo_instruction_utils::async::FindAsyncStart(done);
    if (start != nullptr && start->parent() != done->parent()) {
      return done->operand(0);
    }
    return start;
  }
  switch (done->opcode()) {
    case HloOpcode::kCopyDone:
    case HloOpcode::kSendDone:
    case HloOpcode::kRecvDone: {
      return done->operand(0);
    }
    default: {
      return nullptr;
    }
  }
}

HloInstruction* FindStart(HloInstruction* done) {
  // NOLINTNEXTLINE(*-readability-const-cast): `done` is non-const, so the
  // matching start in the same module is too.
  return const_cast<HloInstruction*>(
      FindStart(static_cast<const HloInstruction*>(done)));
}

const HloInstruction* FindDone(const HloInstruction* start) {
  if (start == nullptr) {
    return nullptr;
  }
  if (start->IsAsyncStart()) {
    return hlo_instruction_utils::async::FindAsyncDone(start);
  }
  HloOpcode done_opcode;
  switch (start->opcode()) {
    case HloOpcode::kCopyStart: {
      done_opcode = HloOpcode::kCopyDone;
      break;
    }
    case HloOpcode::kSend: {
      done_opcode = HloOpcode::kSendDone;
      break;
    }
    case HloOpcode::kRecv: {
      done_opcode = HloOpcode::kRecvDone;
      break;
    }
    default: {
      return nullptr;
    }
  }
  for (const HloInstruction* user : start->users()) {
    if (user->opcode() == done_opcode) {
      return user;
    }
  }
  return nullptr;
}

HloInstruction* FindDone(HloInstruction* start) {
  // NOLINTNEXTLINE(*-readability-const-cast): `start` is non-const, so
  // the matching done in the same module is too.
  return const_cast<HloInstruction*>(
      FindDone(static_cast<const HloInstruction*>(start)));
}

bool IsCustomCallorAsyncOpWithForceDelayAttribute(const HloInstruction* instr) {
  auto attr = instr->get_frontend_attribute("scheduler_hint");
  bool is_custom_call_or_async_op =
      (instr->opcode() == HloOpcode::kCustomCall ||
       instr->opcode() == HloOpcode::kAsyncStart ||
       instr->opcode() == HloOpcode::kAsyncDone);
  return is_custom_call_or_async_op && attr.has_value() &&
         attr.value() == "force_delay";
}

std::string HloEdge::ToString() const {
  return absl::StrCat("\tEdge: ", target_->GetInstr().name(),
                      " latency: ", Latency(), "\n");
}

bool LatencyEstimator::IsAsyncPair(const HloGraphNode& from,
                                   const HloGraphNode& target) const {
  return IsAsyncPair(from.GetInstr(), target.GetInstr());
}

bool LatencyEstimator::IsP2pPair(const HloGraphNode& from,
                                 const HloGraphNode& target) const {
  return IsP2pPair(from.GetInstr(), target.GetInstr());
}

const std::vector<int64_t>& HloGraphNode::GetShareableResourcesOnEdge(
    HloScheduleGraph* g, HloEdge& edge) const {
  if (!edge.SharableResourcesComputed()) {
    std::vector<int64_t> resources;
    if (has_rare_) {
      HloGraphNode& to = edge.Target();
      absl::c_for_each(rare_->released_shareable_resources,
                       [this, &to, &resources](const int64_t resource) {
                         if (to.DoesOccupyShareableResource(resource) &&
                             this->DoesReleaseResource(resource)) {
                           resources.push_back(resource);
                         }
                       });
    }
    g->SetSharableResources(edge, resources);
  }
  return g->GetSharableResources(edge);
}

const std::vector<int64_t>& HloScheduleGraph::GetSharableResources(
    const HloEdge& edge) const {
  return sharable_resources_storage_[edge.sharable_resources_index_];
}

void HloScheduleGraph::SetSharableResources(
    HloEdge& edge, const std::vector<int64_t>& resources) {
  CHECK_LT(edge.sharable_resources_index_, 0);
  if (resources.empty()) {
    // Share the 0th entry, which is an empty vector.
    CHECK(sharable_resources_storage_[0].empty());
    edge.sharable_resources_index_ = 0;
  } else {
    // Store the non-empty vector and record its index on the edge.
    edge.sharable_resources_index_ = sharable_resources_storage_.size();
    sharable_resources_storage_.push_back(resources);
  }
}

bool HloScheduleGraph::IsPredecessorTransitively(
    const HloGraphNode* node, const HloGraphNode* possible_predecessor) {
  absl::flat_hash_set<const HloGraphNode*> visited = {possible_predecessor};
  std::vector<const HloGraphNode*> to_visit_queue = {node};
  while (!to_visit_queue.empty()) {
    const HloGraphNode* curr = to_visit_queue.back();
    to_visit_queue.pop_back();
    if (curr == possible_predecessor) {
      return true;
    }
    if (visited.contains(curr)) {
      continue;
    }
    visited.insert(curr);
    for (const auto& edge : curr->GetPredecessors()) {
      to_visit_queue.push_back(GetNodePtr(&edge.Target().GetInstr()));
    }
  }
  return false;
}

HloScheduleGraph::HloScheduleGraph(
    const std::vector<HloInstruction*>* post_order_instructions,
    std::shared_ptr<const SchedulingContext> scheduling_context,
    std::shared_ptr<const HloReachabilityMap> reachability_map)
    : original_order_(post_order_instructions->begin(),
                      post_order_instructions->end()),
      scheduling_context_(scheduling_context),
      reachability_(std::move(reachability_map)) {
  HloComputation* comp = (*post_order_instructions)[0]->parent();
  if (reachability_ == nullptr) {
    reachability_ = HloReachabilityMap::Build(comp);
  }
  const HloReachabilityMap* reachability = reachability_.get();
  std::vector<const HloInstruction*> while_instrs;
  auto latency_estimator = scheduling_context->GetLatencyEstimator();
  auto async_tracker = scheduling_context->GetAsyncTracker();
  auto alias_analysis = scheduling_context->GetAliasAnalysis();
  bool top_down_scheduling =
      scheduling_context->GetAsyncTracker()->GetConfig().top_down_scheduling;

  const int n_inst = post_order_instructions->size();
  // Allocating the graph nodes. One for each of the instructions in the
  // original instructions order.
  node_storage_.resize(n_inst);

  // Single entry initially, which is a shared entry that is shared by
  // all instructions for which the rare data structures are all empty
  rare_storage_.push_back(std::make_unique<HloGraphNode::Rare>());

  // The first entry of sharable_resources_storage_ is an empty vector,
  // which can be shared by all edges that have an empty sharable resources
  // set.
  sharable_resources_storage_.push_back(std::vector<int64_t>());

  int64_t current_pos = 0;
  for (HloInstruction* instr : *post_order_instructions) {
    auto [new_node_it, inserted] = nodes_.try_emplace(instr, current_pos);
    CHECK(inserted) << "Expected the value to not be already in the map";
    HloGraphNode* n = &node_storage_[current_pos];
    DCHECK_EQ(n, GetNodePtr(instr));
    n->instr_ = instr;
    n->opcode_ = instr->opcode();
    n->is_host_transfer_ =
        (n->opcode_ == HloOpcode::kSend || n->opcode_ == HloOpcode::kSendDone ||
         n->opcode_ == HloOpcode::kRecv ||
         n->opcode_ == HloOpcode::kRecvDone) &&
        static_cast<const HloSendRecvInstruction*>(instr)->is_host_transfer();
    n->original_position_ = current_pos;
    current_pos++;

    n->cost_ = latency_estimator->NodeCost(instr);
    auto resources_span = async_tracker->GetResourcesFromInstruction(*instr);
    ResourcesVector resources(resources_span.begin(), resources_span.end());
    HloGraphNode::Rare r;
    r.released_non_extendable_resources =
        async_tracker->GetReleasedNonextendableResourcesFromVector(resources);
    r.released_shareable_resources =
        async_tracker->GetReleasedShareableResourcesFromVector(resources);
    r.occupied_shareable_resources =
        async_tracker->GetOccupiedShareableResourcesFromVector(resources);
    r.recursive_resources =
        async_tracker->GetNumResourcesPerInstruction(*instr);
    r.resources = resources;

    n->has_recursive_resources_ = !(r.recursive_resources.empty());
    if (!r.recursive_resources.empty() ||                //
        !r.released_non_extendable_resources.empty() ||  //
        !r.released_shareable_resources.empty() ||       //
        !r.occupied_shareable_resources.empty() ||       //
        !r.resources.empty()) {
      // At least one entry is non-empty, so we need an actual
      rare_storage_.push_back(std::make_unique<HloGraphNode::Rare>(r));
      n->rare_ = rare_storage_.back().get();
      n->has_rare_ = true;
    } else {
      // The default entry where all are empty can be shared by this
      // instruction
      n->rare_ = rare_storage_[0].get();
      n->has_rare_ = false;
    }

    n->is_supported_async_done_ = async_tracker->IsSupportedAsyncDone(*instr);
    n->is_supported_async_start_ = async_tracker->IsSupportedAsyncStart(*instr);
    n->has_operand_that_is_supported_async_done_ = absl::c_any_of(
        instr->operands(), [async_tracker](const HloInstruction* i) {
          return async_tracker->IsSupportedAsyncDone(*i);
        });
    n->has_user_that_is_supported_async_start_ = absl::c_any_of(
        instr->users(), [async_tracker](const HloInstruction* i) {
          return async_tracker->IsSupportedAsyncStart(*i);
        });
    n->releases_selective_resource_ =
        async_tracker->ReleasesSelectiveResource(n->GetResources());
    n->occupies_selective_resource_ =
        async_tracker->OccupiesSelectiveResource(n->GetResources());
    n->does_occupy_any_resource_ =
        absl::c_any_of(resources, [](const ResourcePair& resource) {
          return resource.second == ResourceUsageType::kResourceOccupy;
        });
    n->does_release_any_resource_ =
        absl::c_any_of(resources, [](const ResourcePair& resource) {
          return resource.second == ResourceUsageType::kResourceRelease;
        });
    // Gather while instructions for subsequent send-done dependency checks.
    if (instr->opcode() == HloOpcode::kWhile) {
      while_instrs.push_back(instr);
    }

    if (IsCustomCallWithForceEarlyAttribute(instr)) {
      n->SetForceEarly(true);
    }
    if (IsCustomCallorAsyncOpWithForceDelayAttribute(instr)) {
      n->SetForceDelay(true);
      n->SetForceDelayPriority(GetForceDelayPriority(instr));
    }
    if (n->IsSupportedAsyncStart() && HasForceDelayAsyncAttribute(instr)) {
      n->SetForceDelay(true);
      n->SetForceDelayPriority(GetForceDelayPriority(instr));
    }
    if (n->IsSupportedAsyncDone() && HasForceDelayAsyncAttribute(instr)) {
      n->SetForceEarly(true);
    }
    if (top_down_scheduling) {
      n->SetTopDownScheduling(true);
    }
  }

  // num_predecessors[i]: number of predecessors for instruction number "i"
  std::vector<int> num_predecessors(n_inst, 0);
  // num_successors[i]: number of successors for instruction number "i"
  std::vector<int> num_successors(n_inst, 0);

  // We do two passes over the graph.  In phase 0, we just determine how
  // many successor and predecessor edges each node is likely to have (there are
  // some rare exceptions where the first pass can't fully see extra edges
  // added, but these are rare).  This first pass then allows us to lay out
  // storage for all the successor and predecessor edges for all the nodes in
  // the HloSchedulerGraph::predecessor_storage_ and
  // HloSchedulerGraph::successor_storage_ vectors.  This makes the cache
  // locality of traversing the graph via these edges better than if we
  // allocated them individually hanging off each of the HloGraphNode objects.
  //
  // In phase 1, we actually fill in the edge data into the storage arrays
  // we've allocated.
  for (int phase = 0; phase < 2; phase++) {
    // Add dependencies edges between each of the graph nodes.
    for (const HloInstruction* instr : *post_order_instructions) {
      HloGraphNode* instr_node = GetNodePtr(instr);
      // Lambda invoked for each edge or potential edge during phase 0 and phase
      // 1
      auto add_edge = [&](const char* type, HloGraphNode* from,
                          HloGraphNode* to, const LatencyEstimator* estimator) {
        const int from_index = from->original_position_;
        const int to_index = to->original_position_;
        if (phase == 0) {
          // In the first phase, we just count the number of predecessors and
          // successors
          num_successors[from_index]++;
          num_predecessors[to_index]++;
        } else {
          if (estimator != nullptr) {
            // Use overload of AddDependency that uses estimator to look up
            // latency
            HloGraphNode::AddDependency(from, to, estimator);
          } else {
            // Use overload of AddDependency that explicitly specifies latency
            // value
            HloGraphNode::AddDependency(from, to, 1);
          }
        }
      };

      VLOG(10) << "Adding users for " << instr_node->GetInstr().ToString();
      // Add edges that derive from def->use relationships of the HLO graph.
      for (const HloInstruction* user : instr->users()) {
        VLOG(10) << "\tUser: " << user->ToString();
        HloGraphNode* user_node = GetNodePtr(user);
        add_edge("user", instr_node, user_node, latency_estimator.get());
      }
      for (const HloInstruction* ctrl_succ : instr->control_successors()) {
        VLOG(10) << "\tCtrl Successor: " << ctrl_succ->ToString();
        HloGraphNode* ctrl_succ_node = GetNodePtr(ctrl_succ);
        add_edge("ctrl_succ", instr_node, ctrl_succ_node,
                 latency_estimator.get());
      }

      // To make sure an instruction that aliases with the buffer produced
      // by the async-done operation is not scheduled in between the start and
      // the done instruction as that buffer is in flux when the start
      // happens. Add an edge between this instruction and the start in this
      // case.
      if (instr_node->IsSupportedAsyncDone()) {
        const HloInstruction* async_start = FindStart(instr);
        if (alias_analysis != nullptr && async_start != nullptr) {
          for (const HloBuffer* buffer :
               alias_analysis->ComputeBuffersAt(instr, {})) {
            for (const HloValue* value : buffer->values()) {
              if (value->defining_instruction() == instr) {
                continue;
              }
              for (const HloUse& use : value->GetUses()) {
                if (ContainsKey(nodes_, use.instruction)) {
                  // The instruction itself and later ones might be
                  // identified as use.instruction. Add checks here to avoid
                  // adding dependencies for these instructions.
                  if (use.instruction == async_start ||
                      reachability->IsReachable(async_start, use.instruction) ||
                      reachability->IsReachable(use.instruction, async_start) ||
                      reachability->IsReachable(instr, use.instruction)) {
                    continue;
                  }
                  HloGraphNode* pred_node = GetNodePtr(use.instruction);
                  HloGraphNode* start_node = GetNodePtr(async_start);
                  // Ignore token operands as they are not real aliasing.
                  if (use.instruction->operand(use.operand_number)
                          ->shape()
                          .IsToken()) {
                    continue;
                  }
                  if (phase == 0) {
                    // Be conservative and allocate space for a potential
                    // edge. The conditions below mean we might not actually
                    // add this edge, but some of the conditions depend on
                    // having some of the edges in the graph already set up,
                    // which won't really happen until phase 1.  We might
                    // allocate a bit of extra unused space in the edge
                    // storage vectors by being conservative here, but it's
                    // not a big deal.
                    add_edge("async0", pred_node, start_node,
                             nullptr /* unused for phase 0*/);
                    continue;
                  }
                  // If there is already a transitive link between the nodes
                  // the other way then skip adding this one.
                  if (IsPredecessorTransitively(pred_node, start_node)) {
                    continue;
                  }
                  add_edge("async1", pred_node, start_node,
                           nullptr /*==use latency of 1*/);
                }
              }
            }
          }
        }
      }
      // Add dependent edges from send-done operations to while loops which
      // are dependent on the recv-done control predecessor of the send-done.
      // This prevents send-done operations from being scheduled after
      // dependent while loops, which can caused send/recv overlap limits to
      // be violated.
      //
      // Example HLO sequence:
      //
      //   %0 = recv-done --->
      //                     |
      //   %1 = send-done <--|
      //   %2 = while <------|
      //
      if (instr->opcode() == HloOpcode::kSendDone) {
        for (const auto* ctrl_pred : instr->control_predecessors()) {
          if (ctrl_pred->opcode() != HloOpcode::kRecvDone) {
            continue;
          }
          const HloInstruction* dependent_while_instr = nullptr;
          for (const auto* while_hlo : while_instrs) {
            if (!reachability->IsReachable(ctrl_pred, while_hlo)) {
              continue;
            }
            if (dependent_while_instr == nullptr) {
              dependent_while_instr = while_hlo;
              continue;
            }
            if (OriginalInstructionPosition(while_hlo) <
                OriginalInstructionPosition(dependent_while_instr)) {
              dependent_while_instr = while_hlo;
            }
          }
          // Add dependency edge from 'instr' to 'dependent_while_instr'.
          if (dependent_while_instr != nullptr) {
            HloGraphNode* send_done_node = GetNodePtr(instr);
            HloGraphNode* while_node = GetNodePtr(dependent_while_instr);
            add_edge("while", send_done_node, while_node,
                     nullptr /*==use latency of 1*/);
          }
          break;
        }
      }
      // If sends are originally after a while loop, make sure the corresponding
      // recv is also scheduled after the while loop. This ensures the recv
      // and recv-done's life times do not extend across the while loop, which
      // can lead to overlap limit violations. This problem is common with
      // top-down scheduling since the recv tends to be scheduled early due
      // to kScheduleAsyncStart heuristic.
      //   %0 = while <---
      //                  |
      //   %1 = recv -----
      //   %2 = send
      //   %3 = send-done
      if (top_down_scheduling) {
        if (instr->opcode() == HloOpcode::kSendDone) {
          for (const auto* while_hlo : while_instrs) {
            if (!reachability->IsReachable(while_hlo, instr)) {
              continue;
            }
            if (OriginalInstructionPosition(while_hlo) <
                OriginalInstructionPosition(instr)) {
              for (auto* recv_done : instr->control_successors()) {
                if (recv_done->opcode() != HloOpcode::kRecvDone) {
                  continue;
                }
                auto* recv = recv_done->operand(0);
                CHECK(recv->opcode() == HloOpcode::kRecv);
                HloGraphNode* recv_node = GetNodePtr(recv);
                HloGraphNode* while_node = GetNodePtr(while_hlo);
                add_edge("while", while_node, recv_node,
                         nullptr /*==use latency of 1*/);
              }
            }
          }
        }
      }
    }

    if (phase == 0) {
      // After phase 0 runs above, we now know an upper bound on the number of
      // successors and predecessors we will have, and so can allocate space
      // in the predecessors and successor storage vectors.

      int total_predecessors = 0;
      for (const auto& n : num_predecessors) {
        total_predecessors += n;
      }
      int total_successors = 0;
      for (const auto& n : num_successors) {
        total_successors += n;
      }
      predecessors_storage_.resize(total_predecessors);
      successors_storage_.resize(total_successors);

      // Set pointers to reserved ranges of predecessors_storage_ and
      // successors_storage_ within each node
      int next_index = 0;
      int pstart = 0;
      int sstart = 0;
      for (const HloInstruction* instr : *post_order_instructions) {
        HloGraphNode* n = GetNodePtr(instr);
        n->predecessors_.SetEmptyPointingToSharedSpace(
            (pstart < total_predecessors) ? &predecessors_storage_[pstart]
                                          : nullptr,
            num_predecessors[next_index]);
        n->successors_.SetEmptyPointingToSharedSpace(
            (sstart < total_successors) ? &successors_storage_[sstart]
                                        : nullptr,
            num_successors[next_index]);
        pstart += num_predecessors[next_index];
        sstart += num_successors[next_index];
        next_index++;
      }
      CHECK_EQ(next_index, post_order_instructions->size());
    }
  }

  // Initialize ready_nodes_if_scheduled_ for all nodes.
  for (auto& node : node_storage_) {
    node.UpdateReadyNodesIfScheduled();
  }

  // Post process the schedule graph based on the supplied async_tracker.
  async_tracker->PostProcessScheduleGraph(this, latency_estimator.get());

  // Records the current set of edges as 'static', allowing the graph to be
  // reset to this state (e.g., during multiple scheduling passes) by removing
  // any temporary edges added later.
  RecordStaticEdges();
}

std::string HloScheduleGraph::ToString() const {
  std::string result;
  std::vector<std::pair<const HloGraphNode*, int>> stack;
  for (auto& node : node_storage_) {
    if (node.GetPredecessors().empty()) {
      stack.push_back(std::make_pair(&node, 0));
    }
  }
  std::vector<const HloGraphNode*> order;
  absl::flat_hash_set<const HloGraphNode*> visited;
  while (!stack.empty()) {
    auto& val = stack.back();
    if (val.second == val.first->GetSuccessors().size()) {
      order.push_back(val.first);
      stack.pop_back();
      continue;
    }
    const int64_t next_child = val.second++;
    if (visited.insert(&val.first->GetSuccessors()[next_child].Target())
            .second) {
      stack.push_back(
          std::make_pair(&val.first->GetSuccessors()[next_child].Target(), 0));
    }
  }
  for (auto it = order.rbegin(), e = order.rend(); it != e; ++it) {
    absl::StrAppend(
        &result, (*it)->ToString(scheduling_context_->GetAsyncTracker().get()));
  }
  return result;
}

HloGraphNode& HloScheduleGraph::GetNode(const HloInstruction* instr) const {
  auto it = nodes_.find(instr);
  CHECK(it != nodes_.end());
  return node_storage_[it->second];
}

HloGraphNode* HloScheduleGraph::GetNodePtr(const HloInstruction* instr) const {
  auto it = nodes_.find(instr);
  CHECK(it != nodes_.end());
  return &node_storage_[it->second];
}

std::vector<HloGraphNode*> HloScheduleGraph::FindBottomRoots() const {
  std::vector<HloGraphNode*> roots;
  for (const HloInstruction* instr : original_order_) {
    HloGraphNode& node = GetNode(instr);
    if (node.GetOutdegree() == 0) {
      roots.push_back(&node);
    }
  }
  return roots;
}

std::vector<HloGraphNode*> HloScheduleGraph::FindTopRoots() const {
  std::vector<HloGraphNode*> roots;
  for (const HloInstruction* instr : original_order_) {
    HloGraphNode& node = GetNode(instr);
    if (node.GetIndegree() == 0) {
      roots.push_back(&node);
    }
  }
  return roots;
}

void HloScheduleGraph::InitializeGraphAnalysis() {
  absl::flat_hash_map<HloGraphNode*, int> current_rank;
  std::vector<HloGraphNode*> stack;
  for (const HloInstruction* instr : original_order_) {
    HloGraphNode& node = GetNode(instr);
    current_rank[&node] = node.GetIndegree();
    node.SetAsyncDepth(0.0);
    node.SetDepth(0.0);
    node.SetGraphDepth(0);
    if (node.GetIndegree() == 0) {
      stack.push_back(&node);
    }
  }
  while (!stack.empty()) {
    auto* node = stack.back();
    stack.pop_back();
    // If a node occupies a selective resource, it is the closest selective
    // resource occupier to itself and is 0 hops away. Otherwise, the num hops
    // to closest selective resource occupier is the minimum of that of all
    // predecessors plus 1.
    if (scheduling_context_->GetAsyncTracker()->OccupiesSelectiveResource(
            node->GetResources())) {
      node->num_hops_to_closest_selective_resource_occupier_ = 0;
    } else {
      int64_t closest_predecessor_distance =
          std::numeric_limits<int64_t>::max();
      for (auto& pred : node->GetPredecessors()) {
        closest_predecessor_distance = std::min<int64_t>(
            closest_predecessor_distance,
            pred.Target().num_hops_to_closest_selective_resource_occupier_);
      }
      if (closest_predecessor_distance != std::numeric_limits<int64_t>::max()) {
        node->num_hops_to_closest_selective_resource_occupier_ =
            closest_predecessor_distance + 1;
      }
    }
    if (node->IsSupportedAsyncDone()) {
      for (auto& pred : node->GetPredecessors()) {
        node->SetAsyncDepth(
            std::max(pred.Target().GetAsyncDepth() + pred.Latency(),
                     node->GetAsyncDepth()));
        node->SetDepth(std::max(
            pred.Target().GetDepth() + pred.Target().GetCost() + pred.Latency(),
            node->GetDepth()));
        node->SetGraphDepth(
            std::max(pred.Target().GetGraphDepth() + 1, node->GetGraphDepth()));
      }
    } else {
      for (auto& pred : node->GetPredecessors()) {
        node->SetAsyncDepth(
            std::max(pred.Target().GetAsyncDepth(), node->GetAsyncDepth()));
        node->SetDepth(std::max(
            pred.Target().GetDepth() + pred.Target().GetCost() + pred.Latency(),
            node->GetDepth()));
        node->SetGraphDepth(
            std::max(pred.Target().GetGraphDepth() + 1, node->GetGraphDepth()));
      }
    }
    for (auto& succ : node->GetSuccessors()) {
      if (--current_rank[&succ.Target()] == 0) {
        stack.push_back(&succ.Target());
      }
    }
  }
  for (const HloInstruction* instr : original_order_) {
    HloGraphNode& node = GetNode(instr);
    current_rank[&node] = node.GetOutdegree();
    node.SetAsyncHeight(0.0);
    if (node.GetOutdegree() == 0) {
      stack.push_back(&node);
    }
  }
  while (!stack.empty()) {
    auto* node = stack.back();
    stack.pop_back();
    if (node->IsSupportedAsyncStart()) {
      for (auto& succ : node->GetSuccessors()) {
        node->SetAsyncHeight(
            std::max(succ.Target().GetAsyncHeight() + succ.Latency(),
                     node->GetAsyncHeight()));
      }
    } else {
      for (auto& succ : node->GetSuccessors()) {
        node->SetAsyncHeight(
            std::max(succ.Target().GetAsyncHeight(), node->GetAsyncHeight()));
      }
    }
    for (auto& pred : node->GetPredecessors()) {
      if (--current_rank[&pred.Target()] == 0) {
        stack.push_back(&pred.Target());
      }
    }
  }
}

void HloScheduleGraph::AnnotateGraph(
    const AnnotationTracker* annotation_tracker) {
  const HloComputation* comp = original_order_[0]->parent();
  for (int64_t annotation : annotation_tracker->GetAnnotations(comp)) {
    for (const HloInstruction* instr :
         annotation_tracker->GetInstructions(comp, annotation)) {
      HloGraphNode& node = GetNode(instr);
      CHECK_OK(node.SetAnnotation(annotation));
    }
  }
}

std::shared_ptr<const HloAliasAnalysis> SchedulingContext::GetAliasAnalysis()
    const {
  // Lazy initialization of alias analysis on first use.
  if (alias_analysis_ == nullptr) {
    absl::StatusOr<std::unique_ptr<HloAliasAnalysis>> alias_analysis =
        HloAliasAnalysis::Run(module_, alias_info_);
    CHECK_OK(alias_analysis);
    alias_analysis_ = *std::move(alias_analysis);
  }
  return alias_analysis_;
}

AnnotationTracker::AnnotationTracker(const HloModule* module)
    : module_(module) {
  for (const HloComputation* comp : module_->MakeNonfusionComputations()) {
    absl::flat_hash_set<int64_t> annotations;
    for (const HloInstruction* instr : comp->instructions()) {
      if (auto annotation = GetAnnotation(instr)) {
        if (annotations.insert(annotation.value()).second) {
          comp_annotation_map_[comp].push_back(annotation.value());
        }
        annotations_[annotation.value()][comp].push_back(instr);
      }
    }
  }
}

std::optional<int64_t> AnnotationTracker::GetAnnotation(
    const HloInstruction* instr) const {
  if (std::optional<std::string> annotation =
          instr->get_frontend_attribute(kXlaSchedulingGroupIdAttr)) {
    return std::stoi(*annotation);
  }
  return std::nullopt;
}

std::vector<const HloInstruction*> AnnotationTracker::GetInstructions(
    const HloComputation* comp, const int64_t annotation) const {
  if (annotation == kInvalidAnnotation) {
    return {};
  }
  return annotations_.at(annotation).at(comp);
}

void AnnotationTracker::PrintAnnotationSets(int64_t level) const {
  // Iteration order only affects the order of the log lines.
  // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
  for (const auto& [annotation, comp_instr_vector] : annotations_) {
    // NOLINTNEXTLINE(*-custom-deterministic-iteration-order)
    for (const auto& [comp, instrs] : comp_instr_vector) {
      VLOG(level) << "Annotation " << annotation << " has " << instrs.size()
                  << " instructions in computation " << comp->name();
      for (const HloInstruction* instr : instrs) {
        VLOG(level) << "  " << instr->name();
      }
    }
  }
}

HloGraphNode::HloGraphNode(const HloInstruction* i, int64_t original_position)
    : instr_(i), opcode_(i->opcode()), original_position_(original_position) {
  InitBitFields();
  is_host_transfer_ =
      (opcode_ == HloOpcode::kSend || opcode_ == HloOpcode::kSendDone ||
       opcode_ == HloOpcode::kRecv || opcode_ == HloOpcode::kRecvDone) &&
      static_cast<const HloSendRecvInstruction*>(i)->is_host_transfer();
}

void HloGraphNode::UpdateOrAddDependency(HloGraphNode* from, HloGraphNode* to,
                                         LatencyEstimator::TimeCost latency) {
  auto update_latency_if_edge_exists = [&](absl::Span<HloEdge> edges,
                                           HloGraphNode* to,
                                           LatencyEstimator::TimeCost latency) {
    auto it = absl::c_find_if(
        edges, [&](HloEdge edge) { return &edge.Target() == to; });
    if (it != edges.end()) {
      it->SetLatency(latency);
      return true;
    }
    return false;
  };
  if (!update_latency_if_edge_exists(from->GetSuccessors(), to, latency)) {
    from->AddSuccessor(HloEdge(latency, to));
    from->outdegree_++;
  }
  if (!update_latency_if_edge_exists(to->GetPredecessors(), from, latency)) {
    to->AddPredecessor(HloEdge(latency, from));
    to->indegree_++;
  }
}

void HloGraphNode::UpdateOrAddDependency(
    HloGraphNode* from, HloGraphNode* to,
    const LatencyEstimator* latency_estimator) {
  UpdateOrAddDependency(from, to,
                        latency_estimator->GetLatencyBetween(*from, *to));
}

void HloGraphNode::AddDependency(HloGraphNode* from, HloGraphNode* to,
                                 LatencyEstimator::TimeCost latency) {
  to->AddPredecessor(HloEdge(latency, from));
  to->indegree_++;
  from->AddSuccessor(HloEdge(latency, to));
  from->outdegree_++;
}

void HloGraphNode::AddDependency(HloGraphNode* from, HloGraphNode* to,
                                 const LatencyEstimator* latency_estimator) {
  AddDependency(from, to, latency_estimator->GetLatencyBetween(*from, *to));
}

void HloGraphNode::ResetScheduling() {
  predecessors_.RevertToStaticSize();
  successors_.RevertToStaticSize();
  scheduled_ = false;
  indegree_ = predecessors_.size();
  outdegree_ = successors_.size();
  ready_time_ = std::numeric_limits<TimeCost>::max();
  force_delay_after_target_ = false;
  ClearAnnotation();
}

ResourcesVector HloGraphNode::GetNetResources() const {
  if (GetOpcode() != HloOpcode::kWhile) {
    return rare_->resources;
  }
  ResourcesVector result;
  for (const auto& [resource, usage] : rare_->resources) {
    if (usage == ResourceUsageType::kResourceOccupy &&
        !DoesReleaseResource(resource)) {
      result.push_back(std::make_pair(resource, usage));
    }
    if (usage == ResourceUsageType::kResourceRelease &&
        !DoesOccupyResource(resource)) {
      result.push_back(std::make_pair(resource, usage));
    }
  }
  return result;
}

absl::Status HloGraphNode::SetAnnotation(int64_t annotation) {
  TF_RET_CHECK(annotation_ == kInvalidAnnotation)
      << "Instruction " << instr_->name()
      << " has an existing annotation: " << annotation_;
  annotation_ = annotation;
  return absl::OkStatus();
}

std::string HloGraphNode::ToString(const AsyncTracker* async_tracker) const {
  std::string result;
  absl::StrAppend(&result, "Instr: ", instr_->ToShortString(), "\n");
  absl::StrAppend(&result, "ReadyTime: ", ready_time_, "\n");
  absl::StrAppend(&result, "Indegree: ", indegree_, "\n");
  absl::StrAppend(&result, "Outdegree: ", outdegree_, "\n");
  absl::StrAppend(&result, "Cost: ", cost_, "\n");
  absl::StrAppend(&result, "Async Depth: ", async_depth_, "\n");
  absl::StrAppend(&result, "Depth: ", depth_, "\n");
  absl::StrAppend(&result, "Graph Depth: ", graph_depth_, "\n");
  absl::StrAppend(&result, "Force Delay: ", force_delay_, "\n");
  absl::StrAppend(&result, "Force Early: ", force_early_, "\n");
  absl::StrAppend(
      &result, "Force Delay After Target: ", force_delay_after_target_, "\n");
  absl::StrAppend(&result, "Predecessors:\n");
  for (const HloEdge& e : GetPredecessors()) {
    absl::StrAppend(&result, e.ToString());
  }
  absl::StrAppend(&result, "Successors:\n");
  for (const HloEdge& e : GetSuccessors()) {
    absl::StrAppend(&result, e.ToString());
  }
  if (async_tracker != nullptr) {
    absl::StrAppend(&result, "Resources:\n");
    for (const auto& [resource, usage] : rare_->resources) {
      absl::StrAppend(
          &result, "\tResource: ", async_tracker->GetResourceName(resource),
          " usage: ", async_tracker->GetResourceUsageName(usage), "\n");
    }
  }
  return result;
}

void HloGraphNode::InitBitFields() {
  does_occupy_any_resource_ = false;
  does_release_any_resource_ = false;
  is_supported_async_done_ = false;
  is_supported_async_start_ = false;
  has_operand_that_is_supported_async_done_ = false;
  has_user_that_is_supported_async_start_ = false;
  scheduled_ = false;
  valuable_for_selective_overlap_ = true;
  releases_selective_resource_ = false;
  occupies_selective_resource_ = false;
  has_recursive_resources_ = false;
}

void HloScheduleGraph::PrintSizes() const {
  LOG(INFO) << "HloScheduleGraph sizes: node_storage: " << node_storage_.size()
            << " preds: " << predecessors_storage_.size()
            << " succs: " << successors_storage_.size()
            << " rare: " << rare_storage_.size()
            << " sharable_resources: " << sharable_resources_storage_.size()
            << " nodes: " << nodes_.size()
            << " original_order: " << original_order_.size();
}

int64_t HloScheduleGraph::OriginalInstructionPosition(
    const HloInstruction* instr) const {
  auto it = nodes_.find(instr);
  CHECK(it != nodes_.end());
  return it->second;
}

std::vector<double> HloScheduleGraph::GetPreferences() const {
  std::vector<double> preferences;
  preferences.reserve(original_order_.size());
  for (const HloInstruction* instr : original_order_) {
    preferences.push_back(GetNodePtr(instr)->GetPreference());
  }
  return preferences;
}

void HloScheduleGraph::SetPreferences(const std::vector<double>& preferences) {
  CHECK_EQ(preferences.size(), original_order_.size());
  for (int i = 0; i < original_order_.size(); ++i) {
    GetNodePtr(original_order_[i])->SetPreference(preferences[i]);
  }
}

void HloScheduleGraph::RecordStaticEdges() {
  for (auto& pair : nodes_) {
    node_storage_[pair.second].predecessors_.RecordStaticSize();
    node_storage_[pair.second].successors_.RecordStaticSize();
  }
}

void HloScheduleGraph::ResetScheduling() {
  for (auto& pair : nodes_) {
    node_storage_[pair.second].ResetScheduling();
  }
  for (auto& pair : nodes_) {
    node_storage_[pair.second].UpdateReadyNodesIfScheduled();
  }
  for (auto& edge : predecessors_storage_) {
    edge.ResetSharableResources();
  }
  for (auto& edge : successors_storage_) {
    edge.ResetSharableResources();
  }
  sharable_resources_storage_.resize(1);
}

}  // namespace xla
