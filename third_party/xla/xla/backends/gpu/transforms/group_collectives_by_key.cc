/* Copyright 2026 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/gpu/transforms/group_collectives_by_key.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/log/log.h"
#include "absl/log/vlog_is_on.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/backends/gpu/transforms/collectives/collective_domain.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/collective_combiner_utils.h"
#include "xla/service/collective_ops_utils.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/side_effect_util.h"
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla::gpu {
namespace {

struct CollectiveGroup {
  std::string key;
  std::vector<HloInstruction*> collectives;
};

// Trivial ops we drop from the printed reachability chain — they carry no
// useful debugging signal beyond reshape/element-extract bookkeeping.
bool IsTrivialOnPath(const HloInstruction* instr) {
  switch (instr->opcode()) {
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
    case HloOpcode::kBitcast:
    case HloOpcode::kReshape:
    case HloOpcode::kTranspose:
    case HloOpcode::kCopy:
    case HloOpcode::kBroadcast:
    case HloOpcode::kOptimizationBarrier:
      return true;
    default:
      return false;
  }
}

// Reconstruct a dataflow path from `from` to `to` by greedy descent: at each
// step pick any user (or control-successor) that still reaches `to`. Assumes
// reachability.IsReachable(from, to) == true.
std::vector<const HloInstruction*> ReconstructPath(
    const HloInstruction* from, const HloInstruction* to,
    const HloReachabilityMap& reachability) {
  std::vector<const HloInstruction*> path{from};
  const HloInstruction* cur = from;
  while (cur != to) {
    const HloInstruction* next = nullptr;
    for (HloInstruction* u : cur->users()) {
      if (reachability.IsReachable(u, to)) {
        next = u;
        break;
      }
    }
    if (next == nullptr) {
      for (HloInstruction* s : cur->control_successors()) {
        if (reachability.IsReachable(s, to)) {
          next = s;
          break;
        }
      }
    }
    if (next == nullptr) {
      break;  // shouldn't happen if IsReachable was true
    }
    path.push_back(next);
    cur = next;
  }
  return path;
}

std::string FormatReachabilityChain(const HloInstruction* from,
                                    const HloInstruction* to,
                                    const HloReachabilityMap& reachability) {
  std::vector<const HloInstruction*> raw =
      ReconstructPath(from, to, reachability);

  // 1. Drop trivial ops and ops with no op_name metadata (but always keep
  // endpoints).
  std::vector<const HloInstruction*> path;
  path.reserve(raw.size());
  for (size_t i = 0; i < raw.size(); ++i) {
    bool is_endpoint = (i == 0 || i + 1 == raw.size());
    if (is_endpoint ||
        (!IsTrivialOnPath(raw[i]) && !raw[i]->metadata().op_name().empty())) {
      path.push_back(raw[i]);
    }
  }

  // 2. Common op_name prefix to elide. We only consider the two endpoints (the
  // collectives we are reporting on); intermediate fusions / scratch ops can
  // have unrelated op_names like "reshape.1052" that would zero the prefix and
  // defeat the elision.
  std::string prefix;
  if (path.size() >= 2) {
    prefix =
        CommonOpNamePrefix({std::string(path.front()->metadata().op_name()),
                            std::string(path.back()->metadata().op_name())});
  }

  std::string out;
  for (size_t i = 0; i < path.size(); ++i) {
    const HloInstruction* h = path[i];
    absl::string_view op_name = h->metadata().op_name();
    absl::string_view trimmed = op_name.size() >= prefix.size()
                                    ? op_name.substr(prefix.size())
                                    : op_name;
    absl::StrAppend(&out, i == 0 ? "  " : "  -> ", h->name());
    if (!trimmed.empty()) {
      absl::StrAppend(&out, " op=", trimmed);
    }

    // 3. Frontend attributes.
    if (h->has_frontend_attributes()) {
      std::vector<std::string> kvs;
      for (const auto& [k, v] : h->frontend_attributes().map()) {
        kvs.push_back(absl::StrCat(k, "=", v));
      }
      if (!kvs.empty()) {
        absl::StrAppend(&out, " frontend_attributes={",
                        absl::StrJoin(kvs, ", "), "}");
      }
    }
    absl::StrAppend(&out, "\n");
  }
  if (!prefix.empty()) {
    absl::StrAppend(&out, "  (op_name common prefix: '", prefix, "')\n");
  }
  return out;
}

// Builds the embedded computation containing clones of the grouped collectives
// operating on fresh parameters, with a tuple root over their results. The
// computation is added with a neutral base name; AddEmbeddedComputation
// uniquifies it against the module.
HloComputation* BuildGroupComputation(
    HloModule* module, absl::Span<HloInstruction* const> collectives,
    absl::string_view execution_thread) {
  HloComputation::Builder builder("collectives_group");

  std::vector<HloInstruction*> new_collectives;
  new_collectives.reserve(collectives.size());
  int param_idx = 0;
  for (size_t k = 0; k < collectives.size(); ++k) {
    HloInstruction* c = collectives[k];
    std::vector<HloInstruction*> params;
    params.reserve(c->operand_count());
    for (int i = 0; i < c->operand_count(); ++i) {
      params.push_back(builder.AddInstruction(HloInstruction::CreateParameter(
          param_idx++, c->operand(i)->shape(),
          absl::StrCat("c", k, "_operand_", i))));
    }
    // CloneWithNewOperands preserves channel_id, replica_groups, backend config
    // and frontend attributes (including collective_group_key, which we keep on
    // the clone).
    HloInstruction* clone =
        builder.AddInstruction(c->CloneWithNewOperands(c->shape(), params));

    // The scheduling group id applies only at the outer async level; leaving it
    // on the clones makes the latency-hiding scheduler reject the embedded
    // computation for having multiple collective starts in one group. This is
    // the same clone-scoped erase ExplicitCollectivesGroupAsyncWrapper
    // performs.
    clone->erase_frontend_attribute(kXlaSchedulingGroupIdAttr);

    new_collectives.push_back(clone);
  }

  builder.AddInstruction(HloInstruction::CreateTuple(new_collectives));
  HloComputation* group_computation =
      module->AddEmbeddedComputation(builder.Build());
  group_computation->SetExecutionThread(execution_thread);
  return group_computation;
}

// Appends a flat sharding for every leaf of `instruction`. Unknown shardings
// preserve partially annotated tuples without inventing a placement for an
// unannotated member.
void AppendShardingLeaves(const HloInstruction& instruction,
                          std::vector<HloSharding>* shardings) {
  int64_t leaf_count = ShapeUtil::GetLeafCount(instruction.shape());
  if (leaf_count == 0) {
    // Empty tuples may carry one sharding despite having no ShapeTree leaves.
    leaf_count = 1;
  }

  if (!instruction.has_sharding()) {
    shardings->insert(shardings->end(), leaf_count, HloSharding::Unknown());
    return;
  }

  const HloSharding& sharding = instruction.sharding();
  if (sharding.IsTuple()) {
    shardings->insert(shardings->end(), sharding.tuple_elements().begin(),
                      sharding.tuple_elements().end());
  } else {
    shardings->insert(shardings->end(), leaf_count, sharding);
  }
}

absl::Status CreateCollectivesGroup(
    HloComputation* computation, absl::Span<HloInstruction* const> collectives,
    absl::string_view execution_thread) {
  HloModule* module = computation->parent();
  HloComputation* group_comp =
      BuildGroupComputation(module, collectives, execution_thread);
  absl::string_view group_name = group_comp->name();

  std::vector<HloInstruction*> call_operands;
  std::vector<Shape> done_shapes;
  done_shapes.reserve(collectives.size());
  for (HloInstruction* c : collectives) {
    call_operands.insert(call_operands.end(), c->operands().begin(),
                         c->operands().end());
    done_shapes.push_back(c->shape());
  }

  std::vector<const Shape*> param_shapes;
  param_shapes.reserve(call_operands.size());
  for (HloInstruction* op : call_operands) {
    param_shapes.push_back(&op->shape());
  }
  Shape done_shape = ShapeUtil::MakeTupleShape(done_shapes);
  Shape start_shape = ShapeUtil::MakeTupleShape(
      {ShapeUtil::MakeTupleShapeWithPtrs(param_shapes), done_shape});

  HloInstruction* async_start =
      computation->AddInstruction(HloInstruction::CreateAsyncStart(
          start_shape, call_operands, group_comp, execution_thread));
  async_start->SetAndSanitizeName(absl::StrCat(group_name, "-start"));
  HloInstruction* async_done = computation->AddInstruction(
      HloInstruction::CreateAsyncDone(done_shape, async_start));
  async_done->SetAndSanitizeName(absl::StrCat(group_name, "-done"));

  // Merge all frontend attributes of the members. Because every member shares
  // the same collective_group_key, MergeFrontendAttributes deduplicates it to a
  // single value; other shared attributes are preserved. Add the group marker
  // so stream assignment and the scheduler treat this as a launch group.
  FrontendAttributes attrs = MergeFrontendAttributes(collectives);
  (*attrs.mutable_map())[kCollectiveGroupMarkerAttr] = "";
  async_start->set_frontend_attributes(attrs);
  async_done->set_frontend_attributes(attrs);
  group_comp->root_instruction()->set_frontend_attributes(attrs);

  // Preserve metadata and backend config from a representative member.
  async_start->set_metadata(collectives.front()->metadata());
  async_done->set_metadata(collectives.front()->metadata());
  async_start->CopyBackendConfigFrom(collectives.front());
  async_done->CopyBackendConfigFrom(collectives.front());

  bool has_operand_sharding = false;
  for (const HloInstruction* operand : call_operands) {
    has_operand_sharding |= operand->has_sharding();
  }
  bool has_output_sharding = false;
  for (const HloInstruction* collective : collectives) {
    has_output_sharding |= collective->has_sharding();
  }

  std::vector<HloSharding> done_shardings;
  if (has_output_sharding) {
    for (const HloInstruction* collective : collectives) {
      AppendShardingLeaves(*collective, &done_shardings);
    }
    HloSharding done_sharding = HloSharding::Tuple(done_shape, done_shardings);
    async_done->set_sharding(done_sharding);
    group_comp->root_instruction()->set_sharding(std::move(done_sharding));
  }

  if (has_operand_sharding || has_output_sharding) {
    std::vector<HloSharding> start_shardings;
    for (const HloInstruction* operand : call_operands) {
      AppendShardingLeaves(*operand, &start_shardings);
    }
    if (done_shardings.empty()) {
      for (const HloInstruction* collective : collectives) {
        AppendShardingLeaves(*collective, &done_shardings);
      }
    }
    start_shardings.insert(start_shardings.end(), done_shardings.begin(),
                           done_shardings.end());
    async_start->set_sharding(HloSharding::Tuple(start_shape, start_shardings));
  }

  // Rewire each member's uses to a get-tuple-element of the async-done result,
  // preserving per-output sharding, and relay external control dependencies
  // onto the async pair.
  for (size_t i = 0; i < collectives.size(); ++i) {
    HloInstruction* c = collectives[i];
    HloInstruction* replacement = computation->AddInstruction(
        HloInstruction::CreateGetTupleElement(c->shape(), async_done, i));
    if (c->has_sharding()) {
      replacement->set_sharding(c->sharding());
    }
    // Control predecessors of the member gate the whole group's start; control
    // successors wait on the group's done.
    ABSL_RETURN_IF_ERROR(c->CopyAllControlDepsTo(async_start, async_done));
    ABSL_RETURN_IF_ERROR(c->DropAllControlDeps());
    ABSL_RETURN_IF_ERROR(c->ReplaceAllUsesWith(replacement));
    ABSL_RETURN_IF_ERROR(computation->RemoveInstruction(c));
  }

  return absl::OkStatus();
}

// Contracting every group into one async node must leave the dependency graph
// acyclic. Pairwise independence within a group is not sufficient: two groups
// can depend on each other through different, individually independent members.
absl::Status ValidateGroupGraphIsAcyclic(
    absl::Span<const CollectiveGroup> groups,
    const HloReachabilityMap& reachability,
    absl::string_view computation_name) {
  std::vector<std::vector<size_t>> successors(groups.size());
  std::vector<int64_t> in_degree(groups.size(), 0);

  for (size_t i = 0; i < groups.size(); ++i) {
    for (size_t j = 0; j < groups.size(); ++j) {
      if (i == j) {
        continue;
      }

      bool is_reachable = false;
      for (const HloInstruction* from : groups[i].collectives) {
        for (const HloInstruction* to : groups[j].collectives) {
          if (reachability.IsReachable(from, to)) {
            is_reachable = true;
            break;
          }
        }
        if (is_reachable) {
          break;
        }
      }
      if (is_reachable) {
        successors[i].push_back(j);
        ++in_degree[j];
      }
    }
  }

  std::vector<size_t> ready;
  ready.reserve(groups.size());
  for (size_t i = 0; i < groups.size(); ++i) {
    if (in_degree[i] == 0) {
      ready.push_back(i);
    }
  }

  size_t visited = 0;
  while (!ready.empty()) {
    size_t group = ready.back();
    ready.pop_back();
    ++visited;
    for (size_t successor : successors[group]) {
      if (--in_degree[successor] == 0) {
        ready.push_back(successor);
      }
    }
  }

  if (visited != groups.size()) {
    std::vector<absl::string_view> cycle_keys;
    for (size_t i = 0; i < groups.size(); ++i) {
      if (in_degree[i] != 0) {
        cycle_keys.push_back(groups[i].key);
      }
    }
    return FailedPrecondition(
        "Grouping collectives in computation %s would create a dependency "
        "cycle among collective_group_key values {%s}",
        computation_name, absl::StrJoin(cycle_keys, ", "));
  }
  return absl::OkStatus();
}

absl::StatusOr<bool> GroupCollectivesInComputation(
    HloComputation* computation, const HloPredicate& predicate,
    absl::string_view execution_thread) {
  VLOG(1) << "Finding collectives to group in computation "
          << computation->name();

  // A btree_map keeps keys sorted, so groups are processed in a deterministic
  // order without a separate ordering vector.
  using DomainAwareGroupKey =
      std::pair<std::string, CollectiveCommunicationDomain>;
  absl::btree_map<DomainAwareGroupKey, std::vector<HloInstruction*>>
      key_to_collectives;
  for (HloInstruction* instr : computation->instructions()) {
    if (!predicate(instr)) {
      continue;
    }
    std::optional<absl::string_view> key = GetCollectiveGroupKey(*instr);
    if (!key.has_value()) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(CollectiveCommunicationDomain domain,
                     GetCollectiveCommunicationDomain(*instr));
    key_to_collectives[{std::string(*key), domain}].push_back(instr);
  }
  if (key_to_collectives.empty()) {
    return false;
  }

  std::unique_ptr<HloReachabilityMap> reachability =
      HloReachabilityMap::Build(computation);

  // Two-phase: validate and collect every group first, then mutate. Validation
  // reads the reachability map (and, on failure, walks live users() to
  // reconstruct a diagnostic chain), so it must run entirely before any
  // mutation — forming a group inserts async-start/done and GTE instructions
  // that are absent from the map. Deferring all mutation to phase two also
  // guarantees an invalid later key never leaves earlier keys half-transformed.
  std::vector<CollectiveGroup> groups_to_form;
  groups_to_form.reserve(key_to_collectives.size());
  for (auto& [group_key, collectives] : key_to_collectives) {
    const auto& [key, domain] = group_key;
    if (collectives.size() <= 1) {
      LOG(WARNING) << "collective_group_key \"" << key << "\" ("
                   << computation->name() << ", domain=" << domain
                   << ") has only one collective, skipping";
      continue;
    }

    // Sharing a collective_group_key is a frontend assertion that these
    // collectives are independent. Verify it; a violation would create a cycle
    // when we wrap them in a single async call. HloReachabilityMap::Build folds
    // in control-predecessor edges, so this rejects data and control deps.
    for (size_t i = 0; i < collectives.size(); ++i) {
      for (size_t j = i + 1; j < collectives.size(); ++j) {
        const HloInstruction* a = nullptr;
        const HloInstruction* b = nullptr;
        if (reachability->IsReachable(collectives[i], collectives[j])) {
          a = collectives[i];
          b = collectives[j];
        } else if (reachability->IsReachable(collectives[j], collectives[i])) {
          a = collectives[j];
          b = collectives[i];
        }
        if (a != nullptr) {
          std::string chain = FormatReachabilityChain(a, b, *reachability);
          return FailedPrecondition(
              "Collectives %s and %s share collective_group_key=%s but are not "
              "independent (one is reachable from the other).\n"
              "Computation: %s\n Reachability chain %s -> %s:\n%s",
              collectives[i]->name(), collectives[j]->name(), key,
              computation->name(), a->name(), b->name(), chain);
        }
      }
    }

    if (VLOG_IS_ON(1)) {
      std::vector<absl::string_view> names;
      names.reserve(collectives.size());
      for (HloInstruction* c : collectives) {
        names.push_back(c->name());
      }
      VLOG(1) << "Grouping {" << absl::StrJoin(names, ", ")
              << "} with collective_group_key=" << key;
    }

    groups_to_form.push_back({key, std::move(collectives)});
  }

  ABSL_RETURN_IF_ERROR(ValidateGroupGraphIsAcyclic(groups_to_form, *reachability,
                                              computation->name()));

  // Phase two: every group validated against a consistent, mutation-free map,
  // so it is now safe to form them.
  bool changed = false;
  for (const CollectiveGroup& group : groups_to_form) {
    ABSL_RETURN_IF_ERROR(CreateCollectivesGroup(computation, group.collectives,
                                           execution_thread));
    changed = true;
  }
  return changed;
}

}  // namespace

absl::StatusOr<bool> GroupCollectivesByKey::RunImpl(
    HloModule* module,
    const absl::flat_hash_set<absl::string_view>& execution_threads) {
  bool changed = false;
  for (HloComputation* comp :
       module->MakeNonfusionComputations(execution_threads)) {
    // Skip the bodies of async groups (including ones this pass already
    // formed). Their cloned collectives keep collective_group_key, so
    // revisiting them would re-group and break idempotence.
    if (comp->IsAsyncComputation()) {
      continue;
    }
    ABSL_ASSIGN_OR_RETURN(bool comp_changed,
                     GroupCollectivesInComputation(comp, predicate_,
                                                   comp->execution_thread()));
    changed |= comp_changed;
  }
  return changed;
}

}  // namespace xla::gpu
