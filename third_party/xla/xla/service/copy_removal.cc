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

#include "xla/service/copy_removal.h"

#include <cstdint>
#include <functional>
#include <map>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_computation.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/util.h"

namespace xla {
namespace {

struct InstructionInfo {
  HloInstruction* value_definition = nullptr;
  bool is_definition = false;
};

struct ComputationIdLess {
  bool operator()(const HloComputation* a, const HloComputation* b) const {
    return a->unique_id() < b->unique_id();
  }
};

using InstructionMap = HloInstructionMap<InstructionInfo>;
using InstructionEntry = InstructionMap::value_type;
using LiveRangeRegions =
    std::map<const HloComputation*, InstructionMap, ComputationIdLess>;

const InstructionInfo* FindInstruction(const LiveRangeRegions& regions,
                                       HloInstruction* instr) {
  auto comp_it = regions.find(instr->parent());
  if (comp_it == regions.end()) {
    return nullptr;
  }
  auto it = comp_it->second.find(instr);
  return it == comp_it->second.end() ? nullptr : &it->second;
}

// Bitmask encoding the relative execution order of two instructions or regions.
enum RuntimeOrder : uint8_t {
  kNoOverlap = 0,
  kSameInstr = 1 << 0,
  kBeforeStart = 1 << 1,
  kAfterEnd = 1 << 2,
  kBeforeStartOrSameInstr = kBeforeStart | kSameInstr,
  kUnordered = kBeforeStart | kAfterEnd,
};

RuntimeOrder ReverseOrder(RuntimeOrder order) {
  return static_cast<RuntimeOrder>((order & kSameInstr) |
                                   ((order & kBeforeStart) ? kAfterEnd : 0) |
                                   ((order & kAfterEnd) ? kBeforeStart : 0));
}

RuntimeOrder ToRuntimeOrder(HloOrdering::ExecutionConstraint constraint) {
  switch (constraint) {
    case HloOrdering::ExecutionConstraint::kIsSame:
      return kSameInstr;
    case HloOrdering::ExecutionConstraint::kRunBeforeEnd:
      return kBeforeStartOrSameInstr;
    case HloOrdering::ExecutionConstraint::kRunBeforeStart:
      return kBeforeStart;
    case HloOrdering::ExecutionConstraint::kRunAfter:
      return kAfterEnd;
    case HloOrdering::ExecutionConstraint::kRunExclusiveBefore:
    case HloOrdering::ExecutionConstraint::kRunExclusiveAfter:
      return kNoOverlap;
    case HloOrdering::ExecutionConstraint::kUnordered:
      return kUnordered;
  }
}

struct RegionRelation {
  bool overlap = false;
  bool intercept = false;
};

class RelativeLocationAnalysis {
 public:
  RelativeLocationAnalysis(HloOrdering* ordering, const AliasInfo* alias_info)
      : ordering_(ordering), alias_info_(alias_info) {}

  RegionRelation Analyze(const LiveRangeRegions& range1,
                         const LiveRangeRegions& range2,
                         absl::FunctionRef<bool(RegionRelation)> stop_if) {
    RegionRelation result;
    std::vector<HloInstruction*> unordered_ops;
    for (const auto& [comp1, instrs1] : range1) {
      for (const auto& [comp2, instrs2] : range2) {
        if (!ordering_->call_graph().Dominates(comp1, comp2)) {
          continue;
        }
        for (const InstructionEntry& entry2 : instrs2) {
          const bool can_modify = CanIntercept(entry2, range1);
          uint8_t ordered_mask = kNoOverlap;
          bool ordered_intercept = false;
          bool unordered_intercept = false;
          unordered_ops.clear();

          for (const InstructionEntry& entry1 : instrs1) {
            auto [order, intercept] =
                CompareEntries(entry1, entry2, can_modify);
            if (order == kUnordered) {
              unordered_ops.push_back(entry1.first);
              unordered_intercept |= intercept;
            } else {
              ordered_mask |= order;
              ordered_intercept |= intercept;
            }
            RegionRelation early{true, result.intercept || ordered_intercept ||
                                           unordered_intercept};
            if ((ordered_mask & kUnordered) == kUnordered && stop_if(early)) {
              return early;
            }
          }

          if (!ForceRuntimeOrder(unordered_ops, entry2.first, ordered_mask)) {
            result.overlap = true;
            result.intercept |= unordered_intercept;
          } else {
            result.overlap |= (ordered_mask & kUnordered) == kUnordered;
          }
          result.intercept |= ordered_intercept;
          if (stop_if(result)) {
            return result;
          }
        }
      }
    }
    return result;
  }

  bool CommitControlDependencies() {
    if (ctrl_deps_.empty()) {
      return true;
    }
    auto* ordering = dynamic_cast<PredecessorHloOrdering*>(ordering_);
    if (ordering == nullptr) {
      return false;
    }
    absl::flat_hash_map<
        HloComputation*,
        absl::flat_hash_map<const HloInstruction*,
                            absl::flat_hash_set<const HloInstruction*>>>
        updates;
    for (const auto& [succ, preds] : ctrl_deps_) {
      auto& succ_preds = updates[succ->parent()][succ];
      for (HloInstruction* pred : preds) {
        CHECK_OK(pred->AddControlDependencyTo(succ));
        succ_preds.insert(pred);
      }
    }
    for (auto& [comp, comp_updates] : updates) {
      ordering->reachability_map(comp).UpdateMultipleInstructions(
          std::move(comp_updates));
    }
    for (const auto& [succ, preds] : ctrl_deps_) {
      for (HloInstruction* pred : preds) {
        DCHECK_EQ(ordering_->GetExecutionConstraint(succ, pred),
                  HloOrdering::ExecutionConstraint::kRunAfter);
      }
    }
    return true;
  }

 private:
  using InstructionPair = std::pair<HloInstruction*, HloInstruction*>;

  struct SavedEntry {
    RuntimeOrder relation;
    int64_t version;  // Negative for permanent entries; else ctrl_deps_version.
  };

  std::pair<RuntimeOrder, bool> CompareEntries(const InstructionEntry& entry1,
                                               const InstructionEntry& entry2,
                                               bool instr2_can_modify) {
    HloInstruction* def = entry1.second.value_definition;
    HloInstruction* use = entry1.first;
    HloInstruction* instr2 = entry2.first;
    RuntimeOrder order = ComputeRuntimeOrdering(instr2, use);
    if (order == kSameInstr &&
        entry1.second.is_definition != entry2.second.is_definition) {
      order = entry1.second.is_definition ? kBeforeStart : kAfterEnd;
    }
    const HloOpcode op2 = instr2->opcode();
    const bool forced_intercept =
        HloDataflowAnalysis::IsAsynchronousOperationStart(op2) ||
        HloDataflowAnalysis::IsAsynchronousOperationDone(op2) ||
        op2 == HloOpcode::kCollectivePermute;
    if (def == nullptr || !instr2_can_modify) {
      return {order, forced_intercept};
    }
    if (def->opcode() == HloOpcode::kParameter &&
        use == use->parent()->root_instruction()) {
      return {order, true};
    }
    if (use->parent() == def->parent()) {
      // Writes inside a while body do not intercept an outer def/use chain.
      if ((def->opcode() == HloOpcode::kWhile &&
           instr2->parent() == def->while_body() && order == kBeforeStart) ||
          (use->opcode() == HloOpcode::kWhile &&
           instr2->parent() == use->while_body() &&
           ComputeRuntimeOrdering(def, instr2) == kBeforeStart)) {
        return {order, false};
      }
      // When one conditional branch returns `tuple(exp, exp)` (shareable) and
      // another returns a tuple producing op (for example a fusion returning
      // `tuple(abs, negate)`), its distinct outputs must not share a buffer.
      // Because `InstructionInfo` does not track the tuple output index, treat
      // outputs of a tuple producing op in a conditional branch as interfering.
      if (def == instr2 && def->shape().IsTuple() &&
          !def->parent()
               ->caller_instructions(HloOpcode::kConditional)
               .empty()) {
        return {order, true};
      }
    }
    if (!forced_intercept && (order == kBeforeStart || order == kUnordered)) {
      RuntimeOrder def_order = ComputeRuntimeOrdering(instr2, def);
      if (def_order == kAfterEnd || def_order == kUnordered) {
        return {order, true};
      }
    }
    return {order, forced_intercept};
  }

  bool ForceRuntimeOrder(absl::Span<HloInstruction* const> unordered_ops,
                         HloInstruction* instr2, uint8_t desired) {
    if (unordered_ops.empty()) {
      return true;
    }
    if ((desired != kBeforeStart && desired != kAfterEnd) ||
        unordered_ops[0]->parent() != instr2->parent()) {
      return false;
    }
    const bool instr2_first = (desired == kBeforeStart);
    auto can_order = [this](HloInstruction* pred, HloInstruction* succ) {
      return pred != pred->parent()->root_instruction() &&
             !(succ->opcode() == HloOpcode::kCopy &&
               absl::c_any_of(alias_info_->GetInPlaceInputOutputPairs(pred),
                              [&](const auto& pair) {
                                return pred->operand(
                                           pair.first.operand_number) ==
                                       succ->operand(0);
                              }));
    };
    for (HloInstruction* instr1 : unordered_ops) {
      if (!can_order(instr2_first ? instr2 : instr1,
                     instr2_first ? instr1 : instr2)) {
        return false;
      }
    }
    for (HloInstruction* instr1 : unordered_ops) {
      RecordForcedOrder(instr2_first ? instr2 : instr1,
                        instr2_first ? instr1 : instr2);
    }
    return true;
  }

  bool CanIntercept(const InstructionEntry& entry,
                    const LiveRangeRegions& region) const {
    HloInstruction* instr = entry.first;
    if (!entry.second.is_definition) {
      return absl::c_any_of(
          alias_info_->GetInPlaceInputOutputPairs(instr),
          [&](const auto& pair) {
            return FindInstruction(region, instr->mutable_operand(
                                               pair.first.operand_number)) !=
                   nullptr;
          });
    }
    if (instr->opcode() == HloOpcode::kCopy) {
      const InstructionInfo* info = FindInstruction(region, instr);
      return !(info != nullptr && info->value_definition != nullptr &&
               ShapeUtil::Equal(instr->shape(), instr->operand(0)->shape()));
    }
    switch (instr->opcode()) {
      case HloOpcode::kParameter:
      case HloOpcode::kTuple:
      case HloOpcode::kGetTupleElement:
      case HloOpcode::kWhile:
      case HloOpcode::kCall:
      case HloOpcode::kConditional:
        return false;
      default:
        return true;
    }
  }

  std::optional<SavedEntry> LookupSaved(HloInstruction* a,
                                        HloInstruction* b) const {
    const bool swap = a->unique_id() > b->unique_id();
    auto it = saved_relations_.find(swap ? InstructionPair{b, a}
                                         : InstructionPair{a, b});
    if (it == saved_relations_.end()) {
      return std::nullopt;
    }
    RuntimeOrder rel =
        swap ? ReverseOrder(it->second.relation) : it->second.relation;
    return SavedEntry{rel, it->second.version};
  }

  RuntimeOrder SaveRelation(HloInstruction* a, HloInstruction* b,
                            RuntimeOrder rel, int64_t version = -1) {
    const bool swap = a->unique_id() > b->unique_id();
    saved_relations_[swap ? InstructionPair{b, a} : InstructionPair{a, b}] = {
        swap ? ReverseOrder(rel) : rel, version};
    return rel;
  }

  void RecordForcedOrder(HloInstruction* pred, HloInstruction* succ) {
    SaveRelation(pred, succ, kBeforeStart);
    std::vector<HloInstruction*>& preds = ctrl_deps_[succ];
    for (HloInstruction*& existing : preds) {
      if (existing == pred) {
        return;
      }
      std::optional<SavedEntry> saved = LookupSaved(pred, existing);
      if (saved && saved->relation != kUnordered) {
        if (saved->relation == kAfterEnd) {
          existing = pred;
          ++ctrl_deps_version_;
        } else {
          CHECK_EQ(saved->relation, kBeforeStart);
        }
        return;
      }
    }
    preds.push_back(pred);
    ++ctrl_deps_version_;
  }

  RuntimeOrder ComputeRuntimeOrdering(HloInstruction* a, HloInstruction* b) {
    if (a == b) {
      return kSameInstr;
    }
    std::optional<SavedEntry> saved = LookupSaved(a, b);
    if (saved && (saved->version < 0 || saved->version == ctrl_deps_version_)) {
      return saved->relation;
    }
    if (!saved) {
      RuntimeOrder rel =
          ToRuntimeOrder(ordering_->GetExecutionConstraint(a, b));
      if (rel != kUnordered || a->parent() != b->parent()) {
        return SaveRelation(a, b, rel);
      }
    }
    if (!ctrl_deps_.empty()) {
      auto precedes = [&](HloInstruction* x, HloInstruction* y) {
        auto it = ctrl_deps_.find(y);
        return it != ctrl_deps_.end() &&
               absl::c_any_of(it->second, [&](HloInstruction* pred) {
                 RuntimeOrder r = ComputeRuntimeOrdering(x, pred);
                 return r != kNoOverlap && (r & kAfterEnd) == 0;
               });
      };
      if (precedes(a, b)) {
        return SaveRelation(a, b, kBeforeStart);
      }
      if (precedes(b, a)) {
        return SaveRelation(a, b, kAfterEnd);
      }
    }
    return SaveRelation(a, b, kUnordered, ctrl_deps_version_);
  }

  HloOrdering* ordering_;
  const AliasInfo* alias_info_;
  absl::flat_hash_map<InstructionPair, SavedEntry> saved_relations_;
  HloInstructionMap<std::vector<HloInstruction*>> ctrl_deps_;
  int64_t ctrl_deps_version_ = 0;
};

}  // namespace

CopyRemover::CopyRemover(
    const HloModule& module, const HloAliasAnalysis& alias_analysis,
    const AliasInfo* alias_info, HloOrdering* ordering,
    const absl::flat_hash_set<absl::string_view>& execution_threads,
    std::optional<int64_t> view_color)
    : dataflow_(alias_analysis.dataflow_analysis()),
      alias_info_(alias_info),
      ordering_(ordering),
      view_color_(view_color) {
  // Assign postorder instruction ids (parameters first, callees before callers)
  // as a strict weak ordering tie breaker for sorting buffer values.
  absl::flat_hash_map<int64_t, int64_t> instruction_ids;
  int64_t id = 0;
  absl::flat_hash_set<HloComputation*> visited;
  std::function<void(HloComputation*)> assign_ids = [&](HloComputation* comp) {
    if (!visited.insert(comp).second) {
      return;
    }
    for (HloInstruction* param : comp->parameter_instructions()) {
      instruction_ids[param->unique_id()] = id++;
    }
    const HloInstructionSequence* seq = ordering->SequentialOrder(*comp);
    std::vector<HloInstruction*> instructions =
        seq != nullptr ? seq->instructions() : comp->MakeInstructionPostOrder();
    for (HloInstruction* instr : instructions) {
      if (instr->opcode() == HloOpcode::kParameter) {
        continue;
      }
      if (instr->opcode() == HloOpcode::kWhile) {
        assign_ids(instr->while_condition());
        assign_ids(instr->while_body());
      } else {
        for (HloComputation* called : instr->called_computations()) {
          assign_ids(called);
        }
      }
      instruction_ids[instr->unique_id()] = id++;
    }
  };

  if (module.has_entry_computation()) {
    assign_ids(module.entry_computation());
  }
  for (HloComputation* comp : module.MakeComputationSorted(execution_threads)) {
    assign_ids(comp);
  }

  auto get_id = [&](const HloValue* v) -> int64_t {
    const HloInstruction* def = v->defining_instruction();
    return def->parent()->IsDeadComputation()
               ? 0
               : instruction_ids.at(def->unique_id());
  };

  absl::flat_hash_map<const HloValue*, ValueNode*> value_to_node;
  for (const HloBuffer& buffer : alias_analysis.buffers()) {
    if (buffer.values().at(0)->defining_instruction()->IsFused()) {
      continue;
    }
    std::vector<const HloValue*> values = buffer.values();
    absl::c_sort(values, [this, &get_id](const HloValue* a, const HloValue* b) {
      if (a == b) {
        return false;
      }
      const bool a_before_b = get_id(a) < get_id(b);
      const HloValue* first = a_before_b ? a : b;
      const HloValue* second = a_before_b ? b : a;
      if (ordering_->IsDefinedBefore(*first, *second)) {
        return a_before_b;
      }
      if (ordering_->IsDefinedBefore(*second, *first)) {
        return !a_before_b;
      }
      return a_before_b;
    });

    ValueNode* head = nullptr;
    ValueNode* tail = nullptr;
    for (const HloValue* value : values) {
      ValueNode* node = &nodes_.emplace_back(value);
      value_to_node[value] = node;
      node->uses.reserve(value->GetUses().size());
      for (const HloUse& use : value->GetUses()) {
        node->uses.push_back(&use);
      }
      if (view_color_.has_value()) {
        AddViewUses(value, node);
      }
      if (tail == nullptr) {
        head = node;
      } else {
        tail->next = node;
        node->prev = tail;
      }
      tail = node;
    }
    tail->next = head;
    head->prev = tail;
    value_lists_.insert(head);
  }

  for (HloComputation* comp : module.MakeNonfusionComputations()) {
    for (HloInstruction* instr : comp->instructions()) {
      if (instr->opcode() != HloOpcode::kCopy) {
        continue;
      }
      const HloValueSet& src_values = dataflow_.GetValueSet(instr->operand(0));
      if (src_values.values().size() == 1) {
        copy_map_[instr] = {
            value_to_node.at(&src_values.GetUniqueValue()),
            value_to_node.at(&dataflow_.GetUniqueValueAt(instr))};
      }
    }
  }

  XLA_VLOG_LINES(3, ToString());
  DCHECK_OK(Verify());
}

void CopyRemover::AddViewUses(const HloValue* value, ValueNode* node) {
  auto is_view = [this](const HloInstruction* op) {
    return op->shape().IsArray() && op->shape().has_layout() &&
           op->shape().layout().memory_space() == *view_color_;
  };
  if (is_view(value->defining_instruction())) {
    return;
  }
  absl::flat_hash_set<const HloInstruction*> visited;
  std::vector<const HloInstruction*> worklist;
  for (const HloUse& use : value->GetUses()) {
    if (use.operand_number == 0 && is_view(use.instruction) &&
        visited.insert(use.instruction).second) {
      worklist.push_back(use.instruction);
    }
  }
  while (!worklist.empty()) {
    const HloInstruction* view = worklist.back();
    worklist.pop_back();
    for (HloInstruction* user : view->users()) {
      if (is_view(user) && (user->opcode() == HloOpcode::kBitcast ||
                            user->opcode() == HloOpcode::kCopy)) {
        if (visited.insert(user).second) {
          worklist.push_back(user);
        }
        continue;
      }
      for (int64_t i = 0; i < user->operand_count(); ++i) {
        if (user->operand(i) == view) {
          const HloUse& use = view_uses_.emplace_back(user, i, ShapeIndex{});
          view_uses_by_pointer_.insert(&use);
          node->uses.push_back(&use);
        }
      }
    }
  }
}

absl::Status CopyRemover::Verify() const {
  for (const ValueNode* head : value_lists_) {
    const ValueNode* p = head;
    do {
      TF_RET_CHECK(p->prev->next == p);
      TF_RET_CHECK(p->next->prev == p);
      if (auto it = copy_map_.find(p->value->defining_instruction());
          it != copy_map_.end()) {
        TF_RET_CHECK(it->second.dest == p);
      }
      for (const HloUse* use : p->uses) {
        if (use->instruction->opcode() == HloOpcode::kCopy && !IsViewUse(use)) {
          if (auto it = copy_map_.find(use->instruction);
              it != copy_map_.end()) {
            TF_RET_CHECK(it->second.src == p);
          }
        }
      }
      p = p->next;
    } while (p != head);
  }
  return absl::OkStatus();
}

bool CopyRemover::TryElideCopy(
    const HloInstruction* copy, bool use_region_analysis,
    bool insert_post_scheduling_control_dependencies,
    const std::function<bool(const HloInstruction* copy)>&
        should_skip_removal) {
  VLOG(2) << "Trying to remove " << copy->name();
  if (should_skip_removal != nullptr && should_skip_removal(copy)) {
    VLOG(2) << copy->name() << " skipped by should_skip_removal.";
    return false;
  }
  auto copy_it = copy_map_.find(copy);
  if (copy_it == copy_map_.end()) {
    VLOG(2) << copy->name() << " is not removable (not in copy map).";
    return false;
  }
  if (!ShapeUtil::Equal(copy->shape(), copy->operand(0)->shape())) {
    VLOG(2) << copy->name() << " is not removable (shape/layout mismatch).";
    return false;
  }
  ValueNode* src = copy_it->second.src;
  ValueNode* dest = copy_it->second.dest;

  // Check every (s, d) pair across the boundary because LiveRangeBefore is not
  // transitive when values span nonoverlapping computations (such as sibling
  // conditional branches).
  auto all_before = [&](ValueNode* a, ValueNode* b) {
    for (ValueNode* d = b; d != nullptr; d = Next(*d)) {
      for (ValueNode* s = a; s != nullptr; s = Prev(*s)) {
        if (!LiveRangeBefore(*s, *d)) {
          return false;
        }
      }
    }
    return true;
  };
  auto add_control_deps = [](ValueNode* a, ValueNode* b) {
    if (a == nullptr || b == nullptr) {
      return;
    }
    HloInstruction* dst_instr = b->value->instruction();
    for (const HloUse* use : a->uses) {
      if (use->instruction->parent() == dst_instr->parent() &&
          use->instruction != dst_instr) {
        CHECK_OK(use->instruction->AddControlDependencyTo(dst_instr));
      }
    }
  };
  auto can_merge = [&](ValueNode* before1, ValueNode* after1,
                       ValueNode* before2, ValueNode* after2,
                       bool dest_spliced_into_src) {
    if (all_before(before1, after1) && all_before(before2, after2)) {
      if (insert_post_scheduling_control_dependencies) {
        add_control_deps(before1, after1);
        add_control_deps(before2, after2);
      }
      return true;
    }
    if (!use_region_analysis ||
        copy->operand(0)->opcode() == HloOpcode::kBroadcast) {
      VLOG(2) << copy->name() << " live ranges are not strictly ordered.";
      return false;
    }
    if (ValuesInterfere(src, dest, dest_spliced_into_src)) {
      VLOG(2) << copy->name() << " rejected by region interference analysis.";
      return false;
    }
    return true;
  };

  // Given source buffer {s_0, ..., s_n} and destination buffer {d_0, ..., d_m},
  // two interbuffer merges yield a valid linear order after dropping the copy:
  //   1) `dest` is d_0 (copied from s_x): {s_0..s_x, d_1..d_m, s_{x+1}..s_n},
  //      requiring {s_0..s_x} before {d_1..d_m} and {d_0..d_m} before
  //      {s_{x+1}..s_n}.
  //   2) `src` is s_n (copied into d_y): {d_0..d_{y-1}, s_0..s_n,
  //      d_{y+1}..d_m}, requiring {d_0..d_{y-1}} before {s_0..s_n} and
  //      {s_0..s_n} before {d_{y+1}..d_m}.
  if (src->next == dest) {
    // Source and destination already occupy adjacent slots in the same buffer.
  } else if (IsHead(*dest)) {
    // Keep the copy if it feeds a variadic scatter that also reads src so
    // shared operands do not alias in place.
    auto is_variadic_scatter = [](const HloInstruction* op) {
      const HloInstruction* root =
          op->opcode() == HloOpcode::kFusion ? op->fused_expression_root() : op;
      return root->opcode() == HloOpcode::kScatter && root->shape().IsTuple();
    };
    if (ValueNode* next = Next(*dest); next != nullptr) {
      const HloInstruction* consumer = next->value->defining_instruction();
      if (is_variadic_scatter(consumer) &&
          absl::c_any_of(src->uses, [consumer](const HloUse* use) {
            return use->instruction == consumer;
          })) {
        VLOG(2) << copy->name() << " feeds a variadic scatter sharing src.";
        return false;
      }
    }
    if (!can_merge(src, Next(*dest), dest->prev, Next(*src),
                   /*dest_spliced_into_src=*/true)) {
      return false;
    }
    SpliceAfter(dest, src);
  } else if (IsTail(*src)) {
    if (!can_merge(Prev(*dest), src->next, src, Next(*dest),
                   /*dest_spliced_into_src=*/false)) {
      return false;
    }
    SpliceAfter(src->next, Prev(*dest));
  } else {
    // Elide copies between nonadjacent slots in the same buffer when all
    // intervening values follow the copy's uses (such as sibling branches).
    ValueNode* n = src->next;
    while (n != src && n != dest) {
      n = n->next;
    }
    if (n != dest) {
      VLOG(2) << copy->name()
              << " copies middle of source buffer to middle of dest buffer.";
      return false;
    }
    ValueNode merged(src->value);
    merged.uses = dest->uses;
    for (n = src->next; n != dest; n = n->next) {
      if (!LiveRangeBefore(merged, *n)) {
        VLOG(2) << copy->name()
                << " interferes with intervening value in same buffer.";
        return false;
      }
    }
  }

  RemoveCopyValue(dest, src);
  VLOG(2) << "Elided " << copy->name();
  DCHECK_OK(Verify());
  return true;
}

void CopyRemover::RemoveCopyValue(ValueNode* copy_node,
                                  ValueNode* operand_node) {
  HloInstruction* copy = copy_node->value->defining_instruction();
  CHECK_EQ(copy->opcode(), HloOpcode::kCopy);
  CHECK_NE(operand_node, copy_node);
  if (value_lists_.erase(copy_node) > 0) {
    value_lists_.insert(copy_node->next);
  }

  copy_node->prev->next = copy_node->next;
  copy_node->next->prev = copy_node->prev;

  auto it = absl::c_find_if(operand_node->uses, [this, copy](const HloUse* u) {
    return u->instruction == copy && !IsViewUse(u);
  });
  CHECK(it != operand_node->uses.end());
  operand_node->uses.erase(it);

  for (const HloUse* use : copy_node->uses) {
    operand_node->uses.push_back(use);
    if (use->instruction->opcode() == HloOpcode::kCopy && !IsViewUse(use)) {
      if (auto map_it = copy_map_.find(use->instruction);
          map_it != copy_map_.end()) {
        map_it->second.src = operand_node;
      }
    }
  }
  copy_map_.erase(copy);
}

bool CopyRemover::LiveRangeBefore(const ValueNode& a, const ValueNode& b) {
  if (a.uses.empty()) {
    return ordering_->IsDefinedBefore(*a.value, *b.value);
  }
  if (a.value->IsRootOf(b.value->defining_instruction()->parent())) {
    return false;
  }
  return ordering_->UsesBeforeValueDefinition(
      a.uses, *b.value, dataflow_, alias_info_,
      /*use_is_always_before_def_in_same_instr=*/false);
}

void CopyRemover::SpliceAfter(ValueNode* head, ValueNode* insert_after) {
  DCHECK(IsHead(*head));
  value_lists_.erase(head);

  ValueNode* tail = head->prev;
  tail->next = insert_after->next;
  insert_after->next->prev = tail;

  insert_after->next = head;
  head->prev = insert_after;
}

bool CopyRemover::ValuesInterfere(const ValueNode* src, const ValueNode* dest,
                                  bool dest_spliced_into_src) {
  auto build_regions = [this](const ValueNode* start) {
    LiveRangeRegions regions;
    ForEachValueInRange(start, [&](const ValueNode* node) {
      HloInstruction* def = node->value->instruction();
      regions[def->parent()][def].is_definition = true;
      for (const HloUse* use : node->uses) {
        regions[use->instruction->parent()][use->instruction].value_definition =
            def;
      }
    });
    return regions;
  };
  LiveRangeRegions src_regions = build_regions(src);
  LiveRangeRegions dest_regions = build_regions(dest);

  auto interferes = [dest_spliced_into_src](RegionRelation r1,
                                            RegionRelation r2) {
    if (r1.overlap && r2.overlap) {
      return true;
    }
    const bool any_intercept = r1.intercept || r2.intercept;
    return (r1.overlap &&
            (dest_spliced_into_src ? r1.intercept : any_intercept)) ||
           (r2.overlap &&
            (dest_spliced_into_src ? any_intercept : r2.intercept));
  };

  RelativeLocationAnalysis analysis(ordering_, alias_info_);
  RegionRelation r1 =
      analysis.Analyze(src_regions, dest_regions,
                       [&](RegionRelation r) { return interferes(r, {}); });
  if (interferes(r1, {})) {
    return true;
  }
  RegionRelation r2 =
      analysis.Analyze(dest_regions, src_regions,
                       [&](RegionRelation r) { return interferes(r1, r); });
  if (interferes(r1, r2)) {
    return true;
  }
  if (!analysis.CommitControlDependencies()) {
    VLOG(2) << "Could not commit control dependencies for unordered ops.";
    return true;
  }
  return false;
}

void CopyRemover::ForEachValueInRange(
    const ValueNode* start,
    absl::FunctionRef<void(const ValueNode*)> visitor) const {
  const ValueNode* p = start;
  do {
    visitor(p);
    p = p->next;
  } while (p != start);
}

std::string CopyRemover::ToString() const {
  std::string out = "CopyRemover:\n  Def-use chains in each buffer:\n";
  for (const ValueNode* head : value_lists_) {
    absl::StrAppend(&out, "    Buffer defined by ",
                    head->value->ToShortString(), ":\n");
    ForEachValueInRange(head, [&](const ValueNode* node) {
      absl::StrAppend(&out, "      ", node->value->ToShortString(), ", uses: ",
                      absl::StrJoin(node->uses, "; ",
                                    [](std::string* s, const HloUse* use) {
                                      absl::StrAppend(s, use->ToString());
                                    }),
                      "\n");
    });
  }
  absl::StrAppend(&out, "  Potentially removable copies:\n");
  for (const auto& [copy, nodes] : copy_map_) {
    absl::StrAppend(&out, "    ", copy->name(), " : ",
                    nodes.src->value->ToShortString(), " => ",
                    nodes.dest->value->ToShortString(), "\n");
  }
  return out;
}

}  // namespace xla
