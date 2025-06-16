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
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_operand_index.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/analysis/hlo_reachability.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_opcode.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/layout.h"
#include "xla/map_util.h"
#include "xla/service/hlo_buffer.h"
#include "xla/service/hlo_value.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/status_macros.h"
#include "xla/tsl/platform/status.h"
#include "xla/util.h"

using absl::StrAppend;

namespace xla {

// Summarize additional relations into a single runtime ordering, assuming
// both relations are modeling constraints of the same source instruction.
void Relation::UnionRelationFromSameSource(const Relation& rel) {
  CHECK_LE(orders_.size(), 1);
  CHECK_EQ(rel.orders_.size(), 1);
  if (orders_.empty()) {
    orders_.push_back(rel.orders_[0]);
  } else {
    orders_[0] = Union(orders_[0], rel.orders_[0]);
  }
  intercept_def_use_ = intercept_def_use_ || rel.intercept_def_use_;
}

// Summarize additional relations into disjoint runtime orderings, assuming
// the relations are modeling constraints of different source instructions.
void Relation::UnionRelationFromDifferentSource(const Relation& rel) {
  if (rel.orders_.empty()) {
    return;
  }
  CHECK_EQ(rel.orders_.size(), 1);
  intercept_def_use_ = intercept_def_use_ || rel.intercept_def_use_;
  for (auto& local_order : orders_) {
    if (OverwriteIfSubsumes(rel.orders_[0], &local_order)) {
      return;
    }
  }
  orders_.push_back(rel.orders_[0]);
}

Relation::RuntimeOrder Relation::ReverseRuntimeOrder(RuntimeOrder order) {
  switch (order) {
    case kNoOverlap:
    case kSameInstr:
    case kBeforeStartOrAfterEnd:
    case kBeforeOrAfterOrOverlap:
      return order;
    case kBeforeStart:
      return kAfterEnd;
    case kBeforeStartOrSameInstr:
      return kAfterEndOrSameInstr;
    case kAfterEnd:
      return kBeforeStart;
    case kAfterEndOrSameInstr:
      return kBeforeStartOrSameInstr;
  }
}

// Overwrites o1 with o2 if o2 subsumes o1 (as defined above by the Subsume
// function). Return whether o2 is subsumed by the new value in o1.
bool Relation::OverwriteIfSubsumes(RuntimeOrder o2, RuntimeOrder* o1) {
  if (*o1 == o2) {
    return true;
  }
  CHECK_NE(o1, nullptr);
  // Overwrite o1 with o2 if it is subsumed by o2.
  if (Subsumes(o2, *o1)) {
    *o1 = o2;
    return true;
  }
  if (Subsumes(*o1, o2)) {
    // If o2 is already subsumed by o1, do nothing.
    return true;
  }
  // If neither o1 nor o2 is subsumed by the other, return false, so that o2
  // will be inserted as a separate entry representing all possible orderings.
  return false;
}

// Compute locationing constraints between two instructions. Here entry2 is
// the source instruction, in that the returned value describes the relation
// of entry2 in terms of whether it is before or after entry1, and whether it
// can intercept the def-use data flow of entry1.
Relation ComputeRelativeLocation::ComputeBetweenInstructionEntries(
    const InstructionEntry& entry1, const InstructionEntry& entry2,
    bool instr2_can_modify) {
  auto def = entry1.second.value_definition;
  auto use = entry1.first;
  Relation::RuntimeOrder order =
      ComputeRuntimeOrdering(entry2.first, entry1.first);
  if (order == Relation::kSameInstr &&
      entry1.second.is_definition != entry2.second.is_definition) {
    if (entry1.second.is_definition) {
      order = Relation::kBeforeStart;
    } else {
      order = Relation::kAfterEnd;
    }
  }
  bool intercept = AlwaysForceInterception(entry2.first);
  if (def == nullptr || !instr2_can_modify) {
    return Relation(order, intercept);
  }
  // If the definition and use are parameter and return (root) of the parent
  // computation, then any modification is considered intercepting.
  if (def->opcode() == HloOpcode::kParameter &&
      use == use->parent()->root_instruction()) {
    VLOG(3) << "ComputeBetweenInstructionEntries: Setting interception due to "
               "parameter/root relation";
    return Relation(order, true);
  }

  // If the modification is inside the while body, it will not intercept the
  // def-use chain outside of the while body. For the following example, %add
  // does not intercept the def-use chain of %while - %root
  //
  // body = {
  //   ...
  //   add = ...  // modify buffer1
  // }
  // %while = While (param, cond, body) // def buffer1
  // %root = get-tuple-element(%while), index=1 // use buffer1

  if (use->parent() == def->parent() &&
      ComputeRuntimeOrdering(use, entry2.first) == Relation::kAfterEnd &&
      def->opcode() == HloOpcode::kWhile &&
      entry2.first->parent() == def->while_body()) {
    VLOG(3) << "ComputeBetweenInstructionEntries: Setting interception due to "
               "def-while body relation";
    return Relation(order, false);
  }

  if (use->parent() == def->parent() &&
      ComputeRuntimeOrdering(def, entry2.first) == Relation::kBeforeStart &&
      use->opcode() == HloOpcode::kWhile &&
      entry2.first->parent() == use->while_body()) {
    VLOG(3) << "ComputeBetweenInstructionEntries: Setting interception due to "
               "use-while body relation";
    return Relation(order, false);
  }

  // Special case for conditional instruction when in one branch two results
  // can be put in one buffers and another branch returns two results from a
  // multi-output instruction, e.g. fusion or variadic reduction.
  //
  //  branch_0 {
  //    exp = f64[] exp(...)
  //    ROOT tuple = (f64[], f64[]) tuple(exp, exp)
  //  }
  //
  //  fused_computation {
  //    abs = f64[] abs(...)
  //    negate = f64[] negate(...)
  //    ROOT tuple = (f64[], f64[]) tuple(abs, negate)
  //  }
  //
  //  branch_1 {
  //    ROOT fusion = (f64[], f64[]) fusion(...), calls=%fused_computation
  //  }
  //
  //  ENTRY main {
  //    ROOT root = (f64[], f64[]) conditional(...),
  //    branch_computations={%branch_0, %branch_1}
  //  }
  //
  // `branch_0` can use one buffer for both result. `branch_1` must use two
  // different buffers.
  //
  // During live range analysis of results of `branch_0` this function will be
  // called when entry1 and entry2 are different outputs on `fusion` in
  // `branch_1`. `fusion` defines two buffers, but `value_definition` in
  // LiveRangeRegions::InstructionInfo does not track the output index. The
  // analysis will say that they are not interfering and assign the same
  // buffer to both.
  //
  // This check makes sure that outputs of multi-output instructions are
  // always interfering and can not be combined. It can be a false positive
  // when entry1 and entry2 correspond to the same output, but we prefer that
  // over correctness issues.
  //
  // A proper solution would be to track output index in
  // LiveRangeRegions::InstructionInfo.
  if (use->parent() == def->parent() &&
      !def->parent()->caller_instructions(HloOpcode::kConditional).empty() &&
      def == entry2.first && def->shape().IsTuple()) {
    VLOG(3) << "ComputeBetweenInstructionEntries: Setting interception for "
               "multi-output instruction inside conditional branch: "
            << def->name();
    return Relation(order, true);
  }

  if (Relation::UseImpliesInterception(order)) {
    auto order2 = ComputeRuntimeOrdering(entry2.first, def);
    VLOG(3) << "ComputeRuntimeOrdering result: "
            << Relation::GetRuntimeOrderName(order2);
    if (Relation::DefinitionImpliesInterception(order2)) {
      VLOG(3) << "ComputeBetweenInstructionEntries: Setting interception for "
              << def->ToString() << " with use: " << entry1.first->ToString();
      intercept = true;
    }
  }
  return Relation(order, intercept);
}

// Return the relative locations (defined above) of range2 in relation to
// instructions in range1. Return kNoOverlap if range2 is outside of range1.
Relation ComputeRelativeLocation::ComputeBetweenLiveRangeRegions(
    const LiveRangeRegions& range1, const LiveRangeRegions& range2) {
  Relation dir_src_dest;
  for (const auto* computation1 : range1) {
    for (const auto* computation2 : range2) {
      VLOG(3) << "Computing relative location constraints between: ";
      VLOG(3) << "        computation1: " << computation1->name();
      VLOG(3) << "    and computation2: " << computation2->name();
      for (auto instr_entry2 : range2[computation2]) {
        if (!ordering_->call_graph().Dominates(computation1, computation2)) {
          continue;
        }
        VLOG(3) << "              instr2: " << instr_entry2.first->ToString();
        // Saves relations between instr2 and other instructions in range1.
        bool instr2_can_modify = InstructionCanIntercept(instr_entry2, range1);
        Relation instr2_relation;
        std::vector<InstructionEntry> unordered_ops;
        bool unordered_intercept = false;
        for (auto instr_entry1 : range1[computation1]) {
          auto rel = ComputeBetweenInstructionEntries(
              instr_entry1, instr_entry2, instr2_can_modify);
          VLOG(3) << "  Target Instruction: " << instr_entry1.first->name();
          VLOG(3) << "            Relation: " << rel.ToString();
          if (!rel.RuntimeOrderIsUnordered()) {
            instr2_relation.UnionRelationFromSameSource(rel);
          } else {
            unordered_ops.push_back(instr_entry1);
            unordered_intercept |= rel.InterceptDefUse();
          }
          VLOG(3) << "     instr2 relation: " << instr2_relation.ToString();
        }
        // Here instr2_relation is guaranteed to have at most a single entry,
        // because it was initialized to be empty, and has been updated only
        // via instr2_relation.UnionRelationFromSameSource(rel), which
        // maintains that the updated result has only a single entry.
        if (!ForceRuntimeOrder(unordered_ops, instr_entry2,
                               instr2_relation.GetRuntimeOrder())) {
          VLOG(3) << "Unable to force ordering of unordered ops";
          instr2_relation.UnionRelationFromSameSource(
              Relation(Relation::kBeforeStartOrAfterEnd, unordered_intercept));
        }
        dir_src_dest.UnionRelationFromDifferentSource(instr2_relation);
        VLOG(3) << "  Resulting relation: " << dir_src_dest.ToString();
        VLOG(3) << "--------------------------------------------------------";
      }
    }
  }
  return dir_src_dest;
}

// Return whether control dependences, if exist, are added successfully.
bool ComputeRelativeLocation::AddControlDependenceForUnorderedOps() {
  if (ctrl_deps_.empty()) {
    return true;
  }
  PredecessorHloOrdering* ordering =
      dynamic_cast<PredecessorHloOrdering*>(ordering_);
  if (ordering == nullptr) {
    // Support force ordering of unordered-ops only when using predecssor
    // ordering.
    return false;
  }
  for (const auto& comp_it : ctrl_deps_) {
    HloComputation* parent = comp_it.first;
    HloReachabilityMap& reachability_map = ordering->reachability_map(parent);
    for (const auto& instr_it : comp_it.second) {
      HloInstruction* entry1 = instr_it.first;
      for (HloInstruction* entry2 : instr_it.second) {
        VLOG(3) << "   Adding control dependence between:";
        VLOG(3) << "     predecessor: " << entry2->name();
        VLOG(3) << "       successor: " << entry1->name();
        TF_CHECK_OK(entry2->AddControlDependencyTo(entry1));
      }
      reachability_map.UpdateReachabilityThroughInstruction(entry1);
      for (HloInstruction* entry2 : instr_it.second) {
        DCHECK(ordering_->GetExecutionConstraint(entry1, entry2) ==
               HloOrdering::ExecutionConstraint::kRunAfter);
      }
    }
  }
  return true;
}

// Returns whether it is safe to force the desired_relation ordering between
// all operations in unordered_ops and entry2. If safe, save the new enforced
// ordering relations.
bool ComputeRelativeLocation::ForceRuntimeOrder(
    absl::Span<const InstructionEntry> unordered_ops,
    const InstructionEntry entry2, Relation::RuntimeOrder desired_relation) {
  if (unordered_ops.empty()) {
    return true;
  }
  if (desired_relation != Relation::kBeforeStart &&
      desired_relation != Relation::kAfterEnd) {
    VLOG(3) << "      ForceRuntimeOrder: desired_relation is not "
               "kBeforeStart or kAfterEnd";
    return false;
  }
  auto ModifiesNonCopy = [](HloInstruction* instr, const HloInstruction* op) {
    auto in_place = HloDataflowAnalysis::GetInPlaceInputOutputPairs(instr);
    if (in_place.empty()) {
      return false;
    }
    return absl::c_any_of(
        in_place, [&](const std::pair<HloOperandIndex, ShapeIndex>&
                          operand_and_output_index) {
          auto* op2 =
              instr->operand(operand_and_output_index.first.operand_number);
          return (op == nullptr) ? (op2->opcode() == HloOpcode::kCopy)
                                 : (op2 == op);
        });
  };
  for (const InstructionEntry& entry1 : unordered_ops) {
    // Only consider instructions in the same computation.
    if (entry1.first->parent() != entry2.first->parent()) {
      VLOG(3) << "      ForceRuntimeOrder: instructions are not in the same "
                 "computation";
      return false;
    }
    HloInstruction* pred = (desired_relation == Relation::kBeforeStart)
                               ? entry2.first
                               : entry1.first;
    HloInstruction* succ = (desired_relation == Relation::kBeforeStart)
                               ? entry1.first
                               : entry2.first;
    if (pred == pred->parent()->root_instruction()) {
      VLOG(3) << "      ForceRuntimeOrder: predecessor (" << pred->name()
              << ") is the root instruction";
      return false;
    }
    if (succ->opcode() == HloOpcode::kCopy &&
        ModifiesNonCopy(pred, succ->operand(0))) {
      VLOG(3) << "Failed to force unordered op ordering due to copy ordering "
              << " between " << pred->name() << " vs " << succ->name();
      return false;
    }
  }
  for (const InstructionEntry& entry1 : unordered_ops) {
    Save(entry2.first, entry1.first, desired_relation,
         /*is_unordered_originally=*/true);
    VLOG(3) << "      ForceRuntimeOrder: saved unordered relation: ";
    VLOG(3) << "        entry2: " << entry2.first->name();
    VLOG(3) << "        entry1: " << entry1.first->name();
    VLOG(3) << "        relation: "
            << Relation::GetRuntimeOrderName(desired_relation);
  }
  return true;
}

bool ComputeRelativeLocation::AlwaysForceInterception(HloInstruction* instr) {
  // The following communication operations can have some unexpected side
  // effects, when synchronizing across processes. Therefore, we
  // conservatively try provide dedicated buffers to these operations instead
  // of allowing them to share buffers with other operations, as the reuse may
  // cause unexpected interferences.
  if (HloDataflowAnalysis::IsAsynchronousOperationStart(instr->opcode()) ||
      HloDataflowAnalysis::IsAsynchronousOperationDone(instr->opcode())) {
    return true;
  }
  switch (instr->opcode()) {
    // TODO(b/190903339): It appears that collectivePermute needs to be
    // followed by a copy when escaping through a computation root.
    case HloOpcode::kCollectivePermute:
      return true;
    default:
      return false;
  }
}

// Returns whether the given instr may intercept the def-use flow of another
// ongoing live range if its buffer is combined with the other live range.
// The function should return true if instr creates a new HloValue that could
// overwrite an existing HloValue in the combined buffer.
// More specifically, here we are looking for operations that create new
// values, e.g., add, subtract, in contrast to HLOs that merely create
// aliasings among existing values, e.g., tuple, get-tuple-element. Any of the
// new values created by operations such as add or subtract, when included as
// definition operations in a live range, are aliases of the buffer to be
// allocated to the live range and so are treated as they may be modifying the
// targeting buffer.
bool ComputeRelativeLocation::InstructionCanIntercept(
    const InstructionEntry& entry, const LiveRangeRegions& region) {
  auto instr = entry.first;
  if (!entry.second.is_definition) {
    // If the instruction only uses the value, it can intercept only if it
    // modifies the buffer in place.
    for (const auto& operand_and_output_index :
         HloDataflowAnalysis::GetInPlaceInputOutputPairs(instr)) {
      const HloOperandIndex& operand_index = operand_and_output_index.first;
      if (region.contains(
              instr->mutable_operand(operand_index.operand_number))) {
        return true;
      }
    }
    return false;
  }
  switch (instr->opcode()) {
    // If the copy instruction is used to connect two live range regions,
    // it does not overwrite the combined buffer with new values.
    case HloOpcode::kCopy: {
      // Checking the copy simply copies from the other live range with no
      // layout conflicts.
      HloInstruction* operand = instr->mutable_operand(0);
      if (operand->opcode() == HloOpcode::kGetTupleElement) {
        // kGetTupleElement only creates an alias among HloValues and is not
        // included in the live range region. We check its operand instead.
        operand = operand->mutable_operand(0);
      }
      if (region.contains(operand) &&
          ShapeUtil::Equal(instr->shape(), instr->operand(0)->shape())) {
        return false;  // Cannot intercept.
      }
      return true;
    }
    // The following operations merely create aliases among the HloValues.
    case HloOpcode::kParameter:
    case HloOpcode::kTuple:
    case HloOpcode::kGetTupleElement:
    // Here we consider all the compound operations (e.g., conditionals and
    // while loops) as if they do not modify any HloValue, with the argument
    // being that any value modifying operation contained inside will be
    // considered separately to make sure the kIntercept relation being
    // recorded as appropriate. Since the compound operations may or may not
    // modify, not treating them as value modifying would make the algorithm
    // less conservative.
    case HloOpcode::kWhile:
    case HloOpcode::kCall:
    case HloOpcode::kConditional:
      return false;
    default:
      return true;
  }
  return true;
}

ComputeRelativeLocation::SavedRelation ComputeRelativeLocation::AlreadyComputed(
    HloInstruction* op1, HloInstruction* op2) {
  auto p2 = saved_relations_.find(op2);
  if (p2 != saved_relations_.end()) {
    auto p1 = (*p2).second.find(op1);
    if (p1 != (*p2).second.end()) {
      return SavedRelation(kFullyComputed, (*p1).second);
    }
  }
  p2 = saved_relations_.find(op1);
  if (p2 != saved_relations_.end()) {
    auto p1 = (*p2).second.find(op2);
    if (p1 != (*p2).second.end()) {
      return SavedRelation(kPartiallyComputed,
                           Relation::ReverseRuntimeOrder((*p1).second));
    }
  }
  return SavedRelation(kNotComputed, Relation::kNoOverlap);
}

Relation::RuntimeOrder ComputeRelativeLocation::Save(
    HloInstruction* entry1, HloInstruction* entry2,
    const Relation::RuntimeOrder relation, bool is_unordered_originally) {
  CHECK_EQ(AlreadyComputed(entry1, entry2).first, kNotComputed);
  // Do not save unordered relations.
  CHECK_NE(relation, Relation::kBeforeStartOrAfterEnd);
  saved_relations_[entry2][entry1] = relation;
  if (is_unordered_originally) {
    CHECK(relation == Relation::kBeforeStart || relation == Relation::kAfterEnd)
        << relation;
    HloInstruction* pred =
        (relation == Relation::kBeforeStart) ? entry1 : entry2;
    HloInstruction* succ =
        (relation == Relation::kBeforeStart) ? entry2 : entry1;
    VLOG(3) << "Save unordered relation: " << pred->name() << " vs "
            << succ->name();
    CHECK_EQ(succ->parent(), pred->parent());
    auto& dep_vec = ctrl_deps_[succ->parent()][succ];
    for (HloInstruction*& op : dep_vec) {
      auto rel = AlreadyComputed(pred, op);
      if (rel.first != kNotComputed) {
        if (rel.second == Relation::kAfterEnd) {
          op = pred;
        } else {
          CHECK(rel.second == Relation::kBeforeStart);
        }
        return relation;
      }
    }
    VLOG(2) << "Forcing unordered: " << pred->name() << " vs " << succ->name();
    dep_vec.push_back(pred);
  }
  return relation;
}

// Compute the runtime ordering constraints between two instructions.
Relation::RuntimeOrder ComputeRelativeLocation::ComputeRuntimeOrdering(
    HloInstruction* instr1, HloInstruction* instr2) {
  auto saved_relation = AlreadyComputed(instr1, instr2);
  VLOG(3) << "   ComputeRuntimeOrdering: " << instr1->name() << " vs "
          << instr2->name();
  if (saved_relation.first != kNotComputed) {
    VLOG(3) << "   ComputeRuntimeOrdering: Already computed between "
            << instr1->name() << " vs " << instr2->name();
    return saved_relation.second;
  }
  auto constraint = ordering_->GetExecutionConstraint(instr1, instr2);
  switch (constraint) {
    case HloOrdering::ExecutionConstraint::kIsSame:
      return Save(instr1, instr2, Relation::kSameInstr);
    case HloOrdering::ExecutionConstraint::kRunBeforeEnd:
      return Save(instr1, instr2, Relation::kBeforeStartOrSameInstr);
    case HloOrdering::ExecutionConstraint::kRunBeforeStart:
      return Save(instr1, instr2, Relation::kBeforeStart);
    case HloOrdering::ExecutionConstraint::kRunAfter:
      return Save(instr1, instr2, Relation::kAfterEnd);
    case HloOrdering::ExecutionConstraint::kRunExclusiveBefore:
    case HloOrdering::ExecutionConstraint::kRunExclusiveAfter:
      return Save(instr1, instr2, Relation::kNoOverlap);
    case HloOrdering::ExecutionConstraint::kUnordered: {
      if (instr1->parent() != instr2->parent()) {
        return Relation::kBeforeStartOrAfterEnd;
      }
      auto ControlDependenceBefore = [&](HloInstruction* op1,
                                         HloInstruction* op2) {
        auto constraint = ComputeRuntimeOrdering(op1, op2);
        if (constraint == Relation::kBeforeStart ||
            constraint == Relation::kSameInstr ||
            constraint == Relation::kBeforeStartOrSameInstr) {
          return true;
        }
        return false;
      };
      if (!ctrl_deps_.empty()) {
        auto ctrl_deps = ctrl_deps_[instr1->parent()];
        if (absl::c_any_of(ctrl_deps[instr2], [&](HloInstruction* pred2) {
              return ControlDependenceBefore(instr1, pred2);
            })) {
          VLOG(2) << "control-dependent: " << instr1->name() << " vs "
                  << instr2->name();
          return Save(instr1, instr2, Relation::kBeforeStart);
        }
        if (absl::c_any_of(ctrl_deps[instr1], [&](HloInstruction* pred1) {
              return ControlDependenceBefore(instr2, pred1);
            })) {
          VLOG(2) << "control-dependent: " << instr2->name() << " vs "
                  << instr1->name();
          return Save(instr1, instr2, Relation::kAfterEnd);
        }
      }
      // Don't save the result for unordered operations, so they can be
      // refined later.
      return Relation::kBeforeStartOrAfterEnd;
    }
  }
}

CopyRemover::CopyRemover(
    const HloModule& module, const HloAliasAnalysis& alias_analysis,
    HloOrdering* ordering, bool check_live_range_ordering,
    const absl::flat_hash_set<absl::string_view>& execution_threads)
    : dataflow_(alias_analysis.dataflow_analysis()), ordering_(ordering) {
  // Instruction indices based on post order traversal of computations and
  // instructions. Used as an enhancement for getting strict weak ordering
  // used for sorting below.
  absl::flat_hash_map<int, int64_t> instruction_ids;
  int64_t id = 0;

  // Generate instruction ids for all instructions in the module, starting at
  // the entry computation, processing instructions post-order and recursing
  // depth-first into called computations.
  absl::flat_hash_set<HloComputation*> visited;
  std::function<void(HloComputation*)> assign_ids_dfs =
      [&](HloComputation* computation) {
        // Only visit each computation once.
        auto [it, inserted] = visited.insert(computation);
        if (!inserted) {
          return;
        }
        // Assign ids to parameters first to match logic in IsDefinedBefore()
        for (HloInstruction* instruction :
             computation->parameter_instructions()) {
          instruction_ids[instruction->unique_id()] = id++;
        }

        // Use the schedule order if available, otherwise use post order.
        const HloInstructionSequence* seq =
            ordering->SequentialOrder(*computation);
        std::vector<HloInstruction*> instructions =
            seq != nullptr ? seq->instructions()
                           : computation->MakeInstructionPostOrder();

        // Traverse depth-first, assigning ids to caller instructions
        // *after* called computations.
        for (HloInstruction* instruction : instructions) {
          switch (instruction->opcode()) {
            case HloOpcode::kParameter:
              // Parameters are already assigned ids above.
              continue;
            case HloOpcode::kWhile:
              // While condition executes before body.
              assign_ids_dfs(instruction->while_condition());
              assign_ids_dfs(instruction->while_body());
              break;
            default:
              for (HloComputation* called_computation :
                   instruction->called_computations()) {
                assign_ids_dfs(called_computation);
              }
              break;
          }
          instruction_ids[instruction->unique_id()] = id++;
        }
      };

  CHECK(module.has_entry_computation());
  assign_ids_dfs(module.entry_computation());

  // Construct a list for each HLO buffer in the alias analysis. Maintain a
  // map from HloValue to the respective list element representing that
  // value. The map is used to construct the copy info map below.
  absl::flat_hash_map<const HloValue*, ValueNode*> value_to_node;
  // Perform check only if the default dependence-based ordering is used.
  for (const HloBuffer& buffer : alias_analysis.buffers()) {
    // No copies should have been inserted within fused computations, so no
    // need to remove them. HloOrdering isn't compatible with HloValues inside
    // fusions, so skip copy removal for them.
    if (buffer.values().at(0)->defining_instruction()->IsFused()) {
      continue;
    }

    std::vector<const HloValue*> values = buffer.values();
    absl::c_sort(
        values, [this, &instruction_ids](const HloValue* a, const HloValue* b) {
          // IsDefinedBefore() is generally not strict weak ordering required by
          // the sort algorithm, since a may not be comparable to b or c by
          // IsDefinedBefore(), but b and c can be comparable. Such as in:
          //   if () { b = ...; c = b + 1; } else { a = ...; }
          // or
          //   a = param(0) + param(1); b = param(2) + param(3); c = b + 1;
          // So it fails the "incomparability being transitive" requirement by
          // strict weak ordering. We enhance the ordering test by using
          // instruction ids generated by post order visiting of the
          // computations/instructions. All HloValue's are comparable and
          // dependency (thus transitivity) is respected when hlo ordering
          // cannot decide the order.
          if (a == b) {
            return false;
          }
          const bool a_has_smaller_id =
              instruction_ids.at(a->defining_instruction()->unique_id()) <
              instruction_ids.at(b->defining_instruction()->unique_id());
          // Use a_has_smaller_id as a hint for the order between a and b. In
          // case it's right, there is no need for two IsDefinedBefore() tests.
          if (a_has_smaller_id) {
            // Test a is defined before b first.
            if (ordering_->IsDefinedBefore(*a, *b)) {
              return true;
            }
            if (ordering_->IsDefinedBefore(*b, *a)) {
              return false;
            }
          } else {
            // Test b is defined before a first.
            if (ordering_->IsDefinedBefore(*b, *a)) {
              return false;
            }
            if (ordering_->IsDefinedBefore(*a, *b)) {
              return true;
            }
          }

          // Use post order as tie breaker.
          return a_has_smaller_id;
        });

    // Create a list containing all of the values in the buffer.
    AddValueList(values, &value_to_node);
  }

  // Create copy_map_ which contains the source and destination values
  // of all copies.
  CreateCopyMap(module, value_to_node);

  XLA_VLOG_LINES(3, ToString());
  TF_DCHECK_OK(Verify());
}

// Add a list containing the given values to CopyRemover. This
// represents the values contained in a single buffer. For each value in
// 'values' an entry is created in value_to_node which indicates the
// respective ValueNode representing that value.
void CopyRemover::AddValueList(
    absl::Span<const HloValue* const> values,
    absl::flat_hash_map<const HloValue*, ValueNode*>* value_to_node) {
  ValueNode* tail = nullptr;
  ValueNode* head = nullptr;
  for (const HloValue* value : values) {
    auto new_node = new ValueNode(value);
    (*value_to_node)[value] = new_node;

    // Copy the HLO values's uses into the ValueNode for the value. These
    // uses in ValueNode are updated as copies are removed.
    new_node->uses.reserve(value->GetUses().size());
    for (const HloUse& use : value->GetUses()) {
      new_node->uses.push_back(&use);
    }

    // Connect the new node into the linked list.
    if (tail == nullptr) {
      head = new_node;
    } else {
      tail->next = new_node;
      new_node->prev = tail;
    }
    tail = new_node;
  }

  // The linked list is circular so connect the head and tail.
  tail->next = head;
  head->prev = tail;
  value_lists_.insert(head);
}

// This method also fills in copy_map_ which indicates which nodes
// in the value lists corresponding to the source and destination values of
// kCopy instructions. value_to_node should map each HloValue to its
// respective ValueNode.
void CopyRemover::CreateCopyMap(
    const HloModule& module,
    const absl::flat_hash_map<const HloValue*, ValueNode*>& value_to_node) {
  for (HloComputation* computation : module.MakeNonfusionComputations()) {
    for (HloInstruction* instruction : computation->instructions()) {
      // Add copies with unambiguous source values to the map. Copies with
      // ambiguous sources are not removable.
      if (instruction->opcode() == HloOpcode::kCopy) {
        const HloValueSet& src_value_set =
            dataflow_.GetValueSet(instruction->operand(0));
        if (src_value_set.values().size() == 1) {
          CopyNodes& copy_node = copy_map_[instruction];
          copy_node.dest =
              value_to_node.at(&dataflow_.GetUniqueValueAt(instruction));
          copy_node.src = value_to_node.at(&src_value_set.GetUniqueValue());
        }
      }
    }
  }
}

CopyRemover::~CopyRemover() {
  for (const ValueNode* head : value_lists_) {
    const ValueNode* p = head;
    do {
      const ValueNode* tmp = p->next;
      delete p;
      p = tmp;
    } while (p != head);
  }
}

// Verify invariants within the linked lists.
absl::Status CopyRemover::Verify() const {
  for (const ValueNode* head : value_lists_) {
    const ValueNode* p = head;
    do {
      // Verify links between elements are consistent.
      TF_RET_CHECK(p->prev->next == p);
      TF_RET_CHECK(p->next->prev == p);

      const HloInstruction* def = p->value->defining_instruction();
      if (def->opcode() == HloOpcode::kCopy && ContainsKey(copy_map_, def)) {
        TF_RET_CHECK(copy_map_.at(def).dest == p);
      }
      for (const HloUse* use : p->uses) {
        if (use->instruction->opcode() == HloOpcode::kCopy &&
            ContainsKey(copy_map_, use->instruction)) {
          TF_RET_CHECK(copy_map_.at(use->instruction).src == p);
        }
      }

      p = p->next;
    } while (p != head);
  }
  return absl::OkStatus();
}

// Compute the set of instructions where values are alive and organize these
// instructions by separating them into their respective computations.
LiveRangeRegions CopyRemover::ComputeLiveRangeRegions(const ValueNode* head) {
  LiveRangeRegions live_range;

  auto VisitValueNode = [&](const ValueNode* node) {
    HloInstruction* def_op = node->value->instruction();
    HloComputation* def_parent = def_op->parent();
    live_range[def_parent][def_op].is_definition = true;
    for (const auto& use : node->uses) {
      auto* use_op = use->instruction;
      HloComputation* use_parent = use_op->parent();
      live_range[use_parent][use_op].value_definition = def_op;
    }
  };
  ForEachValueInRange(head, VisitValueNode);
  return live_range;
}

bool CopyRemover::IsCopyToFromHost(const HloInstruction* copy) {
  if (copy->shape().has_layout() && copy->operand(0)->shape().has_layout()) {
    if (copy->shape().layout().memory_space() == Layout::kHostMemorySpace &&
        copy->operand(0)->shape().layout().memory_space() !=
            Layout::kHostMemorySpace) {
      return true;
    }
    if (copy->shape().layout().memory_space() != Layout::kHostMemorySpace &&
        copy->operand(0)->shape().layout().memory_space() ==
            Layout::kHostMemorySpace) {
      return true;
    }
  }
  return false;
}
// Try to elide the given copy. Elision of a copy is possible only if no
// live range interference is introduced by the copy's elimination. If
// elision is possible, then the internal state (value lists) are updated,
// and true is returned. Returns false otherwise.
bool CopyRemover::TryElideCopy(
    const HloInstruction* copy, int64_t* region_analysis_limit,
    bool insert_post_scheduling_control_dependencies) {
  VLOG(3) << "TryElideCopy starting for: " << copy->name();
  CHECK_NE(region_analysis_limit, nullptr);

  // Don't elide copies to/from the host.
  if (IsCopyToFromHost(copy)) {
    return false;
  }

  // Don't elide copies that are not in the copy map.
  if (!ContainsKey(copy_map_, copy)) {
    VLOG(2) << copy->name() << " is not removable";
    return false;
  }

  // Don't elide copies with different shapes.
  if (!ShapeUtil::Equal(copy->shape(), copy->operand(0)->shape())) {
    VLOG(2) << copy->name() << " is not removable (shape mismatch)";
    return false;
  }
  const CopyNodes& copy_node = copy_map_.at(copy);
  DCHECK(copy_node.src != nullptr);
  DCHECK(copy_node.dest != nullptr);

  int64_t src_total_read_writes = 0, dst_total_read_writes = 0;
  ForEachValueInRange(copy_node.src, [&](const ValueNode* node) {
    src_total_read_writes += 1 + node->uses.size();
  });
  ForEachValueInRange(copy_node.dest, [&](const ValueNode* node) {
    dst_total_read_writes += 1 + node->uses.size();
  });
  // Use the more accurate region-based live range interference analysis if
  // the live range size is within a given limit (or if no limit is given).
  // Also don't use the new analysis for copies of broadcasts as these copies
  // are cheap and are later removed by replicating the broadcasts.
  bool use_region_analysis =
      copy->operand(0)->opcode() != HloOpcode::kBroadcast &&
      (*region_analysis_limit < 0 ||
       src_total_read_writes * dst_total_read_writes <= *region_analysis_limit);

  *region_analysis_limit = 0;
  VLOG(3) << "Source buffer values: " << ValueListToString(copy_node.src);
  VLOG(3) << "Dest buffer values: " << ValueListToString(copy_node.dest);
  // Checks whether the live range at src is before that defined by dest.
  auto CheckLiveRangeBefore = [&](ValueNode* src, ValueNode* dest) {
    for (ValueNode* next_dest = dest; next_dest != nullptr;
         next_dest = Next(*next_dest)) {
      for (ValueNode* prev_src = src; prev_src != nullptr;
           prev_src = Prev(*prev_src)) {
        if (!LiveRangeBefore(*prev_src, *next_dest)) {
          VLOG(4) << "   CheckLiveRangeBefore - live range of: "
                  << prev_src->value->ToShortString() << ", is not before; "
                  << next_dest->value->ToShortString();
          return false;
        }
      }
    }
    return true;
  };
  auto CheckLiveRangeInterference = [&](ValueNode* src, ValueNode* dest,
                                        const CombineLiveRangeOption option) {
    CHECK_NE(src, nullptr);
    CHECK_NE(dest, nullptr);
    if (!use_region_analysis) {
      VLOG(2) << " TryElideCopy: Configured to not use region-based analysis.";
      return true;
    }
    *region_analysis_limit += src_total_read_writes * dst_total_read_writes;
    if (ValuesInterfere(src, dest, option)) {
      VLOG(2) << " TryElideCopy: Region-based interference is true.";
      return true;
    }
    VLOG(2) << " TryElideCopy: Region-based interference is false.";
    return false;
  };
  auto AddControlDependenciesBetween = [&](ValueNode* src, ValueNode* dst) {
    if (src == nullptr || dst == nullptr) {
      return;
    }
    for (auto use : src->uses) {
      if (use->instruction->parent() != dst->value->instruction()->parent() ||
          use->instruction == dst->value->instruction()) {
        // Don't add control dependencies if the use is in a different
        // computation or if the use is the same as the destination.
        continue;
      }

      VLOG(2)
          << "      AddControlDependenciesBetween: Adding control dependency:";
      VLOG(2) << "      AddControlDependenciesBetween:  From: "
              << use->instruction->ToString();
      VLOG(2) << "      AddControlDependenciesBetween:   Use: "
              << use->ToString();

      VLOG(2) << "      AddControlDependenciesBetween:    To: "
              << dst->value->instruction()->ToShortString();
      VLOG(2) << "      AddControlDependenciesBetween: Value: "
              << dst->value->ToString();

      CHECK_OK(
          use->instruction->AddControlDependencyTo(dst->value->instruction()));
    }
  };

  // A kCopy instruction copies an HLO value from a source buffer and
  // defines an HLO value in a destination buffer. Most generally, the
  // source and destination buffers may each hold more than one value at
  // different points in the computation so we define the following:
  //
  //   Values in source buffer:      {s_0, ..., s_n}
  //   Values in destination buffer: {d_0, ..., d_m}
  //
  // A kCopy instruction between these buffers copies a value s_x in the
  // source buffer and defines a value d_y in the destination buffer. The
  // elision of a copy merges the source and destination buffers together,
  // so the list of values for the source and destination buffers are
  // merged.
  //
  // We handle two different cases for copy elision:
  //
  //  (1) the kCopy defines the first value in the destination buffer (d_0).
  //
  //  (2) the kCopy copies the last value in the source buffer (s_n).
  //
  // For the remaining case where the kCopy copies a not-last value from the
  // source buffer to a not-first value of the destination buffer, the kCopy
  // instruction cannot be removed. This case is generated for example, if
  // the kCopy copies a while body parameter of the loop state at one tuple
  // index to a different tuple index in the while body root. Removal of the
  // copy necessarily results in live range interference of values in the
  // loop state at the two different tuple indices.
  //
  //  We can only perform copy elision if the merged values have
  //  totally ordered live ranges; otherwise the merged buffer would have
  //  live range interference.
  if (copy_node.src->next == copy_node.dest) {
    // In the process of eliding copies, it's possible for a copy to have the
    // same source and destination buffer. In this case, the copy can be
    // safely removed.
    VLOG(2) << "TryElideCopy - copy (" << copy->name()
            << ") has same source / destination buffers";

  } else if (IsHead(*copy_node.dest)) {
    // The copy copies an arbitrary value in the source buffer (call it s_x)
    // and defines d_0, the first value in the destination buffer. After
    // merging, the values in the combined buffer must be strictly ordered
    // as follows** to elide the copy:
    //
    // {s_0, ..., s_x, d_1, ..., d_m, s_{x+1}, ..., s_n}
    //
    // Removing the copy eliminates d_0, and uses of d_0 become uses of
    // s_x. In the above ordering, the live range of d_m will be ordered
    // before the live range of s_{x+1} and the definition and all uses of
    // s_x will be ordered before the definition of d_1. To make sure the
    // copy elision is safe, the following code checks that this ordering is
    // valid --- in particular we check it is safe to order d_m ahead of all
    // the liverages at and after s_{x+1}, and it is safe to order all uses
    // of s_x before the definition of d_1, by checking the live range
    // constraints for each pair --- we cannot skip the later checks because
    // the live range ordering is not guaranteed to be transitive --- while it
    // may be ok to have lr_1 before lr_2, and lr_2 before lv_3 while merging
    // their buffers, it may not be ok to merge the buffers of lr_1 and lv_3,
    // because the exclusiveness relation of non-overlapping computations is
    // not transitive.
    //
    // ** Technically it might be possible to have a non-interfering
    //    non-trivial interleaving of the values of the source and
    //    destination buffers in the resulting order. This can be potentially
    //    supported in the ValuesInterfere function, which performs
    //    interference analysis at a more global scope than the alternative
    //    LiveRangeBefore analysis which requires strict ordering of all live
    //    ranges. Currently, however, this is not yet supported, as
    //    we simply check for the case where *all* values of the destination
    //    buffer (d_1 through d_m) are spliced into the point where the copy
    //    used to be.
    VLOG(2) << "TryElideCopy - copy (" << copy->name()
            << ") defines the first value in its buffer.";
    // Live range of (s_x, s_{x-1},...) must be before 'next_dest' (d_1);
    bool src_use_before_first_dest_def =
        CheckLiveRangeBefore(copy_node.src, Next(*copy_node.dest));
    std::string a = copy_node.src->value->ToShortString();
    std::string b = Next(*copy_node.dest) == nullptr
                        ? "null"
                        : Next(*copy_node.dest)->value->ToShortString();
    VLOG(6) << "TryElideCopy - source uses of value defined by (" << a
            << ") complete before next dest definition (" << b
            << "): " << src_use_before_first_dest_def;
    // Live range of 'last_dest' (d_m) must be before 'next_src' s_{x+1}.
    bool src_def_after_last_dest_use =
        CheckLiveRangeBefore(copy_node.dest->prev, Next(*copy_node.src));
    a = copy_node.dest->prev->value->ToShortString();
    b = Next(*copy_node.src) == nullptr
            ? "null"
            : Next(*copy_node.src)->value->ToShortString();
    VLOG(6) << "TryElideCopy - dest uses of value defined by (" << a
            << ") complete before next src definition (" << b
            << "): " << src_def_after_last_dest_use;

    bool live_range_before =
        src_use_before_first_dest_def && src_def_after_last_dest_use;
    VLOG(3) << "TryElideCopy - LiveRangeBefore result: " << live_range_before;

    // If the live range is before, we can add control dependencies to ensure
    // the ordering. Otherwise, we check for interference (which will
    // also add control dependencies if needed)
    if (live_range_before) {
      if (insert_post_scheduling_control_dependencies) {
        // Ensure that the last uses of the copy source (e.g. s_x) are
        // ordered before the next definition of the copy destination buffer
        // (d_1).
        AddControlDependenciesBetween(copy_node.src, Next(*copy_node.dest));

        // Also ensure that the last uses of the copy destination (e.g. d_m)
        // are ordered before the next definition of the copy source buffer
        // (s_{x+1}).
        AddControlDependenciesBetween(copy_node.dest->prev,
                                      Next(*copy_node.src));
      }
    } else if (CheckLiveRangeInterference(copy_node.src, copy_node.dest,
                                          kMergeFirstDestInSource)) {
      return false;
    }
    VLOG(2) << "TryElideCopy - splicing dest after source.";
    // Splice in destination buffer values list right after 'src'.
    SpliceAfter(copy_node.dest, copy_node.src);
  } else if (IsTail(*copy_node.src)) {
    // The copy copies the last value in the source buffer, s_n, and defines
    // an arbitrary value in the destination buffer, d_y.  After
    // merging, the values in the combined buffer must be strictly ordered
    // as follows** to elide the copy:
    //
    // {d_0, ..., d_{y-1}, s_0, ..., s_n, d_{y+1}, ..., d_m}
    //
    // Removing the copy eliminates d_y, and uses of d_y become uses of
    // s_n. To enforce the above order, the live range of d_{y-1} must be
    // before the live range of s_0, and the live range of s_n must be
    // before the live range of d_{y+1}.
    //
    // ** See comment above in the code handling Case (1).
    VLOG(2) << "TryElideCopy - copy (" << copy->name()
            << ") copies the last value in its buffer.";
    // Live range of d_0, ..., d_{y-1} must be before s_0;
    // Since copy_node.src is tail for this if branch, copy_node.src->next
    // is s0 because the list is circularly linked.
    bool prev_dest_use_before_next_src_def =
        CheckLiveRangeBefore(Prev(*copy_node.dest), copy_node.src->next);

    std::string a = Prev(*copy_node.dest) == nullptr
                        ? "null"
                        : Prev(*copy_node.dest)->value->ToShortString();
    std::string b = copy_node.src->next->value->ToShortString();

    VLOG(6) << "TryElideCopy - prev dest uses of value defined by (" << a
            << ") complete before src definition (" << b
            << "): " << prev_dest_use_before_next_src_def;
    // Live range of 'last_src' must be before next_dest d_{y+1}.
    bool src_use_before_next_dest_def =
        CheckLiveRangeBefore(copy_node.src, Next(*copy_node.dest));

    a = copy_node.src->value->ToShortString();
    b = Next(*copy_node.dest) == nullptr
            ? "null"
            : Next(*copy_node.dest)->value->ToShortString();
    VLOG(6) << "TryElideCopy - src uses of value defined by (" << a
            << ") complete before dest definition (" << b
            << "): " << src_use_before_next_dest_def;

    bool live_range_before =
        prev_dest_use_before_next_src_def && src_use_before_next_dest_def;

    VLOG(2) << "TryElideCopy - LiveRangeBefore result: " << live_range_before;
    // If the live range is before, we can add control dependencies to ensure
    // the ordering. Otherwise, we check for interference (which will
    // also add control dependencies if needed)
    if (live_range_before) {
      if (insert_post_scheduling_control_dependencies) {
        // Ensure that the last uses of the copy source (e.g. s_n) are
        // ordered before the next definition of the copy destination buffer
        // (d_{y+1}).
        AddControlDependenciesBetween(Prev(*copy_node.dest),
                                      copy_node.src->next);
        // Also ensure that the last uses of the copy source (e.g. s_n) are
        // ordered before next definition of the copy destination (e.g.
        // d_{y+1}).
        AddControlDependenciesBetween(copy_node.src, Next(*copy_node.dest));
      }
    } else if (CheckLiveRangeInterference(copy_node.src, copy_node.dest,
                                          kMergeLastSourceInDest)) {
      VLOG(2) << "Region-based analysis concludes interference.";
      return false;
    }
    VLOG(2) << "Splice src after prev of dest.";
    // Splice source buffer values list right after 'prev_dest'.
    SpliceAfter(copy_node.src->next, Prev(*copy_node.dest));
  } else {
    VLOG(2) << copy->name()
            << " copies value in middle of source buffer to value in middle "
               "of destination buffer";
    return false;
  }

  RemoveCopyValue(copy_node.dest);

  XLA_VLOG_LINES(4, ToString());
  TF_DCHECK_OK(Verify());
  VLOG(3) << "TryElideCopy succeeded for: " << copy->name();
  return true;
}

// Delete the given ValueNode associated with a elided kCopy
// instruction. This should be called after splicing the value lists of the
// source and destination buffers together.
void CopyRemover::RemoveCopyValue(ValueNode* copy_value_node) {
  CHECK_EQ(copy_value_node->value->defining_instruction()->opcode(),
           HloOpcode::kCopy);
  ValueNode* operand_node = copy_value_node->prev;
  CHECK(operand_node != copy_value_node);

  VLOG(2) << "Removing copy " << operand_node->value->ToShortString() << " => "
          << copy_value_node->value->ToShortString();

  // Splice out the copy value node.
  operand_node->next = copy_value_node->next;
  copy_value_node->next->prev = operand_node;

  // Patch up uses. Remove use of copy from operand_node uses.
  auto it =
      absl::c_find_if(operand_node->uses, [copy_value_node](const HloUse* use) {
        return use->instruction ==
               copy_value_node->value->defining_instruction();
      });
  CHECK(it != operand_node->uses.end());
  operand_node->uses.erase(it);

  // If the elided copy has any uses which are themselves kCopy instructions
  // then patch up the copy info to reflect the that this kCopy instruction
  // has a different operand (the operand of the elided copy).
  for (const HloUse* copy_use : copy_value_node->uses) {
    operand_node->uses.push_back(copy_use);
    if (copy_use->instruction->opcode() == HloOpcode::kCopy &&
        ContainsKey(copy_map_, copy_use->instruction)) {
      copy_map_.at(copy_use->instruction).src = operand_node;
    }
  }

  // Delete the copy info and the value node.
  copy_map_.erase(copy_value_node->value->defining_instruction());
  delete copy_value_node;
}

// Returns true if the live range of given value 'a' is before the live
// range of 'b'.
//
// We cannot use LiveRangeStrictlyBefore because HloValue::uses() is not
// updated as copies are removed. Also here because the result is used
// to directly drive copy elision, use_is_always_before_def_in_same_instr is
// set to false.
bool CopyRemover::LiveRangeBefore(const ValueNode& a, const ValueNode& b) {
  if (a.uses.empty()) {
    VLOG(2) << "Empty uses for " << *a.value;
    return ordering_->IsDefinedBefore(*a.value, *b.value);
  }
  VLOG(3) << "Checking live ranges before: " << ValueListToString(&a) << " vs "
          << ValueListToString(&b);
  // If any of the positions of the "a" value is a root of the same
  // computation as "b", "a"'s live range cannot be before "b"'s. This catches
  // the cases where the root may not be the last instruction in the
  // computation.
  if (a.value->IsRootOf(b.value->defining_instruction()->parent())) {
    VLOG(3) << "Value is root of the same computation";
    return false;
  }
  return ordering_->UsesBeforeValueDefinition(
      a.uses, *b.value, dataflow_,
      /* use_is_always_before_def_in_same_instr=*/false);
}

// Splices the entire linked list with 'head' as its head right after the
// node 'insert_after' in another linked list.
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
                                  CombineLiveRangeOption merge_location) {
  // Get the entire range of values sharing the buffers in src and dest.
  auto src_live_range = ComputeLiveRangeRegions(src);
  auto dest_live_range = ComputeLiveRangeRegions(dest);

  VLOG(3) << "    ValuesInterfere source value: " << src->value->ToString();
  VLOG(5) << "    ValuesInterfere source live range:\n"
          << src_live_range.ToString();
  VLOG(3) << "    ValuesInterfere destination value: "
          << dest->value->ToString();
  VLOG(5) << "    ValuesInterfere destination live range:\n"
          << dest_live_range.ToString();

  ComputeRelativeLocation relative_location_analysis(ordering_);
  auto rel1 = relative_location_analysis.ComputeBetweenLiveRangeRegions(
      src_live_range, dest_live_range);
  VLOG(3) << "    ValuesInterfere - location of dest in relation to src: ";
  VLOG(3) << "            " << rel1.ToString();

  auto rel2 = relative_location_analysis.ComputeBetweenLiveRangeRegions(
      dest_live_range, src_live_range);
  VLOG(3) << "    ValuesInterfere - location of src in relation to dest: ";
  VLOG(3) << "            " << rel2.ToString();

  // If src and dest are interleaved with each other, they interfere.
  if (rel1.RuntimeOrderOverlap() && rel2.RuntimeOrderOverlap()) {
    VLOG(3) << "    ValuesInterfere: Both relations are overlapped.";
    return true;
  }
  // If src and dest belong to the same group of computations and do not
  // overlap, they do not interfere.
  if (rel1.RuntimeOrderOverlap() || rel2.RuntimeOrderOverlap()) {
    VLOG(3) << "    ValuesInterfere: At least one relation is overlapped.";
    if (rel1.RuntimeOrderOverlap()) {
      VLOG(3) << "    ValuesInterfere: rel1 is overlapped, with interception = "
              << rel1.InterceptDefUse();
      if (rel1.InterceptDefUse() ||
          (merge_location != kMergeFirstDestInSource &&
           rel2.InterceptDefUse())) {
        return true;
      }
    } else {
      VLOG(3) << "    ValuesInterfere: rel2 is overlapped, with interception = "
              << rel2.InterceptDefUse();
      // Here src is at the end of a nested computation inside dest.
      if (rel2.InterceptDefUse() || (merge_location != kMergeLastSourceInDest &&
                                     rel1.InterceptDefUse())) {
        return true;
      }
    }
  }
  if (relative_location_analysis.AddControlDependenceForUnorderedOps()) {
    return false;
  }
  // Disallow removing of copy if control deps cannot be added.
  return true;
}

// Calls `visitor` on each item in the sequence of HloValues starting from
// `element` and wrapping around.
void CopyRemover::ForEachValueInRange(
    const ValueNode* element,
    absl::FunctionRef<void(const ValueNode*)> visitor) {
  const ValueNode* p = element;
  do {
    CHECK_NE(p, nullptr);
    visitor(p);
    p = p->next;
  } while (p != element);
}

std::string CopyRemover::ValueListToString(const ValueNode* element) {
  std::string result = "{";
  auto VisitValueNode = [&](const ValueNode* node) {
    if (result == "{") {
      StrAppend(&result, node->value->ToShortString());
    } else {
      StrAppend(&result, ", ", node->value->ToShortString());
    }
  };
  ForEachValueInRange(element, VisitValueNode);
  StrAppend(&result, "}");
  return result;
}

std::string CopyRemover::ToString() const {
  std::string out = absl::StrCat("CopyRemover:\n");
  StrAppend(&out, "  Def-use chains in each buffer:\n");
  for (const ValueNode* head : value_lists_) {
    StrAppend(&out, "    Buffer defined by ", head->value->ToShortString(),
              ":\n");
    const ValueNode* p = head;
    do {
      StrAppend(&out, "      ", p->value->ToShortString(), ", uses: ",
                absl::StrJoin(p->uses, "; ",
                              [](std::string* s, const HloUse* use) {
                                StrAppend(s, use->ToString());
                              }),
                "\n");

      p = p->next;
    } while (p != head);
  }
  StrAppend(&out, "  Potentially removable copies:\n");
  for (const auto& pair : copy_map_) {
    const HloInstruction* copy = pair.first;
    const CopyNodes& copy_info = pair.second;

    StrAppend(&out, "    ", copy->name(), " : ",
              copy_info.src->value->ToShortString(), " => ",
              copy_info.dest->value->ToShortString(), "\n");
  }
  return out;
}
}  // namespace xla
