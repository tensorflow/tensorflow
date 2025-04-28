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

#ifndef XLA_SERVICE_COPY_REMOVAL_H_
#define XLA_SERVICE_COPY_REMOVAL_H_

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"
#include "xla/map_util.h"
#include "xla/service/hlo_value.h"

namespace xla {
using absl::StrAppend;

class LiveRangeRegions {
 public:
  struct InstructionInfo {
    InstructionInfo() : value_definition(nullptr), is_definition(false) {}

    // The instruction that defines the value being used. It basically saves
    // the defining instruction of each HloValue.
    HloInstruction* value_definition;
    // Whether the instruction defines a new value (or merely uses one). This
    // basically remembers whether the instruction actually creates an HloValue
    // or merely uses one, from a collection of given HloValues. Note that if
    // is_definition = true, it merely says the instruction creates a new
    // HloValue with or without defining a new one. For example, kAdd create a
    // new HloValue (can be value_definition), but tuples or get-tuple-element,
    // create a new HloValue aliasing without defining a new value (cannot be
    // value_definition).
    bool is_definition;

    std::string ToString() const {
      return absl::StrCat(
          "is_definition: ", std::to_string(is_definition),
          ", value_definition: ",
          value_definition ? value_definition->name() : "nullptr");
    }
  };
  // Map instructions that use a value to the defining instruction of the
  // value. Because all values must belong to the same live range, an
  // instruction can have at most a single value-defining instruction;
  // otherwise the multiple incoming active values would share a single
  // buffer, which is not allowed. The value-defining and value-use
  // instructions do not have to belong to the same computation, but the value
  // use needs to be nested within the defining computation.
  typedef HloInstructionMap<InstructionInfo> InstructionMap;
  typedef std::pair<HloInstruction*, InstructionInfo> InstructionEntry;
  // Map each computation to its immediately contained instructions.
  typedef absl::flat_hash_map<const HloComputation*, InstructionMap>
      ComputationMap;

  InstructionMap& operator[](const HloComputation* computation) {
    if (computation_map_.find(computation) == computation_map_.end()) {
      computation_vector_.push_back(computation);
    }
    return computation_map_[computation];
  }

  const InstructionMap& operator[](const HloComputation* computation) const {
    ComputationMap::const_iterator p = computation_map_.find(computation);
    CHECK(p != computation_map_.end());
    return p->second;
  }
  absl::InlinedVector<const HloComputation*, 5>::const_iterator begin() const {
    return computation_vector_.begin();
  }
  absl::InlinedVector<const HloComputation*, 5>::const_iterator end() const {
    return computation_vector_.end();
  }
  int64_t size() const {
    CHECK_EQ(computation_vector_.size(), computation_map_.size());
    return computation_vector_.size();
  }
  bool empty() const { return size() == 0; }
  const HloComputation* Computation(int64_t index) const {
    return computation_vector_[index];
  }
  bool contains(HloInstruction* instr) const {
    CHECK_NE(instr, nullptr);
    auto* computation = instr->parent();
    auto p = computation_map_.find(computation);
    if (p == computation_map_.end()) {
      return false;
    }
    auto instr_map = (*p).second;
    return instr_map.find(instr) != instr_map.end();
  }

  std::string ToString() const {
    std::string result;

    for (const auto* computation : computation_vector_) {
      StrAppend(&result, "computation: ", computation->name(), "\n");
      for (const auto& entry : computation_map_.at(computation)) {
        StrAppend(&result, "  entry: ", entry.first->name(), ", ",
                  entry.second.ToString(), "\n");
      }
    }

    return result;
  }

 private:
  ComputationMap computation_map_;
  absl::InlinedVector<const HloComputation*, 5> computation_vector_;
};

#define RUNTIME_ORDER_LIST(V)                                                  \
  /* Indicates that there is no overlap whatsoever between the two regions. */ \
  V(kNoOverlap, 0)                                                             \
  /* Indicates that the first region includes the same set of instructions     \
    as the second region. */                                                   \
  V(kSameInstr, 1)                                                             \
  /* Indicates that the first region is entirely before the second region      \
    starts. */                                                                 \
  V(kBeforeStart, 2)                                                           \
  /* Indicates that the first region is before the second region ends. */      \
  V(kBeforeStartOrSameInstr, kBeforeStart | kSameInstr)                        \
  /* Indicates that the first region is entirely after the second region       \
    ends. */                                                                   \
  V(kAfterEnd, 4)                                                              \
  /* Indicates that the first region is after the second region                \
    starts, with some instructions before the second region ends. */           \
  V(kAfterEndOrSameInstr, kAfterEnd | kSameInstr)                              \
  /* Indicates that the first region overlaps with the second one, but share   \
    no common instructions. */                                                 \
  V(kBeforeStartOrAfterEnd, kBeforeStart | kAfterEnd)                          \
  /* Indicates that the first region overlaps with the second one, and have    \
    some common instructions. */                                               \
  V(kBeforeOrAfterOrOverlap, kBeforeStart | kAfterEnd | kSameInstr)

// Represent relations between the locations of two regions of instructions,
// each region can include 0-n instructions.
class Relation {
 public:
  enum RuntimeOrder {
#define DECLARE_ENUM(enum_name, enum_value) enum_name = enum_value,
    RUNTIME_ORDER_LIST(DECLARE_ENUM)
#undef DECLARE_ENUM
  };
  Relation() : intercept_def_use_(false) {}
  explicit Relation(RuntimeOrder order, bool intercept_def_use = false)
      : intercept_def_use_(intercept_def_use) {
    orders_.push_back(order);
  }
  Relation(const Relation& that) = default;
  bool operator==(const Relation& that) const {
    return intercept_def_use_ == that.intercept_def_use_ &&
           absl::c_equal(orders_, that.orders_);
  }

  // Return whether the runtime ordering may imply interception, assuming it
  // models the relation between a modifying and a use instruction.
  bool UseImpliesInterception() const {
    CHECK_EQ(orders_.size(), 1);
    return UseImpliesInterception(orders_[0]);
  }
  // Return whether the runtime ordering may imply interception, assuming it
  // models the relation between a modifying and a definition instruction.
  bool DefinitionImpliesInterception() const {
    CHECK_EQ(orders_.size(), 1);
    return DefinitionImpliesInterception(orders_[0]);
  }
  // Return whether the current relation models a modifying instruction that
  // intercepts the dataflow of another live range region.
  bool InterceptDefUse() const { return intercept_def_use_; }
  // Update interception state to the given value.
  void UpdateInterception(bool value) {
    CHECK_EQ(orders_.size(), 1);
    intercept_def_use_ = value;
  }
  Relation::RuntimeOrder GetRuntimeOrder() const {
    if (orders_.empty()) {
      return Relation::kNoOverlap;
    }
    CHECK_EQ(orders_.size(), 1);
    return orders_[0];
  }
  // Return whether the current relation implies two overlapping regions.
  bool RuntimeOrderOverlap() const {
    return absl::c_any_of(orders_, ImpliesOverlap);
  }
  bool RuntimeOrderIsUnordered() const {
    return orders_.size() == 1 && orders_[0] == kBeforeStartOrAfterEnd;
  }
  bool RuntimeOrderIsNoOverlap() const {
    return orders_.empty() || (orders_.size() == 1 && orders_[0] == kNoOverlap);
  }
  bool RuntimeOrderIsRunBefore() const {
    return orders_.size() == 1 && orders_[0] == kBeforeStart;
  }
  bool RuntimeOrderIsRunAfter() const {
    return orders_.size() == 1 && orders_[0] == kAfterEnd;
  }
  std::string ToString() const {
    auto format_order = [](std::string* out, RuntimeOrder order) {
      switch (order) {
#define DECLARE_CASE(enum_name, enum_value) \
  case enum_name:                           \
    absl::StrAppend(out, #enum_name);       \
    break;
        RUNTIME_ORDER_LIST(DECLARE_CASE)
#undef DECLARE_CASE
      }
    };
    return absl::StrCat("Interception = ", intercept_def_use_, " Orders = ",
                        absl::StrJoin(orders_, ", ", format_order), ",");
  }

  static bool DefinitionImpliesInterception(RuntimeOrder definition) {
    return (definition == kAfterEnd || definition == kBeforeStartOrAfterEnd);
  }
  static bool UseImpliesInterception(RuntimeOrder use) {
    return (use == kBeforeStart || use == kBeforeStartOrAfterEnd);
  }

  // Summarize additional relations into a single runtime ordering, assuming
  // both relations are modeling constraints of the same source instruction.
  void UnionRelationFromSameSource(const Relation& rel);

  // Summarize additional relations into disjoint runtime orderings, assuming
  // the relations are modeling constraints of different source instructions.
  void UnionRelationFromDifferentSource(const Relation& rel);

  static Relation::RuntimeOrder ReverseRuntimeOrder(RuntimeOrder order);

 private:
  // Indicate that the second region may intercept the def-use dataflow of the
  // first region, if their buffers are combined.
  bool intercept_def_use_;
  // Remember the different runtime orderings of different instructions.
  absl::InlinedVector<RuntimeOrder, 4> orders_;

  static RuntimeOrder Union(RuntimeOrder o1, RuntimeOrder o2) {
    return static_cast<Relation::RuntimeOrder>(o1 | o2);
  }
  static bool ImpliesOverlap(RuntimeOrder o) {
    return o >= RuntimeOrder::kBeforeStartOrAfterEnd;
  }
  // Returns whether ordering constraint o1 includes o2 as a subset, when they
  // represent runtime orderings (interleavings) of two different regions.
  static bool Subsume(RuntimeOrder o1, RuntimeOrder o2) {
    return Union(o1, o2) == o1;
  }
  // Overwrites o1 with o2 if o2 subsumes o1 (as defined above by the Subsume
  // function). Return whether o2 is subsumed by the new value in o1.
  static bool OverwriteIfSubsume(RuntimeOrder o2, RuntimeOrder* o1);
};

class ComputeRelativeLocation {
 public:
  typedef LiveRangeRegions::InstructionEntry InstructionEntry;
  explicit ComputeRelativeLocation(HloOrdering* ordering)
      : ordering_(ordering) {
    VLOG(3) << "New analysis";
  }

  // Compute locationing constraints between two instructions. Here entry2 is
  // the source instruction, in that the returned value describes the relation
  // of entry2 in terms of whether it is before or after entry1, and whether it
  // can intercept the def-use data flow of entry1.
  Relation Compute(const InstructionEntry& entry1,
                   const InstructionEntry& entry2, bool instr2_can_modify);

  // Return the relative locations (defined above) of range2 in relation to
  // instructions in range1. Return kNoOverlap if range2 is outside of range1.
  Relation Compute(const LiveRangeRegions& range1,
                   const LiveRangeRegions& range2);

  // Return whether control dependences, if exist, are added successfully.
  bool AddControlDependenceForUnorderedOps();

 private:
  enum ComputeStatus {
    kFullyComputed,
    kPartiallyComputed,
    kNotComputed,
  };
  typedef std::pair<ComputeStatus, Relation::RuntimeOrder> SavedRelation;

  // Returns whether it is safe to force the desired_relation ordering between
  // all operations in unordered_ops and entry2. If safe, save the new enforced
  // ordering relations.
  bool ForceRuntimeOrder(absl::Span<const InstructionEntry> unordered_ops,
                         InstructionEntry entry2,
                         Relation::RuntimeOrder desired_relation);

  static bool AlwaysForceInterception(HloInstruction* instr);

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
  bool InstructionCanIntercept(const InstructionEntry& entry,
                               const LiveRangeRegions& region);

  SavedRelation AlreadyComputed(HloInstruction* op1, HloInstruction* op2);

  Relation::RuntimeOrder Save(HloInstruction* entry1, HloInstruction* entry2,
                              Relation::RuntimeOrder relation,
                              bool is_unordered_originally = false);

  // Compute the runtime ordering constraints between two instructions.
  Relation::RuntimeOrder ComputeRuntimeOrdering(HloInstruction* instr1,
                                                HloInstruction* instr2);

  HloOrdering* ordering_;
  absl::flat_hash_map<
      HloInstruction*,
      absl::flat_hash_map<HloInstruction*, Relation::RuntimeOrder>>
      saved_relations_;
  absl::flat_hash_map<
      HloComputation*,
      absl::flat_hash_map<HloInstruction*, std::vector<HloInstruction*>>>
      ctrl_deps_;
};
// Class which tracks the HLO values within each HLO buffer in the module
// during copy removal.
//
// The values are held in a linked list where there is one list for each
// buffer. Removing a copy instruction merges together the values in the
// source buffer of the copy to the destination buffer of the copy. This class
// tracks these value lists as copies are removed from the graph (and value
// lists are merged).
//
// The CopyRemover object is initialized to match the state of
// HloAliasAnalysis. However, as copies are removed this state diverges. The
// values-to-buffer mapping is maintained outside of HloAliasAnalysis because
// a fully updatable alias analysis is very slow.
class CopyRemover {
 public:
  // The values held in a single HLO buffer are represented using a linked
  // list. An element type in this list is ValueNode.
  //
  // This linked list is hand-rolled to enable efficient splicing of lists
  // using only references to list elements without knowing which lists are
  // being spliced. std::list requires a reference to the list object to
  // splice.
  struct ValueNode {
    explicit ValueNode(const HloValue* v) : value(v) {}

    const HloValue* value;

    // The uses are maintained outside of HloValue::uses() because
    // HloValue::uses() is not updatable (a fully updatable dataflow analysis
    // is slow).
    std::vector<const HloUse*> uses;

    // next/prev elements in the linked list. The list is circularly linked so
    // these values are never null for elements in the list.
    ValueNode* prev = nullptr;
    ValueNode* next = nullptr;
  };

  CopyRemover(const HloModule& module, const HloAliasAnalysis& alias_analysis,
              HloOrdering* ordering, bool check_live_range_ordering,
              const absl::flat_hash_set<absl::string_view>& execution_threads);

  // Add a list containing the given values to CopyRemover. This
  // represents the values contained in a single buffer. For each value in
  // 'values' an entry is created in value_to_node which indicates the
  // respective ValueNode representing that value.
  void AddValueList(
      absl::Span<const HloValue* const> values,
      absl::flat_hash_map<const HloValue*, ValueNode*>* value_to_node);

  // This method also fills in copy_map_ which indicates which nodes
  // in the value lists corresponding to the source and destination values of
  // kCopy instructions. value_to_node should map each HloValue to its
  // respective ValueNode.
  void CreateCopyMap(
      const HloModule& module,
      const absl::flat_hash_map<const HloValue*, ValueNode*>& value_to_node);

  ~CopyRemover();

  // Verify invariants within the linked lists.
  absl::Status Verify() const;

  // Compute the set of instructions where values are alive and organize these
  // instructions by separating them into their respective computations.
  LiveRangeRegions ComputeLiveRangeRegions(const ValueNode* head);

  // Try to elide the given copy. Elision of a copy is possible only if no
  // live range interference is introduced by the copy's elimination. If
  // elision is possible, then the internal state (value lists) are updated,
  // and true is returned. Returns false otherwise.
  bool TryElideCopy(const HloInstruction* copy, int64_t* region_analysis_limit,
                    bool insert_post_scheduling_control_dependencies);

  // Delete the given ValueNode associated with a elided kCopy
  // instruction. This should be called after splicing the value lists of the
  // source and destination buffers together.
  void RemoveCopyValue(ValueNode* copy_value_node);

  // Returns true if the live range of given value 'a' is before the live
  // range of 'b'.
  //
  // We cannot use LiveRangeStrictlyBefore because HloValue::uses() is not
  // updated as copies are removed. Also here because the result is used
  // to directly drive copy elision, use_is_always_before_def_in_same_instr is
  // set to false.
  bool LiveRangeBefore(const ValueNode& a, const ValueNode& b);

  // Returns whether 'node' is the last node in its list.
  bool IsTail(const ValueNode& node) const {
    return ContainsKey(value_lists_, node.next);
  }

  // Returns whether 'node' is the first node in its list.
  bool IsHead(const ValueNode& node) const {
    return ContainsKey(value_lists_, &node);
  }

  // Returns the next node in the list after 'node'. If 'node' is the
  // tail, then nullptr is returned.
  ValueNode* Next(const ValueNode& node) const {
    if (IsTail(node)) {
      return nullptr;
    }
    return node.next;
  }

  // Returns the previous node in the list before 'node'. If 'node'
  // is the head, then nullptr is returned.
  ValueNode* Prev(const ValueNode& node) const {
    if (IsHead(node)) {
      return nullptr;
    }
    return node.prev;
  }

  // Splices the entire linked list with 'head' as its head right after the
  // node 'insert_after' in another linked list.
  void SpliceAfter(ValueNode* head, ValueNode* insert_after);

  enum CombineLiveRangeOption {
    kMergeFirstDestInSource = 1,
    kMergeLastSourceInDest = 2
  };

  // This function analyzes all the HloValues that have been grouped together
  // with src to share a single buffer, and all the HloValues that have been
  // similarly grouped together with dest, to determine whether these two groups
  // can be combined, by removing the operation in dest, which makes a copy of
  // the buffer in src.
  bool ValuesInterfere(const ValueNode* src, const ValueNode* dest,
                       CombineLiveRangeOption merge_location);

  // Calls `visitor` on each item in the sequence of HloValues starting from
  // `element`.
  //
  // If element is not head, traverse from element to tail, then wrap
  // around. The ordering is important for live range region analysis.
  void ForEachValueInRange(const ValueNode* element,
                           absl::FunctionRef<void(const ValueNode*)> visitor);
  std::string ValueListToString(const ValueNode* element);

  std::string ToString() const;

 private:
  const HloDataflowAnalysis& dataflow_;
  HloOrdering* ordering_;

  // The heads of all the value lists. Each value list represents the HLO
  // values contained in a particular HLO buffer. The values in the list are
  // in dependency order.
  absl::flat_hash_set<const ValueNode*> value_lists_;

  // Copy removal requires fast access to the value list elements
  // corresponding to the source and destination values of the kCopy
  // instruction. This data structure holds pointers to these elements for
  // each kCopy instruction in the graph.
  struct CopyNodes {
    // The source and destinations values of the kCopy instruction.
    ValueNode* src = nullptr;
    ValueNode* dest = nullptr;
  };
  absl::flat_hash_map<const HloInstruction*, CopyNodes> copy_map_;
};
};  // namespace xla

#endif  // XLA_SERVICE_COPY_REMOVAL_H_
