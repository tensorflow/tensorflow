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
#include <deque>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/functional/function_ref.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/hlo/analysis/alias_info.h"
#include "xla/hlo/analysis/hlo_alias_analysis.h"
#include "xla/hlo/analysis/hlo_dataflow_analysis.h"
#include "xla/hlo/analysis/hlo_ordering.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_module.h"
#include "xla/service/hlo_value.h"

namespace xla {

// Tracks the HLO values within each buffer during copy removal. Eliding a copy
// merges the circular doubly linked value lists of its source and destination
// buffers without paying the cost of a mutable HloAliasAnalysis.
class CopyRemover {
 public:
  struct ElideOptions {
    bool use_region_analysis = false;
    bool insert_post_scheduling_control_dependencies = false;
  };

  // `view_color` is the layout memory space color of address stand in values
  // that alias operand 0 without owning storage (dus_view_color). Readers of a
  // view count as uses of the viewed value at the reader's position.
  CopyRemover(const HloModule& module, const HloAliasAnalysis& alias_analysis,
              const AliasInfo* alias_info, HloOrdering* ordering,
              const absl::flat_hash_set<absl::string_view>& execution_threads,
              std::optional<int64_t> view_color = std::nullopt);

  CopyRemover(const CopyRemover&) = delete;
  CopyRemover& operator=(const CopyRemover&) = delete;

  // Updates internal value lists and returns true if eliminating `copy`
  // introduces no live range interference.
  bool TryElideCopy(const HloInstruction* copy, const ElideOptions& options,
                    const std::function<bool(const HloInstruction* copy)>&
                        should_skip_removal = nullptr);

  std::string ToString() const;

 private:
  struct ValueNode {
    explicit ValueNode(const HloValue* v) : value(v) {}

    const HloValue* value;
    std::vector<const HloUse*> uses;
    ValueNode* prev = nullptr;
    ValueNode* next = nullptr;
  };

  struct CopyNodes {
    ValueNode* src = nullptr;
    ValueNode* dest = nullptr;
  };

  // Records readers of `view_color_` views of `value` as uses on `node`. This
  // flat walk relies on the view creation pass guaranteeing that views are one
  // level deep, are never written through by readers, and never escape into
  // tuples, called computations, or async pairs.
  void AddViewUses(const HloValue* value, ValueNode* node);
  bool IsViewUse(const HloUse* use) const {
    return view_uses_by_pointer_.contains(use);
  }
  absl::Status Verify() const;
  void RemoveCopyValue(ValueNode* copy_node, ValueNode* operand_node);
  bool LiveRangeBefore(const ValueNode& a, const ValueNode& b);
  void SpliceAfter(ValueNode* head, ValueNode* insert_after);
  bool ValuesInterfere(const ValueNode* src, const ValueNode* dest,
                       bool dest_spliced_into_src);
  void ForEachValueInRange(
      const ValueNode* start,
      absl::FunctionRef<void(const ValueNode*)> visitor) const;

  bool IsHead(const ValueNode& node) const {
    return value_lists_.contains(&node);
  }
  bool IsTail(const ValueNode& node) const { return IsHead(*node.next); }
  ValueNode* Next(const ValueNode& node) const {
    return IsTail(node) ? nullptr : node.next;
  }
  ValueNode* Prev(const ValueNode& node) const {
    return IsHead(node) ? nullptr : node.prev;
  }

  const HloDataflowAnalysis& dataflow_;
  const AliasInfo* alias_info_;
  HloOrdering* ordering_;

  std::deque<ValueNode> nodes_;
  absl::flat_hash_set<const ValueNode*> value_lists_;
  absl::flat_hash_map<const HloInstruction*, CopyNodes> copy_map_;

  std::optional<int64_t> view_color_;
  std::deque<HloUse> view_uses_;
  absl::flat_hash_set<const HloUse*> view_uses_by_pointer_;
};

}  // namespace xla

#endif  // XLA_SERVICE_COPY_REMOVAL_H_
