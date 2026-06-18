/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/hlo/transforms/simplifiers/hlo_rematerialization_data_structures.h"

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/algorithm/container.h"
#include "absl/container/flat_hash_map.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_schedule.h"

namespace xla {

absl::Status HloRematInstructionList::UpdateFromSequence(
    const HloInstructionSequence& order, bool preserve_denylist) {
  absl::flat_hash_map<const HloInstruction*, bool> denylist;
  if (preserve_denylist) {
    for (HloInstruction* inst : order.instructions()) {
      if (item_map_.contains(inst) && item_map_[inst]->denylisted) {
        denylist[inst] = true;
      }
    }
  }
  // Clear item map
  item_map_.clear();
  int64_t position = 0;
  HloRematItem* last = nullptr;
  last_skip_node_ = nullptr;
  first_skip_node_ = nullptr;
  for (HloInstruction* inst : order.instructions()) {
    // Add a new item to the linked list.
    auto item = std::make_unique<HloRematItem>();
    if (preserve_denylist && denylist.contains(inst)) {
      item->denylisted = true;
    }
    item->next = nullptr;
    item->prev = last;
    if (last == nullptr) {
      first_ = item.get();
    } else {
      last->next = item.get();
    }
    last = item.get();

    // Initially position numbers are uniquely assigned in order. Later as
    // instructions are added with InsertBefore* methods, some instructions
    // may have duplicate position numbers, but the values will be guaranteed
    // to be monotonically increasing through the list, and so is still useful
    // for quickly(-ish) determining the order of arbitrary instructions in
    // the list.
    item->instruction = inst;
    item->position = position;
    position++;

    item_map_[inst] = std::move(item);
  }
  return absl::OkStatus();
}

void HloRematInstructionList::InsertBeforeInstructions(
    HloRematItem* to_insert,
    absl::Span<HloRematItem* const> before_instructions) {
  VLOG(3) << "InsertBeforeInstructions: " << to_insert->instruction->name()
          << " before {"
          << absl::StrJoin(before_instructions, ", ",
                           [](std::string* out, HloRematItem* item) {
                             absl::StrAppend(out, item->instruction->name());
                           })
          << "}";

  // Find the minimal position number of any instruction in
  // 'before_instructions'.
  CHECK(!before_instructions.empty());
  HloRematItem* min_position_item = nullptr;
  for (HloRematItem* item : before_instructions) {
    if (min_position_item == nullptr ||
        item->position < min_position_item->position) {
      min_position_item = item;
    }
  }

  // Because more than one instruction in 'before_instructions' may have a
  // position number of 'min_position_number', find the first such instruction
  // with position number 'min_position_number'.

  // First find first instruction with the min position.
  CHECK(min_position_item != nullptr);
  while (min_position_item->prev != nullptr &&
         min_position_item->position == min_position_item->prev->position) {
    min_position_item = min_position_item->prev;
  }

  // Now scan forwards until we find one of the before_instructions.
  while (!absl::c_linear_search(before_instructions, min_position_item)) {
    min_position_item = min_position_item->next;
  }
  return InsertBefore(to_insert, min_position_item);
}

void HloRematInstructionList::PromoteNodesToSkip(
    absl::FunctionRef<bool(HloRematItem*)> should_promote) {
  int64_t count = 0;
  for (auto* item = first(); item != nullptr; item = next(item)) {
    if (should_promote(item)) {
      count += 1;
      if (first_skip_node_ == nullptr) {
        first_skip_node_ = item;
      }
      item->is_skip_node = true;
      item->prev_skip_node = last_skip_node_;
      if (last_skip_node_ != nullptr) {
        last_skip_node_->next_skip_node = item;
      }
      last_skip_node_ = item;
    }
  }
  VLOG(1) << " Rematerialization has " << count << " items in express lane";
}

void HloRematInstructionList::InsertAfterInstructions(
    HloRematItem* to_insert,
    absl::Span<HloRematItem* const> after_instructions) {
  VLOG(3) << "InsertAfterInstructions: " << to_insert->instruction->name()
          << " after {"
          << absl::StrJoin(after_instructions, ", ",
                           [](std::string* out, HloRematItem* item) {
                             absl::StrAppend(out, item->instruction->name());
                           })
          << "}";

  // Find the max position number of any instruction in
  // 'after_instructions'.
  CHECK(!after_instructions.empty());
  HloRematItem* max_position_item = nullptr;
  for (HloRematItem* item : after_instructions) {
    if (max_position_item == nullptr ||
        item->position > max_position_item->position) {
      max_position_item = item;
    }
  }
  // No rematerializable instruction should be inserted at the end of the
  // computation.
  CHECK(max_position_item != nullptr);
  CHECK(max_position_item->next != nullptr);
  InsertBeforeInstructions(to_insert, {max_position_item->next});
}

void HloRematInstructionList::InsertBefore(HloRematItem* item,
                                           HloRematItem* before) {
  VLOG(3) << "InsertBefore: " << item->instruction->name() << " before "
          << before->instruction->name();
  // Always place new nodes on express lane for the ease of implementation.
  item->is_skip_node = true;
  // Find the next express node starting from 'before'. Set up the node's
  // express pointers.
  HloRematItem* cursor = before;
  while (cursor != nullptr && !cursor->is_skip_node) {
    cursor = cursor->next;
  }
  CHECK(cursor == nullptr || cursor->is_skip_node);
  if (cursor == nullptr) {
    //
    // last_skip_node_<---+                              : express lane
    //                    |
    //           ...<->`item`<-> .. <-> `cursor`(null)   : slow lane
    //
    // Reached the end. Set the prev_express to last_skip_node, and reset
    // last_skip.
    item->prev_skip_node = last_skip_node_;
    item->next_skip_node = nullptr;
    last_skip_node_ = item;
  } else {
    //
    //     <-+------------+----------------+--------->   : express lane
    //       |            |                |
    // prev_express..<->`item`<-> .. <-> `cursor` <-> ...: slow lane
    //
    // Reached the next skip node, sets up express pointers accordingly.
    CHECK(cursor->is_skip_node);
    item->prev_skip_node = cursor->prev_skip_node;
    if (item->prev_skip_node != nullptr) {
      item->prev_skip_node->next_skip_node = item;
    }
    item->next_skip_node = cursor;
    cursor->prev_skip_node = item;
  }
  if (first_skip_node_ == cursor) {
    first_skip_node_ = item;
  }
  // Insert new item into linked list.
  item->prev = before->prev;
  item->next = before;
  before->prev = item;
  if (item->prev != nullptr) {
    item->prev->next = item;
  } else {
    first_ = item;
  }

  // Assign the same position number to the newly added instruction as
  // 'before'. This guarantees monotonicity of the position numbers, but not
  // uniqueness.
  item->position = before->position;
}
}  // namespace xla
