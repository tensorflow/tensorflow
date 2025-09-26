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

#include "xla/tuple_tree.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "absl/base/optimization.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/statusor.h"

namespace xla {
namespace internal {

namespace {

// Helper function to recursively build the IndexTable for a subtree.
void BuildSubTableImpl(const IndexTable& original_table,
                       const IndexTable::Entry* original_entry,
                       absl::Span<IndexTable::Entry> new_entries,
                       size_t current_new_entry_idx, size_t& next_node_id,
                       size_t& next_children_start_index) {
  IndexTable::Entry& new_entry = new_entries[current_new_entry_idx];
  new_entry.node_id = next_node_id++;
  new_entry.num_children = original_entry->num_children;

  if (original_entry->children_start_id == -1) {
    new_entry.children_start_id = -1;
    return;
  }

  new_entry.children_start_id = next_children_start_index;
  size_t my_children_start = next_children_start_index;
  next_children_start_index += new_entry.num_children;

  const auto& original_entries = *original_table.entries();
  for (size_t i = 0; i < new_entry.num_children; ++i) {
    const IndexTable::Entry* original_child_entry =
        &original_entries[original_entry->children_start_id + i];
    BuildSubTableImpl(original_table, original_child_entry, new_entries,
                      my_children_start + i, next_node_id,
                      next_children_start_index);
  }
}

}  // namespace

size_t IndexTable::CountSubtreeNodes(const IndexTable& table,
                                     const IndexTable::Entry* root_entry) {
  if (!table.entries().has_value() || root_entry == nullptr) {
    return 0;
  }
  if (root_entry->children_start_id == -1) {
    return 1;
  }
  size_t count = 1;
  const auto& entries = *table.entries();
  for (size_t i = 0; i < root_entry->num_children; ++i) {
    const IndexTable::Entry* child_entry =
        &entries[root_entry->children_start_id + i];
    count += CountSubtreeNodes(table, child_entry);
  }
  return count;
}

absl::Status IndexTable::IsSubtreeCompatible(const IndexTable& other_table,
                                             const Entry* other_entry,
                                             const IndexTable& this_table,
                                             const Entry* this_entry) {
  if (other_entry->num_children != this_entry->num_children) {
    return absl::InvalidArgumentError(
        "Subtree structures are incompatible: different number of children");
  }

  if (other_entry->children_start_id == -1) {  // Leaf nodes
    return absl::OkStatus();
  }

  const auto& other_entries = *other_table.entries();
  const auto& this_entries = *this_table.entries();

  for (size_t i = 0; i < other_entry->num_children; ++i) {
    const Entry* other_child =
        &other_entries[other_entry->children_start_id + i];
    const Entry* this_child = &this_entries[this_entry->children_start_id + i];
    TF_RETURN_IF_ERROR(
        IsSubtreeCompatible(other_table, other_child, this_table, this_child));
  }
  return absl::OkStatus();
}

// Computes the total size of all nested tuples in the given tuple shape.
size_t IndexTable::IndexTableTuplesSize(const Shape& shape) {
  DCHECK(shape.IsTuple()) << "Shape must be a tuple";

  size_t size = shape.tuple_shapes().size();
  for (const Shape& subshape : shape.tuple_shapes()) {
    if (ABSL_PREDICT_FALSE(subshape.IsTuple())) {
      size += IndexTableTuplesSize(subshape);
    }
  }

  return size;
}

// Initializes the index table in the given entries span. Span must point into
// the appropriately sized entries storage.
void IndexTable::InitializeIndexTable(const Shape& shape,
                                      absl::Span<IndexTable::Entry> entries,
                                      size_t entry_index, size_t& next_node_id,
                                      size_t& next_children_start_index) {
  IndexTable::Entry& entry = entries[entry_index];
  entry.node_id = next_node_id++;

  // Stop shape traversal once reaching a leaf shape.
  if (!shape.IsTuple()) {
    entry.children_start_id = -1;
    entry.num_children = 0;
    return;
  }

  // The nodes are in depth-first pre-order. However, in order to efficiently
  // lookup indices, we generate the index table using breadth-first.
  entry.children_start_id = next_children_start_index;
  entry.num_children = shape.tuple_shapes().size();

  // Add entry for children first, before recursing, so they are consecutive.
  next_children_start_index += shape.tuple_shapes().size();
  for (size_t i = 0; i < shape.tuple_shapes().size(); ++i) {
    InitializeIndexTable(shape.tuple_shapes(i), entries,
                         entry.children_start_id + i, next_node_id,
                         next_children_start_index);
  }
}

IndexTable::IndexTable(const Shape& shape) {
  if (!shape.IsTuple()) {
    entries_.emplace(1);
    entries_->front() = {0, -1, 0};
    return;
  }

  // Allocate storage for the index table.
  entries_.emplace(1 + IndexTableTuplesSize(shape));

  size_t next_node_id = 0;
  size_t next_children_start_index = 1;
  InitializeIndexTable(shape, absl::MakeSpan(*entries_), 0, next_node_id,
                       next_children_start_index);
}

absl::StatusOr<const IndexTable::Entry*> IndexTable::GetEntry(
    ShapeIndexView index) const {
  if (!entries_.has_value()) {
    return absl::FailedPreconditionError("Index table not initialized");
  }
  const Entry* current = &entries_->front();
  for (int64_t i : index) {
    if (i < 0) {
      return absl::OutOfRangeError("Negative index in ShapeIndex");
    }
    if (current->children_start_id == -1) {
      return absl::InvalidArgumentError("Cannot index into a leaf node");
    }
    if (i >= current->num_children) {
      return absl::NotFoundError("Index out of bounds");
    }
    current = &(*entries_)[current->children_start_id + i];
  }
  return current;
}

absl::StatusOr<IndexTable> IndexTable::CreateFromSubtree(
    const IndexTable& original_table, const ShapeIndex& index) {
  TF_ASSIGN_OR_RETURN(const Entry* root_entry, original_table.GetEntry(index));

  size_t num_nodes = CountSubtreeNodes(original_table, root_entry);
  if (num_nodes == 0) {
    return absl::InvalidArgumentError("Subtree is empty");
  }

  IndexTable new_table;
  new_table.entries_.emplace(num_nodes);

  size_t next_node_id = 0;
  size_t next_children_start_index = 1;
  if (num_nodes > 0) {
    BuildSubTableImpl(original_table, root_entry,
                      absl::MakeSpan(*new_table.entries_), 0, next_node_id,
                      next_children_start_index);
  }

  return new_table;
}

}  // namespace internal
}  // namespace xla
