/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/shape_tree.h"

#include <cstddef>
#include <cstdint>

#include "absl/base/optimization.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {
namespace internal {

// Computes the total size of all nested tuples in the given shape.
//
// If `is_known_tuple` is true, then the shape is known to be a tuple, and we
// can skip the run time check for `IsTuple()`.
template <bool is_known_tuple = false>
static size_t IndexTableTuplesSize(const Shape& shape) {
  if (!is_known_tuple && ABSL_PREDICT_TRUE(!shape.IsTuple())) {
    return 0;
  }

  size_t size = shape.tuple_shapes().size();
  for (const Shape& subshape : shape.tuple_shapes()) {
    if (ABSL_PREDICT_FALSE(subshape.IsTuple())) {
      size += IndexTableTuplesSize</*is_known_tuple=*/true>(subshape);
    }
  }

  return size;
}

// Initializes the index table in the given entries span. Span must point into
// the appropriately sized entries storage.
static void InitializeIndexTable(const Shape& shape,
                                 absl::Span<IndexTable::Entry> entries,
                                 size_t entry_index, size_t& next_node_id,
                                 size_t& next_children_start_index) {
  IndexTable::Entry& entry = entries[entry_index];
  entry.node_id = next_node_id++;

  // Stop shape traversal once reaching a leaf shape.
  if (!shape.IsTuple()) {
    return;
  }

  // The nodes are in depth-first pre-order. However, in order to efficiently
  // lookup indices, we generate the index table using breadth-first.
  entry.children_start_id = next_children_start_index;

  // Add entry for children first, before recursing, so they are consecutive.
  next_children_start_index += shape.tuple_shapes().size();
  for (size_t i = 0; i < shape.tuple_shapes().size(); ++i) {
    InitializeIndexTable(shape.tuple_shapes(i), entries,
                         entry.children_start_id + i, next_node_id,
                         next_children_start_index);
  }
}

IndexTable::IndexTable(const Shape& shape)
    : entries_(1 + IndexTableTuplesSize(shape)) {
  size_t next_node_id = 0;
  size_t next_children_start_index = 1;
  InitializeIndexTable(shape, absl::MakeSpan(entries_), 0, next_node_id,
                       next_children_start_index);
}

const IndexTable::Entry& IndexTable::operator[](ShapeIndexView index) const {
  const Entry* result = &entries_.front();
  for (int64_t i : index) {
    CHECK_GE(result->children_start_id, 0);
    result = &entries_[result->children_start_id + i];
  }
  return *result;
}

}  // namespace internal
}  // namespace xla
