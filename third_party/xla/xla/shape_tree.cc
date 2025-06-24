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

#include "absl/log/check.h"
#include "xla/shape.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {
namespace internal {

IndexTable::IndexTable(const Shape& shape) : entries_(1) {
  size_t next_node_id = 0;
  CreateEntry(entries_[0], shape, next_node_id);
}

// TODO(cjfj): Index table cache?.
void IndexTable::CreateEntry(Entry& entry, const Shape& shape,
                             size_t& next_node_id) {
  entry.node_id = next_node_id++;
  if (!shape.IsTuple()) return;

  // The nodes are in depth-first pre-order. However, in order to efficiently
  // lookup indices, we generate the index table using breadth-first.
  size_t children_start_id = entries_.size();
  entry.children_start_id = children_start_id;
  // Add entry for children first, before recursing, so they are consecutive.
  entries_.resize(entries_.size() + shape.tuple_shapes().size());
  for (size_t i = 0; i < shape.tuple_shapes().size(); ++i) {
    CreateEntry(entries_[children_start_id + i], shape.tuple_shapes(i),
                next_node_id);
  }
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
