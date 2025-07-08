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

#ifndef XLA_RUNTIME_WORK_TILE_SIZE_H_
#define XLA_RUNTIME_WORK_TILE_SIZE_H_

#include <cstdint>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/strings/str_format.h"
#include "absl/strings/str_join.h"

namespace xla {

// Work Tile Size defines defines the number of elements each work item should
// process.
// The tile size may be empty and in such cases it will be treated as a single
// element.
struct WorkTileSize {
  bool operator==(const WorkTileSize& other) const {
    return absl::c_equal(dimensions, other.dimensions);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const WorkTileSize& d) {
    absl::Format(&sink, "WorkTileSize{%s}", absl::StrJoin(d.dimensions, ","));
  }

  absl::InlinedVector<uint64_t, 3> dimensions;
};

}  // namespace xla

#endif  // XLA_RUNTIME_WORK_TILE_SIZE_H_
