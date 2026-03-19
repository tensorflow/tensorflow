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

#ifndef XLA_RUNTIME_WORK_GROUP_H_
#define XLA_RUNTIME_WORK_GROUP_H_

#include <cstdint>

#include "absl/strings/str_format.h"

namespace xla {

// Work group is a collection of work items, that must be able to make progress
// in presence of barriers. In XLA:GPU it corresponds to a block. In XLA:CPU it
// roughly corresponds to a task that runs on a thread pool.

struct NumWorkGroups {
  bool operator==(const NumWorkGroups& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const NumWorkGroups& d) {
    absl::Format(&sink, "NumWorkGroups{%d, %d, %d}", d.x, d.y, d.z);
  }

  uint64_t x = 1;
  uint64_t y = 1;
  uint64_t z = 1;
};

struct WorkGroupId {
  bool operator==(const WorkGroupId& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const WorkGroupId& d) {
    absl::Format(&sink, "WorkGroupId{%d, %d, %d}", d.x, d.y, d.z);
  }

  uint64_t x = 1;
  uint64_t y = 1;
  uint64_t z = 1;
};

}  // namespace xla

#endif  // XLA_RUNTIME_WORK_GROUP_H_
