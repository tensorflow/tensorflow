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

#ifndef XLA_RUNTIME_WORK_ITEM_H_
#define XLA_RUNTIME_WORK_ITEM_H_

#include <cstdint>

#include "absl/strings/str_format.h"

namespace xla {

// Work item is the lowest level of the kernel parallel execution hierarchy in
// XLA. In XLA:GPU it corresponds to a SIMT thread. In XLA:CPU it roughly
// corresponds to one iteration of the kernel loop nest.

struct NumWorkItems {
  bool operator==(const NumWorkItems& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const NumWorkItems& d) {
    absl::Format(&sink, "NumWorkItems{%d, %d, %d}", d.x, d.y, d.z);
  }

  uint64_t x = 1;
  uint64_t y = 1;
  uint64_t z = 1;
};

struct WorkItemId {
  bool operator==(const WorkItemId& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const WorkItemId& d) {
    absl::Format(&sink, "WorkItemId{%d, %d, %d}", d.x, d.y, d.z);
  }

  uint64_t x = 1;
  uint64_t y = 1;
  uint64_t z = 1;
};

}  // namespace xla

#endif  // XLA_RUNTIME_WORK_ITEM_H_
