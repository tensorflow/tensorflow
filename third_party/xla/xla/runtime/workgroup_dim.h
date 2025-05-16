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

#ifndef XLA_RUNTIME_WORKGROUP_DIM_H_
#define XLA_RUNTIME_WORKGROUP_DIM_H_

#include <cstdint>

#include "absl/strings/str_format.h"

namespace xla {

// Dimensionality of an XLA workgroup.
//
// Workgroups mapped to backends specific concepts for parallelizing XLA kernel
// execution, i.e. on GPU backends workgroups are mapped to blocks, and on CPU
// backend workgroups are mapped to parallel tasks (threads).
struct WorkgroupDim {
  bool operator==(const WorkgroupDim& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const WorkgroupDim& d) {
    absl::Format(&sink, "WorkgroupDim{%d, %d, %d}", d.x, d.y, d.z);
  }

  uint64_t x = 1;
  uint64_t y = 1;
  uint64_t z = 1;
};

}  // namespace xla

#endif  // XLA_RUNTIME_WORKGROUP_DIM_H_
