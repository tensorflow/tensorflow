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

#ifndef XLA_RUNTIME_WORK_CLUSTER_H_
#define XLA_RUNTIME_WORK_CLUSTER_H_

#include <cstdint>

#include "absl/strings/str_format.h"

namespace xla {

// Work cluster is a collection of work groups. In XLA:GPU it corresponds to a
// GPU cluster. In XLA:CPU it roughly corresponds to a collection of tasks, that
// would benefit from temporal data locality and should be executed on a same
// set of physical cores (currently it's not used in XLA:CPU).

struct NumWorkClusters {
  bool operator==(const NumWorkClusters& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const NumWorkClusters& d) {
    absl::Format(&sink, "NumWorkClusters{%d, %d, %d}", d.x, d.y, d.z);
  }

  uint64_t x = 1;
  uint64_t y = 1;
  uint64_t z = 1;
};

struct WorkClusterId {
  bool operator==(const WorkClusterId& other) const {
    return x == other.x && y == other.y && z == other.z;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const WorkClusterId& d) {
    absl::Format(&sink, "WorkClusterId{%d, %d, %d}", d.x, d.y, d.z);
  }

  uint64_t x = 1;
  uint64_t y = 1;
  uint64_t z = 1;
};

}  // namespace xla

#endif  // XLA_RUNTIME_WORK_CLUSTER_H_
