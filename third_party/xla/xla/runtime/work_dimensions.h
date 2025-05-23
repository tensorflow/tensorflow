
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

#ifndef XLA_RUNTIME_WORK_DIMENSIONS_H_
#define XLA_RUNTIME_WORK_DIMENSIONS_H_

#include "absl/strings/str_format.h"
#include "xla/runtime/work_cluster.h"
#include "xla/runtime/work_group.h"
#include "xla/runtime/work_item.h"

namespace xla {

// WorkDimensions defines all levels of the parallel execution hierarchy needed
// to launch a kernel.
struct WorkDimensions {
  bool operator==(const WorkDimensions& other) const {
    return num_work_clusters == other.num_work_clusters &&
           num_work_groups == other.num_work_groups &&
           num_work_items == other.num_work_items;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const WorkDimensions& d) {
    absl::Format(&sink, "WorkDimensions{%v, %v, %v}", d.num_work_clusters,
                 d.num_work_groups, d.num_work_items);
  }

  NumWorkClusters num_work_clusters;
  NumWorkGroups num_work_groups;
  NumWorkItems num_work_items;
};

}  // namespace xla
#endif  // XLA_RUNTIME_WORK_DIMENSIONS_H_
