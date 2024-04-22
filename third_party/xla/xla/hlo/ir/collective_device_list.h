/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_HLO_IR_COLLECTIVE_DEVICE_LIST_H_
#define XLA_HLO_IR_COLLECTIVE_DEVICE_LIST_H_

#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Represents a series of devices participating in a collective operation
// (all-gather, all-reduce, etc.). While this directly translates to a list of
// replica groups, it may be used to represent these lists in compact forms.
class CollectiveDeviceList {
 public:
  explicit CollectiveDeviceList(absl::Span<const ReplicaGroup> replica_groups)
      : replica_groups_(SpanToVector(replica_groups)) {}
  explicit CollectiveDeviceList(
      absl::Span<const std::vector<int64_t>> replica_groups);

  explicit CollectiveDeviceList() = default;

  const std::vector<ReplicaGroup>& replica_groups() const {
    return replica_groups_;
  }

 private:
  std::vector<ReplicaGroup> replica_groups_;
};

}  // namespace xla

#endif  // XLA_HLO_IR_COLLECTIVE_DEVICE_LIST_H_
