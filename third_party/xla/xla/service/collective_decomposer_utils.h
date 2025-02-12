/* Copyright 2021 The OpenXLA Authors.

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

#include <functional>

#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/collective_ops_utils.h"

#ifndef XLA_SERVICE_COLLECTIVE_DECOMPOSER_UTILS_H_
#define XLA_SERVICE_COLLECTIVE_DECOMPOSER_UTILS_H_

namespace xla {

absl::StatusOr<std::vector<HloInstruction *>>
CreateStartIndicesForCollectiveDecomposition(
    CollectiveOpGroupMode group_mode,
    absl::Span<const ReplicaGroup> replica_groups, const Shape &shard_shape,
    int64_t shard_dimension, HloComputation *computation,
    std::function<void(Shape &)> update_layout = nullptr);

}  // namespace xla

#endif  // XLA_SERVICE_COLLECTIVE_DECOMPOSER_UTILS_H_
