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

#ifndef XLA_PYTHON_DEBUG_CALLBACK_PARTITIONER_H_
#define XLA_PYTHON_DEBUG_CALLBACK_PARTITIONER_H_

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/custom_call_sharding_helper.h"

namespace xla {

// Partition the custom call according to XLA partitioning. Currently only used
// by `jax.debug.callback`.
// TODO(b/409338207): Pass additional metadata to the custom call e.g.,
// partition id.
class DebugCallbackCustomCallPartitioner : public CustomCallPartitioner {
 public:
  absl::Status Partition(spmd::SpmdPartitioningVisitor* partitioner,
                         HloInstruction* hlo) const override;
};

}  // namespace xla

#endif  // XLA_PYTHON_DEBUG_CALLBACK_PARTITIONER_H_
