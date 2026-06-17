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

#ifndef XLA_PYTHON_CUSTOM_CALL_BATCH_PARTITIONER_H_
#define XLA_PYTHON_CUSTOM_CALL_BATCH_PARTITIONER_H_

#include <optional>

#include "absl/status/status.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/ir/hlo_sharding.h"
#include "xla/service/custom_call_sharding_helper.h"
#include "xla/service/spmd/spmd_partitioner.h"

namespace xla {

class CustomCallBatchPartitioner : public CustomCallPartitioner {
 public:
  bool IsCustomCallShardable(const HloInstruction* instruction) const override {
    return true;
  }
  std::optional<HloSharding> InferShardingFromOperands(
      const HloInstruction* hlo) const override;
  absl::Status Partition(spmd::SpmdPartitioningVisitor* partitioner,
                         HloInstruction* hlo) const override;
};

}  // namespace xla

#endif  // XLA_PYTHON_CUSTOM_CALL_BATCH_PARTITIONER_H_
