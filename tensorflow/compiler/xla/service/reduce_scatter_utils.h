/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_REDUCE_SCATTER_UTILS_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_REDUCE_SCATTER_UTILS_H_

#include "tensorflow/compiler/xla/service/hlo_instructions.h"

namespace xla {

struct ReduceScatterSpec {
  int64_t split_dim;
  int64_t sharded_partitions = 1;
  int64_t sharded_replicas = 1;
  int64_t group_size;
  std::vector<int64_t> original_split_dims;
  HloInstruction* dynamic_slice;
};

// Matches the given all-reduce operation to a reduce-scatter pattern.
absl::optional<ReduceScatterSpec> MatchReduceScatter(
    const HloAllReduceInstruction* ar, int64_t num_partitions,
    int64_t num_replicas, bool allow_multiple_split_dims = false,
    bool allow_intervening_reshape = false, int64_t min_rank = 1);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_REDUCE_SCATTER_UTILS_H_
