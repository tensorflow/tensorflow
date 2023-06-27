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

#ifndef TENSORFLOW_COMPILER_XLA_SERVICE_ALL_REDUCE_KEY_H_
#define TENSORFLOW_COMPILER_XLA_SERVICE_ALL_REDUCE_KEY_H_

#include <tuple>
#include <vector>

#include "tensorflow/compiler/xla/hlo/ir/hlo_instruction.h"
#include "tensorflow/compiler/xla/service/hlo_domain_map.h"

namespace xla {

// Encapsulates all of the properties which must match for two all-reduce
// instructions to be compatible with each other (and hence be possible to
// combine the instructions).
using AllReduceKey =
    std::tuple<HloOpcode, PrimitiveType,
               /*domain metadata id*/ int64_t,
               /*has channel id*/ bool,
               /*use_global_device_ids*/ bool,
               /*replica_groups*/ std::vector<std::vector<int64_t>>>;

std::optional<AllReduceKey> GetAllReduceKey(
    const HloInstruction* instruction, const HloDomainMap* domain_map = nullptr,
    bool ignore_replica_groups = false);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SERVICE_ALL_REDUCE_KEY_H_
