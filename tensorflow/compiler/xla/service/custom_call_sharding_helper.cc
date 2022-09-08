/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/service/custom_call_sharding_helper.h"

#include <memory>
#include <string>
#include <utility>

namespace xla {

HloSharding CustomCallShardingHelper::PropagateUserSharding(
    const HloInstruction* instruction, const HloInstruction* user,
    const HloSharding& sharding) const {
  return sharding;
}
std::optional<HloSharding> CustomCallShardingHelper::InferShardingFromOperands(
    const HloInstruction* instruction) const {
  return std::nullopt;
}
bool CustomCallShardingHelper::IsCustomCallShardable(
    const HloInstruction* instruction) const {
  return false;
}

xla::Status CustomCallPartitioner::Partition(
    spmd::SpmdPartitioningVisitor* partitioner, HloInstruction* hlo) const {
  return xla::Unimplemented("Implement sharding for %s", hlo->ToString());
}

namespace {
absl::flat_hash_map<std::string, std::unique_ptr<CustomCallPartitioner>>&
GetPartitioners() {
  static auto* out =
      new absl::flat_hash_map<std::string,
                              std::unique_ptr<CustomCallPartitioner>>;
  return *out;
}
}  // namespace

const CustomCallPartitioner* GetCustomCallPartitioner(
    const std::string& custom_call_target) {
  auto& partitioners = GetPartitioners();
  auto it = partitioners.find(custom_call_target);
  if (it == partitioners.end()) {
    return nullptr;
  }
  return it->second.get();
}

void RegisterCustomCallPartitioner(
    const std::string& custom_call_target,
    std::unique_ptr<CustomCallPartitioner> partitioner) {
  auto& partitioners = GetPartitioners();
  partitioners.emplace(custom_call_target, std::move(partitioner));
}

}  // namespace xla
