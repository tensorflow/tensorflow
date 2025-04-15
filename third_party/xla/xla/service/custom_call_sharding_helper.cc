/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/service/custom_call_sharding_helper.h"

#include <memory>
#include <string>
#include <utility>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tsl/platform/logging.h"

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

bool CustomCallShardingHelper::CanPropagateShardingToOperands(
    const HloInstruction* instruction) const {
  return true;
}

absl::Status CustomCallPartitioner::Partition(
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

ABSL_CONST_INIT absl::Mutex partitioners_mutex(absl::kConstInit);
}  // namespace

const CustomCallPartitioner* GetCustomCallPartitioner(
    const std::string& custom_call_target) {
  absl::MutexLock partitioners_lock(&partitioners_mutex);
  auto& partitioners = GetPartitioners();
  auto it = partitioners.find(custom_call_target);
  if (it == partitioners.end()) {
    return nullptr;
  }
  return it->second.get();
}

void RegisterCustomCallPartitioner(
    absl::string_view custom_call_target,
    std::unique_ptr<CustomCallPartitioner> partitioner) {
  absl::MutexLock partitioners_lock(&partitioners_mutex);
  auto& partitioners = GetPartitioners();
  // Warn if something has already been registered. We prefer to keep the
  // existing object as other threads are more likely to observe it.
  auto [it, did_insert] =
      partitioners.try_emplace(custom_call_target, std::move(partitioner));
  if (!did_insert) {
    LOG(ERROR) << "Failed to register custom call partitioner for "
               << custom_call_target;
  }
}

}  // namespace xla
