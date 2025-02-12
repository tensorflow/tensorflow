/* Copyright 2023 The OpenXLA Authors.

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

#include "xla/service/gpu/model/fusion_analysis_cache.h"

#include <utility>

#include "absl/synchronization/mutex.h"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/service/gpu/hlo_fusion_analysis.h"

namespace xla::gpu {

const HloFusionAnalysis& HloFusionAnalysisCache::Get(
    const HloInstruction& instruction) {
  {
    absl::MutexLock lock(&mutex_);
    auto it = analyses_.find(instruction.unique_id());
    if (it != analyses_.end()) {
      return it->second;
    }
  }

  HloFusionAnalysis analysis =
      HloFusionAnalysis::Create(instruction, device_info_);
  absl::MutexLock lock(&mutex_);

  // If some other thread created an entry for this key concurrently, return
  // that instead (the other thread is likely using the instance).
  auto it = analyses_.find(instruction.unique_id());
  if (it != analyses_.end()) {
    return it->second;
  }

  return analyses_.emplace(instruction.unique_id(), std::move(analysis))
      .first->second;
}

const HloFusionAnalysis& HloFusionAnalysisCache::Get(
    const HloInstruction& producer, const HloInstruction& consumer) {
  std::pair<int, int> key{producer.unique_id(), consumer.unique_id()};
  {
    absl::MutexLock lock(&mutex_);
    auto it = producer_consumer_analyses_.find(key);
    if (it != producer_consumer_analyses_.end()) {
      return it->second;
    }
  }

  HloFusionAnalysis analysis =
      HloFusionAnalysis::Create(producer, consumer, device_info_);
  absl::MutexLock lock(&mutex_);

  // If some other thread created an entry for this key concurrently, return
  // that instead (the other thread is likely using the instance).
  auto it = producer_consumer_analyses_.find(key);
  if (it != producer_consumer_analyses_.end()) {
    return it->second;
  }

  producers_for_consumers_[consumer.unique_id()].push_back(
      producer.unique_id());
  consumers_for_producers_[producer.unique_id()].push_back(
      consumer.unique_id());
  return producer_consumer_analyses_.emplace(key, std::move(analysis))
      .first->second;
}

void HloFusionAnalysisCache::Invalidate(const HloInstruction& instruction) {
  analyses_.erase(instruction.unique_id());

  if (auto consumers =
          consumers_for_producers_.extract(instruction.unique_id())) {
    for (const auto consumer : consumers.mapped()) {
      producer_consumer_analyses_.erase({instruction.unique_id(), consumer});
    }
  }
  if (auto producers =
          producers_for_consumers_.extract(instruction.unique_id())) {
    for (const auto producer : producers.mapped()) {
      producer_consumer_analyses_.erase({producer, instruction.unique_id()});
    }
  }
}

void HloFusionAnalysisCache::Clear() {
  analyses_.clear();
  producer_consumer_analyses_.clear();
  consumers_for_producers_.clear();
  producers_for_consumers_.clear();
}

}  // namespace xla::gpu
