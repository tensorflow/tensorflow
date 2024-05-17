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
#include "tensorflow/lite/profiling/root_profiler.h"

#include <memory>
#include <utility>
#include <vector>

namespace tflite {
namespace profiling {

void RootProfiler::AddProfiler(Profiler* profiler) {
  if (profiler == nullptr) return;
  profilers_.push_back(profiler);
}

void RootProfiler::AddProfiler(std::unique_ptr<Profiler>&& profiler) {
  if (profiler == nullptr) return;
  owned_profilers_.emplace_back(std::move(profiler));
  profilers_.push_back(owned_profilers_.back().get());
}

uint32_t RootProfiler::BeginEvent(const char* tag, EventType event_type,
                                  int64_t event_metadata1,
                                  int64_t event_metadata2) {
  // TODO(b/238913100): Remove the workaround.
  if (profilers_.size() == 1) {
    return profilers_[0]->BeginEvent(tag, event_type, event_metadata1,
                                     event_metadata2);
  }
  auto id = next_event_id_++;
  std::vector<uint32_t> event_ids;
  event_ids.reserve(profilers_.size());
  for (auto* profiler : profilers_) {
    event_ids.push_back(profiler->BeginEvent(tag, event_type, event_metadata1,
                                             event_metadata2));
  }
  events_.emplace(id, std::move(event_ids));
  return id;
}

void RootProfiler::EndEvent(uint32_t event_handle, int64_t event_metadata1,
                            int64_t event_metadata2) {
  // TODO(b/238913100): Remove the workaround.
  if (profilers_.size() == 1) {
    return profilers_[0]->EndEvent(event_handle, event_metadata1,
                                   event_metadata2);
  }
  if (const auto it = events_.find(event_handle); it != events_.end()) {
    const auto& event_ids = it->second;
    for (auto idx = 0; idx < event_ids.size(); idx++) {
      profilers_[idx]->EndEvent(event_ids[idx], event_metadata1,
                                event_metadata2);
    }
    events_.erase(it);
  }
}

void RootProfiler::EndEvent(uint32_t event_handle) {
  // TODO(b/238913100): Remove the workaround.
  if (profilers_.size() == 1) {
    return profilers_[0]->EndEvent(event_handle);
  }
  if (const auto it = events_.find(event_handle); it != events_.end()) {
    const auto& event_ids = it->second;
    for (auto idx = 0; idx < event_ids.size(); idx++) {
      profilers_[idx]->EndEvent(event_ids[idx]);
    }
    events_.erase(it);
  }
}

void RootProfiler::AddEvent(const char* tag, EventType event_type,
                            uint64_t metric, int64_t event_metadata1,
                            int64_t event_metadata2) {
  for (auto* profiler : profilers_) {
    profiler->AddEvent(tag, event_type, metric, event_metadata1,
                       event_metadata2);
  }
}

void RootProfiler::AddEventWithData(const char* tag, EventType event_type,
                                    const void* data) {
  for (auto* profiler : profilers_) {
    profiler->AddEventWithData(tag, event_type, data);
  }
}

void RootProfiler::RemoveChildProfilers() {
  owned_profilers_.clear();
  profilers_.clear();
  // Previous `BeginEvent` calls will be discarded.
  events_.clear();
}

}  // namespace profiling
}  // namespace tflite
