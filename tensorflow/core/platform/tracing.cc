/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/platform/tracing.h"

#include <array>
#include <atomic>
#include <map>
#include <string>
#include <vector>
#include "tensorflow/core/lib/hash/hash.h"
#include "tensorflow/core/lib/strings/str_util.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/logging.h"

namespace tensorflow {
namespace tracing {
namespace {
std::atomic<uint64> unique_arg{1};
std::atomic<const TraceCollector*> trace_collector;
}  // namespace

const char* GetEventCategoryName(EventCategory category) {
  switch (category) {
    case EventCategory::kScheduleClosure:
      return "ScheduleClosure";
    case EventCategory::kRunClosure:
      return "RunClosure";
    case EventCategory::kCompute:
      return "Compute";
    default:
      return "Unknown";
  }
}

std::array<const EventCollector*, GetNumEventCategories()>
    EventCollector::instances_;

void SetEventCollector(EventCategory category,
                       const EventCollector* collector) {
  EventCollector::instances_[static_cast<unsigned>(category)] = collector;
}

uint64 GetUniqueArg() {
  return unique_arg.fetch_add(1, std::memory_order_relaxed);
}

uint64 GetArgForName(StringPiece name) {
  return Hash64(name.data(), name.size());
}

string TraceCollector::ConcatenateNames(StringPiece first, StringPiece second) {
  std::string result;
  bool has_two_parts = !first.empty() && !second.empty();
  result.reserve(first.size() + second.size() +
                 static_cast<int>(has_two_parts));
  result.append(first.data(), first.size());
  if (has_two_parts) result.append({':'});
  result.append(second.data(), second.size());
  return result;
}

void SetTraceCollector(const TraceCollector* collector) {
  return trace_collector.store(collector, std::memory_order_release);
}

const TraceCollector* GetTraceCollector() {
  return trace_collector.load(std::memory_order_acquire);
}

}  // namespace tracing
}  // namespace tensorflow
