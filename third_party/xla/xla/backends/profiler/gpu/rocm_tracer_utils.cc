/* Copyright 2025 The OpenXLA Authors. All Rights Reserved.

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

#include "xla/backends/profiler/gpu/rocm_tracer_utils.h"

#include <cstdint>
#include <functional>
#include <string>
#include <utility>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "absl/types/span.h"

// for rocprofiler-sdk
namespace xla {
namespace profiler {

//-----------------------------------------------------------------------------
const char* GetRocmTracerEventSourceName(const RocmTracerEventSource& source) {
  switch (source) {
    case RocmTracerEventSource::ApiCallback:
      return "ApiCallback";
      break;
    case RocmTracerEventSource::Activity:
      return "Activity";
      break;
    case RocmTracerEventSource::Invalid:
      return "Invalid";
      break;
    default:
      DCHECK(false);
      return "";
  }
  return "";
}

// FIXME(rocm-profiler): These domain names are not consistent with the
// GetActivityDomainName function
const char* GetRocmTracerEventDomainName(const RocmTracerEventDomain& domain) {
  switch (domain) {
    case RocmTracerEventDomain::HIP_API:
      return "HIP_API";
      break;
    case RocmTracerEventDomain::HIP_OPS:
      return "HIP_OPS";
      break;
    default:
      LOG(WARNING) << "RocmTracerEventDomain::InvalidDomain";
      DCHECK(false);
      return "";
  }
  return "";
}

const char* GetRocmTracerEventTypeName(const RocmTracerEventType& type) {
#define OO(x)                  \
  case RocmTracerEventType::x: \
    return #x;
  switch (type) {
    OO(Kernel)
    OO(MemcpyH2D)
    OO(MemcpyD2H)
    OO(MemcpyD2D)
    OO(MemcpyOther)
    OO(MemoryAlloc)
    OO(MemoryFree)
    OO(Memset)
    OO(Synchronization)
    OO(Generic)
    default: {
    };
  }
#undef OO
  DCHECK(false);
  return "";
}

void AnnotationMap::Add(uint64_t correlation_id, const std::string& annotation,
                        absl::string_view roctx_range,
                        absl::Span<const int64_t> scope_range_ids) {
  // Skip if both fields are empty — nothing to store.
  if (annotation.empty() && roctx_range.empty()) {
    return;
  }
  VLOG(3) << "Add annotation: "
          << " correlation_id=" << correlation_id
          << ", annotation: " << annotation << ", roctx_range: " << roctx_range;
  absl::MutexLock lock(map_.mutex);
  // Decide once per event, so its annotation, ROCTX range and scope range id
  // are stored together or not at all. A slot is reserved for each non-empty
  // string, even one already interned, so the string set never exceeds the cap
  // set by Reset(), and once it is full no sub-map grows. Otherwise
  // scope_range_id_map would keep taking one entry per correlation id for the
  // rest of the session.
  const uint64_t needed =
      (annotation.empty() ? 0 : 1) + (roctx_range.empty() ? 0 : 1);
  if (map_.annotations.size() + needed > max_annotation_strings_) {
    return;
  }
  // Only insert into correlation_map when annotation is non-empty; it may
  // be empty when only a ROCTX range (no XLA AnnotationStack text) is active.
  if (!annotation.empty()) {
    const std::string& interned = *map_.annotations.insert(annotation).first;
    map_.correlation_map.emplace(correlation_id, std::cref(interned));
  }
  if (!roctx_range.empty()) {
    const std::string& interned =
        *map_.annotations.insert(std::string(roctx_range)).first;
    map_.roctx_range_map.emplace(correlation_id, std::cref(interned));
  }
  if (!scope_range_ids.empty()) {
    map_.scope_range_id_map.emplace(correlation_id, scope_range_ids.back());
    if (scope_range_ids.size() > 1) {
      const int64_t* head = scope_range_ids.data();
      const int64_t* curr = &scope_range_ids.back();
      for (; curr > head && !map_.scope_range_id_tree.contains(*curr); --curr) {
        map_.scope_range_id_tree.emplace(*curr, *(curr - 1));
      }
    }
  }
}

absl::string_view AnnotationMap::LookUp(uint64_t correlation_id) {
  absl::MutexLock lock(map_.mutex);
  auto it = map_.correlation_map.find(correlation_id);
  return it != map_.correlation_map.end() ? it->second.get()
                                          : absl::string_view();
}

absl::string_view AnnotationMap::LookUpRoctxRange(uint64_t correlation_id) {
  absl::MutexLock lock(map_.mutex);
  auto it = map_.roctx_range_map.find(correlation_id);
  return it != map_.roctx_range_map.end() ? it->second.get()
                                          : absl::string_view();
}

int64_t AnnotationMap::LookUpScopeRangeId(uint64_t correlation_id) {
  absl::MutexLock lock(map_.mutex);
  auto it = map_.scope_range_id_map.find(correlation_id);
  return it != map_.scope_range_id_map.end() ? it->second : 0;
}

ScopeRangeIdTree AnnotationMap::TakeScopeRangeIdTree() {
  absl::MutexLock lock(map_.mutex);
  return std::move(map_.scope_range_id_tree);
}

void AnnotationMap::Reset(uint64_t max_annotation_strings) {
  absl::MutexLock lock(map_.mutex);
  map_.correlation_map.clear();
  map_.roctx_range_map.clear();
  map_.scope_range_id_map.clear();
  map_.scope_range_id_tree.clear();
  map_.annotations.clear();
  max_annotation_strings_ = max_annotation_strings;
}

}  // namespace profiler
}  // namespace xla
