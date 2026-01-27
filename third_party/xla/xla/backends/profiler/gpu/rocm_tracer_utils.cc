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
#include <string>

#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"

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

void AnnotationMap::Add(uint32_t correlation_id,
                        const std::string& annotation) {
  if (annotation.empty()) {
    return;
  }
  VLOG(3) << "Add annotation: " << " correlation_id=" << correlation_id
          << ", annotation: " << annotation;
  absl::MutexLock lock(map_.mutex);
  if (map_.annotations.size() < max_size_) {
    absl::string_view annotation_str =
        *map_.annotations.insert(annotation).first;
    map_.correlation_map.emplace(correlation_id, annotation_str);
  }
}

absl::string_view AnnotationMap::LookUp(uint32_t correlation_id) {
  absl::MutexLock lock(map_.mutex);
  auto it = map_.correlation_map.find(correlation_id);
  return it != map_.correlation_map.end() ? it->second : absl::string_view();
}

}  // namespace profiler
}  // namespace xla
