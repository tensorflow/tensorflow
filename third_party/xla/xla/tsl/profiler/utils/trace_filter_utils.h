/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_TSL_PROFILER_UTILS_TRACE_FILTER_UTILS_H_
#define XLA_TSL_PROFILER_UTILS_TRACE_FILTER_UTILS_H_

#include <cstdint>

#include "absl/container/flat_hash_set.h"
#include "absl/log/check.h"

namespace tsl {
namespace profiler {

// TraceMeFilter is a bitmap that will be used to filter out TraceMe events
// during recording. Filter will be applied only if record function (e.g.
// TraceMe, ActivityStart, InstantActivity etc.) with filter_mask is called.
// If filter_mask is not passed in the profile request, filter will not be
// applied. Lowest 32 bit are reserved for 3P so this enum should only have
// values in the range [0, 31].
//
// TraceMe events that do not pass a filter_mask carry only the kTraceDefault
// bit. A session with a filter mask therefore records untagged events only if
// its mask includes kTraceDefault, so each new filter is exclusive by default.
// Sessions without a filter mask record all events.
enum class TraceMeFilter {
  kTraceMemory = 0,
  kTraceHostEnqueueCallback = 1,
  // Carried by every TraceMe event that does not pass a filter_mask. Keep in
  // sync with kTraceMeDefaultFilter in traceme.h.
  kTraceDefault = 2,
};

static uint64_t TraceMeFiltersToMask(
    absl::flat_hash_set<tsl::profiler::TraceMeFilter> filter) {
  uint64_t mask = 0;
  for (const tsl::profiler::TraceMeFilter filter : filter) {
    DCHECK_LT(static_cast<uint64_t>(filter), 32);
    mask |= (1ull << static_cast<uint64_t>(filter));
  }
  return mask;
}

}  // namespace profiler
}  // namespace tsl

#endif  // XLA_TSL_PROFILER_UTILS_TRACE_FILTER_UTILS_H_
