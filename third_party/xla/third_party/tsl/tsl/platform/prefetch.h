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

#ifndef TENSORFLOW_TSL_PLATFORM_PREFETCH_H_
#define TENSORFLOW_TSL_PLATFORM_PREFETCH_H_

#include "absl/base/prefetch.h"

namespace tsl {
namespace port {

// Prefetching support.
// Deprecated. Prefer to call absl::Prefetch* directly.

enum PrefetchHint {
  PREFETCH_HINT_T0 = 3,  // Temporal locality
  PREFETCH_HINT_NTA = 0  // No temporal locality
};

template <PrefetchHint hint>
void prefetch(const void* x) {
  absl::PrefetchToLocalCache(x);
}

template <>
inline void prefetch<PREFETCH_HINT_NTA>(const void* x) {
  absl::PrefetchToLocalCacheNta(x);
}

}  // namespace port
}  // namespace tsl

#endif  // TENSORFLOW_TSL_PLATFORM_PREFETCH_H_
