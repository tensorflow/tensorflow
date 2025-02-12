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

#ifndef XLA_TSL_PLATFORM_DEFAULT_TRACING_IMPL_H_
#define XLA_TSL_PLATFORM_DEFAULT_TRACING_IMPL_H_

#ifndef IS_MOBILE_PLATFORM
#include "xla/tsl/profiler/backends/cpu/threadpool_listener_state.h"
#endif
// Stub implementations of tracing functionality.

// Definitions that do nothing for platforms that don't have underlying thread
// tracing support.
#define TRACELITERAL(a) \
  do {                  \
  } while (0)
#define TRACESTRING(s) \
  do {                 \
  } while (0)
#define TRACEPRINTF(format, ...) \
  do {                           \
  } while (0)

namespace tsl {
namespace tracing {

inline bool EventCollector::IsEnabled() {
#ifndef IS_MOBILE_PLATFORM
  return tsl::profiler::threadpool_listener::IsEnabled();
#else
  return false;
#endif
}

}  // namespace tracing
}  // namespace tsl

#endif  // XLA_TSL_PLATFORM_DEFAULT_TRACING_IMPL_H_
