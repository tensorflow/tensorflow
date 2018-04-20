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

#ifndef TENSORFLOW_PLATFORM_DEFAULT_TRACING_IMPL_H_
#define TENSORFLOW_PLATFORM_DEFAULT_TRACING_IMPL_H_

// Stub implementations of tracing functionality.

// IWYU pragma: private, include "third_party/tensorflow/core/platform/tracing.h"
// IWYU pragma: friend third_party/tensorflow/core/platform/tracing.h

#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/tracing.h"

namespace tensorflow {
namespace port {

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

inline uint64 Tracing::UniqueId() { return random::New64(); }
inline bool Tracing::IsActive() { return false; }
inline void Tracing::RegisterCurrentThread(const char* name) {}

// Posts an atomic threadscape event with the supplied category and arg.
inline void Tracing::RecordEvent(EventCategory category, uint64 arg) {
  // TODO(opensource): Implement
}

inline Tracing::ScopedActivity::ScopedActivity(EventCategory category,
                                               uint64 arg) {}

inline Tracing::ScopedActivity::~ScopedActivity() {}

}  // namespace port
}  // namespace tensorflow

#endif  // TENSORFLOW_PLATFORM_DEFAULT_TRACING_IMPL_H_
