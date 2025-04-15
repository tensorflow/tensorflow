/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_TSL_PLATFORM_UNBOUNDED_WORK_QUEUE_H_
#define TENSORFLOW_TSL_PLATFORM_UNBOUNDED_WORK_QUEUE_H_

#include "tsl/platform/platform.h"

// An `UnboundedWorkQueue` feeds potentially-blocking work into a thread-pool
// whose size automatically increases with demand.

#if defined(PLATFORM_GOOGLE)
#include "xla/tsl/platform/google/unbounded_work_queue.h"  // IWYU pragma: export
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) ||    \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_POSIX_IOS) || \
    defined(PLATFORM_GOOGLE_IOS) || defined(PLATFORM_WINDOWS)
#include "xla/tsl/platform/default/unbounded_work_queue.h"  // IWYU pragma: export
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

#endif  // TENSORFLOW_TSL_PLATFORM_UNBOUNDED_WORK_QUEUE_H_
