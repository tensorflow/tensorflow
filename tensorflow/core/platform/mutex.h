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

#ifndef TENSORFLOW_CORE_PLATFORM_MUTEX_H_
#define TENSORFLOW_CORE_PLATFORM_MUTEX_H_

#include "tensorflow/core/platform/platform.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
enum ConditionResult { kCond_Timeout, kCond_MaybeNotified };
}  // namespace tensorflow

// Include appropriate platform-dependent implementations of mutex etc.
#if defined(PLATFORM_GOOGLE)
#include "tensorflow/core/platform/google/mutex.h"
#elif defined(PLATFORM_POSIX) || defined(PLATFORM_POSIX_ANDROID) || \
    defined(PLATFORM_GOOGLE_ANDROID) || defined(PLATFORM_WINDOWS)
#include "tensorflow/core/platform/default/mutex.h"
#else
#error Define the appropriate PLATFORM_<foo> macro for this platform
#endif

// The mutex library included above defines:
//   class mutex;
//   class mutex_lock;
//   class condition_variable;
// It also defines the following:

namespace tensorflow {

// Like "cv->wait(*mu)", except that it only waits for up to "ms" milliseconds.
//
// Returns kCond_Timeout if the timeout expired without this
// thread noticing a signal on the condition variable.  Otherwise may
// return either kCond_Timeout or kCond_MaybeNotified
ConditionResult WaitForMilliseconds(mutex_lock* mu, condition_variable* cv,
                                    int64 ms);
}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_PLATFORM_MUTEX_H_
