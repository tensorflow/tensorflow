/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_UTIL_H_
#define TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_UTIL_H_

#include <functional>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/public/session_options.h"

// TODO(vrv, mrry): Remove this library: its interface circumvents the
// callers' Env and calls Env::Default() directly.

namespace tensorflow {

// Returns a process-wide ThreadPool for scheduling compute operations
// using 'options'.  Caller does not take ownership over threadpool.
thread::ThreadPool* ComputePool(const SessionOptions& options);

// Schedule "closure" in the default thread queue.
void SchedClosure(std::function<void()> closure);

// Schedule "closure" after the given number of microseconds in the
// fixed-size ThreadPool used for non-blocking compute tasks.
void SchedNonBlockingClosureAfter(int64 micros, std::function<void()> closure);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_COMMON_RUNTIME_PROCESS_UTIL_H_
