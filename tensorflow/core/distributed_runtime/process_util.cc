/* Copyright 2016 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/distributed_runtime/process_util.h"

#include <string.h>

#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/host_info.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/tracing.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

namespace {

static thread::ThreadPool* InitComputePool(const SessionOptions& options) {
  int32 inter_op_parallelism_threads =
      options.config.inter_op_parallelism_threads();
  if (inter_op_parallelism_threads == 0) {
    // Default to using the number of cores available in the process.
    inter_op_parallelism_threads = port::NumSchedulableCPUs();
  }

  return new thread::ThreadPool(Env::Default(), "Compute",
                                inter_op_parallelism_threads);
}

}  // namespace

thread::ThreadPool* ComputePool(const SessionOptions& options) {
  static thread::ThreadPool* compute_pool = InitComputePool(options);
  return compute_pool;
}

void SchedClosure(std::function<void()> closure) {
  if (port::Tracing::IsActive()) {
    const uint64 id = port::Tracing::UniqueId();
    port::Tracing::RecordEvent(port::Tracing::EventCategory::kScheduleClosure,
                               id);
    std::function<void()> wrapper = [closure, id]() {
      port::Tracing::ScopedActivity region(
          port::Tracing::EventCategory::kRunClosure, id);
      closure();
    };
    Env::Default()->SchedClosure(wrapper);
  } else {
    Env::Default()->SchedClosure(closure);
  }
}

void SchedNonBlockingClosureAfter(int micros, std::function<void()> closure) {
  Env::Default()->SchedClosureAfter(micros, closure);
}

}  // namespace tensorflow
