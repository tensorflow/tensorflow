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
#include "tensorflow/core/profiler/lib/profiler_lock.h"

#include <atomic>

#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/util/env_var.h"

namespace tensorflow {
namespace profiler {

// Track whether there's an active profiler session.
// Prevents another profiler session from creating ProfilerInterface(s).
std::atomic<bool> session_active = ATOMIC_VAR_INIT(false);

bool AcquireProfilerLock() {
  // Use environment variable to permanently lock the profiler.
  // This allows running TensorFlow under an external profiling tool with all
  // built-in profiling disabled.
  static bool tf_profiler_disabled = [] {
    bool disabled = false;
    ReadBoolFromEnvVar("TF_DISABLE_PROFILING", false, &disabled).IgnoreError();
    return disabled;
  }();
  if (TF_PREDICT_FALSE(tf_profiler_disabled)) {
    LOG(WARNING) << "TensorFlow Profiler is permanently disabled by env var "
                    "TF_DISABLE_PROFILING.";
    return false;
  }
  return !session_active.exchange(true);
}

void ReleaseProfilerLock() { session_active.store(false); }

}  // namespace profiler
}  // namespace tensorflow
