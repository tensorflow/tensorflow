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
#include "tsl/profiler/lib/profiler_lock.h"

#include <atomic>

#include "xla/tsl/util/env_var.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/macros.h"
#include "tsl/platform/statusor.h"

namespace tsl {
namespace profiler {
namespace {

// Track whether there's an active profiler session.
// Prevents another profiler session from creating ProfilerInterface(s).
std::atomic<int> g_session_active = ATOMIC_VAR_INIT(0);

// g_session_active implementation must be lock-free for faster execution of
// the ProfilerLock API.
static_assert(ATOMIC_INT_LOCK_FREE == 2, "Assumed atomic<int> was lock free");

}  // namespace

/*static*/ bool ProfilerLock::HasActiveSession() {
  return g_session_active.load(std::memory_order_relaxed) != 0;
}

/*static*/ StatusOr<ProfilerLock> ProfilerLock::Acquire() {
  // Use environment variable to permanently lock the profiler.
  // This allows running TensorFlow under an external profiling tool with all
  // built-in profiling disabled.
  static bool tf_profiler_disabled = [] {
    bool disabled = false;
    ReadBoolFromEnvVar("TF_DISABLE_PROFILING", false, &disabled).IgnoreError();
    return disabled;
  }();
  if (TF_PREDICT_FALSE(tf_profiler_disabled)) {
    return errors::AlreadyExists(
        "TensorFlow Profiler is permanently disabled by env var "
        "TF_DISABLE_PROFILING.");
  }
  int already_active = g_session_active.exchange(1, std::memory_order_acq_rel);
  if (already_active) {
    return errors::AlreadyExists(kProfilerLockContention);
  }
  return ProfilerLock(/*active=*/true);
}

void ProfilerLock::ReleaseIfActive() {
  if (active_) {
    g_session_active.store(0, std::memory_order_release);
    active_ = false;
  }
}

}  // namespace profiler
}  // namespace tsl
