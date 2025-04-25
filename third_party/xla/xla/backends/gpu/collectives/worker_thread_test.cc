/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/backends/gpu/collectives/worker_thread.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace {

TEST(WorkerThread, ConcurrentCallsToRunGetExecutedSerially) {
  // Issue 100 concurrent calls to Run. The functions, though submitted
  // concurrently, should execute serially.
  tsl::Env* env = tsl::Env::Default();
  const std::string name = "ConcurrentCallsToRunGetExecutedSerially";
  xla::gpu::WorkerThread t(*env, name);

  const int num_threads = 100;
  int x = 0;
  {
    tsl::thread::ThreadPool pool(env, name, num_threads);
    for (int i = 0; i < num_threads; ++i) {
      pool.Schedule([&t, &x]() {
        TF_ASSERT_OK(t.Run([&x]() -> absl::Status {
          x++;
          return absl::OkStatus();
        }));
      });
    }
  }
  EXPECT_EQ(x, num_threads);
}

TEST(WorkerThread, FunctionsRunOnOneThread) {
  tsl::Env* env = tsl::Env::Default();
  xla::gpu::WorkerThread t(*env, "FunctionsRunOnOneThread");

  // Get the thread id of the worker thread.
  int64_t thread_id = 0;
  TF_ASSERT_OK(t.Run([&env, &thread_id]() -> absl::Status {
    thread_id = env->GetCurrentThreadId();
    return absl::OkStatus();
  }));

  // Confirm that every function runs on the same thread.
  for (int i = 0; i < 10; ++i) {
    TF_ASSERT_OK(t.Run([&env, &thread_id]() -> absl::Status {
      EXPECT_EQ(thread_id, env->GetCurrentThreadId());
      return absl::OkStatus();
    }));
  }
}

}  // namespace
