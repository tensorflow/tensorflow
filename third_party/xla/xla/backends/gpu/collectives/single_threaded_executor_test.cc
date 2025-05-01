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

#include "xla/backends/gpu/collectives/single_threaded_executor.h"

#include <cstdint>
#include <string>

#include <gtest/gtest.h>
#include "absl/synchronization/blocking_counter.h"
#include "absl/synchronization/notification.h"
#include "xla/tsl/platform/env.h"
#include "xla/tsl/platform/threadpool.h"

namespace {

TEST(SingleThreadedExecutor, ConcurrentCallsToRunGetExecutedSerially) {
  // Issue 100 concurrent calls to Run. The functions, though submitted
  // concurrently, should execute serially.
  tsl::Env* env = tsl::Env::Default();
  const std::string name = "ConcurrentCallsToRunGetExecutedSerially";
  xla::gpu::SingleThreadedExecutor executor(*env);

  const int num_threads = 100;
  int x = 0;
  {
    tsl::thread::ThreadPool pool(env, name, num_threads);
    absl::BlockingCounter done(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      pool.Schedule([&]() {
        executor.Execute([&]() {
          x++;
          done.DecrementCount();
        });
      });
    }
    done.Wait();
  }
  EXPECT_EQ(x, num_threads);
}

TEST(SingleThreadedExecutor, FunctionsRunOnOneThread) {
  tsl::Env* env = tsl::Env::Default();
  xla::gpu::SingleThreadedExecutor executor(*env);

  // Get the thread id of the worker thread.
  int64_t thread_id = 0;
  absl::Notification done;
  executor.Execute([&]() {
    thread_id = env->GetCurrentThreadId();
    done.Notify();
  });
  done.WaitForNotification();

  // Confirm that every function runs on the same thread.
  absl::BlockingCounter all_done(10);
  for (int i = 0; i < 10; ++i) {
    executor.Execute([&]() {
      EXPECT_EQ(thread_id, env->GetCurrentThreadId());
      all_done.DecrementCount();
    });
  }
  all_done.Wait();
}

}  // namespace
