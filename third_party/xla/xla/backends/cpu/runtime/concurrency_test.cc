/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/backends/cpu/runtime/concurrency.h"

#include <atomic>
#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/synchronization/blocking_counter.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla::cpu {
namespace {

using ConcurrencyTest = ::testing::TestWithParam<int>;

TEST_P(ConcurrencyTest, ScheduleAll) {
  int number_of_tasks = GetParam();

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 10);

  std::vector<int64_t> tasks(number_of_tasks, 0);

  Eigen::ThreadPoolDevice device(thread_pool.AsEigenThreadPool(),
                                 thread_pool.NumThreads());

  absl::BlockingCounter counter(number_of_tasks);

  auto main_thread_id = tsl::Env::Default()->GetCurrentThreadId();
  std::atomic<int> tasks_called_for_host_thread = 0;

  ScheduleAll(&device, number_of_tasks, [&](int64_t index) {
    if (tsl::Env::Default()->GetCurrentThreadId() == main_thread_id) {
      ++tasks_called_for_host_thread;
    }

    tasks[index] += 1;
    counter.DecrementCount();
  });

  counter.Wait();

  // All tasks should be called exactly once.
  ASSERT_TRUE(absl::c_all_of(tasks, [](int64_t task) { return task == 1; }));

  // No task should be called on the host thread.
  EXPECT_EQ(0, tasks_called_for_host_thread);
}

INSTANTIATE_TEST_SUITE_P(ConcurrencyTest, ConcurrencyTest,
                         ::testing::Values(1, 64));

}  // namespace
}  // namespace xla::cpu
