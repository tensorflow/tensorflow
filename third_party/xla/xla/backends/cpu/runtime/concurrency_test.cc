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

#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/synchronization/blocking_counter.h"
#include "tsl/platform/env.h"
#include "tsl/platform/test.h"
#include "tsl/platform/threadpool.h"

namespace xla::cpu {
namespace {

TEST(ConcurrencyTest, ScheduleAll) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test", 10);

  std::vector<int64_t> tasks(64, 0);

  Eigen::ThreadPoolDevice device(thread_pool.AsEigenThreadPool(),
                                 thread_pool.NumThreads());

  absl::BlockingCounter counter(64);
  ScheduleAll(&device, 64, [&](int64_t index) {
    tasks[index] += 1;
    counter.DecrementCount();
  });

  counter.Wait();
  ASSERT_TRUE(absl::c_all_of(tasks, [](int64_t task) { return task == 1; }));
}

}  // namespace
}  // namespace xla::cpu
