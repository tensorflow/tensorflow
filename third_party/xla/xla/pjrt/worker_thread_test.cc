/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/pjrt/worker_thread.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "xla/tsl/concurrency/async_value_ref.h"

namespace xla {
namespace {

TEST(WorkerThreadTest, ScheduleWithOrderedContinuations) {
  tsl::Env* env = tsl::Env::Default();
  std::vector<int> execution_order;
  absl::Mutex mu;

  auto av1 = tsl::MakeUnconstructedAsyncValueRef<int>();
  auto av2 = tsl::MakeUnconstructedAsyncValueRef<int>();
  auto av3 = tsl::MakeUnconstructedAsyncValueRef<int>();

  absl::Notification done;
  absl::Notification done2;
  {
    WorkerThread wt(env, "thread_name");

    wt.ScheduleWithOrderedContinuations([&]() {
      {
        absl::MutexLock lock(&mu);
        execution_order.push_back(1);
      }
      return AsyncValueContinuation{{}, [&]() {
                                      absl::MutexLock lock(&mu);
                                      execution_order.push_back(10);
                                    }};
    });

    wt.ScheduleWithOrderedContinuations([&]() {
      {
        absl::MutexLock lock(&mu);
        execution_order.push_back(2);
      }
      return AsyncValueContinuation{{av1.CopyRCRef()}, [&]() {
                                      done2.Notify();
                                      absl::MutexLock lock(&mu);
                                      execution_order.push_back(20);
                                    }};
    });

    wt.ScheduleWithOrderedContinuations([&]() {
      {
        absl::MutexLock lock(&mu);
        execution_order.push_back(3);
      }
      return AsyncValueContinuation{{av2.CopyRCRef(), av3.CopyRCRef()}, [&]() {
                                      absl::MutexLock lock(&mu);
                                      execution_order.push_back(30);
                                    }};
    });

    wt.ScheduleWithOrderedContinuations([&]() {
      done.Notify();
      {
        absl::MutexLock lock(&mu);
        execution_order.push_back(4);
      }
      return AsyncValueContinuation{{}, [&]() {
                                      absl::MutexLock lock(&mu);
                                      execution_order.push_back(40);
                                    }};
    });

    done.WaitForNotification();
    av1.emplace<int>(100);
    done2.WaitForNotification();
    av2.emplace<int>(200);
    av3.emplace<int>(200);
  }
  EXPECT_THAT(execution_order,
              testing::ElementsAre(1, 10, 2, 3, 4, 20, 30, 40));
}

}  // namespace
}  // namespace xla
