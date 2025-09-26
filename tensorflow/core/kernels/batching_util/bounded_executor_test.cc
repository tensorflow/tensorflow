/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/bounded_executor.h"

#include "absl/functional/bind_front.h"
#include "absl/synchronization/notification.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/statusor.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/threadpool_interface.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace serving {

namespace {
// Tracks the number of concurrently running tasks.
class TaskTracker {
 public:
  // Creates a functor that invokes Run() with the given arguments.
  std::function<void()> MakeTask(int task_id, absl::Duration sleep_duration) {
    return absl::bind_front(&TaskTracker::Run, this, task_id, sleep_duration);
  }

  // Updates run counts, sleeps for a short time and then returns.
  // Exits early if fiber is cancelled.
  void Run(int task_id, absl::Duration sleep_duration) {
    LOG(INFO) << "Entering task " << task_id;
    // Update run counters.
    {
      mutex_lock l(mutex_);
      ++task_count_;
      ++running_count_;
      if (running_count_ > max_running_count_) {
        max_running_count_ = running_count_;
      }
    }

    // Use a sleep loop so we can quickly detect cancellation even when the
    // total sleep time is very large.

    Env::Default()->SleepForMicroseconds(
        absl::ToInt64Microseconds(sleep_duration));
    // Update run counters.
    {
      mutex_lock l(mutex_);
      --running_count_;
    }
    LOG(INFO) << "Task " << task_id << " exiting.";
  }

  // Returns number of tasks that have been run.
  int task_count() {
    mutex_lock l(mutex_);
    return task_count_;
  }

  // Returns number of tasks that are currently running.
  int running_count() {
    mutex_lock l(mutex_);
    return running_count_;
  }

  // Returns the max number of tasks that have run concurrently.
  int max_running_count() {
    mutex_lock l(mutex_);
    return max_running_count_;
  }

 private:
  mutex mutex_;
  int task_count_ = 0;
  int running_count_ = 0;
  int max_running_count_ = 0;
};

TEST(BoundedExecutorTest, InvalidEmptyEnv) {
  BoundedExecutor::Options options;
  options.num_threads = 2;
  options.env = nullptr;
  EXPECT_THAT(BoundedExecutor::Create(options),
              absl_testing::StatusIs(error::INVALID_ARGUMENT,
                                     "options.env must not be nullptr"));
}

TEST(BoundedExecutorTest, InvalidNumThreads) {
  {
    BoundedExecutor::Options options;
    options.num_threads = 0;
    EXPECT_THAT(BoundedExecutor::Create(options),
                absl_testing::StatusIs(error::INVALID_ARGUMENT,
                                       "options.num_threads must be positive"));
  }

  {
    BoundedExecutor::Options options;
    options.num_threads = -1;
    EXPECT_THAT(BoundedExecutor::Create(options),
                absl_testing::StatusIs(error::INVALID_ARGUMENT,
                                       "options.num_threads must be positive"));
  }
}

TEST(BoundedExecutorTest, AddRunsFunctionsEventually) {
  BoundedExecutor::Options options;
  options.num_threads = 2;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, BoundedExecutor::Create(options));

  absl::Notification done0;
  executor->Schedule([&done0] { done0.Notify(); });
  absl::Notification done1;
  executor->Schedule([&done1] { done1.Notify(); });
  done0.WaitForNotification();
  done1.WaitForNotification();

  executor.reset();
}

TEST(BoundedExecutorTest, MaxInflightLimit) {
  BoundedExecutor::Options options;
  options.num_threads = 5;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, BoundedExecutor::Create(options));

  const int num_tasks = 100;
  TaskTracker task_tracker;
  for (int i = 0; i < num_tasks; i++) {
    executor->Schedule(task_tracker.MakeTask(i, absl::Seconds(1)));
  }
  executor.reset();

  EXPECT_EQ(task_tracker.task_count(), num_tasks);
  EXPECT_EQ(task_tracker.max_running_count(), options.num_threads);
  EXPECT_EQ(task_tracker.running_count(), 0);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
