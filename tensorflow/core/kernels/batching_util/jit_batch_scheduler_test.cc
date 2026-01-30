/* Copyright 2024 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/jit_batch_scheduler.h"

#include <cstddef>
#include <functional>
#include <memory>
#include <vector>

#include "absl/status/status.h"
#include "absl/synchronization/notification.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/criticality.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace serving {
namespace {

using tsl::criticality::Criticality;

class FakeTask : public BatchTask {
 public:
  FakeTask(size_t size, Criticality criticality, int id,
           std::function<void(const absl::Status&)> finish_callback = nullptr)
      : size_(size),
        criticality_(criticality),
        id_(id),
        finish_callback_(finish_callback) {}

  ~FakeTask() override = default;

  size_t size() const override { return size_; }
  Criticality criticality() const override { return criticality_; }
  int id() const { return id_; }

  void FinishTask(const absl::Status& status) override {
    if (finish_callback_) {
      finish_callback_(status);
    }
    finished_.Notify();
  }

  void WaitForFinished() { finished_.WaitForNotification(); }

 private:
  const size_t size_;
  const Criticality criticality_;
  const int id_;
  std::function<void(const absl::Status&)> finish_callback_;
  absl::Notification finished_;
};

TEST(JitBatchSchedulerTest, JitOrderingWithPriority) {
  // We want to verify that tasks are assembled "Just-In-Time".
  // Configuration:
  // - 1 execution thread.
  // - Max 1 ready batch.
  // - Max batch size 1 (to simplify counting).

  JitBatchScheduler<FakeTask>::Options options;
  options.max_batch_size = 1;
  options.max_queue_size = 100;
  options.num_batch_threads = 1;
  options.max_ready_batches = 1;

  absl::Notification first_batch_started;
  absl::Notification release_first_batch;

  std::vector<int> processed_ids;
  mutex mu;

  auto callback = [&](std::unique_ptr<Batch<FakeTask>> batch) {
    ASSERT_EQ(1, batch->num_tasks());
    int id = batch->task(0).id();

    if (id == 0) {  // First task blocks
      first_batch_started.Notify();
      release_first_batch.WaitForNotification();
    }

    mutex_lock l(mu);
    processed_ids.push_back(id);
  };

  std::unique_ptr<BatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      JitBatchScheduler<FakeTask>::Create(options, callback, &scheduler));

  // 1. Submit Task 0 (LP). It will be picked up immediately and block
  // execution.
  auto t0 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 0);
  TF_ASSERT_OK(scheduler->Schedule(&t0));

  first_batch_started.WaitForNotification();
  // Now:
  // - Batch(Task 0) is in-flight (blocked).
  // - ready_batches is empty (it was popped).
  // - Assembly thread wakes up because ready_batches is empty.

  // 2. Submit Task 1 (LP). It should be assembled into ready_batches.
  auto t1 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 1);
  TF_ASSERT_OK(scheduler->Schedule(&t1));

  // Give assembly thread a moment to run.
  Env::Default()->SleepForMicroseconds(10000);

  // Now:
  // - Batch(Task 0) is in-flight.
  // - Batch(Task 1) is in ready_batches.
  // - Priority Queue is empty.

  // 3. Submit Task 2 (LP). It should stay in Priority Queue because
  // ready_batches is full (size 1).
  auto t2 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 2);
  TF_ASSERT_OK(scheduler->Schedule(&t2));

  // 4. Submit Task 3 (HP). It should stay in Priority Queue, but sorted ABOVE
  // Task 2.
  auto t3 = std::make_unique<FakeTask>(1, Criticality::kCritical, 3);
  TF_ASSERT_OK(scheduler->Schedule(&t3));

  // 5. Unblock processing.
  release_first_batch.Notify();

  // Wait for all 4 tasks.
  while (true) {
    mutex_lock l(mu);
    if (processed_ids.size() == 4) break;
  }

  // Expected order:
  // 0: Started first.
  // 1: Was already assembled in ready queue.
  // 3: High Priority (assembled next because it jumped queue).
  // 2: Low Priority (assembled last).

  std::vector<int> expected = {0, 1, 3, 2};
  EXPECT_EQ(processed_ids, expected);
}

TEST(JitBatchSchedulerTest, Preemption) {
  JitBatchScheduler<FakeTask>::Options options;
  options.max_batch_size = 1;
  options.max_queue_size = 2;  // Very small queue
  options.num_batch_threads = 1;
  options.max_ready_batches = 100;  // Large ready queue to avoid blocking on
                                    // that, effectively testing queue capacity

  // We want to fill the queue.
  // We need to block assembly so tasks stay in queue.
  // Actually, JitBatchScheduler doesn't have a way to block assembly easily
  // without blocking ready queue. But if we set max_ready_batches to 0, it
  // won't assemble? Code says `ready_batches_.size() >=
  // options_.max_ready_batches`. If max_ready_batches = 0, it stops. But min
  // value is usually 1.

  // Alternative: Set max_ready_batches=0 if the code allows (int).
  // The code check: `while (!stop_ && (task_queue_.empty() ||
  // ready_batches_.size() >= options_.max_ready_batches))` If max_ready_batches
  // is 0, it waits if ready_batches.size() >= 0 (always true). So assembly will
  // never pick up tasks.

  options.max_ready_batches = 0;

  // We need a dummy callback
  auto callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  std::unique_ptr<BatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      JitBatchScheduler<FakeTask>::Create(options, callback, &scheduler));

  // 1. Fill queue with 2 LP tasks.
  auto t1 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 1);
  TF_ASSERT_OK(scheduler->Schedule(&t1));

  auto t2 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 2);
  TF_ASSERT_OK(scheduler->Schedule(&t2));

  EXPECT_EQ(scheduler->NumEnqueuedTasks(), 2);

  // 2. Add HP task. Should succeed and preempt one LP task.
  auto t3 = std::make_unique<FakeTask>(1, Criticality::kCritical, 3);
  TF_ASSERT_OK(scheduler->Schedule(&t3));

  EXPECT_EQ(scheduler->NumEnqueuedTasks(), 2);  // Size stays max

  // 3. Add another HP task. Should succeed.
  auto t4 = std::make_unique<FakeTask>(1, Criticality::kCritical, 4);
  TF_ASSERT_OK(scheduler->Schedule(&t4));

  // 4. Add LP task. Should fail (Unavailable) because it's not higher priority
  // than existing HP tasks.
  auto t5 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 5);
  auto status = scheduler->Schedule(&t5);
  EXPECT_FALSE(status.ok());

  EXPECT_TRUE(absl::IsUnavailable(status));
}

TEST(JitBatchSchedulerTest, PreemptionNotification) {
  JitBatchScheduler<FakeTask>::Options options;
  options.max_batch_size = 1;
  options.max_queue_size = 1;  // Very small queue
  options.num_batch_threads = 1;
  options.max_ready_batches = 0;  // Block assembly

  auto callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};
  std::unique_ptr<BatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      JitBatchScheduler<FakeTask>::Create(options, callback, &scheduler));

  absl::Status t1_status;
  absl::Notification t1_finished;
  auto t1_finish_cb = [&](const absl::Status& status) {
    t1_status = status;
    t1_finished.Notify();
  };

  // 1. Fill queue with 1 LP task.
  auto t1 =
      std::make_unique<FakeTask>(1, Criticality::kSheddable, 1, t1_finish_cb);
  TF_ASSERT_OK(scheduler->Schedule(&t1));
  EXPECT_EQ(scheduler->NumEnqueuedTasks(), 1);

  // 2. Add HP task. Should succeed and preempt the LP task.
  auto t2 = std::make_unique<FakeTask>(1, Criticality::kCritical, 2);
  TF_ASSERT_OK(scheduler->Schedule(&t2));
  EXPECT_EQ(scheduler->NumEnqueuedTasks(), 1);

  // Check that the preempted task t1 was notified with Unavailable.
  t1_finished.WaitForNotification();
  EXPECT_TRUE(absl::IsUnavailable(t1_status));
}

TEST(JitBatchSchedulerTest, MaxBatchSizeGreaterThanOne) {
  JitBatchScheduler<FakeTask>::Options options;
  options.max_batch_size = 2;
  options.max_queue_size = 10;
  options.num_batch_threads = 1;
  options.max_ready_batches = 1;

  std::vector<int> batch_sizes;
  mutex mu;
  auto callback = [&](std::unique_ptr<Batch<FakeTask>> batch) {
    mutex_lock l(mu);
    batch_sizes.push_back(batch->num_tasks());
    for (int i = 0; i < batch->num_tasks(); ++i) {
      batch->mutable_task(i)->FinishTask(absl::OkStatus());
    }
  };

  std::unique_ptr<BatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      JitBatchScheduler<FakeTask>::Create(options, callback, &scheduler));

  auto t1 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 1);
  auto t2 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 2);
  auto t3 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 3);
  TF_ASSERT_OK(scheduler->Schedule(&t1));
  TF_ASSERT_OK(scheduler->Schedule(&t2));
  TF_ASSERT_OK(scheduler->Schedule(&t3));

  // Wait for all tasks to be processed
  Env::Default()->SleepForMicroseconds(20000);  // Give time for processing

  mutex_lock l(mu);
  ASSERT_EQ(batch_sizes.size(), 2);
  EXPECT_EQ(batch_sizes[0], 2);  // t1, t2
  EXPECT_EQ(batch_sizes[1], 1);  // t3
}

TEST(JitBatchSchedulerTest, NumBatchThreadsGreaterThanOne) {
  JitBatchScheduler<FakeTask>::Options options;
  options.max_batch_size = 1;
  options.max_queue_size = 10;
  options.num_batch_threads = 2;
  options.max_ready_batches = 2;

  absl::Notification t1_started, t2_started;
  absl::Notification t1_release, t2_release;
  int processed_count = 0;
  mutex mu;

  auto callback = [&](std::unique_ptr<Batch<FakeTask>> batch) {
    int id = batch->task(0).id();
    if (id == 1) {
      t1_started.Notify();
      t1_release.WaitForNotification();
    } else if (id == 2) {
      t2_started.Notify();
      t2_release.WaitForNotification();
    }
    mutex_lock l(mu);
    processed_count++;
    batch->mutable_task(0)->FinishTask(absl::OkStatus());
  };

  std::unique_ptr<BatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      JitBatchScheduler<FakeTask>::Create(options, callback, &scheduler));

  auto t1 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 1);
  auto t2 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 2);
  TF_ASSERT_OK(scheduler->Schedule(&t1));
  TF_ASSERT_OK(scheduler->Schedule(&t2));

  t1_started.WaitForNotification();
  t2_started.WaitForNotification();
  // Both tasks are now in flight on different threads.

  t1_release.Notify();
  t2_release.Notify();

  Env::Default()->SleepForMicroseconds(10000);  // Allow completion
  mutex_lock l(mu);
  EXPECT_EQ(processed_count, 2);
}

TEST(JitBatchSchedulerTest, MaxReadyBatchesGreaterThanOne) {
  JitBatchScheduler<FakeTask>::Options options;
  options.max_batch_size = 1;
  options.max_queue_size = 10;
  options.num_batch_threads = 1;
  options.max_ready_batches = 2;

  absl::Notification block_execution;
  std::vector<int> processed_ids;
  mutex mu;

  auto callback = [&](std::unique_ptr<Batch<FakeTask>> batch) {
    block_execution.WaitForNotification();
    mutex_lock l(mu);
    processed_ids.push_back(batch->task(0).id());
    batch->mutable_task(0)->FinishTask(absl::OkStatus());
  };

  std::unique_ptr<BatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      JitBatchScheduler<FakeTask>::Create(options, callback, &scheduler));

  // Schedule 3 tasks. With num_batch_threads=1 and max_ready_batches=2:
  // - 1st task will be in flight (but blocked by notification).
  // - 2nd task will be in ready_batches.
  // - 3rd task will be in task_queue.
  auto t1 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 1);
  auto t2 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 2);
  auto t3 = std::make_unique<FakeTask>(1, Criticality::kSheddable, 3);
  TF_ASSERT_OK(scheduler->Schedule(&t1));
  TF_ASSERT_OK(scheduler->Schedule(&t2));
  TF_ASSERT_OK(scheduler->Schedule(&t3));

  Env::Default()->SleepForMicroseconds(10000);  // Allow assembly to catch up

  block_execution.Notify();
  Env::Default()->SleepForMicroseconds(20000);  // Allow all to process

  mutex_lock l(mu);
  EXPECT_EQ(processed_ids.size(), 3);
}

TEST(JitBatchSchedulerTest, CleanShutdown) {
  JitBatchScheduler<FakeTask>::Options options;
  options.max_batch_size = 1;
  options.num_batch_threads = 2;
  options.max_ready_batches = 2;

  absl::Notification release_processing;
  auto callback = [&](std::unique_ptr<Batch<FakeTask>> batch) {
    release_processing.WaitForNotification();
    batch->mutable_task(0)->FinishTask(absl::OkStatus());
  };

  {
    std::unique_ptr<BatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        JitBatchScheduler<FakeTask>::Create(options, callback, &scheduler));

    for (int i = 0; i < 10; ++i) {
      auto task = std::make_unique<FakeTask>(1, Criticality::kSheddable, i);
      scheduler->Schedule(&task)
          .IgnoreError();  // Ignore if queue is full during shutdown
    }
    // Scheduler goes out of scope here, destructor runs.
  }
  release_processing.Notify();
  // If destructor is correct, this should not hang or crash.
  SUCCEED();
}

// TODO(b/265342012): Add test with variable process_batch_callback duration.
// TODO(b/265342012): Add stress test with high concurrency.

}  // namespace
}  // namespace serving
}  // namespace tensorflow
