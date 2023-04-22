/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/serial_device_batch_scheduler.h"

#include "tensorflow/core/kernels/batching_util/fake_clock_env.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"

namespace tensorflow {
namespace serving {
namespace anonymous {

class FakeTask : public BatchTask {
 public:
  explicit FakeTask(size_t size) : size_(size) {}

  ~FakeTask() override = default;

  size_t size() const override { return size_; }

 private:
  const size_t size_;

  TF_DISALLOW_COPY_AND_ASSIGN(FakeTask);
};

// Creates a FakeTask of size 'task_size', and calls 'scheduler->Schedule()' on
// that task. Returns the resulting status.
Status ScheduleTask(size_t task_size, BatchScheduler<FakeTask>* scheduler) {
  std::unique_ptr<FakeTask> task(new FakeTask(task_size));
  Status status = scheduler->Schedule(&task);
  // Schedule() should have consumed 'task' iff it returned Status::OK.
  CHECK_EQ(status.ok(), task == nullptr);
  return status;
}

// Creates a thread that waits on 'start' and then advances the fake clock in
// 'env' in a loop until 'stop' is notified. Useful for allowing objects that
// use the clock to be destroyed.
std::unique_ptr<Thread> CreateFakeClockAdvancerThread(
    test_util::FakeClockEnv* env, Notification* start, Notification* stop) {
  return std::unique_ptr<Thread>(Env::Default()->StartThread(
      {}, "FakeClockAdvancerThread", [env, start, stop] {
        start->WaitForNotification();
        while (!stop->HasBeenNotified()) {
          env->AdvanceByMicroseconds(10);
          Env::Default()->SleepForMicroseconds(10);
        }
      }));
}

TEST(SerialDeviceBatchSchedulerTest, BadOptions) {
  using Scheduler = SerialDeviceBatchScheduler<FakeTask>;
  std::shared_ptr<Scheduler> scheduler;
  Scheduler::Options default_options;
  default_options.get_pending_on_serial_device = []() { return 0; };
  Scheduler::Options options = default_options;
  options.num_batch_threads = 0;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = default_options;
  options.initial_in_flight_batches_limit = 0;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = default_options;
  options.num_batch_threads = 5;
  options.initial_in_flight_batches_limit = 8;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = default_options;
  options.batches_to_average_over = -5;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = default_options;
  options.target_pending = 0;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
}

TEST(SerialDeviceBatchSchedulerTest, InFlightBatchesLimit) {
  SerialDeviceBatchScheduler<FakeTask>::Options options;
  options.num_batch_threads = 3;
  options.initial_in_flight_batches_limit = 2;
  options.batches_to_average_over = 1000;
  options.get_pending_on_serial_device = []() { return 0; };
  mutex mu;
  int processed_batches = 0;
  Notification finish_processing;
  auto queue_callback = [&mu, &processed_batches, &finish_processing](
                            std::unique_ptr<Batch<FakeTask>> batch) {
    ASSERT_TRUE(batch->IsClosed());
    EXPECT_GT(batch->num_tasks(), 0);
    mu.lock();
    int batch_num = ++processed_batches;
    mu.unlock();
    if (batch_num == 2) {
      // Give third batch a chance to process if it's going to.
      Env::Default()->SleepForMicroseconds(1000);
      finish_processing.Notify();
    }
    if (batch_num == 3) {
      ASSERT_TRUE(finish_processing.HasBeenNotified());
    }
    finish_processing.WaitForNotification();
  };
  std::shared_ptr<SerialDeviceBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      SerialDeviceBatchScheduler<FakeTask>::Create(options, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue1;
  std::unique_ptr<BatchScheduler<FakeTask>> queue2;
  std::unique_ptr<BatchScheduler<FakeTask>> queue3;
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue1));
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue2));
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue3));
  // Create 3 batches, only 2 should be processed concurrently.
  TF_ASSERT_OK(ScheduleTask(100, queue1.get()));
  TF_ASSERT_OK(ScheduleTask(100, queue2.get()));
  TF_ASSERT_OK(ScheduleTask(100, queue3.get()));
}

TEST(SerialDeviceBatchSchedulerTest, PendingOnSerialDevice) {
  mutex mu;
  int pending;
  SerialDeviceBatchScheduler<FakeTask>::Options options;
  options.num_batch_threads = 3;
  options.initial_in_flight_batches_limit = 1;
  options.batches_to_average_over = 1;
  options.target_pending = 3;
  options.get_pending_on_serial_device = [&mu, &pending]() {
    mutex_lock l(mu);
    return pending;
  };
  std::shared_ptr<SerialDeviceBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      SerialDeviceBatchScheduler<FakeTask>::Create(options, &scheduler));
  int processed_batches = 0;
  Notification start_processing;
  auto queue_callback = [&mu, &processed_batches, &start_processing, &pending,
                         &scheduler](std::unique_ptr<Batch<FakeTask>> batch) {
    // Be careful with mutex mu to avoid potential deadlock with mutex mu_
    // held in ProcessBatch() and in_flight_batches_limit().
    int batch_num;
    {
      mutex_lock l(mu);
      batch_num = ++processed_batches;
    }
    switch (batch_num) {
      case 1:
        start_processing.WaitForNotification();
        {
          mutex_lock l(mu);
          pending = 3;
        }
        break;
      case 2:
        // Either low traffic or pending at target --> no adjustment.
        CHECK_EQ(scheduler->in_flight_batches_limit(), 1);
        {
          mutex_lock l(mu);
          pending = 1;
        }
        break;
      case 3:
        // Small pending --> 2 additional threads added.
        CHECK_EQ(scheduler->in_flight_batches_limit(), 3);
        {
          mutex_lock l(mu);
          pending = 3;
        }
        break;
      default:
        break;
    }
  };
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));
  // Create 3 batches.
  for (int i = 0; i < 3; i++) {
    TF_ASSERT_OK(ScheduleTask(800, queue.get()));
  }
  start_processing.Notify();
}

TEST(SerialDeviceBatchSchedulerTest, FullBatchSchedulingBoostMicros) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  {
    SerialDeviceBatchScheduler<FakeTask>::Options options;
    options.env = &env;
    options.initial_in_flight_batches_limit = 1;
    options.batches_to_average_over = 1000;
    options.full_batch_scheduling_boost_micros = 10;
    options.get_pending_on_serial_device = []() { return 0; };
    mutex mu;
    int processed_batches = 0;
    auto queue_callback =
        [&mu, &processed_batches](std::unique_ptr<Batch<FakeTask>> batch) {
          ASSERT_TRUE(batch->IsClosed());
          mutex_lock l(mu);
          processed_batches++;
          switch (processed_batches) {
            case 1:
              EXPECT_EQ(1000, batch->size());
              break;
            case 2:
              EXPECT_EQ(100, batch->size());
              break;
            case 3:
              EXPECT_EQ(80, batch->size());
              break;
            default:
              EXPECT_TRUE(false) << "Should only have 3 batches";
          }
        };
    std::shared_ptr<SerialDeviceBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        SerialDeviceBatchScheduler<FakeTask>::Create(options, &scheduler));
    // Make sure batch processing thread has gone to sleep.
    Env::Default()->SleepForMicroseconds(1000);
    SerialDeviceBatchScheduler<FakeTask>::QueueOptions queue_options;
    std::unique_ptr<BatchScheduler<FakeTask>> queue1;
    std::unique_ptr<BatchScheduler<FakeTask>> queue2;
    std::unique_ptr<BatchScheduler<FakeTask>> queue3;
    queue_options.max_batch_size = 1000;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue1));
    queue_options.max_batch_size = 1000;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue2));
    queue_options.max_batch_size = 100;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue3));

    TF_ASSERT_OK(ScheduleTask(100, queue1.get()));
    // First batch - creation time: 0, fullness: 0.1, sched score: -1
    env.AdvanceByMicroseconds(3);
    TF_ASSERT_OK(ScheduleTask(1000, queue2.get()));
    // Second batch - creation time: 3, fullness: 1, sched score: -7
    env.AdvanceByMicroseconds(5);
    TF_ASSERT_OK(ScheduleTask(80, queue3.get()));
    // Third batch - creation time: 8, fullness: .8, sched score: 0
    // Release the batch processing thread.
    env.AdvanceByMicroseconds(1000);
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(SerialDeviceBatchSchedulerTest, DeleteQueue) {
  SerialDeviceBatchScheduler<FakeTask>::Options options;
  options.initial_in_flight_batches_limit = 1;
  options.batches_to_average_over = 1000;
  options.get_pending_on_serial_device = []() { return 0; };
  mutex mu;
  int processed_batches = 0;
  Notification finish_processing;
  auto queue_callback = [&mu, &processed_batches, &finish_processing](
                            std::unique_ptr<Batch<FakeTask>> batch) {
    ASSERT_TRUE(batch->IsClosed());
    EXPECT_GT(batch->num_tasks(), 0);
    finish_processing.WaitForNotification();
    mu.lock();
    processed_batches++;
    mu.unlock();
  };
  std::shared_ptr<SerialDeviceBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      SerialDeviceBatchScheduler<FakeTask>::Create(options, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

  // Enqueue 2 tasks, should result in 2 batches.
  for (int i = 0; i < 2; i++) {
    TF_ASSERT_OK(ScheduleTask(800, queue.get()));
  }
  std::unique_ptr<Thread> queue_deleter(Env::Default()->StartThread(
      {}, "QueueDeleterThread",
      [&queue, &mu, &processed_batches, scheduler]() mutable {
        // Delete queue, should be kept alive until empty.
        queue.reset();
        {
          mutex_lock l(mu);
          // queue may be destroyed before 2nd batch finishes processing.
          EXPECT_GT(processed_batches, 0);
        }
        // Delete scheduler, should be kept alive until all batches processed.
        scheduler.reset();
        mutex_lock l(mu);
        EXPECT_EQ(processed_batches, 2);
      }));
  // Release reference to scheduler, queue and callback above should keep alive.
  scheduler.reset();
  // Give queue_deleter thread time to delete queue.
  Env::Default()->SleepForMicroseconds(1000);
  finish_processing.Notify();
}

TEST(SerialDeviceBatchSchedulerTest, DeleteScheduler) {
  SerialDeviceBatchScheduler<FakeTask>::Options options;
  options.initial_in_flight_batches_limit = 1;
  options.batches_to_average_over = 1000;
  options.get_pending_on_serial_device = []() { return 0; };
  mutex mu;
  int processed_batches = 0;
  Notification start_processing;
  Notification finish_processing;
  auto queue_callback =
      [&mu, &processed_batches, &start_processing,
       &finish_processing](std::unique_ptr<Batch<FakeTask>> batch) {
        ASSERT_TRUE(batch->IsClosed());
        EXPECT_GT(batch->num_tasks(), 0);
        start_processing.WaitForNotification();
        mutex_lock l(mu);
        processed_batches++;
        if (processed_batches == 2) {
          finish_processing.Notify();
        }
      };

  std::shared_ptr<SerialDeviceBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      SerialDeviceBatchScheduler<FakeTask>::Create(options, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

  // Enqueue 2 tasks, should result in 2 batches.
  for (int i = 0; i < 2; i++) {
    TF_ASSERT_OK(ScheduleTask(800, queue.get()));
  }
  // Delete scheduler, should be kept alive until queues are empty.
  scheduler.reset();
  start_processing.Notify();
  finish_processing.WaitForNotification();
}

TEST(SerialDeviceBatchSchedulerTest, QueueCapacityInfo) {
  SerialDeviceBatchScheduler<FakeTask>::Options options;
  options.initial_in_flight_batches_limit = 1;
  options.batches_to_average_over = 1000;
  options.full_batch_scheduling_boost_micros = 1000;
  options.get_pending_on_serial_device = []() { return 0; };
  mutex mu;
  int processed_batches = 0;
  Notification finish_processing;
  auto queue_callback = [&mu, &processed_batches, &finish_processing](
                            std::unique_ptr<Batch<FakeTask>> batch) {
    ASSERT_TRUE(batch->IsClosed());
    EXPECT_GT(batch->num_tasks(), 0);
    mu.lock();
    int batch_num = ++processed_batches;
    mu.unlock();
    if (batch_num == 1) {
      finish_processing.WaitForNotification();
    }
  };
  std::shared_ptr<SerialDeviceBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      SerialDeviceBatchScheduler<FakeTask>::Create(options, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue1;
  std::unique_ptr<BatchScheduler<FakeTask>> queue2;
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue1));
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue2));

  // Blocker task, should schedule first.
  TF_ASSERT_OK(ScheduleTask(800, queue1.get()));
  TF_ASSERT_OK(ScheduleTask(100, queue2.get()));

  EXPECT_EQ(queue2->NumEnqueuedTasks(), 1);
  EXPECT_EQ(queue2->SchedulingCapacity(), 9 * 1000 + 900);
  // Enqueue 2 more tasks, should fall in same batch.
  TF_ASSERT_OK(ScheduleTask(100, queue2.get()));
  TF_ASSERT_OK(ScheduleTask(200, queue2.get()));
  EXPECT_EQ(queue2->NumEnqueuedTasks(), 3);
  EXPECT_EQ(queue2->SchedulingCapacity(), 9 * 1000 + 600);
  // Enqueue 1 more task, should create new batch.
  TF_ASSERT_OK(ScheduleTask(700, queue2.get()));
  EXPECT_EQ(queue2->NumEnqueuedTasks(), 4);
  EXPECT_EQ(queue2->SchedulingCapacity(), 8 * 1000 + 300);
  finish_processing.Notify();
}
}  // namespace anonymous
}  // namespace serving
}  // namespace tensorflow
