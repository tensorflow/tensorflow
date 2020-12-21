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

#include "tensorflow/core/kernels/batching_util/adaptive_shared_batch_scheduler.h"

#include "tensorflow/core/kernels/batching_util/fake_clock_env.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
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

  void set_size(size_t size) { size_ = size; }

 private:
  size_t size_;

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

TEST(AdaptiveSharedBatchSchedulerTest, BadOptions) {
  using Scheduler = AdaptiveSharedBatchScheduler<FakeTask>;
  std::shared_ptr<Scheduler> scheduler;
  Scheduler::Options options;
  options.num_batch_threads = 0;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.initial_in_flight_batches_limit = 0.5;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.num_batch_threads = 5;
  options.initial_in_flight_batches_limit = 8;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.batches_to_average_over = -5;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.min_in_flight_batches_limit = 0;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.min_in_flight_batches_limit = 5;
  options.num_batch_threads = 3;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.initial_in_flight_batches_limit = 1;
  options.min_in_flight_batches_limit = 2;
  options.num_batch_threads = 3;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
}

TEST(AdaptiveSharedBatchSchedulerTest, InFlightBatchesLimit) {
  AdaptiveSharedBatchScheduler<FakeTask>::Options options;
  options.initial_in_flight_batches_limit = 2;
  options.batches_to_average_over = 1000;
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
  std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

  // Enqueue 3 batches.
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  while (queue->NumEnqueuedTasks() > 0) {
  }
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  while (queue->NumEnqueuedTasks() > 0) {
  }
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
}

TEST(AdaptiveSharedBatchSchedulerTest, InFlightBatchesLimitTuning) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  {
    AdaptiveSharedBatchScheduler<FakeTask>::Options options;
    options.env = &env;
    options.initial_in_flight_batches_limit = 2;
    options.batches_to_average_over = 1;
    auto queue_callback = [&env](std::unique_ptr<Batch<FakeTask>> batch) {
      ASSERT_TRUE(batch->IsClosed());
      switch (batch->size()) {
        case 0:
          env.AdvanceByMicroseconds(10);
          break;
        case 1:
          env.AdvanceByMicroseconds(15);
          break;
        case 2:
          env.AdvanceByMicroseconds(10);
          break;
        case 3:
          env.AdvanceByMicroseconds(11);
          break;
      }
    };
    std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    std::unique_ptr<BatchScheduler<FakeTask>> queue;
    TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

    TF_ASSERT_OK(ScheduleTask(0, queue.get()));
    double in_flight_batches_limit = 2;
    while (scheduler->in_flight_batches_limit() == in_flight_batches_limit) {
    }
    // Initial direction will be negative.
    EXPECT_LT(scheduler->in_flight_batches_limit(), in_flight_batches_limit);
    in_flight_batches_limit = scheduler->in_flight_batches_limit();
    TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    while (scheduler->in_flight_batches_limit() == in_flight_batches_limit) {
    }
    // Latency increased -> change direction.
    EXPECT_GT(scheduler->in_flight_batches_limit(), in_flight_batches_limit);
    in_flight_batches_limit = scheduler->in_flight_batches_limit();
    TF_ASSERT_OK(ScheduleTask(2, queue.get()));
    while (scheduler->in_flight_batches_limit() == in_flight_batches_limit) {
    }
    // Latency decreased -> keep going in same direction.
    EXPECT_GT(scheduler->in_flight_batches_limit(), in_flight_batches_limit);
    in_flight_batches_limit = scheduler->in_flight_batches_limit();
    TF_ASSERT_OK(ScheduleTask(3, queue.get()));
    while (scheduler->in_flight_batches_limit() == in_flight_batches_limit) {
    }
    // Latency increased -> change direction.
    EXPECT_LT(scheduler->in_flight_batches_limit(), in_flight_batches_limit);
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(AdaptiveSharedBatchSchedulerTest, FullBatchSchedulingBoostMicros) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  {
    AdaptiveSharedBatchScheduler<FakeTask>::Options options;
    options.env = &env;
    options.initial_in_flight_batches_limit = 1;
    options.num_batch_threads = 1;
    options.batches_to_average_over = 1000;
    options.full_batch_scheduling_boost_micros = 100;
    mutex mu;
    int processed_batches = 0;
    Notification finish_processing;
    auto queue_callback = [&mu, &processed_batches, &finish_processing](
                              std::unique_ptr<Batch<FakeTask>> batch) {
      ASSERT_TRUE(batch->IsClosed());
      finish_processing.WaitForNotification();
      mutex_lock l(mu);
      processed_batches++;
      switch (processed_batches) {
        case 1:
          EXPECT_EQ(100, batch->size());
          break;
        case 2:
          EXPECT_EQ(50, batch->size());
          break;
        case 3:
          EXPECT_EQ(900, batch->size());
          break;
        case 4:
          EXPECT_EQ(200, batch->size());
          break;
        default:
          EXPECT_TRUE(false) << "Should only have 4 batches";
      }
    };
    std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    AdaptiveSharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    std::unique_ptr<BatchScheduler<FakeTask>> queue1;
    std::unique_ptr<BatchScheduler<FakeTask>> queue2;
    queue_options.max_batch_size = 1000;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue1));
    queue_options.max_batch_size = 100;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue2));

    // First batch immediately processed.
    TF_ASSERT_OK(ScheduleTask(100, queue1.get()));
    while (queue1->NumEnqueuedTasks() > 0) {
    }

    TF_ASSERT_OK(ScheduleTask(100, queue1.get()));
    env.AdvanceByMicroseconds(10);
    TF_ASSERT_OK(ScheduleTask(100, queue1.get()));
    env.AdvanceByMicroseconds(10);

    TF_ASSERT_OK(ScheduleTask(50, queue2.get()));
    env.AdvanceByMicroseconds(45);

    TF_ASSERT_OK(ScheduleTask(900, queue1.get()));

    // Second batch - creation time: 0, fullness: 0.2, sched score: -20
    // Third batch - creation time: 20, fullness: 0.5, sched score: -30
    // Fourth batch - creation time: 65, fullness: 0.9, sched score: -25

    finish_processing.Notify();
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(AdaptiveSharedBatchSchedulerTest, DeleteQueue) {
  AdaptiveSharedBatchScheduler<FakeTask>::Options options;
  options.initial_in_flight_batches_limit = 1;
  options.num_batch_threads = 1;
  options.batches_to_average_over = 1000;
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

  auto processed_checker = gtl::MakeCleanup([&mu, &processed_batches] {
    mutex_lock l(mu);
    EXPECT_EQ(processed_batches, 2);
  });
  std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

  // Enqueue 2 tasks, should result in 2 batches.
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  while (queue->NumEnqueuedTasks() > 0) {
  }
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  // Queue destructor should block until second batch has been scheduled.
  Env::Default()->SchedClosureAfter(
      1000, [&finish_processing] { finish_processing.Notify(); });
}

TEST(AdaptiveSharedBatchSchedulerTest, QueueCapacityInfo) {
  AdaptiveSharedBatchScheduler<FakeTask>::Options options;
  options.initial_in_flight_batches_limit = 1;
  options.batches_to_average_over = 1000;
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
  std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(
      AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

  // Enqueue 2 tasks, should result in 2 batches.
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  while (queue->NumEnqueuedTasks() > 0) {
  }
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  // First batch was immediately processed, no longer counts as enqueued.
  EXPECT_EQ(queue->NumEnqueuedTasks(), 1);
  EXPECT_EQ(queue->SchedulingCapacity(), 9 * 1000 + 900);
  // Enqueue 2 more tasks, should fall in same batch.
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  TF_ASSERT_OK(ScheduleTask(200, queue.get()));
  EXPECT_EQ(queue->NumEnqueuedTasks(), 3);
  EXPECT_EQ(queue->SchedulingCapacity(), 9 * 1000 + 600);
  // Enqueue 1 more task, should create new batch and start processing the
  // previous batch.
  TF_ASSERT_OK(ScheduleTask(700, queue.get()));
  EXPECT_EQ(queue->NumEnqueuedTasks(), 1);
  EXPECT_EQ(queue->SchedulingCapacity(), 9 * 1000 + 300);
  finish_processing.Notify();
}

TEST(AdaptiveSharedBatchSchedulerTest, FullBatches) {
  std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(AdaptiveSharedBatchScheduler<FakeTask>::Create({}, &scheduler));
  auto queue_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {
    ASSERT_TRUE(batch->IsClosed());
  };
  AdaptiveSharedBatchScheduler<FakeTask>::QueueOptions queue_options;
  queue_options.max_batch_size = 100;
  queue_options.batch_timeout_micros = 1000000000000;
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue));
  TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  // Full batches should not have to wait batch_timeout_micros.
}

TEST(AdaptiveSharedBatchSchedulerTest, TruncateBatches) {
  mutex mu;
  int processed_batches = 0;
  auto queue_callback =
      [&mu, &processed_batches](std::unique_ptr<Batch<FakeTask>> batch) {
        ASSERT_TRUE(batch->IsClosed());
        mutex_lock l(mu);
        ++processed_batches;
      };
  std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(AdaptiveSharedBatchScheduler<FakeTask>::Create({}, &scheduler));
  std::unique_ptr<BatchScheduler<FakeTask>> queue;

  AdaptiveSharedBatchScheduler<FakeTask>::QueueOptions queue_options;
  queue_options.max_batch_size = 100;
  queue_options.batch_timeout_micros = 1000000;
  queue_options.split_input_task_func =
      [](std::unique_ptr<FakeTask>* input_task, int first_size, int max_size,
         std::vector<std::unique_ptr<FakeTask>>* output_tasks) {
        EXPECT_EQ(first_size, 70);
        output_tasks->push_back(std::move(*input_task));
        int remaining_size = output_tasks->back()->size() - first_size;
        output_tasks->back()->set_size(first_size);
        while (remaining_size > 0) {
          int task_size = std::min(remaining_size, max_size);
          output_tasks->emplace_back(new FakeTask(task_size));
          remaining_size -= task_size;
        }
        return Status::OK();
      };
  TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue));
  TF_ASSERT_OK(ScheduleTask(30, queue.get()));
  TF_ASSERT_OK(ScheduleTask(350, queue.get()));
  // Second task should be split into a task of size 70, 2 tasks of size 100,
  // and one task of size 80.
  while (true) {
    mutex_lock l(mu);
    if (processed_batches == 4) break;
  }
}
}  // namespace anonymous
}  // namespace serving
}  // namespace tensorflow
