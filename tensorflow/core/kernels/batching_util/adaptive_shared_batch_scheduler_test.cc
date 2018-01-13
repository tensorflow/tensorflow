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

TEST(AdaptiveSharedBatchSchedulerTest, Basic) {
  for (const bool delete_scheduler_early : {false, true}) {
    for (const bool delete_queue_1_early : {false, true}) {
      int queue_0_tasks = 0;
      auto queue_0_callback =
          [&queue_0_tasks](std::unique_ptr<Batch<FakeTask>> batch) {
            ASSERT_TRUE(batch->IsClosed());
            EXPECT_GT(batch->num_tasks(), 0);
            for (int i = 0; i < batch->num_tasks(); i++) {
              queue_0_tasks += batch->task(i).size();
            }
          };
      int queue_1_tasks = 0;
      auto queue_1_callback =
          [&queue_1_tasks](std::unique_ptr<Batch<FakeTask>> batch) {
            ASSERT_TRUE(batch->IsClosed());
            EXPECT_GT(batch->num_tasks(), 0);
            for (int i = 0; i < batch->num_tasks(); i++) {
              queue_1_tasks += batch->task(i).size();
            }
          };
      {
        std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
        TF_ASSERT_OK(
            AdaptiveSharedBatchScheduler<FakeTask>::Create({}, &scheduler));

        // Create two queues.
        std::unique_ptr<BatchScheduler<FakeTask>> queue_0;
        TF_ASSERT_OK(scheduler->AddQueue({}, queue_0_callback, &queue_0));
        std::unique_ptr<BatchScheduler<FakeTask>> queue_1;
        TF_ASSERT_OK(scheduler->AddQueue({}, queue_1_callback, &queue_1));

        if (delete_scheduler_early) {
          // Delete our copy of the scheduler. The queues should keep it alive
          // under the covers.
          scheduler = nullptr;
        }
        // Submit tasks to the two queues, and (optionally) remove the queues.
        TF_ASSERT_OK(ScheduleTask(1, queue_0.get()));
        TF_ASSERT_OK(ScheduleTask(2, queue_1.get()));
        TF_ASSERT_OK(ScheduleTask(3, queue_0.get()));
        TF_ASSERT_OK(ScheduleTask(4, queue_1.get()));
        if (delete_queue_1_early) {
          queue_1 = nullptr;
        }
        TF_ASSERT_OK(ScheduleTask(5, queue_0.get()));
      }
      EXPECT_EQ(queue_0_tasks, 9);
      EXPECT_EQ(queue_1_tasks, 6);
    }
  }
}

TEST(AdaptiveSharedBatchSchedulerTest, BadOptions) {
  using Scheduler = AdaptiveSharedBatchScheduler<FakeTask>;
  std::shared_ptr<Scheduler> scheduler;
  Scheduler::Options options;
  options.num_batch_threads = 0;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.min_scheduling_period_micros = 50;
  options.max_scheduling_period_micros = 100;
  options.initial_scheduling_period_micros = 1;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.min_scheduling_period_micros = 50;
  options.max_scheduling_period_micros = 100;
  options.initial_scheduling_period_micros = 1000;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.min_scheduling_period_micros = 100;
  options.max_scheduling_period_micros = 50;
  options.initial_scheduling_period_micros = 75;
  EXPECT_FALSE(Scheduler::Create(options, &scheduler).ok());
  options = Scheduler::Options();
  options.feedback_smoothing_batches = 0;
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
}

TEST(AdaptiveSharedBatchSchedulerTest, ObeysQueueOptions) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  {
    AdaptiveSharedBatchScheduler<FakeTask>::Options options;
    options.initial_scheduling_period_micros = 1000;
    options.env = &env;
    std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    std::unique_ptr<BatchScheduler<FakeTask>> queue_0;
    std::unique_ptr<BatchScheduler<FakeTask>> queue_1;
    int queue_0_tasks = 0;
    int queue_1_tasks = 0;
    auto queue_0_callback = [&queue_0_tasks,
                             &env](std::unique_ptr<Batch<FakeTask>> batch) {
      ASSERT_TRUE(batch->IsClosed());
      EXPECT_GT(batch->num_tasks(), 0);
      for (int i = 0; i < batch->num_tasks(); i++) {
        queue_0_tasks += batch->task(i).size();
      }
      env.SleepForMicroseconds(1);
    };
    auto queue_1_callback = [&queue_1_tasks,
                             &env](std::unique_ptr<Batch<FakeTask>> batch) {
      ASSERT_TRUE(batch->IsClosed());
      EXPECT_GT(batch->num_tasks(), 0);
      for (int i = 0; i < batch->num_tasks(); i++) {
        queue_1_tasks += batch->task(i).size();
      }
      env.SleepForMicroseconds(1);
    };
    AdaptiveSharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    queue_options.max_batch_size = 10;
    queue_options.max_enqueued_batches = 0;
    // Queue must have max_enqueued_batchs > 1.
    EXPECT_FALSE(
        scheduler->AddQueue(queue_options, queue_0_callback, &queue_0).ok());
    queue_options.max_enqueued_batches = 2;
    TF_ASSERT_OK(
        scheduler->AddQueue(queue_options, queue_0_callback, &queue_0));
    EXPECT_EQ(10, queue_0->max_task_size());
    queue_options.max_batch_size = 0;
    // Queue must have max_batch_size > 0.
    EXPECT_FALSE(
        scheduler->AddQueue(queue_options, queue_1_callback, &queue_1).ok());
    queue_options.max_batch_size = 2;
    queue_options.max_enqueued_batches = 1;
    TF_ASSERT_OK(
        scheduler->AddQueue(queue_options, queue_1_callback, &queue_1));

    // Wait for scheduling_thread to sleep.
    env.BlockUntilThreadsAsleep(1);
    // Task larger than max_batch_size shouldn't schedule.
    EXPECT_FALSE(ScheduleTask(15, queue_0.get()).ok());
    TF_ASSERT_OK(ScheduleTask(5, queue_0.get()));
    TF_ASSERT_OK(ScheduleTask(5, queue_0.get()));
    env.AdvanceByMicroseconds(1);

    // Task larger than max_batch_size shouldn't schedule.
    EXPECT_FALSE(ScheduleTask(3, queue_1.get()).ok());
    TF_ASSERT_OK(ScheduleTask(1, queue_1.get()));
    TF_ASSERT_OK(ScheduleTask(1, queue_1.get()));
    env.AdvanceByMicroseconds(1);
    // Exceeds max_enqueued_batches, shouldn't schedule.
    EXPECT_FALSE(ScheduleTask(1, queue_1.get()).ok());

    TF_ASSERT_OK(ScheduleTask(5, queue_0.get()));
    // Exceeds max_enqueued_batches, shouldn't schedule.
    EXPECT_FALSE(ScheduleTask(6, queue_0.get()).ok());
    TF_ASSERT_OK(ScheduleTask(4, queue_0.get()));

    // Batches should be processed in order from oldest to newest.
    env.AdvanceByMicroseconds(1000);
    env.BlockUntilThreadsAsleep(2);
    EXPECT_EQ(queue_0_tasks, 10);
    EXPECT_EQ(queue_1_tasks, 0);

    env.AdvanceByMicroseconds(1000);
    env.BlockUntilThreadsAsleep(2);
    EXPECT_EQ(queue_0_tasks, 10);
    EXPECT_EQ(queue_1_tasks, 2);

    env.AdvanceByMicroseconds(1000);
    env.BlockUntilThreadsAsleep(2);
    EXPECT_EQ(queue_0_tasks, 19);
    EXPECT_EQ(queue_1_tasks, 2);
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(AdaptiveSharedBatchSchedulerTest, RateFeedback) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  {
    double feedback = 0;
    AdaptiveSharedBatchScheduler<FakeTask>::Options options;
    options.initial_scheduling_period_micros = 1000;
    options.min_scheduling_period_micros = 200;
    options.max_scheduling_period_micros = 2000;
    options.env = &env;
    options.scheduling_period_feedback = [&feedback] { return feedback; };
    options.feedback_smoothing_batches = 1;
    std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    std::unique_ptr<BatchScheduler<FakeTask>> queue;
    int scheduled_items = 0;
    auto queue_callback = [&scheduled_items,
                           &env](std::unique_ptr<Batch<FakeTask>> batch) {
      ASSERT_TRUE(batch->IsClosed());
      EXPECT_GT(batch->num_tasks(), 0);
      scheduled_items = 0;
      for (int i = 0; i < batch->num_tasks(); i++) {
        scheduled_items += batch->task(i).size();
      }
      env.SleepForMicroseconds(1);
    };

    TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

    // Wait for scheduling_thread to sleep.
    env.BlockUntilThreadsAsleep(1);
    // Enqueue 6 batches.
    for (int i = 0; i < 6; i++) {
      TF_ASSERT_OK(ScheduleTask(900 + i, queue.get()));
      env.AdvanceByMicroseconds(1);
    }
    feedback = -500;
    env.AdvanceByMicroseconds(994);
    env.BlockUntilThreadsAsleep(2);  // scheduling period = 500 usec.
    EXPECT_EQ(scheduled_items, 900);
    env.AdvanceByMicroseconds(500);
    env.BlockUntilThreadsAsleep(2);  // scheduling period = 250 usec.
    EXPECT_EQ(scheduled_items, 901);
    feedback = 0;
    env.AdvanceByMicroseconds(250);
    env.BlockUntilThreadsAsleep(2);  // scheduling period = 250 usec.
    EXPECT_EQ(scheduled_items, 902);
    feedback = 10000;  // large feedback should hit max_scheduling_period.
    env.AdvanceByMicroseconds(250);
    env.BlockUntilThreadsAsleep(2);  // scheduling period = 2000 usec.
    EXPECT_EQ(scheduled_items, 903);
    feedback = -10000;  // large feedback should hit min_scheduling_period.
    env.AdvanceByMicroseconds(1999);
    // No callback scheduled, only scheduling thread sleeping.
    env.BlockUntilThreadsAsleep(1);
    EXPECT_EQ(scheduled_items, 903);
    env.AdvanceByMicroseconds(1);
    env.BlockUntilThreadsAsleep(2);  // scheduling period = 200 usec.
    EXPECT_EQ(scheduled_items, 904);
    env.AdvanceByMicroseconds(200);
    env.BlockUntilThreadsAsleep(2);
    EXPECT_EQ(scheduled_items, 905);
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(AdaptiveSharedBatchSchedulerTest, FeedbackSmoothing) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  {
    double feedback = 0;
    AdaptiveSharedBatchScheduler<FakeTask>::Options options;
    options.initial_scheduling_period_micros = 1000;
    options.env = &env;
    options.scheduling_period_feedback = [&feedback] { return feedback; };
    options.feedback_smoothing_batches = 3;
    std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    std::unique_ptr<BatchScheduler<FakeTask>> queue;
    int scheduled_items = 0;
    auto queue_callback = [&scheduled_items,
                           &env](std::unique_ptr<Batch<FakeTask>> batch) {
      ASSERT_TRUE(batch->IsClosed());
      EXPECT_GT(batch->num_tasks(), 0);
      scheduled_items = 0;
      for (int i = 0; i < batch->num_tasks(); i++) {
        scheduled_items += batch->task(i).size();
      }
      env.SleepForMicroseconds(1);
    };

    TF_ASSERT_OK(scheduler->AddQueue({}, queue_callback, &queue));

    // Wait for scheduling_thread to sleep.
    env.BlockUntilThreadsAsleep(1);
    // Enqueue 4 batches.
    for (int i = 0; i < 4; i++) {
      TF_ASSERT_OK(ScheduleTask(900 + i, queue.get()));
      env.AdvanceByMicroseconds(1);
    }
    feedback = -300;
    env.AdvanceByMicroseconds(996);
    env.BlockUntilThreadsAsleep(2);
    // ewma_feedback = 100, scheduling_period = 900.
    EXPECT_EQ(scheduled_items, 900);
    env.AdvanceByMicroseconds(899);
    // No callback scheduled, only scheduling thread sleeping.
    env.BlockUntilThreadsAsleep(1);
    EXPECT_EQ(scheduled_items, 900);
    env.AdvanceByMicroseconds(1);
    env.BlockUntilThreadsAsleep(2);
    // ewma_feedback = 167, scheduling_period = 750.
    EXPECT_EQ(scheduled_items, 901);
    env.AdvanceByMicroseconds(749);
    // No callback scheduled, only scheduling thread sleeping.
    env.BlockUntilThreadsAsleep(1);
    EXPECT_EQ(scheduled_items, 901);
    feedback = 1000 / 3.;
    env.AdvanceByMicroseconds(1);
    env.BlockUntilThreadsAsleep(2);
    // emwa_feedback = 0, scheduling_period = 750.
    EXPECT_EQ(scheduled_items, 902);
    env.AdvanceByMicroseconds(749);
    // No callback scheduled, only scheduling thread sleeping.
    env.BlockUntilThreadsAsleep(1);
    EXPECT_EQ(scheduled_items, 902);
    env.AdvanceByMicroseconds(1);
    env.BlockUntilThreadsAsleep(2);
    EXPECT_EQ(scheduled_items, 903);
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(AdaptiveSharedBatchSchedulerTest, QueueCapacityInfo) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  {
    AdaptiveSharedBatchScheduler<FakeTask>::Options options;
    options.initial_scheduling_period_micros = 1000;
    options.env = &env;
    std::shared_ptr<AdaptiveSharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        AdaptiveSharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    std::unique_ptr<BatchScheduler<FakeTask>> queue;
    int scheduled_items = 0;
    auto queue_callback = [&scheduled_items,
                           &env](std::unique_ptr<Batch<FakeTask>> batch) {
      ASSERT_TRUE(batch->IsClosed());
      EXPECT_GT(batch->num_tasks(), 0);
      scheduled_items = 0;
      for (int i = 0; i < batch->num_tasks(); i++) {
        scheduled_items += batch->task(i).size();
      }
      env.SleepForMicroseconds(1);
    };
    AdaptiveSharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    queue_options.max_batch_size = 10;
    queue_options.max_enqueued_batches = 10;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_callback, &queue));

    // Wait for scheduling_thread to sleep.
    env.BlockUntilThreadsAsleep(1);
    // Enqueue 3 tasks.
    EXPECT_EQ(queue->NumEnqueuedTasks(), 0);
    EXPECT_EQ(queue->SchedulingCapacity(), 100);
    TF_ASSERT_OK(ScheduleTask(5, queue.get()));
    EXPECT_EQ(queue->NumEnqueuedTasks(), 1);
    EXPECT_EQ(queue->SchedulingCapacity(), 95);
    env.AdvanceByMicroseconds(1);
    TF_ASSERT_OK(ScheduleTask(6, queue.get()));
    EXPECT_EQ(queue->NumEnqueuedTasks(), 2);
    EXPECT_EQ(queue->SchedulingCapacity(), 84);
    env.AdvanceByMicroseconds(1);
    TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    EXPECT_EQ(queue->NumEnqueuedTasks(), 3);
    EXPECT_EQ(queue->SchedulingCapacity(), 83);

    env.AdvanceByMicroseconds(998);
    env.BlockUntilThreadsAsleep(2);
    EXPECT_EQ(scheduled_items, 5);
    env.AdvanceByMicroseconds(1000);
    env.BlockUntilThreadsAsleep(2);
    EXPECT_EQ(scheduled_items, 7);
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(AdaptiveSharedBatchSchedulerTest, InFlightBatchesImplementation) {
  AdaptiveSharedBatchScheduler<FakeTask>::Options options;
  options.use_in_flight_batches_implementation = true;
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
  for (int i = 0; i < 3; i++) {
    TF_ASSERT_OK(ScheduleTask(100, queue.get()));
  }
}

TEST(AdaptiveSharedBatchSchedulerTest, InFlightBatchesLimitTuning) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  {
    AdaptiveSharedBatchScheduler<FakeTask>::Options options;
    options.env = &env;
    options.use_in_flight_batches_implementation = true;
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
}  // namespace anonymous
}  // namespace serving
}  // namespace tensorflow
