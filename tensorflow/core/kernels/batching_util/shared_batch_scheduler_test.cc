/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/core/kernels/batching_util/shared_batch_scheduler.h"

#include "tensorflow/core/kernels/batching_util/fake_clock_env.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"

namespace tensorflow {
namespace serving {
namespace {

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

TEST(SharedBatchSchedulerTest, Basic) {
  for (int num_batch_threads : {1, 2, 3}) {
    for (const bool delete_scheduler_early : {false, true}) {
      for (const bool delete_queue_1_early : {false, true}) {
        bool queue_0_callback_called = false;
        auto queue_0_callback =
            [&queue_0_callback_called](std::unique_ptr<Batch<FakeTask>> batch) {
              queue_0_callback_called = true;
              ASSERT_TRUE(batch->IsClosed());
              ASSERT_EQ(3, batch->num_tasks());
              EXPECT_EQ(1, batch->task(0).size());
              EXPECT_EQ(3, batch->task(1).size());
              EXPECT_EQ(5, batch->task(2).size());
            };
        bool queue_1_callback_called = false;
        auto queue_1_callback =
            [&queue_1_callback_called](std::unique_ptr<Batch<FakeTask>> batch) {
              queue_1_callback_called = true;
              ASSERT_TRUE(batch->IsClosed());
              ASSERT_EQ(2, batch->num_tasks());
              EXPECT_EQ(2, batch->task(0).size());
              EXPECT_EQ(4, batch->task(1).size());
            };
        {
          SharedBatchScheduler<FakeTask>::Options options;
          options.num_batch_threads = num_batch_threads;
          std::shared_ptr<SharedBatchScheduler<FakeTask>> scheduler;
          TF_ASSERT_OK(
              SharedBatchScheduler<FakeTask>::Create(options, &scheduler));

          // Create two queues.
          SharedBatchScheduler<FakeTask>::QueueOptions queue_options;
          queue_options.input_batch_size_limit = 10;
          queue_options.batch_timeout_micros = 10 * 1000 * 1000;  // 10 seconds
          queue_options.max_enqueued_batches = 2;
          std::unique_ptr<BatchScheduler<FakeTask>> queue_0;
          TF_ASSERT_OK(
              scheduler->AddQueue(queue_options, queue_0_callback, &queue_0));
          std::unique_ptr<BatchScheduler<FakeTask>> queue_1;
          TF_ASSERT_OK(
              scheduler->AddQueue(queue_options, queue_1_callback, &queue_1));

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
        EXPECT_TRUE(queue_0_callback_called);
        EXPECT_TRUE(queue_1_callback_called);
      }
    }
  }
}

TEST(SharedBatchSchedulerTest, ObeyBatchSizeConstraint) {
  // Set up a callback that captures the batches' task sizes.
  mutex mu;
  std::vector<std::vector<size_t>> callback_data;
  auto callback = [&mu,
                   &callback_data](std::unique_ptr<Batch<FakeTask>> batch) {
    ASSERT_TRUE(batch->IsClosed());
    std::vector<size_t> batch_data;
    batch_data.reserve(batch->num_tasks());
    for (int i = 0; i < batch->num_tasks(); ++i) {
      batch_data.push_back(batch->mutable_task(i)->size());
    }
    {
      mutex_lock l(mu);
      callback_data.push_back(batch_data);
    }
  };

  // Run a batch scheduler and inject some tasks.
  {
    SharedBatchScheduler<FakeTask>::Options options;
    options.num_batch_threads = 2;
    std::shared_ptr<SharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(SharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    SharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    queue_options.input_batch_size_limit = 10;
    queue_options.batch_timeout_micros = 10 * 1000 * 1000;  // 10 seconds
    queue_options.max_enqueued_batches = 2;
    std::unique_ptr<BatchScheduler<FakeTask>> queue;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, callback, &queue));

    // First batch.
    TF_ASSERT_OK(ScheduleTask(3, queue.get()));
    TF_ASSERT_OK(ScheduleTask(5, queue.get()));

    // Second batch (due to size overage).
    TF_ASSERT_OK(ScheduleTask(3 /* (3+5) + 3 > 10 */, queue.get()));
    TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    TF_ASSERT_OK(ScheduleTask(6, queue.get()));

    // (Empty third batch, since the second batch exactly hit the size limit,
    // which should never get sent to the callback.)
  }

  // Expect a certain grouping of the tasks into batches.
  ASSERT_EQ(2, callback_data.size());
  ASSERT_TRUE((callback_data[0].size() == 2 && callback_data[1].size() == 3) ||
              (callback_data[0].size() == 3 && callback_data[1].size() == 2));
  const std::vector<size_t>& callback_data_a =
      callback_data[0].size() == 2 ? callback_data[0] : callback_data[1];
  const std::vector<size_t>& callback_data_b =
      callback_data[0].size() == 2 ? callback_data[1] : callback_data[0];
  EXPECT_EQ((std::vector<size_t>{3, 5}), callback_data_a);
  EXPECT_EQ((std::vector<size_t>{3, 1, 6}), callback_data_b);
}

TEST(SharedBatchSchedulerTest, ObeysTimeout) {
  // Set up a fake clock, which only advances when we explicitly tell it to.
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  {
    Notification first_batch_processed, second_batch_processed,
        third_batch_processed;
    auto callback =
        [&first_batch_processed, &second_batch_processed,
         &third_batch_processed](std::unique_ptr<Batch<FakeTask>> batch) {
          ASSERT_TRUE(batch->IsClosed());
          if (batch->size() == 1) {
            first_batch_processed.Notify();
          } else if (batch->size() == 2) {
            second_batch_processed.Notify();
          } else if (batch->size() == 3) {
            third_batch_processed.Notify();
          } else {
            EXPECT_TRUE(false) << "Unexpected batch size";
          }
        };

    SharedBatchScheduler<FakeTask>::Options options;
    options.num_batch_threads = 1;
    options.env = &env;
    std::shared_ptr<SharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(SharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    SharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    queue_options.input_batch_size_limit = 4;
    queue_options.batch_timeout_micros = 10;
    queue_options.max_enqueued_batches = 2;
    std::unique_ptr<BatchScheduler<FakeTask>> queue;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, callback, &queue));

    // Create an underfull batch, and ensure that it gets processed when the
    // clock hits the timeout.
    TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    env.AdvanceByMicroseconds(9);
    Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
    EXPECT_FALSE(first_batch_processed.HasBeenNotified());
    env.AdvanceByMicroseconds(1);
    first_batch_processed.WaitForNotification();

    // Start creating a batch, while leaving the clock well below the timeout.
    // Then submit a new task that overflows into the next batch, causing
    // the original batch to close.
    TF_ASSERT_OK(ScheduleTask(2, queue.get()));
    Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
    EXPECT_FALSE(second_batch_processed.HasBeenNotified());
    TF_ASSERT_OK(ScheduleTask(3, queue.get()));
    second_batch_processed.WaitForNotification();

    // Allow the third batch to hit its timeout, and ensure it gets closed at
    // the right time.
    env.AdvanceByMicroseconds(9);
    Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
    EXPECT_FALSE(third_batch_processed.HasBeenNotified());
    env.AdvanceByMicroseconds(1);
    third_batch_processed.WaitForNotification();

    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(SharedBatchSchedulerTest, ObeysTimeoutWithRealClock) {
  Notification first_batch_processed, second_batch_processed;
  auto callback = [&first_batch_processed, &second_batch_processed](
                      std::unique_ptr<Batch<FakeTask>> batch) {
    ASSERT_TRUE(batch->IsClosed());
    if (batch->size() == 1) {
      first_batch_processed.Notify();
    } else if (batch->size() == 2) {
      second_batch_processed.Notify();
    } else {
      EXPECT_TRUE(false) << "Unexpected batch size";
    }
  };

  SharedBatchScheduler<FakeTask>::Options options;
  options.num_batch_threads = 2;
  std::shared_ptr<SharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(SharedBatchScheduler<FakeTask>::Create(options, &scheduler));
  SharedBatchScheduler<FakeTask>::QueueOptions queue_options;
  queue_options.input_batch_size_limit = 10;
  queue_options.batch_timeout_micros = 100 * 1000;  // 100 milliseconds
  queue_options.max_enqueued_batches = 2;
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_ASSERT_OK(scheduler->AddQueue(queue_options, callback, &queue));

  // Submit a single task that doesn't fill up the batch.
  // Ensure that it gets processed due to the timeout.
  TF_ASSERT_OK(ScheduleTask(1, queue.get()));
  first_batch_processed.WaitForNotification();

  // Do it again.
  TF_ASSERT_OK(ScheduleTask(2, queue.get()));
  second_batch_processed.WaitForNotification();
}

TEST(SharedBatchSchedulerTest,
     WithZeroTimeoutBatchesScheduledAsSoonAsThreadIsAvailable) {
  // Set up a fake clock, and never advance the time.
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  {
    Notification first_batch_processed, second_batch_processed;
    auto callback = [&first_batch_processed, &second_batch_processed](
                        std::unique_ptr<Batch<FakeTask>> batch) {
      ASSERT_TRUE(batch->IsClosed());
      if (batch->size() == 1) {
        first_batch_processed.Notify();
      } else if (batch->size() == 2) {
        second_batch_processed.Notify();
      } else {
        EXPECT_TRUE(false) << "Unexpected batch size";
      }
    };

    SharedBatchScheduler<FakeTask>::Options options;
    options.num_batch_threads = 2;
    options.env = &env;
    std::shared_ptr<SharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(SharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    SharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    // Set a large batch size, so that we don't hit the batch size limit.
    queue_options.input_batch_size_limit = 100;
    // Process a batch as soon as a thread is available.
    queue_options.batch_timeout_micros = 0;
    queue_options.max_enqueued_batches = 2;
    std::unique_ptr<BatchScheduler<FakeTask>> queue;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, callback, &queue));

    TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    first_batch_processed.WaitForNotification();
    TF_ASSERT_OK(ScheduleTask(2, queue.get()));
    second_batch_processed.WaitForNotification();

    // Shut everything down.
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(SharedBatchSchedulerTest, Fairness) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  {
    Notification queue_0_first_batch_scheduled, queue_0_first_batch_proceed,
        queue_0_second_batch_scheduled;
    auto queue_0_callback = [&queue_0_first_batch_scheduled,
                             &queue_0_first_batch_proceed,
                             &queue_0_second_batch_scheduled](
                                std::unique_ptr<Batch<FakeTask>> batch) {
      if (!queue_0_first_batch_scheduled.HasBeenNotified()) {
        queue_0_first_batch_scheduled.Notify();
        queue_0_first_batch_proceed.WaitForNotification();
      } else if (!queue_0_second_batch_scheduled.HasBeenNotified()) {
        queue_0_second_batch_scheduled.Notify();
      }
    };

    Notification queue_1_first_batch_scheduled, queue_1_first_batch_proceed;
    auto queue_1_callback =
        [&queue_1_first_batch_scheduled,
         &queue_1_first_batch_proceed](std::unique_ptr<Batch<FakeTask>> batch) {
          queue_1_first_batch_scheduled.Notify();
          queue_1_first_batch_proceed.WaitForNotification();
        };

    SharedBatchScheduler<FakeTask>::Options options;
    options.num_batch_threads = 1;
    options.env = &env;
    std::shared_ptr<SharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(SharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    SharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    queue_options.input_batch_size_limit = 10;
    queue_options.batch_timeout_micros = 1;
    queue_options.max_enqueued_batches = 100 /* give plenty of room */;
    std::vector<std::unique_ptr<BatchScheduler<FakeTask>>> queues(2);
    TF_ASSERT_OK(
        scheduler->AddQueue(queue_options, queue_0_callback, &queues[0]));
    TF_ASSERT_OK(
        scheduler->AddQueue(queue_options, queue_1_callback, &queues[1]));

    // Enqueue a batch-filling task to queue 0, and wait for it to get
    // scheduled.
    TF_ASSERT_OK(ScheduleTask(10, queues[0].get()));
    env.AdvanceByMicroseconds(1);
    queue_0_first_batch_scheduled.WaitForNotification();

    // Enqueue two more batch-filling tasks to queue 0.
    TF_ASSERT_OK(ScheduleTask(10, queues[0].get()));
    TF_ASSERT_OK(ScheduleTask(10, queues[0].get()));

    // Enqueue one task to queue 1, and then advance the clock so it becomes
    // eligible for scheduling due to the timeout. Ensure that the queue 1 batch
    // gets scheduled before the next queue 0 one.
    TF_ASSERT_OK(ScheduleTask(1, queues[1].get()));
    env.AdvanceByMicroseconds(1);
    queue_0_first_batch_proceed.Notify();
    queue_1_first_batch_scheduled.WaitForNotification();
    Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
    EXPECT_FALSE(queue_0_second_batch_scheduled.HasBeenNotified());

    // Shut everything down.
    queue_1_first_batch_proceed.Notify();
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST(SharedBatchSchedulerTest, ConstMethods) {
  for (const int max_enqueued_batches : {1, 2, 5}) {
    Notification processing, proceed;
    auto callback = [&processing,
                     &proceed](std::unique_ptr<Batch<FakeTask>> batch) {
      if (!processing.HasBeenNotified()) {
        processing.Notify();
      }
      proceed.WaitForNotification();
    };

    SharedBatchScheduler<FakeTask>::Options options;
    options.num_batch_threads = 1;
    std::shared_ptr<SharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(SharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    SharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    queue_options.input_batch_size_limit = 2;
    queue_options.batch_timeout_micros = 0;
    queue_options.max_enqueued_batches = max_enqueued_batches;
    std::unique_ptr<BatchScheduler<FakeTask>> queue;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, callback, &queue));
    EXPECT_EQ(2, queue->max_task_size());
    EXPECT_EQ(0, queue->NumEnqueuedTasks());
    EXPECT_EQ(max_enqueued_batches * 2, queue->SchedulingCapacity());

    // Get one batch going on the thread, and keep the thread blocked until
    // we're done testing the maximum queue length.
    TF_ASSERT_OK(ScheduleTask(2, queue.get()));
    processing.WaitForNotification();
    EXPECT_EQ(0, queue->NumEnqueuedTasks());

    // We should be able to enqueue 'max_enqueued_batches'*2 tasks without
    // issue.
    for (int i = 0; i < max_enqueued_batches; ++i) {
      EXPECT_EQ(i * 2, queue->NumEnqueuedTasks());
      EXPECT_EQ((max_enqueued_batches - i) * 2, queue->SchedulingCapacity());
      TF_ASSERT_OK(ScheduleTask(1, queue.get()));
      EXPECT_EQ((i * 2) + 1, queue->NumEnqueuedTasks());
      EXPECT_EQ((max_enqueued_batches - i) * 2 - 1,
                queue->SchedulingCapacity());
      TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    }
    EXPECT_EQ(max_enqueued_batches * 2, queue->NumEnqueuedTasks());
    EXPECT_EQ(0, queue->SchedulingCapacity());

    // Attempting to enqueue one more task should yield an UNAVAILABLE error.
    Status status = ScheduleTask(1, queue.get());
    ASSERT_FALSE(status.ok());
    EXPECT_EQ(error::UNAVAILABLE, status.code());
    EXPECT_EQ(max_enqueued_batches * 2, queue->NumEnqueuedTasks());
    EXPECT_EQ(0, queue->SchedulingCapacity());

    proceed.Notify();
  }
}

TEST(SharedBatchSchedulerTest, OneFullQueueDoesntBlockOtherQueues) {
  Notification queue_0_processing, queue_0_proceed;
  auto queue_0_callback = [&queue_0_processing, &queue_0_proceed](
                              std::unique_ptr<Batch<FakeTask>> batch) {
    if (!queue_0_processing.HasBeenNotified()) {
      queue_0_processing.Notify();
      queue_0_proceed.WaitForNotification();
    }
  };

  Notification queue_1_first_batch_processed, queue_1_second_batch_processed,
      queue_1_third_batch_processed;
  auto queue_1_callback =
      [&queue_1_first_batch_processed, &queue_1_second_batch_processed,
       &queue_1_third_batch_processed](std::unique_ptr<Batch<FakeTask>> batch) {
        if (batch->size() == 1) {
          queue_1_first_batch_processed.Notify();
        } else if (batch->size() == 2) {
          queue_1_second_batch_processed.Notify();
        } else if (batch->size() == 3) {
          queue_1_third_batch_processed.Notify();
        } else {
          EXPECT_TRUE(false) << "Unexpected batch size";
        }
      };

  SharedBatchScheduler<FakeTask>::Options options;
  options.num_batch_threads = 2;
  std::shared_ptr<SharedBatchScheduler<FakeTask>> scheduler;
  TF_ASSERT_OK(SharedBatchScheduler<FakeTask>::Create(options, &scheduler));
  SharedBatchScheduler<FakeTask>::QueueOptions queue_options;
  queue_options.input_batch_size_limit = 10;
  queue_options.batch_timeout_micros = 0;
  queue_options.max_enqueued_batches = 2;
  std::unique_ptr<BatchScheduler<FakeTask>> queue_0;
  TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_0_callback, &queue_0));
  std::unique_ptr<BatchScheduler<FakeTask>> queue_1;
  TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_1_callback, &queue_1));

  // Clog up queue 0.
  TF_ASSERT_OK(ScheduleTask(1, queue_0.get()));
  queue_0_processing.WaitForNotification();
  Status queue_0_status;
  do {
    queue_0_status = ScheduleTask(1, queue_0.get());
  } while (queue_0_status.ok());
  EXPECT_EQ(error::UNAVAILABLE, queue_0_status.code());

  // Ensure that queue 1 still behaves normally, and lets us process tasks.
  TF_ASSERT_OK(ScheduleTask(1, queue_1.get()));
  queue_1_first_batch_processed.WaitForNotification();
  TF_ASSERT_OK(ScheduleTask(2, queue_1.get()));
  queue_1_second_batch_processed.WaitForNotification();
  TF_ASSERT_OK(ScheduleTask(3, queue_1.get()));
  queue_1_third_batch_processed.WaitForNotification();

  // Let poor queue 0 drain.
  queue_0_proceed.Notify();
}

TEST(SharedBatchSchedulerTest, QueueDestructorBlocksUntilAllTasksProcessed) {
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  {
    int current_batch = 0;
    Notification first_callback_started;
    const int kMaxEnqueuedBatches = 3;
    std::vector<Notification> callback_proceed(kMaxEnqueuedBatches);
    auto callback =
        [&current_batch, &first_callback_started,
         &callback_proceed](std::unique_ptr<Batch<FakeTask>> batch) {
          if (current_batch == 0) {
            first_callback_started.Notify();
          }
          callback_proceed[current_batch].WaitForNotification();
          ++current_batch;
        };

    SharedBatchScheduler<FakeTask>::Options options;
    options.num_batch_threads = 1;
    options.env = &env;
    std::shared_ptr<SharedBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(SharedBatchScheduler<FakeTask>::Create(options, &scheduler));
    SharedBatchScheduler<FakeTask>::QueueOptions queue_options;
    queue_options.input_batch_size_limit = 10;
    queue_options.batch_timeout_micros = 0;
    queue_options.max_enqueued_batches = 2;
    std::unique_ptr<BatchScheduler<FakeTask>> queue;
    TF_ASSERT_OK(scheduler->AddQueue(queue_options, callback, &queue));

    // Clog up the queue.
    int num_enqueued_batches = 0;
    TF_ASSERT_OK(ScheduleTask(10, queue.get()));
    ++num_enqueued_batches;
    env.AdvanceByMicroseconds(1);
    first_callback_started.WaitForNotification();
    for (int i = 0; i < 2; ++i) {
      TF_ASSERT_OK(ScheduleTask(10, queue.get()));
      ++num_enqueued_batches;
    }
    EXPECT_EQ(kMaxEnqueuedBatches, num_enqueued_batches);
    EXPECT_EQ(error::UNAVAILABLE, ScheduleTask(10, queue.get()).code());

    // Destroy the queue. The destructor should block until all tasks have been
    // processed.
    Notification destroy_queue_thread_started, queue_destroyed;
    std::unique_ptr<Thread> destroy_queue_thread(Env::Default()->StartThread(
        {}, "DestroyQueueThread",
        [&queue, &destroy_queue_thread_started, &queue_destroyed] {
          destroy_queue_thread_started.Notify();
          queue = nullptr;
          queue_destroyed.Notify();
        }));
    destroy_queue_thread_started.WaitForNotification();
    for (int i = 0; i < num_enqueued_batches; ++i) {
      Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
      EXPECT_FALSE(queue_destroyed.HasBeenNotified());
      callback_proceed[i].Notify();
    }

    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
