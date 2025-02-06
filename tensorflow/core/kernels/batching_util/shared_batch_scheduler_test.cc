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

#include <cstddef>
#include <memory>
#include <string>
#include <thread>  // NOLINT(build/c++11)
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "absl/base/call_once.h"
#include "absl/container/fixed_array.h"
#include "absl/status/status.h"
#include "absl/time/time.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/kernels/batching_util/batch_scheduler_utils.h"
#include "tensorflow/core/kernels/batching_util/fake_clock_env.h"
#include "tensorflow/core/kernels/batching_util/input_split_metadata.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tensorflow/core/platform/test_benchmark.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tsl/platform/criticality.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::HasSubstr;

class FakeTask : public BatchTask {
 public:
  explicit FakeTask(size_t size, tsl::criticality::Criticality criticality =
                                     tsl::criticality::Criticality::kCritical)
      : size_(size), criticality_(criticality) {}

  ~FakeTask() override = default;

  size_t size() const override { return size_; }

  tsl::criticality::Criticality criticality() const override {
    return criticality_;
  }

 private:
  const size_t size_;
  const tsl::criticality::Criticality criticality_;

  FakeTask(const FakeTask&) = delete;
  void operator=(const FakeTask&) = delete;
};

// Fake task that doesn't inherit BatchTask and doesn't define criticality. The
// shared batch scheduler should still work with this task.
class FakeTaskWithoutCriticality {
 public:
  explicit FakeTaskWithoutCriticality(size_t size) : size_(size) {}

  ~FakeTaskWithoutCriticality() = default;

  size_t size() const { return size_; }

 private:
  const size_t size_;

  FakeTaskWithoutCriticality(const FakeTaskWithoutCriticality&) = delete;
  void operator=(const FakeTaskWithoutCriticality&) = delete;
};

using Queue = BatchScheduler<FakeTask>;
using Scheduler = SharedBatchScheduler<FakeTask>;
using QueueOptions = Scheduler::QueueOptions;
using SplitFunc = std::function<absl::Status(
    std::unique_ptr<FakeTask>* input_task, int first_output_task_size,
    int input_batch_size_limit,
    std::vector<std::unique_ptr<FakeTask>>* output_tasks)>;

// Creates a FakeTask of size 'task_size' and 'criticality', and calls
// 'scheduler->Schedule()' on that task. Returns the resulting status.
// 'criticality' defaults to kCritical.
absl::Status ScheduleTask(size_t task_size, BatchScheduler<FakeTask>* scheduler,
                          tsl::criticality::Criticality criticality =
                              tsl::criticality::Criticality::kCritical) {
  std::unique_ptr<FakeTask> task(new FakeTask(task_size, criticality));
  absl::Status status = scheduler->Schedule(&task);
  // Schedule() should have consumed 'task' iff it returned Status::OK.
  CHECK_EQ(status.ok(), task == nullptr);
  return status;
}

// Helper function similar to the function above. Creates a FakeTask of size
// 'task_size' and calls 'scheduler->Schedule()' on that task. Returns the
// resulting status.
absl::Status ScheduleTaskWithoutCriticality(
    size_t task_size, BatchScheduler<FakeTaskWithoutCriticality>* scheduler) {
  std::unique_ptr<FakeTaskWithoutCriticality> task(
      new FakeTaskWithoutCriticality(task_size));
  absl::Status status = scheduler->Schedule(&task);
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

// Creates a shared-batch-scheduler.
std::shared_ptr<Scheduler> CreateSharedBatchScheduler(
    int num_batch_threads, Env* env = Env::Default(), bool rank_queues = false,
    bool use_global_scheduler = false) {
  Scheduler::Options options;
  options.num_batch_threads = num_batch_threads;
  options.env = env;
  options.rank_queues = rank_queues;
  options.use_global_scheduler = use_global_scheduler;

  std::shared_ptr<Scheduler> shared_batch_scheduler;
  TF_CHECK_OK(Scheduler::Create(options, &shared_batch_scheduler));

  return shared_batch_scheduler;
}

// Creates a queue with the given `queue_options`.
//
// Caller takes ownership of returned queue.
std::unique_ptr<Queue> CreateQueue(
    std::shared_ptr<Scheduler> scheduler, Scheduler::QueueOptions queue_options,
    internal::Queue<FakeTask>::ProcessBatchCallback process_batch_callback) {
  std::unique_ptr<BatchScheduler<FakeTask>> queue;
  TF_CHECK_OK(
      scheduler->AddQueue(queue_options, process_batch_callback, &queue));
  return queue;
}

// Creates QueueOptions based on input parameters.
QueueOptions CreateQueueOptions(size_t max_execution_batch_size,
                                size_t input_batch_size_limit,
                                size_t batch_timeout_micros,
                                size_t max_enqueued_batches,
                                bool enable_large_batch_splitting,
                                SplitFunc split_func,
                                bool enable_priority_queue = false) {
  QueueOptions queue_options;
  queue_options.max_enqueued_batches = max_enqueued_batches;
  queue_options.max_execution_batch_size = max_execution_batch_size;
  queue_options.input_batch_size_limit = input_batch_size_limit;
  queue_options.batch_timeout_micros = batch_timeout_micros;
  queue_options.enable_large_batch_splitting = enable_large_batch_splitting;
  queue_options.enable_priority_queue = enable_priority_queue;
  if (enable_large_batch_splitting) {
    queue_options.split_input_task_func = split_func;
  }
  return queue_options;
}

class SharedBatchSchedulerTestBase {
 public:
  SharedBatchSchedulerTestBase() = default;
  virtual ~SharedBatchSchedulerTestBase() = default;

 protected:
  QueueOptions CreateQueueOptions(size_t max_execution_batch_size,
                                  size_t input_batch_size_limit,
                                  size_t batch_timeout_micros,
                                  size_t max_enqueued_batches,
                                  bool enable_priority_queue = false) {
    return tensorflow::serving::CreateQueueOptions(
        max_execution_batch_size, input_batch_size_limit, batch_timeout_micros,
        max_enqueued_batches, enable_input_batch_split(), get_split_func(),
        enable_priority_queue);
  }
  virtual bool enable_input_batch_split() const = 0;

  SplitFunc get_split_func() const {
    if (enable_input_batch_split()) {
      return
          [](std::unique_ptr<FakeTask>* input_task,
             int open_batch_remaining_slot, int max_batch_size,
             std::vector<std::unique_ptr<FakeTask>>* output_tasks)
              -> absl::Status {
            std::unique_ptr<FakeTask> owned_input_task = std::move(*input_task);
            const int input_task_size = owned_input_task->size();

            const internal::InputSplitMetadata input_split_metadata(
                input_task_size, open_batch_remaining_slot, max_batch_size);

            const absl::FixedArray<int> task_sizes =
                input_split_metadata.task_sizes();
            const int num_batches = task_sizes.size();

            output_tasks->resize(num_batches);
            for (int i = 0; i < num_batches; i++) {
              (*output_tasks)[i] = std::make_unique<FakeTask>(task_sizes[i]);
            }

            return absl::OkStatus();
          };
    }
    return nullptr;
  }
};

class SharedBatchSchedulerTest : public ::testing::TestWithParam<bool>,
                                 public SharedBatchSchedulerTestBase {
 protected:
  bool enable_input_batch_split() const override { return GetParam(); }
};

TEST_P(SharedBatchSchedulerTest, Basic) {
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
          auto scheduler = CreateSharedBatchScheduler(num_batch_threads);

          // Create two queues.

          const size_t input_batch_size_limit = 10;
          const size_t batch_timeout_micros = 1 * 1000 * 1000;  // 1 second
          const size_t max_enqueued_batches = 2;
          const auto queue_options =
              CreateQueueOptions(input_batch_size_limit, input_batch_size_limit,
                                 batch_timeout_micros, max_enqueued_batches);
          auto queue_0 =
              CreateQueue(scheduler, queue_options, queue_0_callback);

          auto queue_1 =
              CreateQueue(scheduler, queue_options, queue_1_callback);

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

TEST_P(SharedBatchSchedulerTest,
       CallbackWithTaskVectorOkWithPriorityQueueEnabled) {
  bool queue_0_callback_called = false;
  auto queue_0_callback = [&queue_0_callback_called](
                              std::unique_ptr<Batch<FakeTask>> batch,
                              std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_0_callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(3, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(3, batch->task(1).size());
    EXPECT_EQ(5, batch->task(2).size());
    EXPECT_EQ(0, tasks.size());
  };
  bool queue_1_callback_called = false;
  auto queue_1_callback = [&queue_1_callback_called](
                              std::unique_ptr<Batch<FakeTask>> batch,
                              std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_1_callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(2, batch->task(0).size());
    EXPECT_EQ(4, batch->task(1).size());
    EXPECT_EQ(0, tasks.size());
  };
  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/3);

    // Create two queues.
    const QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    std::unique_ptr<Queue> queue_0 =
        CreateQueue(scheduler, queue_options, queue_0_callback);
    std::unique_ptr<Queue> queue_1 =
        CreateQueue(scheduler, queue_options, queue_1_callback);

    // Submit tasks to the two queues.
    TF_ASSERT_OK(ScheduleTask(1, queue_0.get()));
    TF_ASSERT_OK(ScheduleTask(2, queue_1.get()));
    TF_ASSERT_OK(ScheduleTask(3, queue_0.get()));
    TF_ASSERT_OK(ScheduleTask(4, queue_1.get()));
    TF_ASSERT_OK(ScheduleTask(5, queue_0.get()));
  }
  EXPECT_TRUE(queue_0_callback_called);
  EXPECT_TRUE(queue_1_callback_called);
}

TEST_P(SharedBatchSchedulerTest,
       CallbackWithTaskVectorOkWithPriorityQueueDisabled) {
  bool queue_0_callback_called = false;
  auto queue_0_callback = [&queue_0_callback_called](
                              std::unique_ptr<Batch<FakeTask>> batch,
                              std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_0_callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(3, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(3, batch->task(1).size());
    EXPECT_EQ(5, batch->task(2).size());
    EXPECT_EQ(0, tasks.size());
  };
  bool queue_1_callback_called = false;
  auto queue_1_callback = [&queue_1_callback_called](
                              std::unique_ptr<Batch<FakeTask>> batch,
                              std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_1_callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(2, batch->task(0).size());
    EXPECT_EQ(4, batch->task(1).size());
    EXPECT_EQ(0, tasks.size());
  };
  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/3);

    // Create two queues.
    const QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/false);
    std::unique_ptr<Queue> queue_0 =
        CreateQueue(scheduler, queue_options, queue_0_callback);
    std::unique_ptr<Queue> queue_1 =
        CreateQueue(scheduler, queue_options, queue_1_callback);

    // Submit tasks to the two queues.
    TF_ASSERT_OK(ScheduleTask(1, queue_0.get()));
    TF_ASSERT_OK(ScheduleTask(2, queue_1.get()));
    TF_ASSERT_OK(ScheduleTask(3, queue_0.get()));
    TF_ASSERT_OK(ScheduleTask(4, queue_1.get()));
    TF_ASSERT_OK(ScheduleTask(5, queue_0.get()));
  }
  EXPECT_TRUE(queue_0_callback_called);
  EXPECT_TRUE(queue_1_callback_called);
}

// The task in the shared batch scheduler template parameter does not define
// criticality priority queue. It should work as if the priority queue is
// disabled.
TEST_P(
    SharedBatchSchedulerTest,
    CallbackWithTaskVectorOkWithPriorityQueueEnabledWithCriticalitylessTask) {
  bool queue_0_callback_called = false;
  auto queue_0_callback =
      [&queue_0_callback_called](
          std::unique_ptr<Batch<FakeTaskWithoutCriticality>> batch,
          std::vector<std::unique_ptr<FakeTaskWithoutCriticality>> tasks) {
        queue_0_callback_called = true;
        ASSERT_TRUE(batch->IsClosed());
        ASSERT_EQ(3, batch->num_tasks());
        EXPECT_EQ(1, batch->task(0).size());
        EXPECT_EQ(3, batch->task(1).size());
        EXPECT_EQ(5, batch->task(2).size());
        EXPECT_EQ(0, tasks.size());
      };
  bool queue_1_callback_called = false;
  auto queue_1_callback =
      [&queue_1_callback_called](
          std::unique_ptr<Batch<FakeTaskWithoutCriticality>> batch,
          std::vector<std::unique_ptr<FakeTaskWithoutCriticality>> tasks) {
        queue_1_callback_called = true;
        ASSERT_TRUE(batch->IsClosed());
        ASSERT_EQ(2, batch->num_tasks());
        EXPECT_EQ(2, batch->task(0).size());
        EXPECT_EQ(4, batch->task(1).size());
        EXPECT_EQ(0, tasks.size());
      };
  {
    SharedBatchScheduler<FakeTaskWithoutCriticality>::Options options;
    options.num_batch_threads = 3;
    options.env = Env::Default();

    std::shared_ptr<SharedBatchScheduler<FakeTaskWithoutCriticality>>
        shared_batch_scheduler;
    TF_CHECK_OK(SharedBatchScheduler<FakeTaskWithoutCriticality>::Create(
        options, &shared_batch_scheduler));

    // Create two queues.

    SharedBatchScheduler<FakeTaskWithoutCriticality>::QueueOptions
        queue_options;
    queue_options.input_batch_size_limit = 10;
    queue_options.batch_timeout_micros = 1000 * 1000;
    queue_options.max_enqueued_batches = 2;
    queue_options.enable_large_batch_splitting = enable_input_batch_split();
    queue_options.split_input_task_func =
        [](std::unique_ptr<FakeTaskWithoutCriticality>* input_task,
           int open_batch_remaining_slot, int max_batch_size,
           std::vector<std::unique_ptr<FakeTaskWithoutCriticality>>*
               output_tasks) -> absl::Status {
      std::unique_ptr<FakeTaskWithoutCriticality> owned_input_task =
          std::move(*input_task);
      const int input_task_size = owned_input_task->size();

      const internal::InputSplitMetadata input_split_metadata(
          input_task_size, open_batch_remaining_slot, max_batch_size);

      const absl::FixedArray<int> task_sizes =
          input_split_metadata.task_sizes();
      const int num_batches = task_sizes.size();

      output_tasks->resize(num_batches);
      for (int i = 0; i < num_batches; i++) {
        (*output_tasks)[i] =
            std::make_unique<FakeTaskWithoutCriticality>(task_sizes[i]);
      }

      return absl::OkStatus();
    };
    queue_options.max_execution_batch_size = 10;
    queue_options.enable_priority_queue = true;

    std::unique_ptr<BatchScheduler<FakeTaskWithoutCriticality>> queue_0;
    TF_CHECK_OK(shared_batch_scheduler->AddQueue(queue_options,
                                                 queue_0_callback, &queue_0));
    std::unique_ptr<BatchScheduler<FakeTaskWithoutCriticality>> queue_1;
    TF_CHECK_OK(shared_batch_scheduler->AddQueue(queue_options,
                                                 queue_1_callback, &queue_1));

    // Submit tasks to the two queues.
    TF_ASSERT_OK(ScheduleTaskWithoutCriticality(1, queue_0.get()));
    TF_ASSERT_OK(ScheduleTaskWithoutCriticality(2, queue_1.get()));
    TF_ASSERT_OK(ScheduleTaskWithoutCriticality(3, queue_0.get()));
    TF_ASSERT_OK(ScheduleTaskWithoutCriticality(4, queue_1.get()));
    TF_ASSERT_OK(ScheduleTaskWithoutCriticality(5, queue_0.get()));
  }
  EXPECT_TRUE(queue_0_callback_called);
  EXPECT_TRUE(queue_1_callback_called);
}

TEST_P(SharedBatchSchedulerTest, ObeyBatchSizeConstraint) {
  // Set up a fake clock, which only advances when we explicitly tell it to.
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);
  // Set up a callback that captures the batches' task sizes.
  mutex mu;
  std::vector<std::vector<size_t>> callback_data;
  Notification all_batches_processed;
  auto callback = [&mu, &callback_data, &all_batches_processed](
                      std::unique_ptr<Batch<FakeTask>> batch) {
    ASSERT_TRUE(batch->IsClosed());
    std::vector<size_t> batch_data;
    batch_data.reserve(batch->num_tasks());
    for (int i = 0; i < batch->num_tasks(); ++i) {
      batch_data.push_back(batch->mutable_task(i)->size());
    }
    {
      mutex_lock l(mu);
      callback_data.push_back(batch_data);
      if (callback_data.size() == 2) {
        all_batches_processed.Notify();
      }
    }
  };

  // Run a batch scheduler and inject some tasks.
  {
    auto scheduler = CreateSharedBatchScheduler(/*num_batch_threads=*/2, &env);

    const size_t input_batch_size_limit = 10;
    const size_t batch_timeout_micros = 10 * 1000;  // 10 milli-seconds
    const size_t max_enqueued_batches = 2;
    auto queue = CreateQueue(
        scheduler,
        CreateQueueOptions(input_batch_size_limit, input_batch_size_limit,
                           batch_timeout_micros, max_enqueued_batches),
        callback);

    if (enable_input_batch_split()) {
      // First batch.
      TF_ASSERT_OK(ScheduleTask(3, queue.get()));
      TF_ASSERT_OK(ScheduleTask(5, queue.get()));

      // Second batch
      // Task spans over first batch and second batch, so contributes two tasks.
      TF_ASSERT_OK(ScheduleTask(3 /* (3+5) + 3 > 10 */, queue.get()));
      TF_ASSERT_OK(ScheduleTask(1, queue.get()));
      TF_ASSERT_OK(ScheduleTask(6, queue.get()));
      TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    } else {
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

    // Advance clock to trigger batch processing.
    env.AdvanceByMicroseconds(20 * 1000);
    all_batches_processed.WaitForNotification();
    // Expect a certain grouping of the tasks into batches.
    if (enable_input_batch_split()) {
      EXPECT_THAT(
          callback_data,
          ::testing::UnorderedElementsAreArray(std::vector<std::vector<size_t>>{
              std::vector<size_t>{3, 5, 2}, std::vector<size_t>{1, 1, 6, 1}}));
    } else {
      EXPECT_THAT(callback_data,
                  ::testing::UnorderedElementsAreArray(
                      std::vector<std::vector<size_t>>{{3, 5}, {3, 1, 6}}));
    }
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerTest, ObeysTimeout) {
  // Set up a fake clock, which only advances when we explicitly tell it to.
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  {
    Notification first_batch_processed, second_batch_processed,
        third_batch_processed;
    bool notify_first_batch = false, notify_second_batch = false,
         notify_third_batch = false;
    auto callback = [&](std::unique_ptr<Batch<FakeTask>> batch) {
      ASSERT_TRUE(batch->IsClosed());
      if (notify_first_batch && (!first_batch_processed.HasBeenNotified())) {
        first_batch_processed.Notify();
        return;
      }
      if (notify_second_batch && (!second_batch_processed.HasBeenNotified())) {
        second_batch_processed.Notify();
        return;
      }
      if (notify_third_batch && (!third_batch_processed.HasBeenNotified())) {
        third_batch_processed.Notify();
        return;
      }

      EXPECT_TRUE(false) << "Unexpected condition";
    };

    auto scheduler = CreateSharedBatchScheduler(1, &env);

    const size_t input_batch_size_limit = 4;
    const size_t batch_timeout_micros = 10;
    const size_t max_enqueued_batches = 2;
    QueueOptions options =
        CreateQueueOptions(input_batch_size_limit, input_batch_size_limit,
                           batch_timeout_micros, max_enqueued_batches);
    auto queue = CreateQueue(scheduler, options, callback);

    // Create an underfull batch, and ensure that it gets processed when the
    // clock hits the timeout.
    TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    env.AdvanceByMicroseconds(9);
    Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
    EXPECT_FALSE(first_batch_processed.HasBeenNotified());
    notify_first_batch = true;
    env.AdvanceByMicroseconds(1);
    first_batch_processed.WaitForNotification();

    // Start creating a batch, while leaving the clock well below the timeout.
    // Then submit a new task that overflows into the next batch, causing
    // the original batch to close.
    TF_ASSERT_OK(ScheduleTask(2, queue.get()));
    Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
    EXPECT_FALSE(second_batch_processed.HasBeenNotified());
    notify_second_batch = true;
    TF_ASSERT_OK(ScheduleTask(3, queue.get()));
    second_batch_processed.WaitForNotification();

    // Allow the third batch to hit its timeout, and ensure it gets closed at
    // the right time.
    env.AdvanceByMicroseconds(9);
    Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
    EXPECT_FALSE(third_batch_processed.HasBeenNotified());
    notify_third_batch = true;
    env.AdvanceByMicroseconds(1);
    third_batch_processed.WaitForNotification();

    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerTest, ObeysTimeoutWithRealClock) {
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

  auto scheduler = CreateSharedBatchScheduler(2);

  const size_t input_batch_size_limit = 10;
  const size_t batch_timeout_micros = 100 * 1000;  // 100 milliseconds
  const size_t max_enqueued_batches = 2;
  auto queue = CreateQueue(
      scheduler,
      CreateQueueOptions(input_batch_size_limit, input_batch_size_limit,
                         batch_timeout_micros, max_enqueued_batches),
      callback);

  // Submit a single task that doesn't fill up the batch.
  // Ensure that it gets processed due to the timeout.
  TF_ASSERT_OK(ScheduleTask(1, queue.get()));
  first_batch_processed.WaitForNotification();

  // Do it again.
  TF_ASSERT_OK(ScheduleTask(2, queue.get()));
  second_batch_processed.WaitForNotification();
}

TEST_P(SharedBatchSchedulerTest,
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

    auto scheduler = CreateSharedBatchScheduler(2, &env);

    // Set a large batch size, so that we don't hit the batch size limit.
    const size_t batch_size_limit = 100;
    // Process a batch as soon as a thread is available.
    const size_t batch_timeout_micros = 0;
    const size_t max_enqueued_batches = 2;
    auto queue = CreateQueue(
        scheduler,
        CreateQueueOptions(batch_size_limit, batch_size_limit,
                           batch_timeout_micros, max_enqueued_batches),
        callback);

    TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    first_batch_processed.WaitForNotification();
    TF_ASSERT_OK(ScheduleTask(2, queue.get()));
    second_batch_processed.WaitForNotification();

    // Shut everything down.
    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerTest, Fairness) {
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

    auto scheduler = CreateSharedBatchScheduler(1, &env);
    size_t input_batch_size_limit = 10;
    QueueOptions queue_options = CreateQueueOptions(
        input_batch_size_limit, input_batch_size_limit,
        1 /* batch_timeout_micros */, 100 /* give plenty of room */);
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

TEST_P(SharedBatchSchedulerTest, ConstMethods) {
  for (const int max_enqueued_batches : {1, 2, 5}) {
    Notification processing, proceed;
    auto callback = [&processing,
                     &proceed](std::unique_ptr<Batch<FakeTask>> batch) {
      if (!processing.HasBeenNotified()) {
        processing.Notify();
      }
      proceed.WaitForNotification();
    };

    auto scheduler = CreateSharedBatchScheduler(/*num_batch_threads*/ 1);

    const size_t input_batch_size_limit = 2;
    const size_t batch_timeout_micros = 0;
    auto queue = CreateQueue(
        scheduler,
        CreateQueueOptions(input_batch_size_limit, input_batch_size_limit,
                           batch_timeout_micros, max_enqueued_batches),
        callback);

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
    EXPECT_THAT(
        ScheduleTask(1, queue.get()),
        testing::StatusIs(error::UNAVAILABLE,
                          HasSubstr("The batch scheduling queue to which this "
                                    "task was submitted is full")));

    EXPECT_EQ(max_enqueued_batches * 2, queue->NumEnqueuedTasks());
    EXPECT_EQ(0, queue->SchedulingCapacity());

    proceed.Notify();
  }
}

TEST_P(SharedBatchSchedulerTest, OneFullQueueDoesntBlockOtherQueues) {
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

  auto scheduler = CreateSharedBatchScheduler(/*num_batch_threads*/ 2);

  const size_t input_batch_size_limit = 10;
  const size_t batch_timeout_micros = 0;
  const size_t max_enqueued_batches = 2;
  QueueOptions queue_options =
      CreateQueueOptions(input_batch_size_limit, input_batch_size_limit,
                         batch_timeout_micros, max_enqueued_batches);

  std::unique_ptr<BatchScheduler<FakeTask>> queue_0;
  TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_0_callback, &queue_0));
  std::unique_ptr<BatchScheduler<FakeTask>> queue_1;
  TF_ASSERT_OK(scheduler->AddQueue(queue_options, queue_1_callback, &queue_1));

  // Clog up queue 0.
  TF_ASSERT_OK(ScheduleTask(1, queue_0.get()));
  queue_0_processing.WaitForNotification();
  absl::Status queue_0_status;
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

TEST_P(SharedBatchSchedulerTest, QueueDestructorBlocksUntilAllTasksProcessed) {
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

    auto scheduler = CreateSharedBatchScheduler(1, &env);

    const size_t batch_size_limit = 10;
    const size_t batch_timeout_micros = 0;
    const size_t max_enqueued_batches = 2;
    QueueOptions queue_options =
        CreateQueueOptions(batch_size_limit, batch_size_limit,
                           batch_timeout_micros, max_enqueued_batches);
    auto queue = CreateQueue(scheduler, queue_options, callback);

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

// Tests that queue configured with zero `max_enqueued_batches` get one queue.
// Note, technically an invalid-argument error should be returned.
// Since existing models (with very low QPS) rely on the rewrite, retain the
// old behavior so such models continue to work.
TEST_P(SharedBatchSchedulerTest, ZeroQueueRewrittenToOneQueue) {
  auto callback = [](std::unique_ptr<Batch<FakeTask>> batch) {
    // do nothing.
  };

  auto scheduler = CreateSharedBatchScheduler(2);

  const size_t input_batch_size_limit = 10;
  const size_t batch_timeout_micros = 100 * 1000;  // 100 milliseconds
  const size_t max_enqueued_batches = 0;
  std::unique_ptr<Queue> queue;
  if (enable_input_batch_split()) {
    EXPECT_THAT(
        scheduler->AddQueue(tensorflow::serving::CreateQueueOptions(
                                input_batch_size_limit, input_batch_size_limit,
                                batch_timeout_micros, max_enqueued_batches,
                                enable_input_batch_split(), get_split_func()),
                            callback, &queue),
        testing::StatusIs(error::INVALID_ARGUMENT,
                          "max_enqueued_batches must be positive; was 0"));
  } else {
    TF_ASSERT_OK(
        scheduler->AddQueue(tensorflow::serving::CreateQueueOptions(
                                input_batch_size_limit, input_batch_size_limit,
                                batch_timeout_micros, max_enqueued_batches,
                                enable_input_batch_split(), get_split_func()),
                            callback, &queue));
    EXPECT_EQ(queue->SchedulingCapacity(), input_batch_size_limit);
  }
}

TEST_P(SharedBatchSchedulerTest, BatchPaddingPolicyBatchDown) {
  // Set up a fake clock, which only advances when we explicitly tell it to.
  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  {
    Notification first_batch_processed;
    Notification second_batch_processed;
    auto callback = [&](std::unique_ptr<Batch<FakeTask>> batch) {
      if (!first_batch_processed.HasBeenNotified()) {
        // This is the main expectation of the test.
        //
        // The scheduler should  have trimmed the batch to a smaller allowed
        // size which requires no padding.
        EXPECT_EQ(batch->size(), 2);

        first_batch_processed.Notify();
        return;
      }

      if (!second_batch_processed.HasBeenNotified()) {
        // Leftovers after the first batch.
        EXPECT_EQ(batch->size(), 1);

        second_batch_processed.Notify();
        return;
      }

      ADD_FAILURE() << "Batch callback must not be invoked more than expected";
    };

    auto scheduler = CreateSharedBatchScheduler(1, &env);

    QueueOptions options =
        CreateQueueOptions(/* max_execution_batch_size= */ 10,
                           /* input_batch_size_limit= */ 10,
                           /* batch_timeout_micros= */ 10,
                           /* max_enqueued_batches= */ 10);

    // The most interesting option for this test.
    options.allowed_batch_sizes = {1, 2, 4, 8};
    options.batch_padding_policy = kBatchDownPolicy;

    auto queue = CreateQueue(scheduler, options, callback);

    // Schedule some tasks and ensure the scheduler calls the callback after a
    // batch timeout has expired.
    TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    TF_ASSERT_OK(ScheduleTask(1, queue.get()));
    env.AdvanceByMicroseconds(options.batch_timeout_micros);
    first_batch_processed.WaitForNotification();

    // Ensure the scheduler correctly updates the starting time of the new
    // batch. We should only wait 2/3 of the batch timeout now.
    auto new_batch_timeout_micros = options.batch_timeout_micros * 2 / 3;
    env.AdvanceByMicroseconds(new_batch_timeout_micros - 1);
    EXPECT_FALSE(second_batch_processed.WaitForNotificationWithTimeout(
        absl::Milliseconds(10)));
    env.AdvanceByMicroseconds(1);
    second_batch_processed.WaitForNotification();

    start_teardown.Notify();
  }
  stop_teardown.Notify();
}

// TODO(b/161857471):
// Add test coverage when input-split and no-split returns differently.
INSTANTIATE_TEST_SUITE_P(Parameter, SharedBatchSchedulerTest,
                         ::testing::Bool());

class SharedBatchSchedulerPriorityTest
    : public ::testing::TestWithParam<
          std::tuple<bool, MixedPriorityBatchingPolicy>>,
      public SharedBatchSchedulerTestBase {
 protected:
  bool enable_input_batch_split() const override {
    return std::get<0>(GetParam());
  }

  MixedPriorityBatchingPolicy mixed_priority_batching_policy() const {
    return std::get<1>(GetParam());
  }
};

TEST_P(SharedBatchSchedulerPriorityTest,
       InvalidLowPriorityTaskWithPriorityQueueEnabled) {
  bool queue_callback_called = false;
  auto queue_callback = [&queue_callback_called](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_called = true;
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/3);

    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/100, /*input_batch_size_limit=*/100,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 1;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 1;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        mixed_priority_batching_policy();
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    EXPECT_THAT(
        ScheduleTask(10, queue.get(),
                     tsl::criticality::Criticality::kSheddablePlus),
        testing::StatusIs(
            absl::StatusCode::kUnavailable,
            HasSubstr(
                "The low priority task queue to which this task was submitted "
                "has max_execution_batch_size=1 and the task size is 10")));
  }
  EXPECT_FALSE(queue_callback_called);
}

TEST_P(SharedBatchSchedulerPriorityTest,
       InvalidLowPriorityTaskWithQueueFullWithPriorityQueueEnabledNew) {
  Notification processing, proceed;
  auto queue_callback = [&processing, &proceed](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    if (!processing.HasBeenNotified()) {
      processing.Notify();
    }
    proceed.WaitForNotification();
  };

  std::shared_ptr<Scheduler> scheduler =
      CreateSharedBatchScheduler(/*num_batch_threads=*/1);

  QueueOptions queue_options = CreateQueueOptions(
      /*max_execution_batch_size=*/100, /*input_batch_size_limit=*/100,
      /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
      /*enable_priority_queue=*/true);
  queue_options.low_priority_queue_options.max_execution_batch_size = 10;
  queue_options.low_priority_queue_options.batch_timeout_micros =
      1 * 1000 * 1000;
  queue_options.low_priority_queue_options.input_batch_size_limit = 10;
  queue_options.low_priority_queue_options.max_enqueued_batches = 2;
  queue_options.mixed_priority_batching_policy =
      mixed_priority_batching_policy();
  std::unique_ptr<Queue> queue =
      CreateQueue(scheduler, queue_options, queue_callback);

  // Schedule one task and block the thread.
  TF_ASSERT_OK(ScheduleTask(5, queue.get(),
                            tsl::criticality::Criticality::kCriticalPlus));
  processing.WaitForNotification();
  ASSERT_EQ(0, queue->NumEnqueuedTasks());

  // Adding tasks up to size 20 should be fine.
  TF_ASSERT_OK(ScheduleTask(10, queue.get(),
                            tsl::criticality::Criticality::kSheddablePlus));
  ASSERT_EQ(1, queue->NumEnqueuedTasks());
  TF_ASSERT_OK(ScheduleTask(10, queue.get(),
                            tsl::criticality::Criticality::kSheddablePlus));
  ASSERT_EQ(2, queue->NumEnqueuedTasks());

  // Adding one more task should result in an error.
  EXPECT_THAT(
      ScheduleTask(1, queue.get(),
                   tsl::criticality::Criticality::kSheddablePlus),
      testing::StatusIs(
          absl::StatusCode::kUnavailable,
          HasSubstr("The low priority task queue to which this task was "
                    "submitted does not have the capacity to handle this task; "
                    "currently the low priority queue has 20 tasks enqueued "
                    "and the submitted task size is 1 while "
                    "max_enqueued_batches=2 and max_execution_batch_size=10")));

  // Unblock the thread.
  proceed.Notify();
}

TEST_P(SharedBatchSchedulerPriorityTest,
       CallbackWithTaskVectorOkWithPriorityQueueDisabledWithPrioritySet) {
  bool queue_callback_called = false;
  auto queue_callback = [&queue_callback_called](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(3, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(3, batch->task(1).size());
    EXPECT_EQ(5, batch->task(2).size());
    EXPECT_EQ(0, tasks.size());
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/3);

    // Create a queue with the priority queue disabled.
    const QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/false);
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    // Submit tasks to the queue.
    TF_ASSERT_OK(ScheduleTask(1, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(5, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
  }
  EXPECT_TRUE(queue_callback_called);
}

TEST_P(SharedBatchSchedulerPriorityTest,
       LowPriorityTaskOnlyAtMaxBatchSizeWithPriorityQueueEnabled) {
  bool queue_callback_called = false;
  auto queue_callback = [&queue_callback_called](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(3, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(3, batch->task(1).size());
    EXPECT_EQ(5, batch->task(2).size());
    EXPECT_TRUE(tasks.empty());
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/3);

    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/100, /*input_batch_size_limit=*/100,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 9;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        mixed_priority_batching_policy();
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    // Submit low priority tasks to fill up the max batch size.
    TF_ASSERT_OK(ScheduleTask(1, queue.get(),
                              tsl::criticality::Criticality::kSheddablePlus));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kSheddablePlus));
    TF_ASSERT_OK(ScheduleTask(5, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
  }
  EXPECT_TRUE(queue_callback_called);
}

TEST_P(SharedBatchSchedulerPriorityTest,
       LowPriorityTaskOnlyAtTimeoutWithPriorityQueueEnabled) {
  bool queue_callback_called = false;
  auto queue_callback = [&queue_callback_called](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(3, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(3, batch->task(1).size());
    EXPECT_EQ(5, batch->task(2).size());
    EXPECT_TRUE(tasks.empty());
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/3);

    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/100, /*input_batch_size_limit=*/100,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 20;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        mixed_priority_batching_policy();
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    // Submit low priority tasks that wouldn't fill up the max batch size, but
    // they should still be scheduled due to timeout.
    TF_ASSERT_OK(ScheduleTask(1, queue.get(),
                              tsl::criticality::Criticality::kSheddablePlus));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kSheddablePlus));
    TF_ASSERT_OK(ScheduleTask(5, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
  }
  EXPECT_TRUE(queue_callback_called);
}

// Lazy split is to be removed. The mixed priority batching is only supported
// when the lazy split is not enabled.
INSTANTIATE_TEST_SUITE_P(
    Parameter, SharedBatchSchedulerPriorityTest,
    ::testing::Values(
        std::make_tuple(
            /*enable_input_batch_split=*/true,
            MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize),
        std::make_tuple(/*enable_input_batch_split=*/true,
                        MixedPriorityBatchingPolicy::
                            kLowPriorityPaddingWithNextAllowedBatchSize),
        std::make_tuple(
            /*enable_input_batch_split=*/false,
            MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize),
        std::make_tuple(/*enable_input_batch_split=*/false,
                        MixedPriorityBatchingPolicy::
                            kLowPriorityPaddingWithNextAllowedBatchSize),
        std::make_tuple(
            /*enable_input_batch_split=*/false,
            MixedPriorityBatchingPolicy::kPriorityIsolation),
        std::make_tuple(/*enable_input_batch_split=*/false,
                        MixedPriorityBatchingPolicy::kPriorityIsolation),
        std::make_tuple(/*enable_input_batch_split=*/false,
                        MixedPriorityBatchingPolicy::kPriorityMerge),
        std::make_tuple(/*enable_input_batch_split=*/true,
                        MixedPriorityBatchingPolicy::kPriorityMerge)));

using SharedBatchSchedulerPriorityPolicyTest = SharedBatchSchedulerTest;

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       HighPriorityBatchPaddedUptoMaxBatchSize) {
  bool queue_callback_called = false;
  auto queue_callback = [&queue_callback_called](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    // Skip if this was called already, which is the low priority task only
    // batch case.
    if (queue_callback_called) return;

    queue_callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(3, batch->task(1).size());
    EXPECT_EQ(2, tasks.size());
    EXPECT_EQ(3, tasks[0]->size());
    EXPECT_EQ(3, tasks[1]->size());
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/3);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    // Submit tasks to the queue. The high priority batch is padded by the low
    // priority tasks only up to 10, which is the max batch size.
    TF_ASSERT_OK(ScheduleTask(1, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
  }
  EXPECT_TRUE(queue_callback_called);
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       HighPriorityBatchPaddedUptoMaxAvailableBatchSize) {
  bool queue_callback_called = false;
  auto queue_callback = [&queue_callback_called](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(3, batch->task(1).size());
    EXPECT_EQ(1, tasks.size());
    EXPECT_EQ(3, tasks[0]->size());
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/3);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    // Submit tasks to the queue. The high priority batch is padded by the low
    // priority tasks only up to 7, since there is no more available low
    // priority task.
    TF_ASSERT_OK(ScheduleTask(1, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
  }
  EXPECT_TRUE(queue_callback_called);
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       HighPriorityBatchPaddedUptoNextAllowedBatchSize) {
  bool queue_callback_called = false;
  auto queue_callback = [&queue_callback_called](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    // Skip if this was called already, which is the low priority task only
    // batch case.
    if (queue_callback_called) return;

    queue_callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(3, batch->task(1).size());
    EXPECT_EQ(2, tasks.size());
    EXPECT_EQ(2, tasks[0]->size());
    EXPECT_EQ(2, tasks[1]->size());
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/3);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.allowed_batch_sizes = {2, 8, 16};
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy = MixedPriorityBatchingPolicy::
        kLowPriorityPaddingWithNextAllowedBatchSize;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    // Submit tasks to the queue. The high priority batch is padded by the low
    // priority tasks only up to 8, which is the next allowed batch size.
    TF_ASSERT_OK(ScheduleTask(1, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(2, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
    TF_ASSERT_OK(ScheduleTask(2, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
    TF_ASSERT_OK(ScheduleTask(2, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
  }
  EXPECT_TRUE(queue_callback_called);
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       HighPriorityBatchNotPaddedWhenAllowedBatchSizeMissing) {
  bool queue_callback_called = false;
  auto queue_callback = [&queue_callback_called](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    // Skip if this was called already, which is the low priority task only
    // batch case.
    if (queue_callback_called) return;

    queue_callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(3, batch->task(1).size());
    EXPECT_EQ(0, tasks.size());
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/3);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy = MixedPriorityBatchingPolicy::
        kLowPriorityPaddingWithNextAllowedBatchSize;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    // Submit tasks to the queue. The high priority batch is padded by the low
    // priority tasks up to 10, which is the max batch size because the allowed
    // batch sizes are missing.
    TF_ASSERT_OK(ScheduleTask(1, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(2, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
    TF_ASSERT_OK(ScheduleTask(2, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
    TF_ASSERT_OK(ScheduleTask(2, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
  }
  EXPECT_TRUE(queue_callback_called);
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       HighPriorityBatchNotPaddedWithLowPriorityTasks) {
  int queue_callback_counter = 0;
  auto queue_callback = [&queue_callback_counter](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    // The first batch must be high priority only batch.
    if (queue_callback_counter++ == 0) {
      ASSERT_TRUE(batch->IsClosed());
      ASSERT_EQ(2, batch->num_tasks());
      EXPECT_EQ(1, batch->task(0).size());
      EXPECT_EQ(3, batch->task(1).size());
      return;
    }

    // The next batch must be the low priority only batch.
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(3, batch->task(0).size());
    EXPECT_EQ(3, batch->task(1).size());
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/3);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kPriorityIsolation;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    // Submit tasks to the queue. The high priority batch and the low priority
    // batch are scheduled separately.
    TF_ASSERT_OK(ScheduleTask(1, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
  }
  EXPECT_EQ(queue_callback_counter, 2);
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       PriorityMergedLowPriorityTimeout) {
  // With the kPriorityMerge strategy, a pure low priority batch can be
  // scheduled after the timeout has elapsed.

  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  Notification first_batch_processed;
  int queue_callback_counter = 0;
  auto queue_callback = [&queue_callback_counter, &first_batch_processed](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_counter++;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(1, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(queue_callback_counter, 1);
    first_batch_processed.Notify();
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/1, &env);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kPriorityMerge;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    TF_ASSERT_OK(ScheduleTask(1, queue.get(),
                              tsl::criticality::Criticality::kSheddable));

    Env::Default()->SleepForMicroseconds(100 * 1000 /* 100ms */);
    EXPECT_EQ(queue_callback_counter, 0);

    env.AdvanceByMicroseconds(2 * 1000 * 1000);
    first_batch_processed.WaitForNotification();
    EXPECT_EQ(queue_callback_counter, 1);

    start_teardown.Notify();
  }

  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       PriorityMergedHighPriorityTimeout) {
  // With the kPriorityMerge strategy, a pure high priority batch can be
  // scheduled after the timeout has elapsed.

  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  Notification first_batch_processed;
  int queue_callback_counter = 0;
  auto queue_callback = [&queue_callback_counter, &first_batch_processed](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_counter++;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(1, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(queue_callback_counter, 1);
    first_batch_processed.Notify();
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/1, &env);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kPriorityMerge;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    TF_ASSERT_OK(ScheduleTask(1, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));

    Env::Default()->SleepForMicroseconds(100 * 1000 /* 100ms */);
    EXPECT_EQ(queue_callback_counter, 0);

    env.AdvanceByMicroseconds(2 * 1000 * 1000);
    first_batch_processed.WaitForNotification();
    EXPECT_EQ(queue_callback_counter, 1);

    start_teardown.Notify();
  }

  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       PriorityMergedLowPriorityCompleteBatch) {
  // With the kPriorityMerge strategy, a pure low priority batch can be
  // scheduled immediately if the max_execution_batch_size is hit.

  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  Notification first_batch_processed;
  auto queue_callback = [&first_batch_processed](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(1, batch->num_tasks());
    EXPECT_EQ(10, batch->task(0).size());
    EXPECT_FALSE(first_batch_processed.HasBeenNotified());
    first_batch_processed.Notify();
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/1, &env);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kPriorityMerge;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    TF_ASSERT_OK(ScheduleTask(10, queue.get(),
                              tsl::criticality::Criticality::kSheddable));

    // NOTE: The fake env clock is not advanced.
    first_batch_processed.WaitForNotification();

    start_teardown.Notify();
  }

  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       PriorityMergedHighPriorityCompleteBatch) {
  // With the kPriorityMerge strategy, a pure high priority batch can be
  // scheduled immediately if the max_execution_batch_size is hit.

  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  Notification first_batch_processed;
  auto queue_callback = [&first_batch_processed](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(1, batch->num_tasks());
    EXPECT_EQ(10, batch->task(0).size());
    EXPECT_FALSE(first_batch_processed.HasBeenNotified());
    first_batch_processed.Notify();
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/1, &env);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kPriorityMerge;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    TF_ASSERT_OK(ScheduleTask(10, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));

    // NOTE: The fake env clock is not advanced.
    first_batch_processed.WaitForNotification();

    start_teardown.Notify();
  }

  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       PriorityMergedMixedPriorityBatchAfterHighPriorityTimeout) {
  // With the kPriorityMerge strategy, if there are both low and high priority
  // tasks, the timeout will be measured from the earliest task (in this case a
  // low priority task). After the timeout elapsed the high and low pri tasks
  // will be concatenated together.

  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  Notification first_batch_processed;
  int queue_callback_counter = 0;
  auto queue_callback = [&queue_callback_counter, &first_batch_processed](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_counter++;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(3, batch->task(0).size());  // High pri task
    EXPECT_EQ(2, batch->task(1).size());  // Low pri task
    EXPECT_FALSE(first_batch_processed.HasBeenNotified());
    first_batch_processed.Notify();
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/1, &env);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kPriorityMerge;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    env.AdvanceByMicroseconds(500 * 1000);
    TF_ASSERT_OK(ScheduleTask(2, queue.get(),
                              tsl::criticality::Criticality::kSheddable));

    Env::Default()->SleepForMicroseconds(100 * 1000 /* 100ms */);
    EXPECT_EQ(queue_callback_counter, 0);

    env.AdvanceByMicroseconds(501 * 1000);
    first_batch_processed.WaitForNotification();
    EXPECT_EQ(queue_callback_counter, 1);

    start_teardown.Notify();
  }

  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       PriorityMergedMixedPriorityBatchAfterLowPriorityTimeout) {
  // With the kPriorityMerge strategy, if there are both low and high priority
  // tasks, the timeout will be measured from the earliest task (in this case a
  // low priority task). After the timeout elapsed the high and low pri tasks
  // will be concatenated together.

  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  Notification first_batch_processed;
  int queue_callback_counter = 0;
  auto queue_callback = [&queue_callback_counter, &first_batch_processed](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_counter++;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(3, batch->task(0).size());  // High pri task
    EXPECT_EQ(2, batch->task(1).size());  // Low pri task
    EXPECT_FALSE(first_batch_processed.HasBeenNotified());
    first_batch_processed.Notify();
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/1, &env);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kPriorityMerge;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    TF_ASSERT_OK(ScheduleTask(2, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
    env.AdvanceByMicroseconds(500 * 1000);
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));

    Env::Default()->SleepForMicroseconds(100 * 1000 /* 100ms */);
    EXPECT_EQ(queue_callback_counter, 0);

    env.AdvanceByMicroseconds(501 * 1000);
    first_batch_processed.WaitForNotification();
    EXPECT_EQ(queue_callback_counter, 1);

    start_teardown.Notify();
  }

  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       PriorityMergedMixedPriorityCompleteBatch) {
  // With the kPriorityMerge strategy, if there are enough of low + high
  // priority inputs to form a complete batch, it will be scheduled immediately.

  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  Notification first_batch_processed;
  auto queue_callback = [&first_batch_processed](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(3, batch->num_tasks());
    EXPECT_EQ(5, batch->task(0).size());  // High pri task
    EXPECT_EQ(2, batch->task(1).size());  // Low pri task
    EXPECT_EQ(3, batch->task(2).size());  // Low pri task
    EXPECT_FALSE(first_batch_processed.HasBeenNotified());
    first_batch_processed.Notify();
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/1, &env);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kPriorityMerge;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    TF_ASSERT_OK(ScheduleTask(2, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
    env.AdvanceByMicroseconds(1 * 1000);
    TF_ASSERT_OK(ScheduleTask(3, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
    env.AdvanceByMicroseconds(1 * 1000);
    TF_ASSERT_OK(ScheduleTask(5, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));

    first_batch_processed.WaitForNotification();

    start_teardown.Notify();
  }

  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       PriorityMergedMixedPriorityBatchSplitting) {
  // With the kPriorityMerge strategy, if there are too many low and high
  // priority inputs, the high priority tasks and part of the low pri tasks will
  // be scheduled immediately and the remaining low pri tasks overflowing the
  // max execution batch size will be scheduled later.

  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  Notification first_batch_processed, second_batch_processed;
  int queue_callback_counter = 0;
  auto queue_callback = [this, &queue_callback_counter, &first_batch_processed,
                         &second_batch_processed](
                            std::unique_ptr<Batch<FakeTask>> batch,
                            std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_counter++;
    ASSERT_TRUE(batch->IsClosed());
    EXPECT_LE(queue_callback_counter, 2);

    if (enable_input_batch_split()) {
      if (queue_callback_counter == 1) {
        ASSERT_EQ(2, batch->num_tasks());
        EXPECT_EQ(6, batch->task(0).size());  // High pri task
        EXPECT_EQ(4, batch->task(1).size());  // Low pri task (batch split)
        first_batch_processed.Notify();
      }

      if (queue_callback_counter == 2) {
        ASSERT_EQ(1, batch->num_tasks());
        EXPECT_EQ(3, batch->task(0).size());  // Low pri remainder
        second_batch_processed.Notify();
      }
    } else {
      if (queue_callback_counter == 1) {
        ASSERT_EQ(1, batch->num_tasks());
        EXPECT_EQ(6, batch->task(0).size());  // High pri task
        first_batch_processed.Notify();
      }

      if (queue_callback_counter == 2) {
        ASSERT_EQ(1, batch->num_tasks());
        EXPECT_EQ(7, batch->task(0).size());  // Low pri remainder
        second_batch_processed.Notify();
      }
    }
  };

  {
    std::shared_ptr<Scheduler> scheduler =
        CreateSharedBatchScheduler(/*num_batch_threads=*/1, &env);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kPriorityMerge;
    std::unique_ptr<Queue> queue =
        CreateQueue(scheduler, queue_options, queue_callback);

    TF_ASSERT_OK(ScheduleTask(7, queue.get(),
                              tsl::criticality::Criticality::kSheddable));
    env.AdvanceByMicroseconds(1 * 1000);
    TF_ASSERT_OK(ScheduleTask(6, queue.get(),
                              tsl::criticality::Criticality::kCriticalPlus));

    Env::Default()->SleepForMicroseconds(100 * 1000 /* 100ms */);
    first_batch_processed.WaitForNotification();
    EXPECT_EQ(queue_callback_counter, 1);

    env.AdvanceByMicroseconds(2 * 1000 * 1000);
    second_batch_processed.WaitForNotification();
    EXPECT_EQ(queue_callback_counter, 2);

    start_teardown.Notify();
  }

  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       PriorityMergedRankingMultipleQueues) {
  // Testing that kPriorityMerge strategy + rank_queues when combined will total
  // order the processing of batches.

  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  /*
  Total ordering will be:
  - Queue 3 (high pri, time = 3)
  - Queue 4 (high + low pri, time = 4)
  - Queue 2 (low pri, time = 1)
  - Queue 1 (low pri, time = 2)
  */
  Notification last_batch_processed;
  int queue_callback_counter = 0;
  auto queue_callback1 = [&queue_callback_counter, &last_batch_processed](
                             std::unique_ptr<Batch<FakeTask>> batch,
                             std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_counter++;
    ASSERT_TRUE(batch->IsClosed());

    EXPECT_EQ(queue_callback_counter, 4);
    ASSERT_EQ(1, batch->num_tasks());
    EXPECT_EQ(5, batch->task(0).size());
    last_batch_processed.Notify();
  };

  auto queue_callback2 = [&queue_callback_counter](
                             std::unique_ptr<Batch<FakeTask>> batch,
                             std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_counter++;
    ASSERT_TRUE(batch->IsClosed());

    EXPECT_EQ(queue_callback_counter, 3);
    ASSERT_EQ(1, batch->num_tasks());
    EXPECT_EQ(4, batch->task(0).size());
  };

  auto queue_callback3 = [&queue_callback_counter](
                             std::unique_ptr<Batch<FakeTask>> batch,
                             std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_counter++;
    ASSERT_TRUE(batch->IsClosed());

    EXPECT_EQ(queue_callback_counter, 1);
    ASSERT_EQ(1, batch->num_tasks());
    EXPECT_EQ(6, batch->task(0).size());
  };

  auto queue_callback4 = [&queue_callback_counter](
                             std::unique_ptr<Batch<FakeTask>> batch,
                             std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_counter++;
    ASSERT_TRUE(batch->IsClosed());

    EXPECT_EQ(queue_callback_counter, 2);
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(1, batch->task(0).size());
    EXPECT_EQ(2, batch->task(1).size());
  };

  {
    std::shared_ptr<Scheduler> scheduler = CreateSharedBatchScheduler(
        /*num_batch_threads=*/1, &env, /*rank_queues=*/true);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kPriorityMerge;
    std::unique_ptr<Queue> queue1 =
        CreateQueue(scheduler, queue_options, queue_callback1);

    std::unique_ptr<Queue> queue2 =
        CreateQueue(scheduler, queue_options, queue_callback2);

    std::unique_ptr<Queue> queue3 =
        CreateQueue(scheduler, queue_options, queue_callback3);

    std::unique_ptr<Queue> queue4 =
        CreateQueue(scheduler, queue_options, queue_callback4);

    Env::Default()->SleepForMicroseconds(100 * 1000 /* 100ms */);
    EXPECT_EQ(queue_callback_counter, 0);

    TF_ASSERT_OK(ScheduleTask(4, queue2.get(),
                              tsl::criticality::Criticality::kSheddable));
    env.AdvanceByMicroseconds(1 * 1000);

    TF_ASSERT_OK(ScheduleTask(5, queue1.get(),
                              tsl::criticality::Criticality::kSheddable));
    env.AdvanceByMicroseconds(1 * 1000);

    TF_ASSERT_OK(ScheduleTask(6, queue3.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    env.AdvanceByMicroseconds(1 * 1000);

    TF_ASSERT_OK(ScheduleTask(1, queue4.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(2, queue4.get(),
                              tsl::criticality::Criticality::kSheddable));
    env.AdvanceByMicroseconds(1 * 1000);

    Env::Default()->SleepForMicroseconds(100 * 1000 /* 100ms */);
    EXPECT_EQ(queue_callback_counter, 0);

    env.AdvanceByMicroseconds(2 * 1000 * 1000);
    last_batch_processed.WaitForNotification();
    EXPECT_EQ(queue_callback_counter, 4);

    start_teardown.Notify();
  }

  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest, PriorityMergedRankingFullBatch) {
  // Testing that kPriorityMerge strategy + rank_queues when there is >=1 full
  // batches (so PeekBatchPriority() shouldn't just look at the last partially
  // full one).

  test_util::FakeClockEnv env(Env::Default());
  Notification start_teardown, stop_teardown;
  std::unique_ptr<Thread> teardown_thread =
      CreateFakeClockAdvancerThread(&env, &start_teardown, &stop_teardown);

  Notification first_batch_processed;
  int queue_callback_counter = 0;
  auto queue_callback1 = [&queue_callback_counter, &first_batch_processed](
                             std::unique_ptr<Batch<FakeTask>> batch,
                             std::vector<std::unique_ptr<FakeTask>> tasks) {
    queue_callback_counter++;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(1, batch->num_tasks());
    EXPECT_EQ(queue_callback_counter == 1 ? 10 : 1, batch->task(0).size());
    if (queue_callback_counter == 1) {
      first_batch_processed.Notify();
    }
  };

  {
    std::shared_ptr<Scheduler> scheduler = CreateSharedBatchScheduler(
        /*num_batch_threads=*/1, &env, /*rank_queues=*/true);

    // Create a queue with the priority queue enabled.
    QueueOptions queue_options = CreateQueueOptions(
        /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
        /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2,
        /*enable_priority_queue=*/true);
    queue_options.low_priority_queue_options.max_execution_batch_size = 10;
    queue_options.low_priority_queue_options.batch_timeout_micros =
        1 * 1000 * 1000;
    queue_options.low_priority_queue_options.input_batch_size_limit = 10;
    queue_options.low_priority_queue_options.max_enqueued_batches = 2;
    queue_options.mixed_priority_batching_policy =
        MixedPriorityBatchingPolicy::kPriorityMerge;
    std::unique_ptr<Queue> queue1 =
        CreateQueue(scheduler, queue_options, queue_callback1);

    Env::Default()->SleepForMicroseconds(100 * 1000 /* 100ms */);
    EXPECT_EQ(queue_callback_counter, 0);

    TF_ASSERT_OK(ScheduleTask(10, queue1.get(),
                              tsl::criticality::Criticality::kCriticalPlus));
    TF_ASSERT_OK(ScheduleTask(1, queue1.get(),
                              tsl::criticality::Criticality::kCriticalPlus));

    first_batch_processed.WaitForNotification();
    EXPECT_EQ(queue_callback_counter, 1);

    start_teardown.Notify();
  }

  stop_teardown.Notify();
}

TEST_P(SharedBatchSchedulerPriorityPolicyTest,
       ReuseGlobalSchedulerAcrossCreations) {
  Scheduler* scheduler_ptr = nullptr;

  {
    std::shared_ptr<Scheduler> scheduler1 =
        CreateSharedBatchScheduler(1, Env::Default(), /*rank_queues=*/false,
                                   /*use_global_scheduler=*/true);

    std::shared_ptr<Scheduler> scheduler2 =
        CreateSharedBatchScheduler(2, Env::Default(), /*rank_queues=*/false,
                                   /*use_global_scheduler=*/true);

    scheduler_ptr = scheduler1.get();

    EXPECT_EQ(scheduler1.get(), scheduler2.get());
  }

  // Allow time for any async destruction to complete.
  Env::Default()->SleepForMicroseconds(100 * 1000 /* 100ms */);

  std::shared_ptr<Scheduler> scheduler3 = CreateSharedBatchScheduler(
      3, Env::Default(), /*rank_queues=*/false, /*use_global_scheduler=*/true);
  EXPECT_EQ(scheduler_ptr, scheduler3.get());

  // Attempt to create a queue. If this crashes then the global scheduler was
  // incorrectly destroyed.
  QueueOptions queue_options = CreateQueueOptions(
      /*max_execution_batch_size=*/10, /*input_batch_size_limit=*/10,
      /*batch_timeout_micros=*/1 * 1000 * 1000, /*max_enqueued_batches=*/2);
  std::unique_ptr<Queue> queue1 =
      CreateQueue(scheduler3, queue_options,
                  [](std::unique_ptr<Batch<FakeTask>>,
                     std::vector<std::unique_ptr<FakeTask>>) {});

  // Future use_global_scheduler=false calls should continue to create distinct
  // schedulers.
  std::shared_ptr<Scheduler> scheduler4 = CreateSharedBatchScheduler(
      4, Env::Default(), /*rank_queues=*/false, /*use_global_scheduler=*/false);
  EXPECT_NE(scheduler_ptr, scheduler4.get());
}

// Lazy split is to be removed. The mixed priority batching is only supported
// when the lazy split is not enabled.
INSTANTIATE_TEST_SUITE_P(Parameter, SharedBatchSchedulerPriorityPolicyTest,
                         ::testing::Bool());

#ifdef PLATFORM_GOOGLE
// This benchmark relies on https://github.com/google/benchmark features,
// (in particular, `Benchmark::ThreadRange`) not available in open-sourced TF
//  codebase.

static std::vector<std::unique_ptr<Queue>>* queues =
    new std::vector<std::unique_ptr<Queue>>();

// Store queue labels, which are used to label benchmark results.
static std::vector<std::string>* queue_labels = new std::vector<std::string>();

// Create queues and add them to `queues` to keep them alive.
// Adds labels in `queue_labels`.
void CreateQueues() {
  // The split function is guaranteed (in the context of test) to process task
  // of size one, so it adds `input_task` into `output_tasks` directly, and
  // simulates a computation that takes some cpu cycles and time to complete.
  auto split_func_for_size_one_task =
      [](std::unique_ptr<FakeTask>* input_task, int open_batch_remaining_slot,
         int max_batch_size,
         std::vector<std::unique_ptr<FakeTask>>* output_tasks) -> absl::Status {
    output_tasks->push_back(std::move(*input_task));

    Notification notify;
    std::thread busy_waiter([&] {
      while (!notify.HasBeenNotified()) {
      }
    });

    std::thread notifier([&] {
      Env::Default()->SleepForMicroseconds(1);
      notify.Notify();
    });
    busy_waiter.join();
    notifier.join();
    return absl::OkStatus();
  };

  internal::Queue<FakeTask>::ProcessBatchCallback process_batch_callback =
      [](std::unique_ptr<Batch<FakeTask>> task) {
        // process_batch_callback is supposed to take ownership of `task`.
        // do nothing since `task` will be freed up when the callback returns.
      };
  const size_t max_execution_batch_size = 64;
  const size_t input_batch_size_limit = 128;
  const size_t batch_timeout_micros = 10;
  // Each queue has its own shared-batch-scheduler with the same parameter, so
  // scheduling behavior are approximately the same.
  queues->push_back(CreateQueue(
      CreateSharedBatchScheduler(5),
      CreateQueueOptions(max_execution_batch_size, input_batch_size_limit,
                         batch_timeout_micros, INT_MAX /* unbounded queue */,
                         true /* enable_large_batch_splitting */,
                         split_func_for_size_one_task),
      process_batch_callback));
  queue_labels->push_back(std::string("EagerSplit"));

  queues->push_back(CreateQueue(
      CreateSharedBatchScheduler(5),
      CreateQueueOptions(max_execution_batch_size, input_batch_size_limit,
                         batch_timeout_micros, INT_MAX /* unbounded queue */,
                         false /* enable_large_batch_splitting */,

                         nullptr /* no func */),
      process_batch_callback));
  queue_labels->push_back(std::string("NoSplit"));
}

void BM_QueueSchedule(::testing::benchmark::State& state) {
  static absl::once_flag once;
  absl::call_once(once, []() { CreateQueues(); });

  const int queue_index = state.range(1);
  Queue* queue = (*queues)[queue_index].get();

  const string label = strings::StrCat(state.threads(), "-Threads",
                                       (*queue_labels)[queue_index]);
  state.SetLabel(label);
  for (auto s : state) {
    for (int i = 0; i < state.range(0); i++) {
      auto batch_task = std::make_unique<FakeTask>(1);

      auto status = queue->Schedule(&batch_task);
      tensorflow::testing::DoNotOptimize(status);
    }
  }
}

BENCHMARK(BM_QueueSchedule)->Apply([](benchmark::internal::Benchmark* b) {
  b->ThreadRange(1,
                 port::NumSchedulableCPUs() * tensorflow::port::CPUIDNumSMT());

  for (int queue_index : {0, 1, 2}) {
    b->ArgPair(10000, queue_index);
  }
});

#endif  // PLATFORM_GOOGLE

}  // namespace
}  // namespace serving
}  // namespace tensorflow
