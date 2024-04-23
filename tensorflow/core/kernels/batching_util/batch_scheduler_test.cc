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

#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <tuple>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/platform/test.h"
#include "tsl/platform/criticality.h"

namespace tensorflow {
namespace serving {
namespace {

using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::Property;

TEST(MixedPriorityBatchingPolicyTest, InvalidAttrValueError) {
  EXPECT_THAT(
      GetMixedPriorityBatchingPolicy("invalid_attr_value"),
      testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          ::testing::HasSubstr(
              "Unknown mixed priority batching policy: invalid_attr_value")));
}

using MixedPriorityBatchingPolicyParameterizedTest = ::testing::TestWithParam<
    std::tuple<std::string, MixedPriorityBatchingPolicy>>;

TEST_P(MixedPriorityBatchingPolicyParameterizedTest,
       GetMixedPriorityBatchingPolicySuccess) {
  auto [attr_name, policy] = GetParam();
  EXPECT_THAT(GetMixedPriorityBatchingPolicy(attr_name),
              testing::IsOkAndHolds(Eq(policy)));
}

INSTANTIATE_TEST_SUITE_P(
    Parameter, MixedPriorityBatchingPolicyParameterizedTest,
    ::testing::Values(
        std::make_tuple(
            /*attr_name=*/kLowPriorityPaddingWithMaxBatchSizeAttrValue,
            /*policy=*/MixedPriorityBatchingPolicy::
                kLowPriorityPaddingWithMaxBatchSize),
        std::make_tuple(
            /*attr_name=*/kLowPriorityPaddingWithNextAllowedBatchSizeAttrValue,
            /*policy=*/MixedPriorityBatchingPolicy::
                kLowPriorityPaddingWithNextAllowedBatchSize),
        std::make_tuple(
            /*attr_name=*/kPriorityIsolationAttrValue,
            /*policy=*/MixedPriorityBatchingPolicy::kPriorityIsolation)));

class FakeTask : public BatchTask {
 public:
  explicit FakeTask(size_t size) : size_(size) {}

  ~FakeTask() override = default;

  size_t size() const override { return size_; }

 private:
  const size_t size_;

  FakeTask(const FakeTask&) = delete;
  void operator=(const FakeTask&) = delete;
};

TEST(TaskCriticalityTest, CriticalityDefaultsToCritical) {
  FakeTask fake_task(0);
  EXPECT_EQ(fake_task.criticality(), tsl::criticality::Criticality::kCritical);
}

TEST(TaskQueueTest, EmptyTaskQueue) {
  TaskQueue<FakeTask> task_queue;

  EXPECT_TRUE(task_queue.empty());
  EXPECT_EQ(0, task_queue.num_tasks());
  EXPECT_EQ(0, task_queue.size());
}

TEST(TaskQueueTest, AddTaskToTaskQueue) {
  TaskQueue<FakeTask> task_queue;

  task_queue.AddTask(std::make_unique<FakeTask>(1), 1);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(1, task_queue.num_tasks());
  EXPECT_EQ(1, task_queue.size());
}

TEST(TaskQueueTest, AddTasksToTaskQueue) {
  TaskQueue<FakeTask> task_queue;

  task_queue.AddTask(std::make_unique<FakeTask>(1), 1);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(1, task_queue.num_tasks());
  EXPECT_EQ(1, task_queue.size());

  task_queue.AddTask(std::make_unique<FakeTask>(2), 2);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(2, task_queue.num_tasks());
  EXPECT_EQ(3, task_queue.size());

  task_queue.AddTask(std::make_unique<FakeTask>(3), 3);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(3, task_queue.num_tasks());
  EXPECT_EQ(6, task_queue.size());
}

TEST(TaskQueueTest, RemoveTaskFromTaskQueueWithSingleTask) {
  TaskQueue<FakeTask> task_queue;

  task_queue.AddTask(std::make_unique<FakeTask>(1), 1);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(1, task_queue.num_tasks());
  EXPECT_EQ(1, task_queue.size());

  EXPECT_THAT(task_queue.RemoveTask(),
              Pointee(Property(&FakeTask::size, Eq(1))));
  EXPECT_TRUE(task_queue.empty());
  EXPECT_EQ(0, task_queue.num_tasks());
  EXPECT_EQ(0, task_queue.size());
}

TEST(TaskQueueTest, RemoveTaskFromTaskQueueWithMultipleTasks) {
  TaskQueue<FakeTask> task_queue;

  task_queue.AddTask(std::make_unique<FakeTask>(2), 1);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(1, task_queue.num_tasks());
  EXPECT_EQ(2, task_queue.size());

  task_queue.AddTask(std::make_unique<FakeTask>(1), 2);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(2, task_queue.num_tasks());
  EXPECT_EQ(3, task_queue.size());

  EXPECT_THAT(task_queue.RemoveTask(),
              Pointee(Property(&FakeTask::size, Eq(2))));
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(1, task_queue.num_tasks());
  EXPECT_EQ(1, task_queue.size());
}

TEST(TaskQueueTest, RemoveTasksFromTaskQueue) {
  TaskQueue<FakeTask> task_queue;

  task_queue.AddTask(std::make_unique<FakeTask>(1), 1);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(1, task_queue.num_tasks());
  EXPECT_EQ(1, task_queue.size());

  task_queue.AddTask(std::make_unique<FakeTask>(2), 2);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(2, task_queue.num_tasks());
  EXPECT_EQ(3, task_queue.size());

  task_queue.AddTask(std::make_unique<FakeTask>(3), 3);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(3, task_queue.num_tasks());
  EXPECT_EQ(6, task_queue.size());

  // The first two tasks are removed because they sum up to the size 3 as
  // specified.
  EXPECT_THAT(task_queue.RemoveTask(3),
              ElementsAre(Pointee(Property(&FakeTask::size, Eq(1))),
                          Pointee(Property(&FakeTask::size, Eq(2)))));
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(1, task_queue.num_tasks());
  EXPECT_EQ(3, task_queue.size());
}

TEST(TaskQueueTest, RemoveTasksFewerThanArgFromTaskQueue) {
  TaskQueue<FakeTask> task_queue;

  task_queue.AddTask(std::make_unique<FakeTask>(1), 1);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(1, task_queue.num_tasks());
  EXPECT_EQ(1, task_queue.size());

  task_queue.AddTask(std::make_unique<FakeTask>(2), 2);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(2, task_queue.num_tasks());
  EXPECT_EQ(3, task_queue.size());

  task_queue.AddTask(std::make_unique<FakeTask>(3), 3);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(3, task_queue.num_tasks());
  EXPECT_EQ(6, task_queue.size());

  // The last task is not removed since that will end up removing tasks that
  // sum up to the size larger than 5.
  EXPECT_THAT(task_queue.RemoveTask(5),
              ElementsAre(Pointee(Property(&FakeTask::size, Eq(1))),
                          Pointee(Property(&FakeTask::size, Eq(2)))));
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(1, task_queue.num_tasks());
  EXPECT_EQ(3, task_queue.size());
}

TEST(TaskQueueTest, RemoveAllTasksWhenArgGreaterThanTaskSize) {
  TaskQueue<FakeTask> task_queue;

  task_queue.AddTask(std::make_unique<FakeTask>(1), 1);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(1, task_queue.num_tasks());
  EXPECT_EQ(1, task_queue.size());

  task_queue.AddTask(std::make_unique<FakeTask>(2), 2);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(2, task_queue.num_tasks());
  EXPECT_EQ(3, task_queue.size());

  task_queue.AddTask(std::make_unique<FakeTask>(3), 3);
  EXPECT_FALSE(task_queue.empty());
  EXPECT_EQ(3, task_queue.num_tasks());
  EXPECT_EQ(6, task_queue.size());

  // All tasks upto the size 6 shoule be remove when the size 8 is specified.
  EXPECT_THAT(task_queue.RemoveTask(8),
              ElementsAre(Pointee(Property(&FakeTask::size, Eq(1))),
                          Pointee(Property(&FakeTask::size, Eq(2))),
                          Pointee(Property(&FakeTask::size, Eq(3)))));
  EXPECT_TRUE(task_queue.empty());
  EXPECT_EQ(0, task_queue.num_tasks());
  EXPECT_EQ(0, task_queue.size());
}

TEST(TaskQueueTest, EarliestStartTimeWithEmptyQueue) {
  TaskQueue<FakeTask> task_queue;
  EXPECT_FALSE(task_queue.EarliestTaskStartTime().has_value());
}

TEST(TaskQueueTest, EarliestStartTimeWithMultipleTasksInQueue) {
  TaskQueue<FakeTask> task_queue;

  task_queue.AddTask(std::make_unique<FakeTask>(1), 1);
  task_queue.AddTask(std::make_unique<FakeTask>(2), 2);

  std::optional<uint64_t> result = task_queue.EarliestTaskStartTime();
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(*result, 1);
}

TEST(TaskQueueTest, EarliestStartTimeAfterTaskRemoval) {
  TaskQueue<FakeTask> task_queue;

  task_queue.AddTask(std::make_unique<FakeTask>(1), 1);
  task_queue.AddTask(std::make_unique<FakeTask>(2), 2);
  task_queue.AddTask(std::make_unique<FakeTask>(3), 3);

  std::optional<uint64_t> result = task_queue.EarliestTaskStartTime();
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(*result, 1);

  EXPECT_THAT(task_queue.RemoveTask(3),
              ElementsAre(Pointee(Property(&FakeTask::size, Eq(1))),
                          Pointee(Property(&FakeTask::size, Eq(2)))));

  result = task_queue.EarliestTaskStartTime();
  EXPECT_TRUE(result.has_value());
  EXPECT_EQ(*result, 3);
}

TEST(BatchTest, Basic) {
  Batch<FakeTask> batch;

  EXPECT_EQ(0, batch.num_tasks());
  EXPECT_TRUE(batch.empty());
  EXPECT_EQ(0, batch.size());
  EXPECT_FALSE(batch.IsClosed());

  auto task0 = new FakeTask(3);
  batch.AddTask(std::unique_ptr<FakeTask>(task0));

  EXPECT_EQ(1, batch.num_tasks());
  EXPECT_FALSE(batch.empty());
  EXPECT_EQ(task0->size(), batch.size());
  EXPECT_EQ(task0->size(), batch.task(0).size());
  EXPECT_FALSE(batch.IsClosed());

  auto task1 = new FakeTask(7);
  batch.AddTask(std::unique_ptr<FakeTask>(task1));

  EXPECT_EQ(2, batch.num_tasks());
  EXPECT_FALSE(batch.empty());
  EXPECT_EQ(task0->size() + task1->size(), batch.size());
  EXPECT_EQ(task1->size(), batch.task(1).size());
  EXPECT_EQ(task1->size(), batch.mutable_task(1)->size());
  EXPECT_FALSE(batch.IsClosed());

  batch.Close();
  EXPECT_TRUE(batch.IsClosed());

  EXPECT_EQ(2, batch.num_tasks());
  EXPECT_FALSE(batch.empty());
  EXPECT_EQ(task0->size() + task1->size(), batch.size());
  EXPECT_EQ(task0->size(), batch.task(0).size());
  EXPECT_EQ(task1->size(), batch.task(1).size());

  EXPECT_EQ(7, batch.RemoveTask()->size());
  EXPECT_EQ(3, batch.size());
  EXPECT_EQ(3, batch.RemoveTask()->size());
  EXPECT_EQ(0, batch.size());
  EXPECT_TRUE(batch.empty());
}

TEST(BatchTest, WaitUntilClosed) {
  Batch<FakeTask> batch;
  batch.AddTask(std::unique_ptr<FakeTask>(new FakeTask(3)));
  EXPECT_FALSE(batch.IsClosed());

  std::unique_ptr<Thread> close_thread(
      Env::Default()->StartThread(ThreadOptions(), "test", [&batch]() {
        Env::Default()->SleepForMicroseconds(100);
        batch.Close();
      }));
  batch.WaitUntilClosed();
  EXPECT_TRUE(batch.IsClosed());
}

TEST(BatchTest, DeletionBlocksUntilClosed) {
  Batch<FakeTask>* batch = new Batch<FakeTask>;
  batch->AddTask(std::unique_ptr<FakeTask>(new FakeTask(3)));
  EXPECT_FALSE(batch->IsClosed());

  Notification do_delete, deleted;
  std::unique_ptr<Thread> delete_thread(Env::Default()->StartThread(
      ThreadOptions(), "test", [&batch, &do_delete, &deleted]() {
        do_delete.WaitForNotification();
        delete batch;
        deleted.Notify();
      }));
  do_delete.Notify();
  Env::Default()->SleepForMicroseconds(10 * 1000 /* 10 milliseconds */);
  EXPECT_FALSE(deleted.HasBeenNotified());
  batch->Close();
  deleted.WaitForNotification();
}

TEST(BatchTest, RemoveAllTasks) {
  Batch<FakeTask> batch;

  auto task0 = new FakeTask(3);
  batch.AddTask(std::unique_ptr<FakeTask>(task0));

  auto task1 = new FakeTask(7);
  batch.AddTask(std::unique_ptr<FakeTask>(task1));

  batch.Close();
  EXPECT_TRUE(batch.IsClosed());

  std::vector<std::unique_ptr<FakeTask>> tasks_in_batch =
      batch.RemoveAllTasks();
  EXPECT_EQ(2, tasks_in_batch.size());
  EXPECT_TRUE(batch.empty());

  EXPECT_EQ(task0, tasks_in_batch[0].get());
  EXPECT_EQ(task1, tasks_in_batch[1].get());

  // RemoveAllTasks returns empty vector from the second call and on, since
  // batch is closed.
  EXPECT_THAT(batch.RemoveAllTasks(), ::testing::IsEmpty());  // second call
  EXPECT_THAT(batch.RemoveAllTasks(), ::testing::IsEmpty());  // third call
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
