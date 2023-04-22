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

#include "tensorflow/core/kernels/batching_util/basic_batch_scheduler.h"

#include <utility>

#include "tensorflow/core/kernels/batching_util/batch_scheduler.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/test.h"

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

// Creates a FakeTask of size 'task_size', and calls 'scheduler->Schedule()'
// on that task. Returns the resulting status.
Status ScheduleTask(size_t task_size, BatchScheduler<FakeTask>* scheduler) {
  std::unique_ptr<FakeTask> task(new FakeTask(task_size));
  Status status = scheduler->Schedule(&task);
  // Schedule() should have consumed 'task' iff it returned Status::OK.
  CHECK_EQ(status.ok(), task == nullptr);
  return status;
}

// Since BasicBatchScheduler is implemented as a thin wrapper around
// SharedBatchScheduler, we only do some basic testing. More comprehensive
// testing is done in shared_batch_scheduler_test.cc.

TEST(BasicBatchSchedulerTest, Basic) {
  bool callback_called = false;
  auto callback = [&callback_called](std::unique_ptr<Batch<FakeTask>> batch) {
    callback_called = true;
    ASSERT_TRUE(batch->IsClosed());
    ASSERT_EQ(2, batch->num_tasks());
    EXPECT_EQ(3, batch->task(0).size());
    EXPECT_EQ(5, batch->task(1).size());
  };
  {
    BasicBatchScheduler<FakeTask>::Options options;
    options.max_batch_size = 10;
    options.batch_timeout_micros = 100 * 1000;  // 100 milliseconds
    options.num_batch_threads = 1;
    options.max_enqueued_batches = 3;
    std::unique_ptr<BasicBatchScheduler<FakeTask>> scheduler;
    TF_ASSERT_OK(
        BasicBatchScheduler<FakeTask>::Create(options, callback, &scheduler));
    EXPECT_EQ(10, scheduler->max_task_size());
    EXPECT_EQ(0, scheduler->NumEnqueuedTasks());
    EXPECT_EQ(3 * 10, scheduler->SchedulingCapacity());
    TF_ASSERT_OK(ScheduleTask(3, scheduler.get()));
    EXPECT_EQ(1, scheduler->NumEnqueuedTasks());
    EXPECT_EQ((3 * 10) - 3, scheduler->SchedulingCapacity());
    TF_ASSERT_OK(ScheduleTask(5, scheduler.get()));
    EXPECT_EQ(2, scheduler->NumEnqueuedTasks());
    EXPECT_EQ((3 * 10) - (3 + 5), scheduler->SchedulingCapacity());
  }
  EXPECT_TRUE(callback_called);
}

}  // namespace
}  // namespace serving
}  // namespace tensorflow
