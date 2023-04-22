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

#include "tensorflow/core/platform/env.h"
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

}  // namespace
}  // namespace serving
}  // namespace tensorflow
