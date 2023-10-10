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
#include "tensorflow/core/tfrt/runtime/tf_threadpool_concurrent_work_queue.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/tfrt/utils/thread_pool.h"
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/support/latch.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

const int32_t kNumThreads = 2;

class TfThreadpoolWorkQueueTest : public ::testing::Test {
 protected:
  TfThreadpoolWorkQueueTest()
      : tf_threadpool_cwq_(CreateDefaultTfThreadPoolWorkQueue(
            /*num_inter_op_threads=*/kNumThreads,
            /*num_intra_op_threads=*/kNumThreads)) {}
  std::unique_ptr<TfThreadPoolWorkQueue> tf_threadpool_cwq_;
};

TEST_F(TfThreadpoolWorkQueueTest, GetParallelismLevelOk) {
  EXPECT_GT(tf_threadpool_cwq_->GetParallelismLevel(), 0);
}

TEST_F(TfThreadpoolWorkQueueTest, GetNameOk) {
  EXPECT_EQ(tf_threadpool_cwq_->name(), "TfThreadPoolWorkQueue");
}

TEST_F(TfThreadpoolWorkQueueTest, InitializeRequestOk) {
  tfrt::RequestContextBuilder ctx_builder(/*host=*/nullptr,
                                          /*resource_context=*/nullptr);
  auto queue = tf_threadpool_cwq_->InitializeRequest(/*request_id=*/0);
  TF_ASSERT_OK(queue.status());
  EXPECT_NE(*queue, nullptr);
  EXPECT_NE((*queue)->GetIntraOpThreadPool(), nullptr);
}

TEST_F(TfThreadpoolWorkQueueTest, IsInWorkerThreadOk) {
  EXPECT_TRUE(tf_threadpool_cwq_->IsInWorkerThread());
}

TEST_F(TfThreadpoolWorkQueueTest, RunningBlockingTask) {
  tfrt::latch latch(10);
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    tf_threadpool_cwq_->AddBlockingTask(tfrt::TaskFunction([&n, &m, &latch] {
                                          {
                                            tensorflow::mutex_lock lock(m);
                                            ++n;
                                          }
                                          latch.count_down();
                                        }),
                                        true);
  }
  latch.wait();
  EXPECT_EQ(n, 10);
}

TEST_F(TfThreadpoolWorkQueueTest, RunningNonBlockingTask) {
  tfrt::latch latch(10);
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    tf_threadpool_cwq_->AddTask(tfrt::TaskFunction([&n, &m, &latch] {
      {
        tensorflow::mutex_lock lock(m);
        ++n;
      }
      latch.count_down();
    }));
  }
  latch.wait();
  EXPECT_EQ(n, 10);
}

TEST_F(TfThreadpoolWorkQueueTest, RunningMixedTask) {
  tfrt::latch latch(20);
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    tf_threadpool_cwq_->AddTask(tfrt::TaskFunction([&n, &m, &latch] {
      {
        tensorflow::mutex_lock lock(m);
        ++n;
      }
      latch.count_down();
    }));
    tf_threadpool_cwq_->AddBlockingTask(tfrt::TaskFunction([&n, &m, &latch] {
                                          {
                                            tensorflow::mutex_lock lock(m);
                                            ++n;
                                          }
                                          latch.count_down();
                                        }),
                                        true);
  }
  latch.wait();
  EXPECT_EQ(n, 20);
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
