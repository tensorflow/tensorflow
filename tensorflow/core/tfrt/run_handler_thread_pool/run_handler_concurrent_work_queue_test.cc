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
#include "tensorflow/core/tfrt/run_handler_thread_pool/run_handler_concurrent_work_queue.h"

#include <cstdio>
#include <memory>

#include <gtest/gtest.h>
#include "absl/strings/str_cat.h"
#include "absl/time/time.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/status_matchers.h"
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tfrt/host_context/concurrent_work_queue.h"  // from @tf_runtime
#include "tfrt/host_context/diagnostic.h"  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/host_allocator.h"  // from @tf_runtime
#include "tfrt/host_context/host_context.h"  // from @tf_runtime
#include "tfrt/host_context/task_function.h"  // from @tf_runtime
#include "tfrt/support/mutex.h"  // from @tf_runtime

namespace tfrt {
namespace tf {
namespace {

const int kNumMainThreads = 1;
const int kNumComplementaryThreads = 1;

class RunHandlerThreadWorkQueueTest : public ::testing::Test {
 protected:
  void SetUp() override {
    RunHandlerThreadWorkQueue::Options options;
    options.num_complementary_threads = kNumComplementaryThreads;
    options.num_main_threads = kNumMainThreads;
    options.init_timeout_ms = 100;
    pool_ = std::make_unique<RunHandlerThreadWorkQueue>(options);

    // decoded_diagnostic_handler does nothing.
    auto decoded_diagnostic_handler = [&](const DecodedDiagnostic& diag) {};

    std::unique_ptr<ConcurrentWorkQueue> work_queue =
        CreateSingleThreadedWorkQueue();
    std::unique_ptr<HostAllocator> host_allocator = CreateMallocAllocator();
    host_ = std::make_unique<HostContext>(decoded_diagnostic_handler,
                                          std::move(host_allocator),
                                          std::move(work_queue));
    RequestContextBuilder req_ctx_builder{host_.get(),
                                          /*resource_context=*/nullptr};
    auto queue = pool_->InitializeRequest(/*request_id=*/100);
    TF_CHECK_OK(queue.status());
    queue_ = std::move(*queue);
    auto req_ctx = std::move(req_ctx_builder).build();
    ASSERT_TRUE(static_cast<bool>(req_ctx));
    exec_ctx_ = std::make_unique<ExecutionContext>(std::move(*req_ctx));
  }

  std::unique_ptr<RunHandlerThreadWorkQueue> pool_;
  std::unique_ptr<tensorflow::tfrt_stub::WorkQueueInterface> queue_;
  std::unique_ptr<HostContext> host_;
  std::unique_ptr<ExecutionContext> exec_ctx_;
};

TEST_F(RunHandlerThreadWorkQueueTest, RunningBlockingTask) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    ASSERT_FALSE(pool_->AddBlockingTask(TaskFunction([&n, &m] {
                                          tensorflow::mutex_lock lock(m);
                                          ++n;
                                        }),
                                        true));
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 10);
}

TEST_F(RunHandlerThreadWorkQueueTest, RunningBlockingTaskNoExecCtx) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    pool_->AddBlockingTask(TaskFunction([&n, &m] {
                             tensorflow::mutex_lock lock(m);
                             ++n;
                           }),
                           true);
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 10);
}

TEST_F(RunHandlerThreadWorkQueueTest, RunningBlockingTaskNoQueueing) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    ASSERT_FALSE(pool_->AddBlockingTask(TaskFunction([&n, &m] {
                                          tensorflow::mutex_lock lock(m);
                                          ++n;
                                        }),
                                        false));
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 10);
}

TEST_F(RunHandlerThreadWorkQueueTest, RunningNonBlockingTask) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    queue_->AddTask(TaskFunction([&n, &m] {
      tensorflow::mutex_lock lock(m);
      ++n;
    }));
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 10);
}

TEST_F(RunHandlerThreadWorkQueueTest, RunningNonBlockingTaskWithNoExecCtx) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    pool_->AddTask(TaskFunction([&n, &m] {
      tensorflow::mutex_lock lock(m);
      ++n;
    }));
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 10);
}

TEST_F(RunHandlerThreadWorkQueueTest, RunningMixedTask) {
  int n = 0;
  tensorflow::mutex m;
  for (int i = 0; i < 10; ++i) {
    queue_->AddTask(TaskFunction([&n, &m] {
      tensorflow::mutex_lock lock(m);
      ++n;
    }));
    ASSERT_FALSE(pool_->AddBlockingTask(TaskFunction([&n, &m] {
                                          tensorflow::mutex_lock lock(m);
                                          ++n;
                                        }),
                                        true));
  }
  pool_->Quiesce();
  EXPECT_EQ(n, 20);
}

TEST_F(RunHandlerThreadWorkQueueTest, NameReturnsValidString) {
  EXPECT_EQ(queue_->name(), "run_handler");
}

TEST_F(RunHandlerThreadWorkQueueTest, GetParallelismLevelOk) {
  EXPECT_EQ(queue_->GetParallelismLevel(),
            kNumComplementaryThreads + kNumMainThreads);
}

TEST_F(RunHandlerThreadWorkQueueTest, IsWorkerThreadOk) {
  EXPECT_TRUE(queue_->IsInWorkerThread());
}

TEST_F(RunHandlerThreadWorkQueueTest, NoHandlerReturnsError) {
  RunHandlerThreadWorkQueue::Options options;
  options.num_complementary_threads = 0;
  options.num_main_threads = 0;
  options.init_timeout_ms = 1;
  options.max_concurrent_handler = 0;
  auto queue = std::make_unique<RunHandlerThreadWorkQueue>(options);
  tfrt::RequestContextBuilder ctx_builder(nullptr, nullptr);
  EXPECT_THAT(
      queue->InitializeRequest(/*request_id=*/100),
      tensorflow::testing::StatusIs(
          tensorflow::error::INTERNAL,
          "Could not obtain RunHandler for request after waiting for 1 ms."));
}

}  // namespace
}  // namespace tf
}  // namespace tfrt
