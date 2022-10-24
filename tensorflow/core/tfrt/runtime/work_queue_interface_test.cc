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
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"

#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/tfrt/utils/thread_pool.h"
#include "tfrt/cpp_tests/test_util.h""  // from @tf_runtime
#include "tfrt/host_context/execution_context.h"  // from @tf_runtime
#include "tfrt/host_context/task_function.h"  // from @tf_runtime

namespace tensorflow {
namespace tfrt_stub {
namespace {

TEST(DefaultWorkQueueWrapperTest, Name) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  auto work_queue_ptr = work_queue.get();
  auto work_queue_wrapper = WrapDefaultWorkQueue(std::move(work_queue));

  EXPECT_EQ(work_queue_wrapper->name(), work_queue_ptr->name());
}

TEST(DefaultWorkQueueWrapperTest, AddTask_OnlyTask) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  auto work_queue_wrapper = WrapDefaultWorkQueue(std::move(work_queue));

  auto av = tfrt::MakeUnconstructedAsyncValueRef<int>().ReleaseRCRef();
  work_queue_wrapper->AddTask(
      tfrt::TaskFunction([av] { av->emplace<int>(0); }));
  work_queue_wrapper->Await(std::move(av));
}

TEST(DefaultWorkQueueWrapperTest, AddBlockingTask_TaskAndAllowQueueing) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  auto work_queue_wrapper = WrapDefaultWorkQueue(std::move(work_queue));

  auto av = tfrt::MakeUnconstructedAsyncValueRef<int>().ReleaseRCRef();
  std::thread thread{[&] {
    auto work = work_queue_wrapper->AddBlockingTask(
        tfrt::TaskFunction([&] { av->emplace<int>(0); }),
        /*allow_queuing=*/true);
  }};
  work_queue_wrapper->Await(std::move(av));
  thread.join();
}

TEST(DefaultWorkQueueWrapperTest, GetParallelismLevel) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  auto work_queue_ptr = work_queue.get();
  auto work_queue_wrapper = WrapDefaultWorkQueue(std::move(work_queue));

  EXPECT_EQ(work_queue_wrapper->GetParallelismLevel(),
            work_queue_ptr->GetParallelismLevel());
}

TEST(DefaultWorkQueueWrapperTest, IsInWorkerThread) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  auto work_queue_ptr = work_queue.get();
  auto work_queue_wrapper = WrapDefaultWorkQueue(std::move(work_queue));

  EXPECT_EQ(work_queue_wrapper->IsInWorkerThread(),
            work_queue_ptr->IsInWorkerThread());
}

TEST(DefaultWorkQueueWrapperTest, IntraOpThreadPool) {
  auto work_queue = tfrt::CreateSingleThreadedWorkQueue();
  TfThreadPool intra_op_thread_pool(/*name=*/"tf_intra",
                                    /*num_threads=*/1);
  auto work_queue_wrapper =
      WrapDefaultWorkQueue(std::move(work_queue), &intra_op_thread_pool);

  thread::ThreadPoolInterface* got_intra_op_threadpool;
  auto statusor_queue = work_queue_wrapper->InitializeRequest(
      /*request_context_builder=*/nullptr, &got_intra_op_threadpool);
  TF_ASSERT_OK(statusor_queue.status());
  EXPECT_NE(statusor_queue.value(), nullptr);

  EXPECT_EQ(got_intra_op_threadpool, &intra_op_thread_pool);
}

}  // namespace
}  // namespace tfrt_stub
}  // namespace tensorflow
