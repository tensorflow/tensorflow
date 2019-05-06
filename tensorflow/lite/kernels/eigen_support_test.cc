/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/eigen_support.h"

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/lite/kernels/internal/optimized/eigen_spatial_convolutions.h"

namespace tflite {
namespace eigen_support {

struct TestTfLiteContext : public TfLiteContext {
  TestTfLiteContext() {
    recommended_num_threads = -1;
    external_context = nullptr;
    GetExternalContext = GetExternalContextImpl;
    SetExternalContext = SetExternalContextImpl;
  }

  static void SetExternalContextImpl(TfLiteContext* context,
                                     TfLiteExternalContextType type,
                                     TfLiteExternalContext* external_context) {
    static_cast<TestTfLiteContext*>(context)->external_context =
        external_context;
  }

  static TfLiteExternalContext* GetExternalContextImpl(
      TfLiteContext* context, TfLiteExternalContextType type) {
    return static_cast<TestTfLiteContext*>(context)->external_context;
  }

  TfLiteExternalContext* external_context;
};

TEST(EigenSupport, Default) {
  TestTfLiteContext context;
  IncrementUsageCounter(&context);
  ASSERT_NE(context.external_context, nullptr);
  EXPECT_EQ(context.external_context->type, kTfLiteEigenContext);

  auto thread_pool_device = GetThreadPoolDevice(&context);
  ASSERT_NE(thread_pool_device, nullptr);
  EXPECT_EQ(thread_pool_device->numThreads(), 4);

  DecrementUsageCounter(&context);
}

TEST(EigenSupport, SingleThreaded) {
  TestTfLiteContext context;
  context.recommended_num_threads = 1;
  IncrementUsageCounter(&context);

  auto thread_pool_device = GetThreadPoolDevice(&context);
  ASSERT_NE(thread_pool_device, nullptr);
  EXPECT_EQ(thread_pool_device->numThreads(), 1);
  EXPECT_EQ(thread_pool_device->numThreadsInPool(), 1);

  bool executed = false;
  auto notification =
      thread_pool_device->enqueue([&executed]() { executed = true; });
  ASSERT_NE(notification, nullptr);
  notification->Wait();
  delete notification;
  EXPECT_TRUE(executed);

  DecrementUsageCounter(&context);
}

TEST(EigenSupport, MultiThreaded) {
  TestTfLiteContext context;
  context.recommended_num_threads = 2;
  IncrementUsageCounter(&context);

  auto thread_pool_device = GetThreadPoolDevice(&context);
  ASSERT_NE(thread_pool_device, nullptr);
  EXPECT_EQ(thread_pool_device->numThreads(), 2);

  bool executed = false;
  auto notification =
      thread_pool_device->enqueue([&executed]() { executed = true; });
  ASSERT_NE(notification, nullptr);
  notification->Wait();
  delete notification;
  EXPECT_TRUE(executed);

  DecrementUsageCounter(&context);
}

TEST(EigenSupport, NumThreadsChanged) {
  TestTfLiteContext context;
  context.recommended_num_threads = 1;
  IncrementUsageCounter(&context);

  auto thread_pool_device = GetThreadPoolDevice(&context);
  ASSERT_NE(thread_pool_device, nullptr);
  EXPECT_EQ(thread_pool_device->numThreads(), 1);

  context.recommended_num_threads = 3;
  ASSERT_NE(context.external_context, nullptr);
  context.external_context->Refresh(&context);
  thread_pool_device = GetThreadPoolDevice(&context);
  ASSERT_NE(thread_pool_device, nullptr);
  EXPECT_EQ(thread_pool_device->numThreads(), 3);

  DecrementUsageCounter(&context);
}

TEST(EigenSupport, RefCounting) {
  TestTfLiteContext context;
  EXPECT_EQ(context.external_context, nullptr);

  IncrementUsageCounter(&context);
  EXPECT_NE(context.external_context, nullptr);

  IncrementUsageCounter(&context);
  EXPECT_NE(context.external_context, nullptr);

  DecrementUsageCounter(&context);
  EXPECT_NE(context.external_context, nullptr);

  DecrementUsageCounter(&context);
  EXPECT_EQ(context.external_context, nullptr);
}

}  // namespace eigen_support
}  // namespace tflite

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
