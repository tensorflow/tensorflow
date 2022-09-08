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

#include "absl/synchronization/mutex.h"
#include "tensorflow/compiler/xla/stream_executor/multi_platform_manager.h"
#include "tensorflow/compiler/xla/stream_executor/platform.h"
#include "tensorflow/compiler/xla/stream_executor/stream.h"
#include "tensorflow/compiler/xla/stream_executor/stream_executor.h"
#include "tensorflow/core/lib/core/status_test_util.h"
#include "tensorflow/core/platform/test.h"

namespace se = stream_executor;

TEST(HostStream, EnforcesFIFOOrder) {
  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName("Host").value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  se::Stream stream(executor);
  stream.Init();

  absl::Mutex mu;
  int expected = 0;
  bool ok = true;
  for (int i = 0; i < 2000; ++i) {
    stream.ThenDoHostCallback([i, &mu, &expected, &ok]() {
      absl::MutexLock lock(&mu);
      if (expected != i) {
        ok = false;
      }
      ++expected;
    });
  }
  TF_ASSERT_OK(stream.BlockHostUntilDone());
  absl::MutexLock lock(&mu);
  EXPECT_TRUE(ok);
}

TEST(HostStream, ReportsHostCallbackError) {
  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName("Host").value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  se::Stream stream(executor);
  stream.Init();

  stream.ThenDoHostCallbackWithStatus(
      []() { return se::port::InternalError("error!"); });

  se::port::Status status = stream.BlockHostUntilDone();
  ASSERT_EQ(status.code(), tensorflow::error::INTERNAL);
  ASSERT_EQ(status.error_message(), "error!");
}

TEST(HostStream, ReportsFirstHostCallbackError) {
  se::Platform* platform =
      se::MultiPlatformManager::PlatformWithName("Host").value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  se::Stream stream(executor);
  stream.Init();

  stream.ThenDoHostCallbackWithStatus(
      []() { return se::port::InternalError("error 1"); });
  stream.ThenDoHostCallbackWithStatus(
      []() { return se::port::InternalError("error 2"); });

  // "error 2" is just lost.
  ASSERT_EQ(stream.BlockHostUntilDone().error_message(), "error 1");
}
