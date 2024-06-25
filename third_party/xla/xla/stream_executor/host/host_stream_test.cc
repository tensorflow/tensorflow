/* Copyright 2019 The OpenXLA Authors.

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

#include "absl/status/status.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "tsl/lib/core/status_test_util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace se = stream_executor;

TEST(HostStream, EnforcesFIFOOrder) {
  se::Platform* platform =
      se::PlatformManager::PlatformWithName("Host").value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  absl::Mutex mu;
  int expected = 0;
  bool ok = true;
  for (int i = 0; i < 2000; ++i) {
    TF_ASSERT_OK(stream->DoHostCallback([i, &mu, &expected, &ok]() {
      absl::MutexLock lock(&mu);
      if (expected != i) {
        ok = false;
      }
      ++expected;
    }));
  }
  TF_ASSERT_OK(stream->BlockHostUntilDone());
  absl::MutexLock lock(&mu);
  EXPECT_TRUE(ok);
}

TEST(HostStream, ReportsHostCallbackError) {
  se::Platform* platform =
      se::PlatformManager::PlatformWithName("Host").value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  TF_ASSERT_OK(stream->DoHostCallbackWithStatus(
      []() { return absl::InternalError("error!"); }));

  auto status = stream->BlockHostUntilDone();
  ASSERT_EQ(status.code(), tsl::error::INTERNAL);
  ASSERT_EQ(status.message(), "error!");
}

TEST(HostStream, ReportsFirstHostCallbackError) {
  se::Platform* platform =
      se::PlatformManager::PlatformWithName("Host").value();
  se::StreamExecutor* executor = platform->ExecutorForDevice(0).value();
  TF_ASSERT_OK_AND_ASSIGN(auto stream, executor->CreateStream());
  TF_ASSERT_OK(stream->DoHostCallbackWithStatus(
      []() { return absl::InternalError("error 1"); }));
  TF_ASSERT_OK(stream->DoHostCallbackWithStatus(
      []() { return absl::InternalError("error 2"); }));

  // "error 2" is just lost.
  ASSERT_EQ(stream->BlockHostUntilDone().message(), "error 1");
}
