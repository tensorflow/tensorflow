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

#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status_matchers.h"
#include "absl/synchronization/mutex.h"
#include "xla/stream_executor/device_address.h"
#include "xla/stream_executor/event.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/statusor.h"

namespace se = stream_executor;

TEST(HostStream, EnforcesFIFOOrder) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("Host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());
  absl::Mutex mu;
  int expected = 0;
  bool ok = true;
  for (int i = 0; i < 2000; ++i) {
    ASSERT_OK(stream->DoHostCallback([i, &mu, &expected, &ok]() {
      absl::MutexLock lock(mu);
      if (expected != i) {
        ok = false;
      }
      ++expected;
    }));
  }
  ASSERT_OK(stream->BlockHostUntilDone());
  absl::MutexLock lock(mu);
  EXPECT_TRUE(ok);
}

TEST(HostStream, Memset32) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("Host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());

  uint32_t pattern = 0x12345678;
  std::vector<uint32_t> buffer(4, 0);
  se::DeviceAddressBase location(buffer.data(),
                                 buffer.size() * sizeof(uint32_t));

  ASSERT_OK(
      stream->Memset32(&location, pattern, buffer.size() * sizeof(uint32_t)));
  ASSERT_OK(stream->BlockHostUntilDone());

  for (int i = 0; i < 4; ++i) {
    EXPECT_EQ(buffer[i], pattern);
  }
}

TEST(HostStream, ReusedEvent) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("Host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream,
                       executor->CreateStream());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Event> event,
                       executor->CreateEvent());

  ASSERT_OK(stream->RecordEvent(event.get()));
  ASSERT_OK(stream->WaitFor(event.get()));

  ASSERT_OK(stream->RecordEvent(event.get()));
  ASSERT_OK(stream->WaitFor(event.get()));
  EXPECT_EQ(event->PollForStatus(), se::Event::Status::kComplete);
  ASSERT_OK(stream->BlockHostUntilDone());
}

TEST(HostStream, WaitFor) {
  ASSERT_OK_AND_ASSIGN(se::Platform * platform,
                       se::PlatformManager::PlatformWithName("Host"));
  ASSERT_OK_AND_ASSIGN(se::StreamExecutor * executor,
                       platform->ExecutorForDevice(0));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream1,
                       executor->CreateStream());
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<se::Stream> stream2,
                       executor->CreateStream());

  absl::Mutex mu;
  bool stream1_done = false;
  ASSERT_OK(stream1->DoHostCallback([&mu, &stream1_done]() {
    absl::MutexLock lock(mu);
    stream1_done = true;
  }));

  ASSERT_OK(stream2->WaitFor(stream1.get()));
  ASSERT_OK(stream2->BlockHostUntilDone());

  absl::MutexLock lock(mu);
  EXPECT_TRUE(stream1_done);
}
