/* Copyright 2024 The OpenXLA Authors.

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

#include "xla/stream_executor/stream_finder.h"

#include "absl/status/status.h"
#include "xla/stream_executor/mock_platform.h"
#include "xla/stream_executor/mock_stream.h"
#include "xla/stream_executor/mock_stream_executor.h"
#include "xla/test.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

using testing::Return;
namespace stream_executor {
namespace {

TEST(StreamFinderTest, FindStreamFailsWithNoExecutors) {
  MockStreamExecutor stream_executor;
  MockPlatform platform;
  EXPECT_CALL(platform, VisibleDeviceCount()).WillOnce(Return(0));
  EXPECT_FALSE(FindStream(&platform, nullptr).ok());
}

TEST(StreamFinderTest, FindStreamFailsWithNoMatchingStream) {
  MockStreamExecutor stream_executor;
  MockPlatform platform;
  EXPECT_CALL(platform, VisibleDeviceCount()).WillOnce(Return(1));
  EXPECT_CALL(platform, FindExisting(0)).WillOnce(Return(&stream_executor));
  void *gpu_stream = reinterpret_cast<void *>(0x1234);
  EXPECT_CALL(stream_executor, FindAllocatedStream(gpu_stream))
      .WillOnce(Return(nullptr));
  EXPECT_FALSE(FindStream(&platform, gpu_stream).ok());
}

TEST(StreamFinderTest, FindStreamSucceeds) {
  MockStreamExecutor stream_executor0;
  MockStreamExecutor stream_executor1;
  MockPlatform platform;
  EXPECT_CALL(platform, VisibleDeviceCount()).WillOnce(Return(2));
  EXPECT_CALL(platform, FindExisting(0)).WillOnce(Return(&stream_executor0));
  EXPECT_CALL(platform, FindExisting(1)).WillOnce(Return(&stream_executor1));
  void *gpu_stream = reinterpret_cast<void *>(0x1234);
  MockStream stream;
  EXPECT_CALL(stream_executor0, FindAllocatedStream(gpu_stream))
      .WillOnce(Return(nullptr));
  EXPECT_CALL(stream_executor1, FindAllocatedStream(gpu_stream))
      .WillOnce(Return(&stream));
  TF_ASSERT_OK_AND_ASSIGN(auto found_stream, FindStream(&platform, gpu_stream));
  EXPECT_EQ(found_stream, &stream);
}

TEST(StreamFinderTest, OnlyExecutor1Exists) {
  MockStreamExecutor stream_executor1;
  MockPlatform platform;
  EXPECT_CALL(platform, VisibleDeviceCount()).WillOnce(Return(2));
  EXPECT_CALL(platform, FindExisting(0))
      .WillRepeatedly(Return(absl::NotFoundError("Nope")));
  EXPECT_CALL(platform, FindExisting(1)).WillOnce(Return(&stream_executor1));
  void *gpu_stream = reinterpret_cast<void *>(0x1234);
  MockStream stream;
  EXPECT_CALL(stream_executor1, FindAllocatedStream(gpu_stream))
      .WillOnce(Return(&stream));
  TF_ASSERT_OK_AND_ASSIGN(auto found_stream, FindStream(&platform, gpu_stream));
  EXPECT_EQ(found_stream, &stream);
}
}  // namespace
}  // namespace stream_executor
