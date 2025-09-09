/* Copyright 2025 The OpenXLA Authors.

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

#include "xla/stream_executor/sycl/sycl_event.h"

#include <gtest/gtest.h>
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/sycl/sycl_platform_id.h"

namespace stream_executor::gpu {
namespace {

const int kDefaultDeviceOrdinal = 0;

class SyclEventTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(Platform * platform,
                            stream_executor::PlatformManager::PlatformWithId(
                                stream_executor::sycl::kSyclPlatformId));
    TF_ASSERT_OK_AND_ASSIGN(executor_,
                            platform->ExecutorForDevice(kDefaultDeviceOrdinal));
  }

  StreamExecutor* executor_;
};

// TODO (intel-tf): Add a test for events with dependencies once SyclStream
// class is supported.
TEST_F(SyclEventTest, CreateEvent) {
  TF_ASSERT_OK_AND_ASSIGN(SyclEvent event, SyclEvent::Create(executor_));

  ::sycl::event sycl_event = event.GetEvent();

  // Expect the event to be complete immediately after creation
  // since it has no dependencies.
  EXPECT_EQ(event.PollForStatus(), Event::Status::kComplete);
}

TEST_F(SyclEventTest, MoveEvent) {
  TF_ASSERT_OK_AND_ASSIGN(SyclEvent orig_sycl_event,
                          SyclEvent::Create(executor_));

  // Make a copy of the event wrapper handle to check after move.
  ::sycl::event orig_event = orig_sycl_event.GetEvent();

  // Move the event to a new SyclEvent instance.
  SyclEvent moved_sycl_event = std::move(orig_sycl_event);

  // The moved event should still be valid.
  ::sycl::event moved_event = moved_sycl_event.GetEvent();
  EXPECT_EQ(moved_event, orig_event);
}

}  // namespace

}  // namespace stream_executor::gpu
