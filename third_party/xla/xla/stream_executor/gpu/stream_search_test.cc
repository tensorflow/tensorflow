/* Copyright 2022 The OpenXLA Authors.

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

#include "absl/status/statusor.h"
#include "xla/service/platform_util.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/stream.h"
#include "xla/stream_executor/stream_executor.h"
#include "xla/stream_executor/stream_finder.h"
#include "tsl/platform/statusor.h"
#include "tsl/platform/test.h"

namespace stream_executor {
namespace {

class StreamSearchTest : public ::testing::Test {
 public:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(platform_, xla::PlatformUtil::GetPlatform("GPU"));
  }

  Platform* platform_;
};

TEST_F(StreamSearchTest, NoMatchBadPtr) {
  void* bad_ptr = reinterpret_cast<void*>(0xdeadbeef);

  EXPECT_FALSE(FindStream(platform_, bad_ptr).ok());
}

TEST_F(StreamSearchTest, FoundPrevExecutor) {
  int number_devices = platform_->VisibleDeviceCount();
  EXPECT_GT(number_devices, 0);
  TF_ASSERT_OK_AND_ASSIGN(
      StreamExecutor * executor,
      platform_->ExecutorForDevice(number_devices > 1 ? 1 : 0));

  TF_ASSERT_OK_AND_ASSIGN(auto s, executor->CreateStream());
  TF_ASSERT_OK_AND_ASSIGN(auto s2, executor->CreateStream());

  void* gpu_ptr = s->platform_specific_handle().stream;
  void* gpu_ptr_2 = s2->platform_specific_handle().stream;

  TF_ASSERT_OK_AND_ASSIGN(Stream * found1, FindStream(platform_, gpu_ptr));
  EXPECT_EQ(found1, s.get());
  TF_ASSERT_OK_AND_ASSIGN(Stream * found2, FindStream(platform_, gpu_ptr_2));
  EXPECT_EQ(found2, s2.get());
}

}  // namespace
}  // namespace stream_executor
