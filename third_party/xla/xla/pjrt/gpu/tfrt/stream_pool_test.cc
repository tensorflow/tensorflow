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

#include "xla/pjrt/gpu/tfrt/stream_pool.h"

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "xla/service/backend.h"
#include "xla/stream_executor/platform.h"
#include "xla/stream_executor/platform_manager.h"
#include "xla/stream_executor/stream_executor.h"

namespace xla {
namespace {
constexpr absl::string_view kTestPlatformName = "CUDA";

class StreamPoolTest : public ::testing::Test {
 protected:
  void SetUp() override {
    platform_ =
        xla::se::PlatformManager::PlatformWithName(kTestPlatformName).value();
    xla::BackendOptions backend_options;
    backend_options.set_platform(platform_);
    backend_ = xla::Backend::CreateBackend(backend_options).value();
    device_ordinal_ = 0;
    executor_ = backend_->stream_executor(device_ordinal_).value();
  }

  stream_executor::Platform* platform_;
  std::unique_ptr<xla::Backend> backend_;
  int device_ordinal_;
  se::StreamExecutor* executor_;
};

TEST_F(StreamPoolTest, Borrow) {
  BoundedStreamPool pool(executor_, 1);
  EXPECT_EQ(pool.GetAvailableStreamNum(), 1);
  {
    ASSERT_OK_AND_ASSIGN(BoundedStreamPool::Handle handle, pool.Borrow());
    EXPECT_NE(handle.get(), nullptr);
    EXPECT_EQ(handle.get(), &*handle);
    EXPECT_EQ(pool.GetAvailableStreamNum(), 0);
  }
  EXPECT_EQ(pool.GetAvailableStreamNum(), 1);
}

}  // namespace
}  // namespace xla
