/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/stream_executor/abi/runtime_abi_version_manager.h"

#include <string>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/status_matchers.h"
#include "absl/strings/string_view.h"
#include "xla/stream_executor/abi/runtime_abi_version.pb.h"

namespace stream_executor {
namespace {

using absl_testing::StatusIs;
using ::testing::HasSubstr;

constexpr absl::string_view kPlatformName = "test_platform";

TEST(RuntimeAbiVersionManagerTest, GetInstance) {
  EXPECT_EQ(&RuntimeAbiVersionManager::GetInstance(),
            &RuntimeAbiVersionManager::GetInstance());
}

TEST(RuntimeAbiVersionManagerTest, RegisterRuntimeAbiVersionFactory) {
  RuntimeAbiVersionManager runtime_abi_version_manager;
  EXPECT_OK(runtime_abi_version_manager.RegisterRuntimeAbiVersionFactory(
      std::string(kPlatformName),
      [](absl::string_view) { return absl::InternalError("test_error"); }));
  EXPECT_THAT(runtime_abi_version_manager.RegisterRuntimeAbiVersionFactory(
                  std::string(kPlatformName),
                  [](absl::string_view) { return absl::OkStatus(); }),
              StatusIs(absl::StatusCode::kAlreadyExists,
                       HasSubstr("RuntimeAbiVersionFactory for platform")));
}

TEST(RuntimeAbiVersionManagerTest, GetRuntimeAbiVersionCallsFactory) {
  RuntimeAbiVersionManager runtime_abi_version_manager;
  RuntimeAbiVersionProto proto;
  proto.set_platform_name(kPlatformName);
  EXPECT_THAT(runtime_abi_version_manager.GetRuntimeAbiVersion(
                  RuntimeAbiVersionProto()),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("RuntimeAbiVersionFactory for platform")));

  EXPECT_OK(runtime_abi_version_manager.RegisterRuntimeAbiVersionFactory(
      std::string(kPlatformName),
      [](absl::string_view) { return absl::InternalError("test_error"); }));

  EXPECT_THAT(runtime_abi_version_manager.GetRuntimeAbiVersion(proto),
              StatusIs(absl::StatusCode::kInternal, HasSubstr("test_error")));
}

}  // namespace
}  // namespace stream_executor
