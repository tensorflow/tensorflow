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

#include <memory>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/pjrt/plugin/test/plugin_test_fixture.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/test.h"

namespace {

using ::tsl::testing::StatusIs;
using ::xla::PluginTestFixture;

TEST_F(PluginTestFixture, PluginReportsValidName) {
  auto platform_name = client_->platform_name();
  ASSERT_FALSE(platform_name.empty());
  LOG(INFO) << "Plugin reports platform_name: " << platform_name;
}

TEST(PluginRegistrationTest, InvalidPluginName) {
  absl::string_view invalid_plugin_name = "invalid_plugin";
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client =
      xla::GetCApiClient(invalid_plugin_name, {}, nullptr);
  ASSERT_THAT(client.status(), StatusIs(absl::StatusCode::kNotFound));
}
}  // namespace
