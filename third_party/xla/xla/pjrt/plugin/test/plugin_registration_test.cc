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
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_api.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/platform/status_matchers.h"
#include "xla/tsl/platform/statusor.h"
#include "xla/tsl/platform/test.h"

namespace {

using ::tsl::testing::StatusIs;

absl::StatusOr<std::string> GetRegisteredPluginName() {
  TF_ASSIGN_OR_RETURN(std::vector<std::string> pjrt_apis,
                      pjrt::GetRegisteredPjrtApis());
  if (pjrt_apis.size() != 1) {
    return absl::InvalidArgumentError(
        "Expected exactly one plugin to be registered.");
  }
  return pjrt_apis[0];
}

TEST(PluginRegistrationTest, PluginReportsValidName) {
  TF_ASSERT_OK_AND_ASSIGN(absl::string_view plugin_name,
                          GetRegisteredPluginName());
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::PjRtClient> client,
                          xla::GetCApiClient(plugin_name, {}, nullptr));
  ASSERT_FALSE(client->platform_name().empty());
  LOG(INFO) << "Plugin: " << plugin_name
            << " Platform name: " << client->platform_name();
}

TEST(PluginRegistrationTest, InvalidPluginName) {
  absl::string_view invalid_plugin_name = "invalid_plugin";
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> client =
      xla::GetCApiClient(invalid_plugin_name, {}, nullptr);
  ASSERT_THAT(client.status(), StatusIs(absl::StatusCode::kNotFound));
}
}  // namespace
