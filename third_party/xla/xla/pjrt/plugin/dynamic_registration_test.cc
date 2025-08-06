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

#include "xla/pjrt/plugin/dynamic_registration.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/log/log.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/pjrt_c_api_client.h"
#include "xla/pjrt/pjrt_client.h"
#include "xla/tsl/lib/core/status_test_util.h"
#include "xla/tsl/platform/test.h"

namespace {

// This test links in the example plugin and registers it prior to the test
// running. For this reason, we depend on the global state of the pjrt
// registration map.

TEST(DynamicRegistrationTest, RegisteredDynamicPjrtPluginSucceeds) {
  absl::StatusOr<std::unique_ptr<xla::PjRtClient>> c_api_client =
      xla::GetCApiClient("myplugin");
  TF_EXPECT_OK(c_api_client);
}

TEST(DynamicRegistrationTest, RegistrationFailsWithoutEnvVar) {
  EXPECT_FALSE(RegisterDynamicPjrtPlugin("myplugin", "BAD_FAKE_ENV_VAR").ok());
}

}  // namespace
