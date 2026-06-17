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

#include "xla/pjrt/plugin/static_registration.h"

#include <memory>

#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "xla/pjrt/c/pjrt_c_api.h"
#include "xla/tsl/lib/core/status_test_util.h"

namespace {

// Note: the tests in this file depend somewhat on the global state of the pjrt
// registration map. Conflicts can occur if the same plugin name is used in
// multiple tests in the same process.

TEST(StaticRegistrationTest, RegisterStaticPjrtPluginSucceeds) {
  constexpr absl::string_view kPluginName = "test_plugin_succeeds";
  auto plugin_api = std::make_unique<PJRT_Api>();
  TF_EXPECT_OK(RegisterStaticPjrtPlugin(kPluginName, plugin_api.get()));
}

TEST(StaticRegistrationTest, RegisterStaticPjrtPluginTwiceFails) {
  constexpr absl::string_view kPluginName = "test_plugin_second_time";
  auto plugin_api = std::make_unique<PJRT_Api>();
  TF_EXPECT_OK(RegisterStaticPjrtPlugin(kPluginName, plugin_api.get()));
  EXPECT_TRUE(absl::IsAlreadyExists(
      RegisterStaticPjrtPlugin(kPluginName, plugin_api.get())));
}

TEST(StaticRegistrationTest, RegisterStaticPjrtPluginNullptrFails) {
  constexpr absl::string_view kPluginName = "test_plugin_with_nullptr";
  EXPECT_TRUE(
      absl::IsInvalidArgument(RegisterStaticPjrtPlugin(kPluginName, nullptr)));
}

}  // namespace
