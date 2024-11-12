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

#include "xla/pjrt/plugin/example_plugin/myplugin_cpp_pjrt.h"

#include "tsl/platform/status_matchers.h"
#include "tsl/platform/test.h"

namespace {

using ::myplugin_pjrt::CreateMyPluginPjrtClient;

TEST(MyPluginCPPTest, HasDeviceCount) {
  auto client = CreateMyPluginPjrtClient();
  EXPECT_EQ(client->device_count(), 42);
}

TEST(MyPluginCPPTest, GetHloCostAnalysis) {
  auto client = CreateMyPluginPjrtClient();

  EXPECT_THAT(client->GetHloCostAnalysis(),
              testing::Not(::tsl::testing::IsOk()));
}

}  // namespace
