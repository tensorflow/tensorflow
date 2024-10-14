// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "tensorflow/lite/experimental/lrt/core/plugin_manager.h"

#include <vector>

#include <gtest/gtest.h>
#include "testing/base/public/unique-test-directory.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"

namespace {

using ::lrt::internal::LrtPluginManager;
using ::lrt::testing::TouchTestFile;

constexpr absl::string_view kTestPluginSearchPath =
    "third_party/tensorflow/lite/experimental/lrt/vendors/examples";

TEST(PluginManagerTest, LoadTestPlugin) {
  std::vector<LrtPluginManager> plugins;

  ASSERT_STATUS_OK(
      LrtPluginManager::LoadPlugins({kTestPluginSearchPath}, plugins));

  ASSERT_EQ(plugins.size(), 1);
  EXPECT_STREQ(plugins[0].Api()->soc_manufacturer(), "ExampleSocManufacturer");
  ASSERT_STATUS_OK(plugins[0].FreeLib());
}

TEST(PluginManagerTest, LoadTestPluginWithMalformed) {
  std::vector<LrtPluginManager> plugins;
  const auto dir = testing::UniqueTestDirectory();
  TouchTestFile("notLibLrt.so", dir);

  ASSERT_STATUS_OK(
      LrtPluginManager::LoadPlugins({kTestPluginSearchPath, dir}, plugins));

  ASSERT_EQ(plugins.size(), 1);
  EXPECT_STREQ(plugins[0].Api()->soc_manufacturer(), "ExampleSocManufacturer");
  ASSERT_STATUS_OK(plugins[0].FreeLib());
}

TEST(PluginManagerTest, MultipleValidPlugins) {
  std::vector<LrtPluginManager> plugins;

  ASSERT_STATUS_OK(LrtPluginManager::LoadPlugins(
      {kTestPluginSearchPath, kTestPluginSearchPath}, plugins));

  ASSERT_EQ(plugins.size(), 2);
  EXPECT_STREQ(plugins[0].Api()->soc_manufacturer(), "ExampleSocManufacturer");
  EXPECT_STREQ(plugins[1].Api()->soc_manufacturer(), "ExampleSocManufacturer");
  ASSERT_STATUS_OK(plugins[0].FreeLib());
  ASSERT_STATUS_OK(plugins[1].FreeLib());
}

}  // namespace
