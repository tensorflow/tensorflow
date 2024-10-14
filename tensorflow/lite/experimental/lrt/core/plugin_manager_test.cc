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

#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "testing/base/public/unique-test-directory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"

namespace {

using ::lrt::internal::LrtPluginManager;
using ::lrt::testing::TouchTestFile;

constexpr absl::string_view kTestPluginSearchPath =
    "third_party/tensorflow/lite/experimental/lrt/vendors/examples";

constexpr absl::string_view kTestManufacturer = "ExampleSocManufacturer";

bool PluginOk(LrtPluginManager& plugin) {
  return kTestManufacturer == plugin.Api().soc_manufacturer() &&
         plugin.Api().num_supported_models(plugin.PluginHandle()) == 1;
}

TEST(PluginManagerTest, LoadTestPlugin) {
  std::vector<LrtPluginManager> plugins;

  ASSERT_STATUS_OK(
      LrtPluginManager::LoadPlugins({kTestPluginSearchPath}, plugins));
  plugins.front().DumpPluginInfo();

  ASSERT_EQ(plugins.size(), 1);
  EXPECT_TRUE(PluginOk(plugins[0]));
}

TEST(PluginManagerTest, LoadTestPluginWithMalformed) {
  std::vector<LrtPluginManager> plugins;
  const auto dir = testing::UniqueTestDirectory();
  TouchTestFile("notLibLrt.so", dir);

  ASSERT_STATUS_OK(
      LrtPluginManager::LoadPlugins({kTestPluginSearchPath, dir}, plugins));

  ASSERT_EQ(plugins.size(), 1);
  EXPECT_TRUE(PluginOk(plugins[0]));
}

TEST(PluginManagerTest, MultipleValidPlugins) {
  std::vector<LrtPluginManager> plugins;

  ASSERT_STATUS_OK(LrtPluginManager::LoadPlugins(
      {kTestPluginSearchPath, kTestPluginSearchPath}, plugins));

  ASSERT_EQ(plugins.size(), 2);
  EXPECT_TRUE(PluginOk(plugins[0]));
  EXPECT_TRUE(PluginOk(plugins[1]));
}

TEST(PluginManagerTest, MoveAssign) {
  std::vector<LrtPluginManager> plugins;

  ASSERT_STATUS_OK(
      LrtPluginManager::LoadPlugins({kTestPluginSearchPath}, plugins));

  ASSERT_EQ(plugins.size(), 1);
  EXPECT_TRUE(PluginOk(plugins.front()));

  LrtPluginManager other = std::move(plugins.front());

  EXPECT_EQ(plugins.front().Api().soc_manufacturer, nullptr);
  EXPECT_EQ(plugins.front().PluginHandle(), nullptr);
  EXPECT_EQ(plugins.front().CompiledResultHandle(), nullptr);

  EXPECT_TRUE(PluginOk(other));
}

TEST(PluginManagerTest, MoveConstruct) {
  std::vector<LrtPluginManager> plugins;

  ASSERT_STATUS_OK(
      LrtPluginManager::LoadPlugins({kTestPluginSearchPath}, plugins));

  ASSERT_EQ(plugins.size(), 1);
  EXPECT_TRUE(PluginOk(plugins.front()));

  LrtPluginManager other(std::move(plugins.front()));

  EXPECT_EQ(plugins.front().Api().soc_manufacturer, nullptr);
  EXPECT_EQ(plugins.front().PluginHandle(), nullptr);
  EXPECT_EQ(plugins.front().CompiledResultHandle(), nullptr);

  EXPECT_TRUE(PluginOk(other));
}

}  // namespace
