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

#include "tensorflow/lite/experimental/lrt/core/compiler_plugin/compiler_plugin.h"

#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "testing/base/public/unique-test-directory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"
#include "tensorflow/lite/experimental/lrt/tools/dump.h"

namespace {

using ::litert::internal::CompilerPlugin;
using ::litert::testing::TouchTestFile;

constexpr absl::string_view kTestPluginSearchPath =
    "third_party/tensorflow/lite/experimental/lrt/vendors/examples";

constexpr absl::string_view kTestManufacturer = "ExampleSocManufacturer";
constexpr absl::string_view kTestModels = "ExampleSocModel";

TEST(CompilerPluginTest, LoadTestPlugin) {
  ASSERT_RESULT_OK_MOVE(CompilerPlugin::VecT plugins,
                        CompilerPlugin::LoadPlugins({kTestPluginSearchPath}));

  ASSERT_EQ(plugins.size(), 1);
  EXPECT_EQ(plugins.front().SocManufacturer(), kTestManufacturer);
  ASSERT_EQ(plugins.front().SocModels().size(), 1);
  EXPECT_EQ(plugins.front().SocModels().front(), kTestModels);
}

TEST(CompilerPluginTest, LoadTestPluginWithMalformed) {
  const auto dir = testing::UniqueTestDirectory();
  TouchTestFile("notLibLiteRt.so", dir);

  ASSERT_RESULT_OK_MOVE(CompilerPlugin::VecT plugins,
                        CompilerPlugin::LoadPlugins({kTestPluginSearchPath}));

  ASSERT_EQ(plugins.size(), 1);
  EXPECT_EQ(plugins.front().SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, MultipleValidPlugins) {
  ASSERT_RESULT_OK_MOVE(CompilerPlugin::VecT plugins,
                        CompilerPlugin::LoadPlugins(
                            {kTestPluginSearchPath, kTestPluginSearchPath}));

  ASSERT_EQ(plugins.size(), 2);
  EXPECT_EQ(plugins.front().SocManufacturer(), kTestManufacturer);
  EXPECT_EQ(plugins.back().SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, MoveAssign) {
  ASSERT_RESULT_OK_MOVE(CompilerPlugin::VecT plugins,
                        CompilerPlugin::LoadPlugins({kTestPluginSearchPath}));

  ASSERT_EQ(plugins.size(), 1);
  EXPECT_EQ(plugins.front().SocManufacturer(), kTestManufacturer);

  CompilerPlugin other = std::move(plugins.front());

  EXPECT_EQ(other.SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, MoveConstruct) {
  ASSERT_RESULT_OK_MOVE(CompilerPlugin::VecT plugins,
                        CompilerPlugin::LoadPlugins({kTestPluginSearchPath}));

  ASSERT_EQ(plugins.size(), 1);
  EXPECT_EQ(plugins.front().SocManufacturer(), kTestManufacturer);

  CompilerPlugin other(std::move(plugins.front()));

  EXPECT_EQ(other.SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, SocModels) {
  ASSERT_RESULT_OK_MOVE(CompilerPlugin::VecT plugins,
                        CompilerPlugin::LoadPlugins({kTestPluginSearchPath}));
  ASSERT_EQ(plugins.size(), 1);
  EXPECT_EQ(plugins.front().SocManufacturer(), kTestManufacturer);

  EXPECT_THAT(plugins.front().SocModels(),
              ::testing::ElementsAreArray({kTestModels}));
}

TEST(CompilerPluginTest, PartitionModel) {
  ASSERT_RESULT_OK_MOVE(CompilerPlugin::VecT plugins,
                        CompilerPlugin::LoadPlugins({kTestPluginSearchPath}));
  ASSERT_EQ(plugins.size(), 1);
  EXPECT_EQ(plugins.front().SocManufacturer(), kTestManufacturer);

  auto model = litert::testing::LoadTestFileModel("mul_simple.tflite");

  ASSERT_RESULT_OK_ASSIGN(auto ops, plugins.front().PartitionModel(*model));
  EXPECT_EQ(ops.size(), 2);
}

TEST(CompilerPluginTest, CompileModel) {
  ASSERT_RESULT_OK_MOVE(CompilerPlugin::VecT plugins,
                        CompilerPlugin::LoadPlugins({kTestPluginSearchPath}));
  ASSERT_EQ(plugins.size(), 1);
  EXPECT_EQ(plugins.front().SocManufacturer(), kTestManufacturer);

  auto model = litert::testing::LoadTestFileModel("mul_simple.tflite");
  ASSERT_RESULT_OK_ASSIGN(auto subgraph, graph_tools::GetSubgraph(model.get()));

  std::ostringstream byte_code_out;
  std::vector<std::string> call_info_out;
  ASSERT_STATUS_OK(plugins.front().Compile(kTestModels, {subgraph},
                                           byte_code_out, call_info_out));

  EXPECT_GT(byte_code_out.str().size(), 0);
  EXPECT_EQ(call_info_out.size(), 1);
}

TEST(CompilerPluginTest, Dump) {
  ASSERT_RESULT_OK_MOVE(CompilerPlugin::VecT plugins,
                        CompilerPlugin::LoadPlugins({kTestPluginSearchPath}));
  ASSERT_EQ(plugins.size(), 1);

  std::stringstream dump;
  litert::internal::Dump(plugins.front(), dump);

  ASSERT_EQ(dump.view(),
            "SocManufacturer: ExampleSocManufacturer\nSocModels: { "
            "ExampleSocModel }\n");
}

}  // namespace
