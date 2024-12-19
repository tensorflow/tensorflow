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

#include "tensorflow/lite/experimental/litert/compiler/plugin/compiler_plugin.h"

#include <array>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "testing/base/public/unique-test-directory.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"
#include "tensorflow/lite/experimental/litert/core/filesystem.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/test_macros.h"
#include "tensorflow/lite/experimental/litert/tools/dump.h"

namespace litert::internal {
namespace {

using ::testing::HasSubstr;
using ::testing::UniqueTestDirectory;

constexpr absl::string_view kTestPluginSearchPath =
    "third_party/tensorflow/lite/experimental/litert/vendors/examples";

constexpr absl::string_view kTestManufacturer = "ExampleSocManufacturer";
constexpr absl::string_view kTestModels = "ExampleSocModel";

TEST(CompilerPluginTest, LoadTestPlugin) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);
  ASSERT_EQ(plugins->front().SocModels().size(), 1);
  EXPECT_EQ(plugins->front().SocModels().front(), kTestModels);
}

TEST(CompilerPluginTest, LoadTestPluginWithMalformed) {
  const auto dir = UniqueTestDirectory();
  Touch(Join({dir, "notLibLiteRt.so"}));

  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, MultipleValidPlugins) {
  auto plugins = CompilerPlugin::LoadPlugins(
      {kTestPluginSearchPath, kTestPluginSearchPath});

  ASSERT_EQ(plugins->size(), 2);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);
  EXPECT_EQ(plugins->back().SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, MoveAssign) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  CompilerPlugin other = std::move(plugins->front());

  EXPECT_EQ(other.SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, MoveConstruct) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  CompilerPlugin other(std::move(plugins->front()));

  EXPECT_EQ(other.SocManufacturer(), kTestManufacturer);
}

TEST(CompilerPluginTest, SocModels) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  EXPECT_THAT(plugins->front().SocModels(),
              ::testing::ElementsAreArray({kTestModels}));
}

TEST(CompilerPluginTest, Partition) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  auto model = testing::LoadTestFileModel("mul_simple.tflite");
  auto subgraph = model.MainSubgraph();
  auto ops = plugins->front().Partition(*subgraph);
  ASSERT_TRUE(ops);

  EXPECT_EQ(ops->size(), 2);
}

TEST(CompilerPluginTest, Compile) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  EXPECT_EQ(plugins->front().SocManufacturer(), kTestManufacturer);

  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  auto& model = *model_wrap.Get();

  auto result = plugins->front().Compile(model.Subgraphs());
  ASSERT_TRUE(result);

  auto byte_code = result->ByteCode();
  ASSERT_TRUE(byte_code && byte_code->Size() > 0);

  auto num_calls = result->NumCalls();
  ASSERT_TRUE(num_calls);
  ASSERT_EQ(*num_calls, 1);

  auto call_info = result->CallInfo(0);
  ASSERT_TRUE(call_info);
  ASSERT_FALSE(call_info->empty());
}

TEST(CompilerPluginTest, Dump) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);

  std::stringstream dump;
  Dump(plugins->front(), dump);

  ASSERT_EQ(dump.view(),
            "SocManufacturer: ExampleSocManufacturer\nSocModels: { "
            "ExampleSocModel }\n");
}

TEST(PartitionModelTest, Simple) {
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  auto& model = *model_wrap.Get();

  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();

  auto partition_result = PartitionModel(plugin, model);
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 1);

  const auto& [ops, subgraphs] = *partition_result;

  EXPECT_EQ(ops.size(), 1);
  EXPECT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);

  EXPECT_EQ(subgraphs.Size(), 1);
  EXPECT_EQ(subgraphs.Elements().front()->Ops().size(), 2);
}

TEST(PartitionModelTest, MultiSubgraph) {
  auto model_wrap = testing::LoadTestFileModel("multi_subgraph_mul.tflite");
  auto& model = *model_wrap.Get();

  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();

  auto partition_result = PartitionModel(plugin, model);
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 2);

  const auto& [ops, subgraphs] = *partition_result;

  EXPECT_EQ(ops.size(), 2);
  EXPECT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_EQ(ops.back()->OpCode(), kLiteRtOpCodeTflCustom);

  EXPECT_EQ(subgraphs.Size(), 2);
  EXPECT_EQ(subgraphs.Elements().front()->Ops().size(), 1);
  EXPECT_EQ(subgraphs.Elements().back()->Ops().size(), 1);
}

TEST(ApplyTest, Simple) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  ASSERT_TRUE(ApplyPlugin(plugins->front(), model));
  ASSERT_EQ(model.NumSubgraphs(), 1);

  auto& subgraph = *model.MainSubgraph();
  ASSERT_EQ(subgraph.Ops().size(), 1);

  EXPECT_EQ(subgraph.Op(0).OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_THAT(subgraph.Op(0).CustomOptions().StrView(),
              HasSubstr(kByteCodeMetadataKey));

  EXPECT_TRUE(model.FindMetadata(kByteCodeMetadataKey));
  EXPECT_TRUE(model.FindMetadata(kLiteRtBuildStampKey));
}

TEST(ApplyTest, MultiSubgraph) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  auto model_wrap = testing::LoadTestFileModel("multi_subgraph_mul.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  ASSERT_TRUE(ApplyPlugin(plugins->front(), model));
  ASSERT_EQ(model.NumSubgraphs(), 2);

  auto& subgraph = model.Subgraph(0);
  ASSERT_EQ(subgraph.Ops().size(), 1);
  EXPECT_EQ(subgraph.Op(0).OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_THAT(subgraph.Op(0).CustomOptions().StrView(),
              HasSubstr(kByteCodeMetadataKey));

  auto& subgraph2 = model.Subgraph(1);
  ASSERT_EQ(subgraph2.Ops().size(), 1);
  EXPECT_EQ(subgraph2.Op(0).OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_THAT(subgraph2.Op(0).CustomOptions().StrView(),
              HasSubstr(kByteCodeMetadataKey));

  EXPECT_TRUE(model.FindMetadata(kByteCodeMetadataKey));
  EXPECT_TRUE(model.FindMetadata(kLiteRtBuildStampKey));
}

TEST(ApplyTest, ApplyPlugins) {
  litert::Environment::Destroy();

  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  const std::array environment_options = {
      litert::Environment::Option{
          /*.tag=*/litert::Environment::OptionTag::CompilerPluginLibraryPath,
          /*.value=*/kTestPluginSearchPath,
      },
  };
  ASSERT_TRUE(litert::Environment::Create(environment_options));

  LiteRtHwAccelerators compilation_options = static_cast<LiteRtHwAccelerators>(
      kLiteRtHwAccelatorCpu | kLiteRtHwAccelatorGpu | kLiteRtHwAccelatorNpu);
  auto new_flatbuffer =
      litert::internal::ApplyPlugins(&model, compilation_options);
  ASSERT_TRUE(new_flatbuffer);

  ASSERT_EQ(model.NumSubgraphs(), 1);

  auto& subgraph = *model.MainSubgraph();
  ASSERT_EQ(subgraph.Ops().size(), 1);

  EXPECT_EQ(subgraph.Op(0).OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_THAT(subgraph.Op(0).CustomOptions().StrView(),
              HasSubstr(kByteCodeMetadataKey));

  EXPECT_TRUE(model.FindMetadata(kByteCodeMetadataKey));
  EXPECT_TRUE(model.FindMetadata(kLiteRtBuildStampKey));

  litert::Environment::Destroy();
}

}  // namespace
}  // namespace litert::internal
