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
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_environment.h"
#include "tensorflow/lite/experimental/litert/cc/litert_op_options.h"
#include "tensorflow/lite/experimental/litert/core/build_stamp.h"
#include "tensorflow/lite/experimental/litert/core/filesystem.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/matchers.h"
#include "tensorflow/lite/experimental/litert/tools/dump.h"

namespace litert::internal {
namespace {

using testing::UniqueTestDirectory;

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
  const auto dir = UniqueTestDirectory::Create();
  ASSERT_TRUE(dir);
  Touch(Join({dir->Str(), "notLibLiteRt.so"}));

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

TEST(CompilerPluginTest, SetFlags) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  LITERT_ASSERT_OK(plugins->front().SetFlags(CompilerFlags()));
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

  auto result = plugins->front().Compile(&model);
  ASSERT_TRUE(result);

  auto byte_code = result->ByteCode();
  ASSERT_TRUE(byte_code && byte_code->Size() > 0);

  auto num_calls = result->NumCalls();
  ASSERT_TRUE(num_calls);
  ASSERT_EQ(*num_calls, 1);

  auto call_info = result->CallInfo(0);
  ASSERT_TRUE(call_info);
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

TEST(PartitionModelTest, PartitionDirect) {
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  auto& model = *model_wrap.Get();

  std::vector<LiteRtOpWithPartitionIndex> selected_ops = {
      {model.MainSubgraph()->Ops().front(), 0},
      {model.MainSubgraph()->Ops().back(), 0}};

  auto partition_result = PartitionModelDirect(std::move(selected_ops), model);
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

TEST(PartitionModelTest, MultiSubgraphWithSelectedSubgraphs) {
  auto model_wrap = testing::LoadTestFileModel("multi_subgraph_mul.tflite");
  auto& model = *model_wrap.Get();

  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();

  auto partition_result = PartitionModel(plugin, model, {1});
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 2);

  const auto& [ops, subgraphs] = *partition_result;

  EXPECT_EQ(ops.size(), 1);
  EXPECT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);

  EXPECT_EQ(subgraphs.Size(), 1);
  EXPECT_EQ(subgraphs.Elements().front()->Ops().size(), 1);
}

TEST(PartitionModelTest, CstMultiSubgraph) {
  auto model_wrap = testing::LoadTestFileModel("multi_use_cst.tflite");
  auto& model = *model_wrap.Get();
  ASSERT_EQ(model.MainSubgraph()->Ops().size(), 3);

  std::vector<LiteRtOpWithPartitionIndex> selected_ops = {
      {model.MainSubgraph()->Ops().front(), 0},
      {model.MainSubgraph()->Ops().back(), 0},
  };
  auto partition_result = PartitionModelDirect(std::move(selected_ops), model);
  ASSERT_TRUE(partition_result);

  const auto& [ops, subgraphs] = *partition_result;

  EXPECT_EQ(ops.size(), 2);
  EXPECT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_EQ(ops.back()->OpCode(), kLiteRtOpCodeTflCustom);

  EXPECT_EQ(subgraphs.Size(), 2);
  EXPECT_EQ(subgraphs.Elements().front()->Ops().size(), 1);
  EXPECT_EQ(subgraphs.Elements().back()->Ops().size(), 1);

  const auto& cst_1 =
      subgraphs.Elements().front()->Ops().front()->Input(1).Weights();
  const auto& cst_2 =
      subgraphs.Elements().back()->Ops().front()->Input(1).Weights();

  // Both weights should have the same object managed by the same buffer
  // manager.
  ASSERT_EQ(cst_1.GetBufferManager(), model.Buffers());
  ASSERT_EQ(cst_2.GetBufferManager(), model.Buffers());
  ASSERT_GT(cst_1.Buffer().Size(), 0);
  ASSERT_GT(cst_2.Buffer().Size(), 0);
  EXPECT_EQ(cst_1.GetBufferId(), cst_2.GetBufferId());
  ASSERT_EQ(cst_1.Buffer().Data(), cst_2.Buffer().Data());
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

  auto* op = subgraph.Ops().front();

  EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_TRUE(model.FindOpAsset(op));

  EXPECT_TRUE(model.FindMetadata(kLiteRtBuildStampKey));
}

TEST(ApplyTest, WithPartition) {
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  auto& model = *model_wrap.Get();

  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  auto& plugin = plugins->front();

  auto partition_result = PartitionModel(plugin, model);
  ASSERT_TRUE(partition_result);
  ASSERT_EQ(model.NumSubgraphs(), 1);

  ASSERT_TRUE(ApplyPluginWithPartition(plugins->front(), model,
                                       std::move(*partition_result)));

  auto& subgraph = model.Subgraph(0);
  ASSERT_EQ(subgraph.Ops().size(), 1);

  auto* op = subgraph.Ops().front();

  EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_TRUE(model.FindOpAsset(op));
}

TEST(ApplyTest, MultiSubgraph) {
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});
  ASSERT_EQ(plugins->size(), 1);
  auto model_wrap = testing::LoadTestFileModel("multi_subgraph_mul.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  ASSERT_TRUE(ApplyPlugin(plugins->front(), model));
  ASSERT_EQ(model.NumSubgraphs(), 2);

  {
    auto& subgraph = model.Subgraph(0);
    ASSERT_EQ(subgraph.Ops().size(), 1);

    auto* op = subgraph.Ops().front();

    EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
    EXPECT_TRUE(model.FindOpAsset(op));
  }

  {
    auto& subgraph = model.Subgraph(1);
    ASSERT_EQ(subgraph.Ops().size(), 1);

    auto* op = subgraph.Ops().front();

    EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
    EXPECT_TRUE(model.FindOpAsset(op));
  }

  EXPECT_TRUE(model.FindMetadata(kLiteRtBuildStampKey));
}

TEST(ApplyTest, ApplyPlugins) {
  auto model_wrap = testing::LoadTestFileModel("mul_simple.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();

  const std::array environment_options = {
      litert::Environment::Option{
          /*.tag=*/litert::Environment::OptionTag::CompilerPluginLibraryDir,
          /*.value=*/kTestPluginSearchPath,
      },
  };
  auto env = litert::Environment::Create(environment_options);
  ASSERT_TRUE(env);

  LiteRtHwAccelerators compilation_options = static_cast<LiteRtHwAccelerators>(
      kLiteRtHwAcceleratorCpu | kLiteRtHwAcceleratorGpu |
      kLiteRtHwAcceleratorNpu);
  auto result =
      litert::internal::ApplyPlugins(env->Get(), &model, compilation_options);
  ASSERT_TRUE(result);

  ASSERT_EQ(model.NumSubgraphs(), 1);

  auto& subgraph = *model.MainSubgraph();
  ASSERT_EQ(subgraph.Ops().size(), 1);

  auto* op = subgraph.Ops().front();

  EXPECT_EQ(op->OpCode(), kLiteRtOpCodeTflCustom);
  EXPECT_TRUE(model.FindOpAsset(op));

  EXPECT_TRUE(model.FindMetadata(kLiteRtBuildStampKey));
}

TEST(PartitionTest, MappedCompositeOp) {
  auto model_wrap = testing::LoadTestFileModel("rms_norm_composite.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);
  // One new subgraph for the consumed composite op only, decomp not consumed.
  ASSERT_EQ(partition_result->second.Size(), 1);
}

TEST(PartitionTest, SimpleNpuCallComposite) {
  auto model_wrap = testing::LoadTestFileModel("simple_composite.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  auto* decomp = model.Subgraphs()[1];

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);

  auto& ops = partition_result->first;
  ASSERT_EQ(ops.size(), 1);
  ASSERT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);

  auto& sgs = partition_result->second;
  ASSERT_EQ(sgs.Size(), 1);
  ASSERT_EQ(sgs.Elements().front(), decomp);
}

TEST(PartitionTest, MultiNpuCallComposite) {
  auto model_wrap = testing::LoadTestFileModel("multi_composite.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  ASSERT_EQ(model.NumSubgraphs(), 4);
  auto* decomp1 = model.Subgraphs()[1];
  auto* non_npu_call_decomop = model.Subgraphs()[2];
  auto* decomp2 = model.Subgraphs()[3];

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);

  {
    // Subgraphs to be compiled will be moved to the result from the model.
    // Non-npu-call decompositions will be reindexed.
    ASSERT_EQ(model.NumSubgraphs(), 2);
    ASSERT_EQ(model.Subgraphs()[1], non_npu_call_decomop);
    auto opts = GetOptionsAs<CompositeOptions>(model.Subgraph(0).Ops()[1]);
    ASSERT_TRUE(opts);
    ASSERT_EQ(opts->subgraph, 1);
  }

  {
    // All npu call ops are now dispatch ops.
    auto& ops = partition_result->first;

    ASSERT_EQ(ops.size(), 2);
    auto* first_dispatch_op = ops.front();
    auto* second_dispatch_op = ops.back();

    ASSERT_EQ(first_dispatch_op->OpCode(), kLiteRtOpCodeTflCustom);
    ASSERT_EQ(first_dispatch_op, model.Subgraphs()[0]->Ops().front());

    ASSERT_EQ(second_dispatch_op->OpCode(), kLiteRtOpCodeTflCustom);
    ASSERT_EQ(second_dispatch_op, model.Subgraphs()[0]->Ops().back());
  }

  {
    // Bodies to compile are the decompositions of npu call ops.
    auto& sgs = partition_result->second;

    ASSERT_EQ(sgs.Size(), 2);
    ASSERT_EQ(sgs.Elements().front(), decomp1);
    ASSERT_EQ(sgs.Elements().back(), decomp2);
  }
}

TEST(PartitionTest, NestedNpuCallComposite) {
  auto model_wrap = testing::LoadTestFileModel("nested_composite.tflite");
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap.Get();
  auto plugins = CompilerPlugin::LoadPlugins({kTestPluginSearchPath});

  ASSERT_EQ(model.NumSubgraphs(), 3);

  auto partition_result = PartitionModel(plugins->front(), model);
  ASSERT_TRUE(partition_result);

  auto& ops = partition_result->first;
  ASSERT_EQ(ops.size(), 1);
  ASSERT_EQ(ops.front()->OpCode(), kLiteRtOpCodeTflCustom);

  auto& sgs = partition_result->second;
  ASSERT_EQ(sgs.Size(), 1);
  ASSERT_EQ(sgs.Elements().front()->Op(0).OpCode(), kLiteRtOpCodeShloComposite);
}

}  // namespace
}  // namespace litert::internal
