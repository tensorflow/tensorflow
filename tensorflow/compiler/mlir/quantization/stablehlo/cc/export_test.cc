/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/quantization/stablehlo/cc/export.h"

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/core/framework/graph.pb.h"
#include "tensorflow/core/framework/node_def.pb.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/protobuf/saver.pb.h"
#include "tsl/platform/protobuf.h"  // IWYU pragma: keep
#include "tsl/platform/status_matchers.h"
#include "tsl/platform/statusor.h"

namespace stablehlo::quantization {
namespace {

using ::tensorflow::AssetFileDef;
using ::tensorflow::GraphDef;
using ::tensorflow::SaverDef;
using ::tensorflow::quantization::ExportedModel;
using ::testing::HasSubstr;
using ::testing::IsEmpty;
using ::testing::SizeIs;
using ::testing::StrEq;
using ::tsl::protobuf::TextFormat;
using ::tsl::testing::StatusIs;

TEST(CreateExportedModelTest, CreateExportedModelBasicFieldsSet) {
  GraphDef graph_def{};
  ASSERT_TRUE(
      TextFormat::ParseFromString(R"pb(node { name: "foo" })pb", &graph_def));

  const ExportedModel exported_model =
      CreateExportedModel(std::move(graph_def), "init_node_name",
                          "checkpoint_dir", /*saver_def=*/std::nullopt,
                          /*function_aliases=*/{}, /*asset_file_defs=*/{});
  ASSERT_THAT(exported_model.graph_def().node(), SizeIs(1));
  EXPECT_THAT(exported_model.graph_def().node()[0].name(), StrEq("foo"));

  EXPECT_THAT(exported_model.init_node_name(), StrEq("init_node_name"));
  EXPECT_THAT(exported_model.checkpoint_dir(), StrEq("checkpoint_dir"));
  EXPECT_FALSE(exported_model.has_saver_def());
  EXPECT_THAT(exported_model.function_aliases(), IsEmpty());
  EXPECT_THAT(exported_model.asset_file_defs(), IsEmpty());
}

TEST(CreateExportedModelTest, CreateExportedModelWithAddedFunctionAliases) {
  const ExportedModel exported_model = CreateExportedModel(
      GraphDef(), /*init_node_name=*/"", /*checkpoint_dir=*/"",
      /*saver_def=*/std::nullopt,
      /*function_aliases=*/{{"func1", "alias1"}, {"func2", "alias2"}},
      /*asset_file_defs=*/{});
  ASSERT_THAT(exported_model.function_aliases(), SizeIs(2));
  EXPECT_TRUE(exported_model.function_aliases().contains("func1"));
  EXPECT_THAT(exported_model.function_aliases().at("func1"), StrEq("alias1"));
  EXPECT_TRUE(exported_model.function_aliases().contains("func2"));
  EXPECT_THAT(exported_model.function_aliases().at("func2"), StrEq("alias2"));
}

TEST(CreateExportedModelTest, CreateExportedModelWithAddedAssetFileDefs) {
  AssetFileDef asset1;
  ASSERT_TRUE(
      TextFormat::ParseFromString(R"pb(filename: "fname1")pb", &asset1));

  AssetFileDef asset2;
  ASSERT_TRUE(
      TextFormat::ParseFromString(R"pb(filename: "fname2")pb", &asset2));

  const ExportedModel exported_model = CreateExportedModel(
      GraphDef(), /*init_node_name=*/"", /*checkpoint_dir=*/"",
      /*saver_def=*/std::nullopt, /*function_aliases=*/{},
      /*asset_file_defs=*/{asset1, asset2});
  ASSERT_THAT(exported_model.asset_file_defs(), SizeIs(2));
  EXPECT_THAT(exported_model.asset_file_defs()[0].filename(), StrEq("fname1"));
  EXPECT_THAT(exported_model.asset_file_defs()[1].filename(), StrEq("fname2"));
}

TEST(CreateExportedModelTest, CreateExportedModelWithAddedSaverDef) {
  SaverDef saver_def;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(filename_tensor_name: "my_file")pb", &saver_def));

  const ExportedModel exported_model = CreateExportedModel(
      GraphDef(), /*init_node_name=*/"", /*checkpoint_dir=*/"", saver_def,
      /*function_aliases=*/{}, /*asset_file_defs=*/{});
  EXPECT_THAT(exported_model.saver_def().filename_tensor_name(), "my_file");
}

TEST(CreateSaverDefTest, CreateValidSaverDef) {
  // Needs to have a _Arg node with an attribute "tf_saved_model.index_path" =
  // ["__tf_file_prefix"].
  GraphDef graph_def;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(node {
             name: "foo",
             op: "_Arg",
             attr {
               key: "tf_saved_model.index_path",
               value { list { s: "__tf_file_prefix" } }
             }
           })pb",
      &graph_def));

  // Restore op's name should start with "restore_op" and the save op's name
  // should start with "tf_quant__save_op".
  const std::vector<std::string> control_ret_node_names = {
      "restore_op_0", "tf_quant__save_op_0"};

  TF_ASSERT_OK_AND_ASSIGN(const std::optional<SaverDef> saver_def,
                          CreateSaverDef(control_ret_node_names, graph_def));
  ASSERT_NE(saver_def, std::nullopt);
  EXPECT_THAT(saver_def->version(), SaverDef::V2);
  EXPECT_THAT(saver_def->restore_op_name(), "restore_op_0");
  EXPECT_THAT(saver_def->filename_tensor_name(), "foo:0");
  EXPECT_THAT(saver_def->save_tensor_name(), "tf_quant__save_op_0:0");
}

TEST(CreateSaverDefTest, ReturnsNulloptIfNoSaverDefRelatedNodesExist) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::optional<SaverDef> saver_def,
      CreateSaverDef(/*control_ret_node_names=*/{}, GraphDef()));
  EXPECT_EQ(saver_def, std::nullopt);
}

TEST(CreateSaverDefTest, ReturnsErrorStatusIfSaverDefNodesPartiallyExist) {
  // An _Arg node missing the attribute "tf_saved_model.index_path" =
  // ["__tf_file_prefix"].
  GraphDef graph_def;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(node { name: "foo", op: "_Arg" })pb", &graph_def));

  // Restore op's name should start with "restore_op" and the save op's name
  // should start with "tf_quant__save_op".
  const std::vector<std::string> control_ret_node_names = {
      "restore_op_0", "tf_quant__save_op_0"};

  const absl::StatusOr<std::optional<SaverDef>> saver_def =
      CreateSaverDef(control_ret_node_names, graph_def);
  EXPECT_THAT(
      saver_def,
      StatusIs(
          absl::StatusCode::kInternal,
          HasSubstr(
              "should be either all empty strings or all non-empty strings")));
}

}  // namespace
}  // namespace stablehlo::quantization
