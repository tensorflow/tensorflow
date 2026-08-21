/* Copyright 2026 The OpenXLA Authors.

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

#include "xla/hlo/tools/hlo_diff/render/model_explorer_url_generator.h"

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "third_party/json/include/nlohmann/json_fwd.hpp"
#include "third_party/json/src/json.hpp"
#include "xla/hlo/ir/hlo_instruction.h"
#include "xla/hlo/testlib/hlo_hardware_independent_test_base.h"
#include "xla/hlo/testlib/verified_hlo_module.h"
#include "xla/hlo/tools/hlo_diff/hlo_diff_result.h"
#include "xla/hlo/tools/hlo_diff/proto/diff_result.pb.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/direct_hlo_to_json_graph_convert.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_adapter/schema_structs.h"
#include "xla/hlo/tools/hlo_diff/render/hlo_gumgraph_model_explorer_renderer.h"
#include "xla/tsl/platform/env.h"
#include "tsl/platform/path.h"

namespace xla {
namespace hlo_diff {
namespace {

using ::nlohmann::json;
using ::testing::HasSubstr;
using ::testing::status::StatusIs;
using ::tooling::visualization_client::GetInstructionId;
using ::tooling::visualization_client::Graph;
using ::tooling::visualization_client::GraphCollection;
using ::tooling::visualization_client::Subgraph;

std::string UrlDecode(absl::string_view url) {
  std::string unescaped;
  unescaped.reserve(url.size());
  for (size_t i = 0; i < url.size(); ++i) {
    if (url[i] == '%' && i + 2 < url.size()) {
      unsigned int hex;
      if (absl::SimpleHexAtoi(url.substr(i + 1, 2), &hex)) {
        unescaped.push_back(static_cast<char>(hex));
        i += 2;
        continue;
      }
    } else if (url[i] == '+') {
      unescaped.push_back(' ');
      continue;
    }
    unescaped.push_back(url[i]);
  }
  return unescaped;
}

class ModelExplorerUrlGeneratorTest : public HloHardwareIndependentTestBase {};

TEST_F(ModelExplorerUrlGeneratorTest, GenerateMeUrl) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true
ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true
ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  DiffResult diff_result;
  ASSERT_OK_AND_ASSIGN(MeJson result,
                       RenderMe(*module_l, *module_r, diff_result));

  std::string me_json_path = tsl::io::JoinPath(testing::TempDir(), "me.json");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), me_json_path,
                                   result.DumpGraphCollectionJson()));
  std::string sync_nav_path = "path/to/me.sync_nav.json";
  std::string node_data_path = "path/to/me.node_data.json";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MeUrlGenerator> url_generator,
      MeUrlGenerator::Create(&result.graph_collection, me_json_path,
                             sync_nav_path, node_data_path));
  std::string url = url_generator->Generate();

  EXPECT_TRUE(absl::StartsWith(url, "http://localhost:8080/?data="));

  std::string data_param = UrlDecode(url.substr(url.find("data=") + 5));
  json data = json::parse(data_param);

  ASSERT_TRUE(data.contains("models"));
  ASSERT_EQ(data["models"].size(), 1);
  EXPECT_EQ(data["models"][0]["url"], me_json_path);

  ASSERT_TRUE(data.contains("uiState"));
  ASSERT_TRUE(data["uiState"].contains("paneStates"));
  ASSERT_EQ(data["uiState"]["paneStates"].size(), 2);
  EXPECT_THAT(data["uiState"]["paneStates"][0]["selectedCollectionLabel"],
              HasSubstr("me.json (left)"));
  EXPECT_EQ(data["uiState"]["paneStates"][0]["selectedGraphId"], "entry_left");
  EXPECT_THAT(data["uiState"]["paneStates"][1]["selectedCollectionLabel"],
              HasSubstr("me.json (right)"));
  EXPECT_EQ(data["uiState"]["paneStates"][1]["selectedGraphId"], "entry_right");

  ASSERT_TRUE(data.contains("sync"));
  EXPECT_EQ(data["sync"]["mode"], "from_cns");
  EXPECT_EQ(data["sync"]["cnsPath"], sync_nav_path);

  ASSERT_TRUE(data.contains("nodeData"));
  ASSERT_EQ(data["nodeData"].size(), 1);
  EXPECT_EQ(data["nodeData"][0], node_data_path);

  // No selected nodes.
  EXPECT_FALSE(data["uiState"]["paneStates"][0].contains("selectedNodeId"));
  EXPECT_FALSE(data["uiState"]["paneStates"][1].contains("selectedNodeId"));
}

TEST_F(ModelExplorerUrlGeneratorTest, GenerateMeUrlWithSelectedChangedNodes) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true
ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true
ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  DiffResult diff_result;
  HloInstruction* left_p0 =
      module_l->entry_computation()->parameter_instruction(0);
  HloInstruction* right_p0 =
      module_r->entry_computation()->parameter_instruction(0);
  diff_result.changed_instructions[left_p0] = right_p0;
  ASSERT_OK_AND_ASSIGN(MeJson result,
                       RenderMe(*module_l, *module_r, diff_result));
  auto [left_selected_node_id, right_selected_node_id] =
      MeUrlGenerator::SelectInitialSelectedNodes(diff_result);

  EXPECT_EQ(left_selected_node_id, absl::StrCat(GetInstructionId(left_p0)));
  EXPECT_EQ(right_selected_node_id, absl::StrCat(GetInstructionId(right_p0)));

  std::string me_json_path = tsl::io::JoinPath(testing::TempDir(), "me.json");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), me_json_path,
                                   result.DumpGraphCollectionJson()));
  std::string sync_nav_path = "path/to/me.sync_nav.json";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MeUrlGenerator> url_generator,
      MeUrlGenerator::Create(&result.graph_collection, me_json_path,
                             sync_nav_path, /*node_data_path=*/""));
  std::string url = url_generator->GenerateWithSelectedNodes(
      left_selected_node_id, right_selected_node_id);

  EXPECT_TRUE(absl::StartsWith(url, "http://localhost:8080/?data="));

  std::string data_param = UrlDecode(url.substr(url.find("data=") + 5));
  json data = json::parse(data_param);

  EXPECT_EQ(data["uiState"]["paneStates"][0]["selectedNodeId"],
            left_selected_node_id);
  EXPECT_EQ(data["uiState"]["paneStates"][1]["selectedNodeId"],
            right_selected_node_id);
}

TEST_F(ModelExplorerUrlGeneratorTest, GenerateMeUrlWithSelectedUnmatchedNodes) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true
ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true
ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  DiffResult diff_result;
  HloInstruction* left_p0 =
      module_l->entry_computation()->parameter_instruction(0);
  HloInstruction* right_p0 =
      module_r->entry_computation()->parameter_instruction(0);
  diff_result.left_module_unmatched_instructions.insert(left_p0);
  diff_result.right_module_unmatched_instructions.insert(right_p0);
  ASSERT_OK_AND_ASSIGN(MeJson result,
                       RenderMe(*module_l, *module_r, diff_result));
  auto [left_selected_node_id, right_selected_node_id] =
      MeUrlGenerator::SelectInitialSelectedNodes(diff_result);

  EXPECT_EQ(left_selected_node_id, absl::StrCat(GetInstructionId(left_p0)));
  EXPECT_EQ(right_selected_node_id, absl::StrCat(GetInstructionId(right_p0)));

  std::string me_json_path = tsl::io::JoinPath(testing::TempDir(), "me.json");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), me_json_path,
                                   result.DumpGraphCollectionJson()));
  std::string sync_nav_path = "path/to/me.sync_nav.json";
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MeUrlGenerator> url_generator,
      MeUrlGenerator::Create(&result.graph_collection, me_json_path,
                             sync_nav_path, /*node_data_path=*/""));
  std::string url = url_generator->GenerateWithSelectedNodes(
      left_selected_node_id, right_selected_node_id);

  EXPECT_TRUE(absl::StartsWith(url, "http://localhost:8080/?data="));

  std::string data_param = UrlDecode(url.substr(url.find("data=") + 5));
  json data = json::parse(data_param);

  EXPECT_EQ(data["uiState"]["paneStates"][0]["selectedNodeId"],
            left_selected_node_id);
  EXPECT_EQ(data["uiState"]["paneStates"][1]["selectedNodeId"],
            right_selected_node_id);
}

TEST_F(ModelExplorerUrlGeneratorTest, GenerateMeUrlWithCustomBaseUrl) {
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_l,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true
ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<xla::VerifiedHloModule> module_r,
                       ParseAndReturnVerifiedModule(R"(
HloModule module, is_scheduled=true
ENTRY entry {
  parameter.0 = f32[] parameter(0)
  parameter.1 = f32[] parameter(1)
  add.0 = f32[] add(parameter.0, parameter.1)
}
)"));
  DiffResult diff_result;
  ASSERT_OK_AND_ASSIGN(MeJson result,
                       RenderMe(*module_l, *module_r, diff_result));

  std::string me_json_path = tsl::io::JoinPath(testing::TempDir(), "me.json");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), me_json_path,
                                   result.DumpGraphCollectionJson()));
  std::string sync_nav_path = "path/to/me.sync_nav.json";
  std::vector<std::string> node_data_paths = {"path/to/me.node_data.json"};
  ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<MeUrlGenerator> url_generator,
      MeUrlGenerator::Create(&result.graph_collection, me_json_path,
                             sync_nav_path, node_data_paths, "http://my-url"));
  std::string url = url_generator->Generate();

  EXPECT_TRUE(absl::StartsWith(url, "http://my-url/?data="));
}

TEST_F(ModelExplorerUrlGeneratorTest, ReturnErrorForNullGraphCollection) {
  EXPECT_THAT(MeUrlGenerator::Create(
                  /*graph_collection=*/nullptr, /*me_json_path=*/"",
                  /*sync_nav_path=*/"", /*node_data_path=*/""),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("graph_collection is null")));
}

TEST_F(ModelExplorerUrlGeneratorTest, ReturnErrorForInvalidGraphCollection) {
  GraphCollection graph_collection;
  graph_collection.graphs.push_back(Graph{"left", {Subgraph("left")}});
  EXPECT_THAT(MeUrlGenerator::Create(&graph_collection,
                                     /*me_json_path=*/"",
                                     /*sync_nav_path=*/"",
                                     /*node_data_path=*/""),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("The graph collection doesn't contain exactly "
                                 "two graphs.")));
}

TEST_F(ModelExplorerUrlGeneratorTest, ReturnErrorForInvalidLeftGraph) {
  GraphCollection graph_collection;
  graph_collection.graphs.push_back(Graph{"left", {}});
  graph_collection.graphs.push_back(Graph{"right", {Subgraph("right")}});
  EXPECT_THAT(MeUrlGenerator::Create(&graph_collection,
                                     /*me_json_path=*/"",
                                     /*sync_nav_path=*/"",
                                     /*node_data_path=*/""),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("The graph collection doesn't contain exactly "
                                 "one subgraph in the first graph.")));
}

TEST_F(ModelExplorerUrlGeneratorTest, ReturnErrorForInvalidRightGraph) {
  GraphCollection graph_collection;
  graph_collection.graphs.push_back(Graph{"left", {Subgraph("left")}});
  graph_collection.graphs.push_back(Graph{"right", {}});
  EXPECT_THAT(MeUrlGenerator::Create(&graph_collection,
                                     /*me_json_path=*/"",
                                     /*sync_nav_path=*/"",
                                     /*node_data_path=*/""),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("The graph collection doesn't contain exactly "
                                 "one subgraph in the second graph.")));
}

TEST_F(ModelExplorerUrlGeneratorTest, ReturnErrorForEmptyMeJsonPath) {
  GraphCollection graph_collection;
  graph_collection.graphs.push_back(Graph{"left", {Subgraph("left")}});
  graph_collection.graphs.push_back(Graph{"right", {Subgraph("right")}});
  EXPECT_THAT(MeUrlGenerator::Create(&graph_collection,
                                     /*me_json_path=*/"",
                                     /*sync_nav_path=*/"",
                                     /*node_data_path=*/""),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("me_json_path is empty")));
}

TEST_F(ModelExplorerUrlGeneratorTest, ReturnErrorForLargeJson) {
  // Create a large file.
  std::string me_json_path = tsl::io::JoinPath(testing::TempDir(), "me.json");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), me_json_path,
                                   std::string(512 * 1024 * 1024 + 1, 'a')));
  GraphCollection graph_collection;
  graph_collection.graphs.push_back(Graph{"left", {Subgraph("left")}});
  graph_collection.graphs.push_back(Graph{"right", {Subgraph("right")}});
  EXPECT_THAT(
      MeUrlGenerator::Create(&graph_collection, me_json_path,
                             /*sync_nav_path=*/"", /*node_data_path=*/""),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("The diff results are too large for Model Explorer")));
}

}  // namespace
}  // namespace hlo_diff
}  // namespace xla
