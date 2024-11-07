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

#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_support.h"
#include "tensorflow/lite/experimental/litert/core/graph_tools.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"
#include "tensorflow/lite/experimental/litert/core/util/buffer_ref.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/test/common.h"

namespace litert::internal {
namespace {

using ::litert::OwningBufferRef;
using ::litert::internal::VerifyFlatbuffer;

Model LoadModelThroughRoundTrip(std::string_view path) {
  auto model = litert::testing::LoadTestFileModel(path);

  OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();

  LITERT_CHECK_STATUS_OK_MSG(
      LiteRtSerializeModel(model.Release(), &data, &size, &offset),
      "Failed to serialize model");

  // Reload model.
  LiteRtModel result = nullptr;
  LITERT_CHECK_STATUS_OK_MSG(
      LiteRtLoadModelFromMemory(buf.Data(), buf.Size(), &result),
      "Failed to re load model");

  return Model::CreateFromOwnedHandle(result);
}

class TestWithPath : public ::testing::TestWithParam<std::string_view> {};

class TopologyTest : public ::testing::TestWithParam<LiteRtModel> {
 public:
  static std::vector<LiteRtModel> MakeTestModels(
      const std::vector<std::string>& paths) {
    std::vector<LiteRtModel> result;

    for (auto p : paths) {
      result.push_back(litert::testing::LoadTestFileModel(p).Release());
      result.push_back(LoadModelThroughRoundTrip(p).Release());
    }

    return result;
  }
};

TEST(LiteRtModelTest, TestLoadTestDataBadFilepath) {
  LiteRtModel model = nullptr;
  ASSERT_STATUS_HAS_CODE(LiteRtLoadModelFromFile("bad_path", &model),
                         kLiteRtStatusErrorFileIO);
}

TEST(LiteRtModelTest, TestLoadTestDataBadFileData) {
  // NOLINTBEGIN
#ifndef NDEBUG
  // In debug mode, flatbuffers will `assert` while verifying. This will
  // cause this test to crash (as expected).
  GTEST_SKIP();
#endif
  std::filesystem::path test_file_path(::testing::TempDir());
  test_file_path.append("bad_file.txt");

  std::ofstream bad_file;
  bad_file.open(test_file_path.c_str());
  bad_file << "not_tflite";
  bad_file.close();

  LiteRtModel model = nullptr;
  ASSERT_STATUS_HAS_CODE(
      LiteRtLoadModelFromFile(test_file_path.c_str(), &model),
      kLiteRtStatusErrorInvalidFlatbuffer);
  // NOLINTEND
}

TEST(TestSerializeModel, TestMetadata) {
  auto model = litert::testing::LoadTestFileModel("add_simple.tflite");

  constexpr static std::string_view kMetadataName = "an_soc_manufacturer";
  constexpr static std::string_view kMetadataData = "My_Meta_Data";

  ASSERT_STATUS_OK(model.Get()->PushMetadata(
      kMetadataName, OwningBufferRef<uint8_t>(kMetadataData)));

  ASSERT_RESULT_OK_ASSIGN(auto serialized, SerializeModel(std::move(model)));
  EXPECT_TRUE(VerifyFlatbuffer(serialized.Span()));
  ASSERT_RESULT_OK_MOVE(auto re_loaded, LoadModelFromMemory(serialized));
  ASSERT_RESULT_OK_ASSIGN(auto metadata,
                          re_loaded.Get()->FindMetadata(kMetadataName));
  EXPECT_EQ(metadata.StrView(), kMetadataData);
}

using AddSimpleTest = TopologyTest;

TEST_P(AddSimpleTest, TestBuildModelAddSimple) {
  Model model = Model::CreateFromOwnedHandle(GetParam());

  // func(arg0)
  //  output = tfl.add(arg0, arg0)
  //  return(output)
  //

  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph, GetSubgraph(model.Get()));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_inputs, GetSubgraphInputs(subgraph));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_outputs, GetSubgraphOutputs(subgraph));

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_RESULT_OK_ASSIGN(auto ops, GetSubgraphOps(subgraph));
  ASSERT_TRUE(ValidateTopology(ops));

  ASSERT_EQ(ops.size(), 1);
  auto op = ops[0];

  RankedTypeInfo float_2by2_type(kLiteRtElementTypeFloat32, {2, 2});
  ASSERT_TRUE(MatchOpType(op, {float_2by2_type, float_2by2_type},
                          {float_2by2_type}, kLiteRtOpCodeTflAdd));

  ASSERT_RESULT_OK_ASSIGN(auto op_inputs, GetOpIns(op));
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs[0], subgraph_inputs[0]);
  ASSERT_EQ(op_inputs[0], op_inputs[1]);

  ASSERT_RESULT_OK_ASSIGN(auto op_out, GetOnlyOpOut(op));
  ASSERT_EQ(op_out, subgraph_outputs[0]);

  ASSERT_TRUE(MatchNoWeights(subgraph_outputs[0]));
  ASSERT_TRUE(MatchNoWeights(subgraph_inputs[0]));
}

INSTANTIATE_TEST_SUITE_P(
    AddSimpleTests, AddSimpleTest,
    ::testing::ValuesIn(TopologyTest::MakeTestModels({"add_simple.tflite"})));

using AddCstTest = TopologyTest;

TEST_P(AddCstTest, TestBuildModelAddCst) {
  Model model = Model::CreateFromOwnedHandle(GetParam());

  // func(arg0)
  //  cst = ConstantTensor([1, 2, 3, 4])
  //  output = tfl.add(arg0, cst)
  //  return(output)
  //

  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph, GetSubgraph(model.Get()));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_inputs, GetSubgraphInputs(subgraph));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_outputs, GetSubgraphOutputs(subgraph));

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_RESULT_OK_ASSIGN(auto ops, GetSubgraphOps(subgraph));
  ASSERT_TRUE(ValidateTopology(ops));

  ASSERT_EQ(ops.size(), 1);
  auto op = ops[0];

  RankedTypeInfo float_2by2_type(kLiteRtElementTypeFloat32, {4});
  ASSERT_TRUE(MatchOpType(op, {float_2by2_type, float_2by2_type},
                          {float_2by2_type}, kLiteRtOpCodeTflAdd));

  ASSERT_RESULT_OK_ASSIGN(auto op_inputs, GetOpIns(op));
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs[0], subgraph_inputs[0]);
  ASSERT_TRUE(MatchWeights(op_inputs[1],
                           absl::Span<const float>({1.0, 2.0, 3.0, 4.0})));

  ASSERT_RESULT_OK_ASSIGN(auto op_out, GetOnlyOpOut(op));
  ASSERT_EQ(op_out, subgraph_outputs[0]);

  ASSERT_TRUE(MatchNoWeights(subgraph_outputs[0]));
  ASSERT_TRUE(MatchNoWeights(subgraph_inputs[0]));
}

INSTANTIATE_TEST_SUITE_P(
    AddCstTests, AddCstTest,
    ::testing::ValuesIn(TopologyTest::MakeTestModels({"add_cst.tflite"})));

using SimpleMultiOpTest = TopologyTest;

TEST_P(SimpleMultiOpTest, TestBuildModelSimpleMultiAdd) {
  Model model = Model::CreateFromOwnedHandle(GetParam());

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph, GetSubgraph(model.Get()));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_inputs, GetSubgraphInputs(subgraph));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_outputs, GetSubgraphOutputs(subgraph));

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_RESULT_OK_ASSIGN(auto ops, GetSubgraphOps(subgraph));
  ASSERT_TRUE(ValidateTopology(ops));

  ASSERT_EQ(ops.size(), 4);
  for (auto op : ops) {
    ASSERT_RESULT_OK_ASSIGN(auto inputs, GetOpIns(op));
    ASSERT_EQ(inputs.size(), 2);
    ASSERT_EQ(inputs[0], inputs[1]);
  }

  RankedTypeInfo float_2by2_type(kLiteRtElementTypeFloat32, {2, 2});

  ASSERT_TRUE(MatchOpType(ops[2], {float_2by2_type, float_2by2_type},
                          {float_2by2_type}, kLiteRtOpCodeTflMul));
}

INSTANTIATE_TEST_SUITE_P(SimpleMultiOpTests, SimpleMultiOpTest,
                         ::testing::ValuesIn(TopologyTest::MakeTestModels(
                             {"simple_multi_op.tflite"})));

}  // namespace
}  // namespace litert::internal
