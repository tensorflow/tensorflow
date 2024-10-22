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

// NOLINTNEXTLINE
#include <filesystem>
#include <fstream>
#include <string>
#include <string_view>
#include <vector>

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>
#include "llvm/ADT/ArrayRef.h"
#include "llvm/ADT/SmallVector.h"
#include "tensorflow/lite/experimental/lrt/c/litert_common.h"
#include "tensorflow/lite/experimental/lrt/c/litert_model.h"
#include "tensorflow/lite/experimental/lrt/c/litert_op_code.h"
#include "tensorflow/lite/experimental/lrt/cc/litert_support.h"
#include "tensorflow/lite/experimental/lrt/core/graph_tools.h"
#include "tensorflow/lite/experimental/lrt/core/litert_model_init.h"
#include "tensorflow/lite/experimental/lrt/core/util/buffer_ref.h"
#include "tensorflow/lite/experimental/lrt/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/lrt/test/common.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace {

using ::graph_tools::GetMetadata;
using ::litert::BufferRef;
using ::litert::OwningBufferRef;
using ::litert::internal::VerifyFlatbuffer;

inline UniqueLiteRtModel LoadModelThroughRoundTrip(std::string_view path) {
  auto model = litert::testing::LoadTestFileModel(path);

  OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();

  LITERT_CHECK_STATUS_OK_MSG(
      SerializeModel(model.release(), &data, &size, &offset),
      "Failed to serialize model");

  // Reload model.
  LiteRtModel result = nullptr;
  LITERT_CHECK_STATUS_OK_MSG(LoadModel(buf.Data(), buf.Size(), &result),
                             "Failed to re load model");

  return UniqueLiteRtModel(result);
}

class TestWithPath : public ::testing::TestWithParam<std::string_view> {};

class TopologyTest : public ::testing::TestWithParam<LiteRtModel> {
 public:
  static std::vector<LiteRtModel> MakeTestModels(
      const std::vector<std::string>& paths) {
    std::vector<LiteRtModel> result;

    for (auto p : paths) {
      result.push_back(litert::testing::LoadTestFileModel(p).release());
      result.push_back(LoadModelThroughRoundTrip(p).release());
    }

    return result;
  }
};

TEST(LiteRtModelTest, TestLoadTestDataBadFilepath) {
  LiteRtModel model = nullptr;
  ASSERT_STATUS_HAS_CODE(LoadModelFromFile("bad_path", &model),
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
  ASSERT_STATUS_HAS_CODE(LoadModelFromFile(test_file_path.c_str(), &model),
                         kLiteRtStatusErrorInvalidFlatbuffer);
  // NOLINTEND
}

TEST(TestSerializeModel, TestAllocations) {
  auto model = litert::testing::LoadTestFileModel("add_simple.tflite");

  OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();

  ASSERT_STATUS_OK(SerializeModel(model.release(), &data, &size, &offset));
  EXPECT_TRUE(VerifyFlatbuffer(data + offset, size - offset));
}

TEST(TestSerializeModel, TestMetadata) {
  auto model = litert::testing::LoadTestFileModel("add_simple.tflite");

  constexpr static std::string_view kMetadataName = "an_soc_manufacturer";
  constexpr static std::string_view kMetadataData = "My_Meta_Data";

  ASSERT_STATUS_OK(AppendMetadata(model.get(), kMetadataData.data(),
                                  kMetadataData.size(), kMetadataName.data()));
  ASSERT_RESULT_OK_ASSIGN(auto m_buffer,
                          GetMetadata(model.get(), kMetadataName));
  EXPECT_EQ(m_buffer.StrView(), kMetadataData);

  OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();

  ASSERT_STATUS_OK(SerializeModel(model.release(), &data, &size, &offset));
  EXPECT_TRUE(VerifyFlatbuffer(buf.Span()));

  auto new_model = tflite::UnPackModel(buf.Data());

  ASSERT_NE(new_model, nullptr);
  ASSERT_GT(new_model->metadata.size(), 0);

  tflite::MetadataT* fb_metadata = nullptr;
  for (auto& m : new_model->metadata) {
    if (m->name == kMetadataName) {
      fb_metadata = m.get();
      break;
    }
  }
  ASSERT_NE(fb_metadata, nullptr);
  ASSERT_GE(fb_metadata->buffer, 0);
  ASSERT_LT(fb_metadata->buffer, new_model->buffers.size());

  tflite::BufferT* metadata_buffer =
      new_model->buffers.at(fb_metadata->buffer).get();
  BufferRef metadata_buf(metadata_buffer->data.data(),
                         metadata_buffer->data.size());

  EXPECT_EQ(metadata_buf.StrView(), kMetadataData);
}

TEST(TestSerializeModel, TestCustomOpCode) {
  auto model = litert::testing::LoadTestFileModel("add_simple.tflite");

  constexpr static std::string_view kCustomCode = "MyCustomCode";
  ASSERT_STATUS_OK(RegisterCustomOpCode(model.get(), kCustomCode.data()));

  OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();

  ASSERT_STATUS_OK(SerializeModel(model.release(), &data, &size, &offset));
  EXPECT_TRUE(VerifyFlatbuffer(buf.Span()));
  auto new_model = tflite::UnPackModel(buf.Data());

  tflite::OperatorCodeT* custom_op_code = nullptr;
  for (auto& c : new_model->operator_codes) {
    if (c->custom_code == kCustomCode) {
      custom_op_code = c.get();
      break;
    }
  }
  ASSERT_NE(custom_op_code, nullptr);
  ASSERT_EQ(custom_op_code->custom_code, kCustomCode);
  ASSERT_EQ(custom_op_code->builtin_code, tflite::BuiltinOperator_CUSTOM);
}

TEST_P(TestWithPath, TestConstructDestroy) {
  UniqueLiteRtModel model = litert::testing::LoadTestFileModel(GetParam());
}

TEST_P(TestWithPath, TestConstructDestroyRoundTrip) {
  UniqueLiteRtModel model = LoadModelThroughRoundTrip(GetParam());
}

INSTANTIATE_TEST_SUITE_P(InstTestWithPath, TestWithPath,
                         ::testing::Values("add_simple.tflite",
                                           "add_cst.tflite",
                                           "simple_multi_op.tflite"));

using AddSimpleTest = TopologyTest;

TEST_P(AddSimpleTest, TestBuildModelAddSimple) {
  UniqueLiteRtModel model(GetParam());

  // func(arg0)
  //  output = tfl.add(arg0, arg0)
  //  return(output)
  //

  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_inputs,
                          graph_tools::GetSubgraphInputs(subgraph));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_outputs,
                          graph_tools::GetSubgraphOutputs(subgraph));

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  ASSERT_TRUE(graph_tools::ValidateTopology(ops));

  ASSERT_EQ(ops.size(), 1);
  auto op = ops[0];

  graph_tools::RankedTypeInfo float_2by2_type(kLiteRtElementTypeFloat32,
                                              {2, 2});
  ASSERT_TRUE(graph_tools::MatchOpType(op, {float_2by2_type, float_2by2_type},
                                       {float_2by2_type}, kLiteRtOpCodeTflAdd));

  ASSERT_RESULT_OK_ASSIGN(auto op_inputs, graph_tools::GetOpIns(op));
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs[0], subgraph_inputs[0]);
  ASSERT_EQ(op_inputs[0], op_inputs[1]);

  ASSERT_RESULT_OK_ASSIGN(auto op_out, graph_tools::GetOnlyOpOut(op));
  ASSERT_EQ(op_out, subgraph_outputs[0]);

  ASSERT_TRUE(graph_tools::MatchNoWeights(subgraph_outputs[0]));
  ASSERT_TRUE(graph_tools::MatchNoWeights(subgraph_inputs[0]));
}

INSTANTIATE_TEST_SUITE_P(
    AddSimpleTests, AddSimpleTest,
    ::testing::ValuesIn(TopologyTest::MakeTestModels({"add_simple.tflite"})));

using AddCstTest = TopologyTest;

TEST_P(AddCstTest, TestBuildModelAddCst) {
  UniqueLiteRtModel model(GetParam());

  // func(arg0)
  //  cst = ConstantTensor([1, 2, 3, 4])
  //  output = tfl.add(arg0, cst)
  //  return(output)
  //

  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_inputs,
                          graph_tools::GetSubgraphInputs(subgraph));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_outputs,
                          graph_tools::GetSubgraphOutputs(subgraph));

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  ASSERT_TRUE(graph_tools::ValidateTopology(ops));

  ASSERT_EQ(ops.size(), 1);
  auto op = ops[0];

  graph_tools::RankedTypeInfo float_2by2_type(kLiteRtElementTypeFloat32, {4});
  ASSERT_TRUE(graph_tools::MatchOpType(op, {float_2by2_type, float_2by2_type},
                                       {float_2by2_type}, kLiteRtOpCodeTflAdd));

  ASSERT_RESULT_OK_ASSIGN(auto op_inputs, graph_tools::GetOpIns(op));
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs[0], subgraph_inputs[0]);
  ASSERT_TRUE(graph_tools::MatchWeights(
      op_inputs[1], llvm::ArrayRef<float>{1.0, 2.0, 3.0, 4.0}));

  ASSERT_RESULT_OK_ASSIGN(auto op_out, graph_tools::GetOnlyOpOut(op));
  ASSERT_EQ(op_out, subgraph_outputs[0]);

  ASSERT_TRUE(graph_tools::MatchNoWeights(subgraph_outputs[0]));
  ASSERT_TRUE(graph_tools::MatchNoWeights(subgraph_inputs[0]));
}

INSTANTIATE_TEST_SUITE_P(
    AddCstTests, AddCstTest,
    ::testing::ValuesIn(TopologyTest::MakeTestModels({"add_cst.tflite"})));

using SimpleMultiOpTest = TopologyTest;

TEST_P(SimpleMultiOpTest, TestBuildModelSimpleMultiAdd) {
  UniqueLiteRtModel model(GetParam());

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  ASSERT_RESULT_OK_ASSIGN(LiteRtSubgraph subgraph,
                          graph_tools::GetSubgraph(model.get()));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_inputs,
                          graph_tools::GetSubgraphInputs(subgraph));
  ASSERT_RESULT_OK_ASSIGN(auto subgraph_outputs,
                          graph_tools::GetSubgraphOutputs(subgraph));

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_RESULT_OK_ASSIGN(auto ops, graph_tools::GetSubgraphOps(subgraph));
  ASSERT_TRUE(graph_tools::ValidateTopology(ops));

  ASSERT_EQ(ops.size(), 4);
  for (auto op : ops) {
    ASSERT_RESULT_OK_ASSIGN(auto inputs, graph_tools::GetOpIns(op));
    ASSERT_EQ(inputs.size(), 2);
    ASSERT_EQ(inputs[0], inputs[1]);
  }

  graph_tools::RankedTypeInfo float_2by2_type(kLiteRtElementTypeFloat32,
                                              {2, 2});

  ASSERT_TRUE(graph_tools::MatchOpType(ops[2],
                                       {float_2by2_type, float_2by2_type},
                                       {float_2by2_type}, kLiteRtOpCodeTflMul));
}

INSTANTIATE_TEST_SUITE_P(SimpleMultiOpTests, SimpleMultiOpTest,
                         ::testing::ValuesIn(TopologyTest::MakeTestModels(
                             {"simple_multi_op.tflite"})));

}  // namespace
