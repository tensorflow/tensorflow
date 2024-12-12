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
#include <utility>
#include <vector>

#include <gmock/gmock.h>  // IWYU pragma: keep
#include <gtest/gtest.h>
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_model.h"
#include "tensorflow/lite/experimental/litert/c/litert_op_code.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_element_type.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model_predicates.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_file_test_util.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/test_macros.h"
#include "tensorflow/lite/experimental/litert/test/test_models.h"
#include "tensorflow/lite/experimental/litert/tools/dump.h"

namespace litert::internal {
namespace {

using ::litert::testing::ValidateTopology;

Model LoadModelThroughRoundTrip(absl::string_view path) {
  auto model = litert::testing::LoadTestFileModel(path);

  OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();

  LITERT_CHECK_STATUS_OK(
      LiteRtSerializeModel(model.Release(), &data, &size, &offset));

  // Reload model.
  LiteRtModel result = nullptr;
  LITERT_CHECK_STATUS_OK(
      LiteRtCreateModelFromBuffer(buf.Data(), buf.Size(), &result));

  return Model::CreateFromOwnedHandle(result);
}

class TestWithModelPath : public ::testing::TestWithParam<absl::string_view> {
 protected:
  std::string GetTestModelPath() const {
    return testing::GetTestFilePath(GetParam());
  }
};

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
  LITERT_ASSERT_STATUS_HAS_CODE(LiteRtCreateModelFromFile("bad_path", &model),
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
  LITERT_ASSERT_STATUS_HAS_CODE(
      LiteRtCreateModelFromFile(test_file_path.c_str(), &model),
      kLiteRtStatusErrorInvalidFlatbuffer);
  // NOLINTEND
}

TEST(TestSerializeModel, TestMetadata) {
  auto model = litert::testing::LoadTestFileModel("add_simple.tflite");

  constexpr static absl::string_view kMetadataName = "an_soc_manufacturer";
  constexpr static absl::string_view kMetadataData = "My_Meta_Data";

  LITERT_ASSERT_STATUS_OK(model.Get()->PushMetadata(
      kMetadataName, OwningBufferRef<uint8_t>(kMetadataData)));

  auto serialized = SerializeModel(std::move(model));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));

  auto re_loaded = LoadModelFromBuffer(*serialized);
  auto metadata = re_loaded->get()->FindMetadata(kMetadataName);
  EXPECT_EQ(metadata->StrView(), kMetadataData);
}

using AddSimpleTest = TopologyTest;

TEST_P(AddSimpleTest, TestBuildModelAddSimple) {
  Model model = Model::CreateFromOwnedHandle(GetParam());

  // func(arg0)
  //  output = tfl.add(arg0, arg0)
  //  return(output)
  //

  auto subgraph = model.MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_TRUE(ValidateTopology(ops));

  ASSERT_EQ(ops.size(), 1);
  const auto& op = ops.front();

  const TensorTypeInfo float_2by2_type(ElementType::Float32, {2, 2});
  ASSERT_TRUE(
      MatchOpType(op, {float_2by2_type, float_2by2_type}, {float_2by2_type}));
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflAdd);

  const auto op_inputs = op.Inputs();
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs.front().Get(), subgraph_inputs.front().Get());
  ASSERT_EQ(op_inputs.front().Get(), op_inputs.back().Get());

  const auto op_outputs = op.Outputs();
  ASSERT_EQ(op_outputs.size(), 1);
  ASSERT_EQ(op_outputs.front().Get(), subgraph_outputs.front().Get());

  ASSERT_FALSE(subgraph_outputs.front().IsConstant());
  ASSERT_FALSE(subgraph_inputs.front().IsConstant());
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

  auto subgraph = model.MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_TRUE(ValidateTopology(ops));

  ASSERT_EQ(ops.size(), 1);
  const auto& op = ops.front();

  const TensorTypeInfo float_by4_type(ElementType::Float32, {4});
  ASSERT_TRUE(
      MatchOpType(op, {float_by4_type, float_by4_type}, {float_by4_type}));
  EXPECT_EQ(op.Code(), kLiteRtOpCodeTflAdd);

  const auto op_inputs = op.Inputs();
  ASSERT_EQ(op_inputs.size(), 2);
  ASSERT_EQ(op_inputs.front().Get(), subgraph_inputs.front().Get());
  ASSERT_TRUE(MatchWeights(op_inputs.back(),
                           absl::Span<const float>({1.0, 2.0, 3.0, 4.0})));

  const auto op_outputs = op.Outputs();
  ASSERT_EQ(op_outputs.size(), 1);
  ASSERT_EQ(op_outputs.front().Get(), subgraph_outputs.front().Get());

  ASSERT_FALSE(subgraph_outputs.front().IsConstant());
  ASSERT_FALSE(subgraph_inputs.front().IsConstant());
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

  auto subgraph = model.MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  ASSERT_TRUE(ValidateTopology(ops));
  ASSERT_EQ(ops.size(), 4);

  for (const auto& op : ops) {
    const auto inputs = op.Inputs();
    ASSERT_EQ(inputs.size(), 2);
    ASSERT_EQ(inputs.front().Get(), inputs.back().Get());
  }

  const TensorTypeInfo float_2by2_type(ElementType::Float32, {2, 2});

  ASSERT_TRUE(MatchOpType(ops.at(2), {float_2by2_type, float_2by2_type},
                          {float_2by2_type}));
  EXPECT_EQ(ops.at(2).Code(), kLiteRtOpCodeTflMul);
}

INSTANTIATE_TEST_SUITE_P(SimpleMultiOpTests, SimpleMultiOpTest,
                         ::testing::ValuesIn(TopologyTest::MakeTestModels(
                             {"simple_multi_op.tflite"})));

using ModelLoadOpCheckTest = TestWithModelPath;

TEST_P(ModelLoadOpCheckTest, CheckOps) {
  const auto model_path = GetTestModelPath();

  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(model_path);
  ASSERT_TRUE(flatbuffer);
  auto expected_fb = flatbuffer->get()->Unpack();

  auto model = LoadModelFromFile(model_path);
  ASSERT_TRUE(model);

  const auto& subgraph = model->get()->MainSubgraph();
  const auto& ops = subgraph.ops;

  const auto& fb_subgraph = *expected_fb->subgraphs.front();
  const auto& fb_ops = fb_subgraph.operators;
  const auto& fb_tensors = fb_subgraph.tensors;

  ASSERT_EQ(ops.size(), fb_ops.size());

  auto get_tfl_tensor = [&](uint32_t ind) -> const TflTensor& {
    return *fb_tensors.at(ind);
  };

  for (auto i = 0; i < ops.size(); ++i) {
    Dump(*ops.at(i));
    ASSERT_TRUE(EqualsFbOp(*ops.at(i), *fb_ops.at(i), get_tfl_tensor));
  }
}

INSTANTIATE_TEST_SUITE_P(ModelLoadQuantizedOpCheckTest, ModelLoadOpCheckTest,
                         ::testing::ValuesIn(kAllQModels));

INSTANTIATE_TEST_SUITE_P(ModelLoadDynamicOpCheckTest, ModelLoadOpCheckTest,
                         ::testing::ValuesIn({static_cast<absl::string_view>(
                             "dynamic_shape_tensor.tflite")}));

INSTANTIATE_TEST_SUITE_P(
    ModelLoadStaticOpCheckTest, ModelLoadOpCheckTest,
    ::testing::ValuesIn({static_cast<absl::string_view>("one_mul.tflite")}));

using ModelSerializeOpCheckTest = TestWithModelPath;

TEST_P(ModelSerializeOpCheckTest, CheckOps) {
  const auto model_path = GetTestModelPath();

  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(model_path);
  ASSERT_TRUE(flatbuffer);
  auto expected_fb = flatbuffer->get()->Unpack();

  auto model = LoadModelFromFile(model_path);
  ASSERT_TRUE(model);

  auto serialized = SerializeModel(std::move(**model));
  auto serialized_fb = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(serialized_fb);
  auto actual_fb = serialized_fb->get()->Unpack();

  const auto& expected_fb_subgraph = *expected_fb->subgraphs.front();
  const auto& expected_fb_ops = expected_fb_subgraph.operators;
  const auto& expected_fb_tensors = expected_fb_subgraph.tensors;

  const auto& actual_fb_subgraph = *actual_fb->subgraphs.front();
  const auto& actual_fb_ops = actual_fb_subgraph.operators;
  const auto& actual_fb_tensors = actual_fb_subgraph.tensors;

  ASSERT_EQ(expected_fb_ops.size(), actual_fb_ops.size());
  for (auto i = 0; i < actual_fb_ops.size(); ++i) {
    const auto& expected = *expected_fb_ops.at(i);
    const auto& actual = *actual_fb_ops.at(i);
    EXPECT_EQ(expected.inputs.size(), actual.inputs.size());
    EXPECT_EQ(expected.outputs.size(), actual.outputs.size());
  }

  ASSERT_EQ(expected_fb_tensors.size(), actual_fb_tensors.size());
  for (auto i = 0; i < actual_fb_tensors.size(); ++i) {
    const auto& expected = *expected_fb_tensors.at(i);
    const auto& actual = *actual_fb_tensors.at(i);

    EXPECT_EQ(actual.type, expected.type);
    EXPECT_EQ(actual.shape, expected.shape);
    EXPECT_EQ(actual.shape_signature, expected.shape_signature);

    const auto expected_q_params = expected.quantization.get();
    const auto actual_q_params = actual.quantization.get();

    const auto neither_quantized =
        !IsQuantized(expected_q_params) && !IsQuantized(actual_q_params);
    const auto both_per_tensor = IsPerTensorQuantized(expected_q_params) &&
                                 IsPerTensorQuantized(actual_q_params);
    ASSERT_TRUE(neither_quantized || both_per_tensor);

    if (both_per_tensor) {
      const auto expected_per_tensor = AsPerTensorQparams(expected_q_params);
      const auto actual_per_tensor = AsPerTensorQparams(actual_q_params);
      EXPECT_EQ(*expected_per_tensor, *actual_per_tensor);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    ModelSerializeStaticOpCheckTest, ModelSerializeOpCheckTest,
    ::testing::ValuesIn({static_cast<absl::string_view>("one_mul.tflite")}));

INSTANTIATE_TEST_SUITE_P(ModelSerializeDynamicOpCheckTest,
                         ModelSerializeOpCheckTest,
                         ::testing::ValuesIn({static_cast<absl::string_view>(
                             "dynamic_shape_tensor.tflite")}));

INSTANTIATE_TEST_SUITE_P(ModelSerializeQuantizedOpCheckTest,
                         ModelSerializeOpCheckTest,
                         ::testing::ValuesIn(kAllQModels));

}  // namespace
}  // namespace litert::internal
