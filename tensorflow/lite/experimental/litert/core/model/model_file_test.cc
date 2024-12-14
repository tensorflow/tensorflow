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

#include <array>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <functional>
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
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model_predicates.h"
#include "tensorflow/lite/experimental/litert/core/byte_code_util.h"
#include "tensorflow/lite/experimental/litert/core/model/graph_validation.h"
#include "tensorflow/lite/experimental/litert/core/model/model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_file_test_util.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/experimental/litert/core/model/model_serialize.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/experimental/litert/test/common.h"
#include "tensorflow/lite/experimental/litert/test/test_macros.h"
#include "tensorflow/lite/experimental/litert/test/test_models.h"

namespace litert::internal {
namespace {

using ::litert::testing::GetTestFilePath;
using ::testing::Each;
using ::testing::ElementsAreArray;
using ::testing::FloatEq;
using ::testing::Values;

using ModelFactory = std::function<Expected<Model>()>;

static constexpr absl::string_view kAddSimple = "add_simple.tflite";
static constexpr absl::string_view kAddCst = "add_cst.tflite";
static constexpr absl::string_view kDynamicShapeModel =
    "dynamic_shape_tensor.tflite";
static constexpr absl::string_view kSimpleMultiOp = "simple_multi_op.tflite";
static constexpr absl::string_view kOneMul = "one_mul.tflite";
static constexpr absl::string_view kSimpleMultiSubgraph =
    "multi_subgraph.tflite";
static constexpr absl::string_view kCstMultiSubgraph =
    "cst_multi_subgraph.tflite";

// Load a model, then serialize and re-load. Used to test serialization.
Expected<Model> LoadModelThroughRoundTrip(absl::string_view filename) {
  auto model = Model::CreateFromFile(GetTestFilePath(filename));
  if (!model) {
    return model.Error();
  }

  OwningBufferRef buf;
  auto [data, size, offset] = buf.GetWeak();

  LITERT_EXPECT_OK(
      LiteRtSerializeModel(model->Release(), &data, &size, &offset));

  // Reload model.
  LiteRtModel result = nullptr;
  LITERT_EXPECT_OK(
      LiteRtCreateModelFromBuffer(buf.Data(), buf.Size(), &result));

  return Model::CreateFromOwnedHandle(result);
}

ModelFactory MakeRoundTripFactory(absl::string_view filename) {
  return [=]() { return LoadModelThroughRoundTrip(filename); };
}

ModelFactory MakeLoadFactory(absl::string_view filename) {
  return [=]() { return Model::CreateFromFile(GetTestFilePath(filename)); };
}

// Test fixture parameterized by a file path to test model.
class TestWithModelPath : public ::testing::TestWithParam<absl::string_view> {
 protected:
  std::string GetTestModelPath() const {
    return testing::GetTestFilePath(GetParam());
  }
};

// Test fixture pareterized by a function that loads a model.
class TestWithModelFactory : public ::testing::TestWithParam<ModelFactory> {
 protected:
  Expected<Model> LoadModel() { return GetParam()(); }
};

// Simple tests
//===---------------------------------------------------------------------------

TEST(ModelLoadTest, BadFilepath) {
  LiteRtModel model = nullptr;
  LITERT_ASSERT_STATUS_HAS_CODE(LiteRtCreateModelFromFile("bad_path", &model),
                                kLiteRtStatusErrorFileIO);
}

TEST(ModelLoadTest, BadFileData) {
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

TEST(ModelLoadTest, WithMetadata) {
  constexpr static std::string_view kMetadataName = "an_soc_manufacturer";
  constexpr static std::string_view kMetadataData = "My_Meta_Data";

  auto flatbuffer =
      FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(kAddSimple));
  auto tfl_model = flatbuffer->get()->Unpack();
  PushMetadata(kMetadataName, *tfl_model,
               BufferRef<uint8_t>(kMetadataData.data(), kMetadataData.size()));
  auto serialialized = SerializeFlatbuffer(*tfl_model);

  auto litert_model = LoadModelFromBuffer(serialialized);
  ASSERT_TRUE(litert_model);

  auto metadata = litert_model->get()->FindMetadata(kMetadataName);
  ASSERT_TRUE(metadata);
  EXPECT_EQ(metadata->StrView(), kMetadataData);
}

TEST(ModelSerializeTest, WithMetadata) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);

  constexpr static absl::string_view kMetadataName = "an_soc_manufacturer";
  constexpr static absl::string_view kMetadataData = "My_Meta_Data";

  LITERT_ASSERT_STATUS_OK(model.Get()->PushMetadata(
      kMetadataName, OwningBufferRef<uint8_t>(kMetadataData)));

  auto serialized = SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));

  auto re_loaded = LoadModelFromBuffer(*serialized);
  auto metadata = re_loaded->get()->FindMetadata(kMetadataName);
  EXPECT_EQ(metadata->StrView(), kMetadataData);
}

TEST(ModelLoadTest, WithSignature) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);
  auto& litert_model = *model.Get();

  auto signature =
      litert_model.FindSignature(LiteRtSignatureT::kDefaultSignatureKey);
  ASSERT_TRUE(signature);

  EXPECT_EQ(signature->get().InputNames().size(), 1);
  EXPECT_EQ(signature->get().OutputNames().size(), 1);
  EXPECT_EQ(&signature->get().GetSubgraph(), litert_model.MainSubgraph());
}

TEST(ModelSerializeTest, WithSignature) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);
  auto& litert_model = *model.Get();

  static constexpr char kInput[] = "foo";
  static constexpr char kOutput[] = "bar";
  static constexpr char kKey[] = "newKey";

  LiteRtSignatureT signature(litert_model.MainSubgraph(), {kInput}, {kOutput},
                             kKey);
  litert_model.EmplaceSignature(std::move(signature));

  auto serialized = SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));

  auto re_loaded = LoadModelFromBuffer(*serialized);
  auto re_loaded_signature = re_loaded->get()->FindSignature(kKey);
  ASSERT_TRUE(re_loaded_signature);
  const auto& sig = re_loaded_signature->get();

  const auto& inputs = sig.InputNames();
  const auto& outputs = sig.OutputNames();
  EXPECT_THAT(inputs, ElementsAreArray({kInput}));
  EXPECT_THAT(outputs, ElementsAreArray({kOutput}));
  EXPECT_EQ(&sig.GetSubgraph(), re_loaded->get()->MainSubgraph());
}

TEST(ModelSerializeTest, WithMetadataByteCode) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);
  auto& litert_model = *model.Get();

  static constexpr absl::string_view kManufacturer = "Dodge";
  static constexpr absl::string_view kModel = "Dart";
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr auto kSerialization = Serialization::kMetadata;

  // TODO(@lukeboyer) consider wrapping the tag & push metadata for npu
  // in a helper function somewhere.
  {
    auto build_stamp = MakeBuildStamp(kManufacturer, kModel, kSerialization);
    litert_model.PushMetadata(kLiteRtBuildStampKey, *build_stamp);
    litert_model.PushMetadata(kByteCodeMetadataKey, kByteCode);
  }

  auto serialized = SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));
  auto re_loaded = LoadModelFromBuffer(*serialized);
  ASSERT_TRUE(re_loaded);
  auto& re_loaded_model = **re_loaded;

  auto build_stamp =
      ParseBuildStamp(*re_loaded_model.FindMetadata(kLiteRtBuildStampKey));
  ASSERT_TRUE(build_stamp);

  EXPECT_EQ(std::get<0>(*build_stamp), kManufacturer);
  EXPECT_EQ(std::get<1>(*build_stamp), kModel);
  EXPECT_EQ(std::get<2>(*build_stamp), kSerialization);

  auto byte_code = re_loaded_model.FindMetadata(kByteCodeMetadataKey);
  ASSERT_TRUE(byte_code);
  EXPECT_EQ(byte_code->StrView(), kByteCode);
}

TEST(ModelSerializeTest, WithAppendByteCode) {
  auto model = litert::testing::LoadTestFileModel(kAddSimple);
  auto& litert_model = *model.Get();

  static constexpr absl::string_view kManufacturer = "Honda";
  static constexpr absl::string_view kModel = "Civic";
  static constexpr absl::string_view kByteCode = "SOME_BYTE_CODE";
  static constexpr auto kSerialization = Serialization::kAppend;

  {
    auto build_stamp = MakeBuildStamp(kManufacturer, kModel, kSerialization);
    litert_model.PushMetadata(kLiteRtBuildStampKey, *build_stamp);
    litert_model.PushMetadata(kByteCodeMetadataKey, kByteCode);
  }

  auto serialized = SerializeModel(std::move(*model.Get()));
  EXPECT_TRUE(VerifyFlatbuffer(serialized->Span()));
  auto re_loaded = LoadModelFromBuffer(*serialized);
  ASSERT_TRUE(re_loaded);
  auto& re_loaded_model = **re_loaded;

  auto build_stamp =
      ParseBuildStamp(*re_loaded_model.FindMetadata(kLiteRtBuildStampKey));
  ASSERT_TRUE(build_stamp);

  EXPECT_EQ(std::get<0>(*build_stamp), kManufacturer);
  EXPECT_EQ(std::get<1>(*build_stamp), kModel);
  EXPECT_EQ(std::get<2>(*build_stamp), kSerialization);

  auto byte_code_metadata = re_loaded_model.FindMetadata(kByteCodeMetadataKey);
  ASSERT_TRUE(byte_code_metadata);
  auto byte_code_offset = ParseByteCodePlaceholder(*byte_code_metadata);
  ASSERT_TRUE(byte_code_offset);

  const auto offset = std::get<0>(*byte_code_offset);
  const auto size = std::get<1>(*byte_code_offset);

  ASSERT_EQ(offset + size, serialized->Size());
  EXPECT_EQ(serialized->StrView().substr(offset, size), kByteCode);
}

// Tests that explicitly check litert graph structure.
//===---------------------------------------------------------------------------

using AddSimpleTest = TestWithModelFactory;

TEST_P(AddSimpleTest, CheckGraph) {
  auto model = LoadModel();
  ASSERT_TRUE(model);

  // func(arg0)
  //  output = tfl.add(arg0, arg0)
  //  return(output)
  //

  auto subgraph = model->MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  const auto& internal_ops = subgraph->Get()->Ops();
  ASSERT_TRUE(
      ValidateLocalTopology(internal_ops.cbegin(), internal_ops.cend()));
  ASSERT_TRUE(ValidateSubgraphIO(*subgraph->Get()));

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

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, AddSimpleTest,
                         Values(MakeLoadFactory(kAddSimple)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, AddSimpleTest,
                         Values(MakeRoundTripFactory(kAddSimple)));

using AddCstTest = TestWithModelFactory;

TEST_P(AddCstTest, CheckGraph) {
  auto model = LoadModel();
  ASSERT_TRUE(model);

  // func(arg0)
  //  cst = ConstantTensor([1, 2, 3, 4])
  //  output = tfl.add(arg0, cst)
  //  return(output)
  //

  auto subgraph = model->MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  const auto& internal_ops = subgraph->Get()->Ops();
  ASSERT_TRUE(
      ValidateLocalTopology(internal_ops.cbegin(), internal_ops.cend()));
  ASSERT_TRUE(ValidateSubgraphIO(*subgraph->Get()));

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

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, AddCstTest,
                         Values(MakeLoadFactory(kAddCst)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, AddCstTest,
                         Values(MakeRoundTripFactory(kAddCst)));

using SimpleMultiOpTest = TestWithModelFactory;

TEST_P(SimpleMultiOpTest, CheckGraph) {
  auto model = LoadModel();
  ASSERT_TRUE(model);

  // func.func @main(arg0)
  //   0 = tfl.add arg0, arg0
  //   1 = tfl.mul 0, 0
  //   2 = tfl.mul 1, 1
  //   3 = tfl.add 2, 2
  //   return 3

  auto subgraph = model->MainSubgraph();
  const auto subgraph_inputs = subgraph->Inputs();
  const auto subgraph_outputs = subgraph->Outputs();
  const auto ops = subgraph->Ops();

  ASSERT_EQ(subgraph_inputs.size(), 1);
  ASSERT_EQ(subgraph_outputs.size(), 1);

  const auto& internal_ops = subgraph->Get()->Ops();
  ASSERT_TRUE(
      ValidateLocalTopology(internal_ops.cbegin(), internal_ops.cend()));
  ASSERT_TRUE(ValidateSubgraphIO(*subgraph->Get()));

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

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, SimpleMultiOpTest,
                         Values(MakeLoadFactory(kSimpleMultiOp)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, SimpleMultiOpTest,
                         Values(MakeRoundTripFactory(kSimpleMultiOp)));

using SimpleMultiSubgraphTest = TestWithModelFactory;

TEST_P(SimpleMultiSubgraphTest, CheckGraph) {
  auto model_wrap = LoadModel();
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap->Get();

  ASSERT_EQ(model.NumSubgraphs(), 3);

  {
    auto& main = *model.MainSubgraph();
    EXPECT_EQ(main.NumInputs(), 1);
    EXPECT_EQ(main.NumOutputs(), 1);
    EXPECT_EQ(main.Ops().size(), 1);
    EXPECT_EQ(main.Tensors().size(), 3);
    auto& op = main.Op(0);
    auto* cst = op.Inputs().back();
    auto data = Tensor(cst).WeightsData<float>();
    ASSERT_TRUE(data);
    EXPECT_THAT(*data, Each(FloatEq(-1.0)));
    EXPECT_TRUE(ValidateLocalTopology(main.Ops().cbegin(), main.Ops().cend()));
    EXPECT_TRUE(ValidateSubgraphIO(main));
  }

  {
    auto& func1 = model.Subgraph(1);
    EXPECT_EQ(func1.NumInputs(), 1);
    EXPECT_EQ(func1.NumOutputs(), 1);
    EXPECT_EQ(func1.Ops().size(), 1);
    EXPECT_EQ(func1.Tensors().size(), 3);
    auto& op = func1.Op(0);
    auto* cst = op.Inputs().back();
    auto data = Tensor(cst).WeightsData<float>();
    ASSERT_TRUE(data);
    EXPECT_THAT(*data, Each(FloatEq(1.0)));
    EXPECT_TRUE(
        ValidateLocalTopology(func1.Ops().cbegin(), func1.Ops().cend()));
    EXPECT_TRUE(ValidateSubgraphIO(func1));
  }

  {
    auto& func2 = model.Subgraph(2);
    EXPECT_EQ(func2.NumInputs(), 1);
    EXPECT_EQ(func2.NumOutputs(), 1);
    EXPECT_EQ(func2.Ops().size(), 1);
    EXPECT_EQ(func2.Tensors().size(), 3);
    auto& op = func2.Op(0);
    auto* cst = op.Inputs().back();
    auto data = Tensor(cst).WeightsData<float>();
    ASSERT_TRUE(data);
    EXPECT_THAT(*data, Each(FloatEq(2.0)));
    EXPECT_TRUE(
        ValidateLocalTopology(func2.Ops().cbegin(), func2.Ops().cend()));
    EXPECT_TRUE(ValidateSubgraphIO(func2));
  }
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, SimpleMultiSubgraphTest,
                         Values(MakeLoadFactory(kSimpleMultiSubgraph)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, SimpleMultiSubgraphTest,
                         Values(MakeRoundTripFactory(kSimpleMultiSubgraph)));

// Test when flatbuffer export has optimized multiple tensors to share the
// same buffer.
using MultiSubgraphDupeConstTest = TestWithModelFactory;

TEST_P(MultiSubgraphDupeConstTest, CheckGraph) {
  static constexpr std::array kWeights = {1.0, 2.0, 3.0, 4.0};

  auto model_wrap = LoadModel();
  ASSERT_TRUE(model_wrap);
  auto& model = *model_wrap->Get();

  ASSERT_EQ(model.NumSubgraphs(), 2);

  {
    ASSERT_EQ(model.Subgraph(0).Ops().size(), 1);
    ASSERT_EQ(model.Subgraph(0).Tensors().size(), 3);
    auto& cst = model.Subgraph(0).Op(0).Input(1);
    Tensor t(&cst);
    EXPECT_THAT(*t.WeightsData<float>(), ElementsAreArray(kWeights));
  }

  {
    ASSERT_EQ(model.Subgraph(1).Ops().size(), 1);
    ASSERT_EQ(model.Subgraph(1).Tensors().size(), 3);
    auto& cst = model.Subgraph(1).Op(0).Input(1);
    Tensor t(&cst);
    EXPECT_THAT(*t.WeightsData<float>(), ElementsAreArray(kWeights));
  }
}

INSTANTIATE_TEST_SUITE_P(ModelLoadTests, MultiSubgraphDupeConstTest,
                         Values(MakeLoadFactory(kCstMultiSubgraph)));

INSTANTIATE_TEST_SUITE_P(ModelSerializeTests, MultiSubgraphDupeConstTest,
                         Values(MakeRoundTripFactory(kCstMultiSubgraph)));

// Tests that programatically check litert against tflite models.
//===---------------------------------------------------------------------------

using ModelLoadOpCheckTest = TestWithModelPath;

TEST_P(ModelLoadOpCheckTest, CheckOps) {
  const auto model_path = GetTestModelPath();

  auto flatbuffer = FlatbufferWrapper::CreateFromTflFile(model_path);
  ASSERT_TRUE(flatbuffer);
  auto expected_fb = flatbuffer->get()->Unpack();

  auto model = LoadModelFromFile(model_path);
  ASSERT_TRUE(model);

  const auto* subgraph = model->get()->MainSubgraph();
  const auto& ops = subgraph->Ops();

  const auto& fb_subgraph = *expected_fb->subgraphs.front();
  const auto& fb_ops = fb_subgraph.operators;
  const auto& fb_tensors = fb_subgraph.tensors;

  ASSERT_EQ(ops.size(), fb_ops.size());

  auto get_tfl_tensor = [&](uint32_t ind) -> const TflTensor& {
    return *fb_tensors.at(ind);
  };

  for (auto i = 0; i < ops.size(); ++i) {
    ASSERT_TRUE(EqualsFbOp(*ops.at(i), *fb_ops.at(i), get_tfl_tensor));
  }
}

INSTANTIATE_TEST_SUITE_P(ModelLoadQuantizedOpCheckTest, ModelLoadOpCheckTest,
                         ::testing::ValuesIn(kAllQModels));

INSTANTIATE_TEST_SUITE_P(ModelLoadDynamicOpCheckTest, ModelLoadOpCheckTest,
                         ::testing::ValuesIn({kDynamicShapeModel}));

using ModelSerializeOpCheckTest = TestWithModelPath;

TEST_P(ModelSerializeOpCheckTest, CheckOps) {
  const auto model_path = GetTestModelPath();

  // Save the initial fb for comparison.
  auto expected_fb_data = FlatbufferWrapper::CreateFromTflFile(model_path);
  ASSERT_TRUE(expected_fb_data);
  auto expected_fb = expected_fb_data->get()->Unpack();

  // Round trip the model.
  auto model = LoadModelFromFile(model_path);
  ASSERT_TRUE(model);
  auto serialized = SerializeModel(std::move(**model));

  auto actual_fb_data = FlatbufferWrapper::CreateFromBuffer(*serialized);
  ASSERT_TRUE(actual_fb_data);
  auto actual_fb = actual_fb_data->get()->Unpack();

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

INSTANTIATE_TEST_SUITE_P(ModelSerializeOpCheckTest, ModelSerializeOpCheckTest,
                         ::testing::ValuesIn({kOneMul, kDynamicShapeModel}));

INSTANTIATE_TEST_SUITE_P(ModelSerializeQuantizedOpCheckTest,
                         ModelSerializeOpCheckTest,
                         ::testing::ValuesIn(kAllQModels));

}  // namespace
}  // namespace litert::internal
