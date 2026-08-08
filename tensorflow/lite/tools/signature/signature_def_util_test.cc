/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/tools/signature/signature_def_util.h"

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <utility>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/status/status.h"
#include "absl/strings/string_view.h"
#include "flatbuffers/buffer.h"  // from @flatbuffers
#include "flatbuffers/flatbuffer_builder.h"  // from @flatbuffers
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/protobuf/meta_graph.pb.h"
#include "tensorflow/lite/core/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace {

using ::tensorflow::kClassifyMethodName;
using ::tensorflow::kDefaultServingSignatureDefKey;
using ::tensorflow::kPredictMethodName;
using ::tensorflow::SignatureDef;
using ::testing::ElementsAre;
using ::testing::EqualsProto;
using ::testing::Pair;
using ::testing::status::StatusIs;

constexpr absl::string_view kSignatureInput = "input";
constexpr absl::string_view kSignatureOutput = "output";
constexpr absl::string_view kTestFilePath =
    "tensorflow/lite/testdata/add.bin";

SignatureDef GetTestSignatureDef() {
  SignatureDef signature_def;
  tensorflow::TensorInfo input_tensor;
  tensorflow::TensorInfo output_tensor;
  input_tensor.set_name(kSignatureInput);
  output_tensor.set_name(kSignatureOutput);
  signature_def.set_method_name(kClassifyMethodName);
  (*signature_def.mutable_inputs())[kSignatureInput] = std::move(input_tensor);
  (*signature_def.mutable_outputs())[kSignatureOutput] =
      std::move(output_tensor);
  return signature_def;
}

class SimpleSignatureDefUtilTest : public testing::Test {
 protected:
  void SetUp() override {
    flatbuffer_model_ =
        FlatBufferModel::BuildFromFile(std::string(kTestFilePath).c_str());
    if (!flatbuffer_model_) {
      GTEST_SKIP() << "Failed to load model";
    }
    model_ = flatbuffer_model_->GetModel();
    if (!model_) {
      GTEST_SKIP() << "Failed to get model";
    }
  }

  std::unique_ptr<FlatBufferModel> flatbuffer_model_;
  const Model* model_;
};

TEST_F(SimpleSignatureDefUtilTest, SetSignatureDefTest) {
  SignatureDef expected_signature_def = GetTestSignatureDef();
  std::string model_output;
  const std::map<std::string, SignatureDef> expected_signature_def_map = {
      {std::string(kDefaultServingSignatureDefKey), expected_signature_def}};
  ASSERT_OK(
      SetSignatureDefMap(model_, expected_signature_def_map, &model_output));
  const Model* add_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_TRUE(HasSignatureDef(add_model, kDefaultServingSignatureDefKey));
  std::map<std::string, SignatureDef> test_signature_def_map;
  ASSERT_OK(GetSignatureDefMap(add_model, &test_signature_def_map));
  EXPECT_THAT(test_signature_def_map,
              ElementsAre(Pair(std::string(kDefaultServingSignatureDefKey),
                               EqualsProto(expected_signature_def))));
}

TEST_F(SimpleSignatureDefUtilTest, OverwriteSignatureDefTest) {
  SignatureDef expected_signature_def = GetTestSignatureDef();
  std::string model_output;
  std::map<std::string, SignatureDef> expected_signature_def_map = {
      {std::string(kDefaultServingSignatureDefKey), expected_signature_def}};
  ASSERT_OK(
      SetSignatureDefMap(model_, expected_signature_def_map, &model_output));
  const Model* add_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_TRUE(HasSignatureDef(add_model, kDefaultServingSignatureDefKey));
  std::map<std::string, SignatureDef> test_signature_def_map;
  ASSERT_OK(GetSignatureDefMap(add_model, &test_signature_def_map));
  EXPECT_THAT(test_signature_def_map,
              ElementsAre(Pair(std::string(kDefaultServingSignatureDefKey),
                               EqualsProto(expected_signature_def))));
  expected_signature_def.set_method_name(std::string(kPredictMethodName));
  expected_signature_def_map.erase(std::string(kDefaultServingSignatureDefKey));
  static constexpr absl::string_view kTestSignatureDefKey = "ServingTest";
  expected_signature_def_map[std::string(kTestSignatureDefKey)] =
      expected_signature_def;
  ASSERT_OK(
      SetSignatureDefMap(add_model, expected_signature_def_map, &model_output));
  const Model* final_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_FALSE(HasSignatureDef(final_model, kDefaultServingSignatureDefKey));
  EXPECT_TRUE(HasSignatureDef(final_model, kTestSignatureDefKey));
  ASSERT_OK(GetSignatureDefMap(final_model, &test_signature_def_map));
  EXPECT_THAT(test_signature_def_map,
              ElementsAre(Pair(std::string(kTestSignatureDefKey),
                               EqualsProto(expected_signature_def))));
}

TEST_F(SimpleSignatureDefUtilTest, GetSignatureDefTest) {
  std::map<std::string, SignatureDef> test_signature_def_map;
  EXPECT_OK(GetSignatureDefMap(model_, &test_signature_def_map));
  EXPECT_TRUE(test_signature_def_map.empty());
  EXPECT_FALSE(HasSignatureDef(model_, kDefaultServingSignatureDefKey));
}

TEST_F(SimpleSignatureDefUtilTest, ClearSignatureDefTest) {
  const uint32_t expected_num_buffers = model_->buffers()->size();
  SignatureDef expected_signature_def = GetTestSignatureDef();
  std::string model_output;
  std::map<std::string, SignatureDef> expected_signature_def_map = {
      {std::string(kDefaultServingSignatureDefKey), expected_signature_def}};
  ASSERT_OK(
      SetSignatureDefMap(model_, expected_signature_def_map, &model_output));
  const Model* add_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_TRUE(HasSignatureDef(add_model, kDefaultServingSignatureDefKey));
  std::map<std::string, SignatureDef> test_signature_def_map;
  ASSERT_OK(GetSignatureDefMap(add_model, &test_signature_def_map));
  SignatureDef test_signature_def =
      test_signature_def_map[std::string(kDefaultServingSignatureDefKey)];
  EXPECT_THAT(test_signature_def, EqualsProto(expected_signature_def));
  ASSERT_OK(ClearSignatureDefMap(add_model, &model_output));
  const Model* clear_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_FALSE(HasSignatureDef(clear_model, kDefaultServingSignatureDefKey));
  EXPECT_EQ(expected_num_buffers + 1, clear_model->buffers()->size());
}

TEST_F(SimpleSignatureDefUtilTest, SetSignatureDefErrorsTest) {
  std::map<std::string, SignatureDef> test_signature_def_map;
  std::string model_output;
  EXPECT_THAT(SetSignatureDefMap(model_, test_signature_def_map, &model_output),
              StatusIs(absl::StatusCode::kInvalidArgument));
  SignatureDef test_signature_def;
  test_signature_def_map[std::string(kDefaultServingSignatureDefKey)] =
      test_signature_def;
  EXPECT_THAT(SetSignatureDefMap(model_, test_signature_def_map, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(SimpleSignatureDefUtilTest, GetSignatureDefErrorsTest) {
  auto mutable_model = std::make_unique<ModelT>();
  model_->UnPackTo(mutable_model.get(), nullptr);
  uint32_t buffer_id = mutable_model->buffers.size();
  auto buffer = std::make_unique<BufferT>();
  buffer->data = {0, 1};
  mutable_model->buffers.emplace_back(std::move(buffer));
  auto sigdef_metadata = std::make_unique<MetadataT>();
  sigdef_metadata->buffer = buffer_id;
  sigdef_metadata->name = kSignatureDefsMetadataName;
  mutable_model->metadata.emplace_back(std::move(sigdef_metadata));

  flatbuffers::FlatBufferBuilder builder;
  flatbuffers::Offset<Model> packed_model =
      Model::Pack(builder, mutable_model.get());
  FinishModelBuffer(builder, packed_model);

  const Model* invalid_model =
      flatbuffers::GetRoot<Model>(builder.GetBufferPointer());

  EXPECT_FALSE(HasSignatureDef(invalid_model, kDefaultServingSignatureDefKey));
  std::map<std::string, SignatureDef> test_signature_def_map;
  EXPECT_THAT(GetSignatureDefMap(invalid_model, &test_signature_def_map),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace tflite
