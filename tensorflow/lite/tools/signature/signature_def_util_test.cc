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

#include <string>

#include <gtest/gtest.h>
#include "tensorflow/cc/saved_model/signature_constants.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/lite/core/c/c_api.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/testing/util.h"

namespace tflite {
namespace {

using tensorflow::kClassifyMethodName;
using tensorflow::kDefaultServingSignatureDefKey;
using tensorflow::kPredictMethodName;
using tensorflow::SignatureDef;
using tensorflow::Status;

constexpr char kSignatureInput[] = "input";
constexpr char kSignatureOutput[] = "output";
constexpr char kTestFilePath[] = "tensorflow/lite/testdata/add.bin";

class SimpleSignatureDefUtilTest : public testing::Test {
 protected:
  void SetUp() override {
    flatbuffer_model_ = FlatBufferModel::BuildFromFile(kTestFilePath);
    ASSERT_NE(flatbuffer_model_, nullptr);
    model_ = flatbuffer_model_->GetModel();
    ASSERT_NE(model_, nullptr);
  }

  SignatureDef GetTestSignatureDef() {
    auto signature_def = SignatureDef();
    tensorflow::TensorInfo input_tensor;
    tensorflow::TensorInfo output_tensor;
    *input_tensor.mutable_name() = kSignatureInput;
    *output_tensor.mutable_name() = kSignatureOutput;
    *signature_def.mutable_method_name() = kClassifyMethodName;
    (*signature_def.mutable_inputs())[kSignatureInput] = input_tensor;
    (*signature_def.mutable_outputs())[kSignatureOutput] = output_tensor;
    return signature_def;
  }
  std::unique_ptr<FlatBufferModel> flatbuffer_model_;
  const Model* model_;
};

TEST_F(SimpleSignatureDefUtilTest, SetSignatureDefTest) {
  SignatureDef expected_signature_def = GetTestSignatureDef();
  std::string model_output;
  const std::map<string, SignatureDef> expected_signature_def_map = {
      {kDefaultServingSignatureDefKey, expected_signature_def}};
  EXPECT_EQ(
      ::tensorflow::OkStatus(),
      SetSignatureDefMap(model_, expected_signature_def_map, &model_output));
  const Model* add_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_TRUE(HasSignatureDef(add_model, kDefaultServingSignatureDefKey));
  std::map<string, SignatureDef> test_signature_def_map;
  EXPECT_EQ(::tensorflow::OkStatus(),
            GetSignatureDefMap(add_model, &test_signature_def_map));
  SignatureDef test_signature_def =
      test_signature_def_map[kDefaultServingSignatureDefKey];
  EXPECT_EQ(expected_signature_def.SerializeAsString(),
            test_signature_def.SerializeAsString());
}

TEST_F(SimpleSignatureDefUtilTest, OverwriteSignatureDefTest) {
  auto expected_signature_def = GetTestSignatureDef();
  std::string model_output;
  std::map<string, SignatureDef> expected_signature_def_map = {
      {kDefaultServingSignatureDefKey, expected_signature_def}};
  EXPECT_EQ(
      ::tensorflow::OkStatus(),
      SetSignatureDefMap(model_, expected_signature_def_map, &model_output));
  const Model* add_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_TRUE(HasSignatureDef(add_model, kDefaultServingSignatureDefKey));
  std::map<string, SignatureDef> test_signature_def_map;
  EXPECT_EQ(::tensorflow::OkStatus(),
            GetSignatureDefMap(add_model, &test_signature_def_map));
  SignatureDef test_signature_def =
      test_signature_def_map[kDefaultServingSignatureDefKey];
  EXPECT_EQ(expected_signature_def.SerializeAsString(),
            test_signature_def.SerializeAsString());
  *expected_signature_def.mutable_method_name() = kPredictMethodName;
  expected_signature_def_map.erase(
      expected_signature_def_map.find(kDefaultServingSignatureDefKey));
  constexpr char kTestSignatureDefKey[] = "ServingTest";
  expected_signature_def_map[kTestSignatureDefKey] = expected_signature_def;
  EXPECT_EQ(
      ::tensorflow::OkStatus(),
      SetSignatureDefMap(add_model, expected_signature_def_map, &model_output));
  const Model* final_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_FALSE(HasSignatureDef(final_model, kDefaultServingSignatureDefKey));
  EXPECT_EQ(::tensorflow::OkStatus(),
            GetSignatureDefMap(final_model, &test_signature_def_map));
  EXPECT_NE(expected_signature_def.SerializeAsString(),
            test_signature_def.SerializeAsString());
  EXPECT_TRUE(HasSignatureDef(final_model, kTestSignatureDefKey));
  EXPECT_EQ(::tensorflow::OkStatus(),
            GetSignatureDefMap(final_model, &test_signature_def_map));
  test_signature_def = test_signature_def_map[kTestSignatureDefKey];
  EXPECT_EQ(expected_signature_def.SerializeAsString(),
            test_signature_def.SerializeAsString());
}

TEST_F(SimpleSignatureDefUtilTest, GetSignatureDefTest) {
  std::map<string, SignatureDef> test_signature_def_map;
  EXPECT_EQ(::tensorflow::OkStatus(),
            GetSignatureDefMap(model_, &test_signature_def_map));
  EXPECT_FALSE(HasSignatureDef(model_, kDefaultServingSignatureDefKey));
}

TEST_F(SimpleSignatureDefUtilTest, ClearSignatureDefTest) {
  const int expected_num_buffers = model_->buffers()->size();
  auto expected_signature_def = GetTestSignatureDef();
  std::string model_output;
  std::map<string, SignatureDef> expected_signature_def_map = {
      {kDefaultServingSignatureDefKey, expected_signature_def}};
  EXPECT_EQ(
      ::tensorflow::OkStatus(),
      SetSignatureDefMap(model_, expected_signature_def_map, &model_output));
  const Model* add_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_TRUE(HasSignatureDef(add_model, kDefaultServingSignatureDefKey));
  SignatureDef test_signature_def;
  std::map<string, SignatureDef> test_signature_def_map;
  EXPECT_EQ(::tensorflow::OkStatus(),
            GetSignatureDefMap(add_model, &test_signature_def_map));
  test_signature_def = test_signature_def_map[kDefaultServingSignatureDefKey];
  EXPECT_EQ(expected_signature_def.SerializeAsString(),
            test_signature_def.SerializeAsString());
  EXPECT_EQ(::tensorflow::OkStatus(),
            ClearSignatureDefMap(add_model, &model_output));
  const Model* clear_model = flatbuffers::GetRoot<Model>(model_output.data());
  EXPECT_FALSE(HasSignatureDef(clear_model, kDefaultServingSignatureDefKey));
  EXPECT_EQ(expected_num_buffers, clear_model->buffers()->size());
}

TEST_F(SimpleSignatureDefUtilTest, SetSignatureDefErrorsTest) {
  std::map<string, SignatureDef> test_signature_def_map;
  std::string model_output;
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(
      SetSignatureDefMap(model_, test_signature_def_map, &model_output)));
  SignatureDef test_signature_def;
  test_signature_def_map[kDefaultServingSignatureDefKey] = test_signature_def;
  EXPECT_TRUE(tensorflow::errors::IsInvalidArgument(
      SetSignatureDefMap(model_, test_signature_def_map, nullptr)));
}

}  // namespace
}  // namespace tflite
