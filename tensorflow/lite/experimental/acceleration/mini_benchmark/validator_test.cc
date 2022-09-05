/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/validator.h"

#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/configuration/configuration.pb.h"
#include "tensorflow/lite/experimental/acceleration/configuration/configuration_generated.h"
#include "tensorflow/lite/experimental/acceleration/configuration/proto_to_flatbuffer.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_validation_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_loader.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/custom_validation_embedder.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/schema/mutable/schema_generated.h"

// Note that these tests are not meant to be completely exhaustive, but to test
// error propagation.

namespace tflite {
namespace acceleration {
namespace {

using flatbuffers::FlatBufferBuilder;

class ValidatorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string validation_model_path = MiniBenchmarkTestHelper::DumpToTempFile(
        "mobilenet_quant_with_validation.tflite",
        g_tflite_acceleration_embedded_mobilenet_validation_model,
        g_tflite_acceleration_embedded_mobilenet_validation_model_len);
    ASSERT_TRUE(!validation_model_path.empty());
    validation_model_loader_ =
        std::make_unique<PathModelLoader>(validation_model_path);

    std::string plain_model_path = MiniBenchmarkTestHelper::DumpToTempFile(
        "mobilenet_quant.tflite",
        g_tflite_acceleration_embedded_mobilenet_model,
        g_tflite_acceleration_embedded_mobilenet_model_len);
    ASSERT_TRUE(!plain_model_path.empty());
    plain_model_loader_ = std::make_unique<PathModelLoader>(plain_model_path);

    compute_settings_fbb_.Finish(CreateComputeSettings(compute_settings_fbb_));
    default_compute_settings_ = flatbuffers::GetRoot<ComputeSettings>(
        compute_settings_fbb_.GetBufferPointer());
  }

  std::unique_ptr<ModelLoader> validation_model_loader_;
  std::unique_ptr<ModelLoader> plain_model_loader_;
  FlatBufferBuilder compute_settings_fbb_;
  const ComputeSettings* default_compute_settings_;
};

TEST_F(ValidatorTest, HappyPathOnCpuWithEmbeddedValidation) {
  ASSERT_EQ(validation_model_loader_->Init(), kMinibenchmarkSuccess);
  int model_output_size = validation_model_loader_->GetModel()
                              ->GetModel()
                              ->subgraphs()
                              ->Get(0)
                              ->outputs()
                              ->size();

  Validator validator(std::move(validation_model_loader_),
                      default_compute_settings_);
  Validator::Results results;
  EXPECT_EQ(validator.RunValidation(&results), kMinibenchmarkSuccess);
  EXPECT_TRUE(results.ok);
  EXPECT_GE(results.metrics.size(), 0);
  EXPECT_EQ(results.delegate_error, 0);
  EXPECT_EQ(results.actual_inference_output.size(), model_output_size);
}

TEST_F(ValidatorTest, HappyPathOnCpuWithCustomValidation) {
  // Setup.
  ASSERT_EQ(plain_model_loader_->Init(), kMinibenchmarkSuccess);
  ASSERT_EQ(validation_model_loader_->Init(), kMinibenchmarkSuccess);
  const SubGraph* main_model =
      plain_model_loader_->GetModel()->GetModel()->subgraphs()->Get(0);

  const int model_output_size = main_model->outputs()->size();
  int model_input_byte_size = 1;
  for (int shape_i :
       *main_model->tensors()->Get(main_model->inputs()->Get(0))->shape()) {
    model_input_byte_size *= shape_i;
  }

  int batch_size = 5;
  // Create model with input.
  FlatBufferBuilder model_with_input;
  CustomValidationEmbedder embedder(
      *plain_model_loader_->GetModel()->GetModel(), batch_size,
      {std::vector<uint8_t>(batch_size * model_input_byte_size, 1)});
  EXPECT_EQ(embedder.BuildModel(model_with_input), kMinibenchmarkSuccess);
  // Dump the model with input to temp.
  std::string model_path = MiniBenchmarkTestHelper::DumpToTempFile(
      "mobilenet_quant_with_input.tflite", model_with_input.GetBufferPointer(),
      model_with_input.GetSize());
  ASSERT_TRUE(!model_path.empty());
  auto model_loader = std::make_unique<PathModelLoader>(model_path);

  // Execute.
  Validator validator(std::move(model_loader), default_compute_settings_);
  // Verify.
  Validator::Results results;
  EXPECT_EQ(validator.RunValidation(&results), kMinibenchmarkSuccess);
  EXPECT_FALSE(results.ok);
  EXPECT_EQ(results.metrics.size(), 0);
  EXPECT_EQ(results.delegate_error, 0);
  EXPECT_EQ(results.actual_inference_output.size(), model_output_size);
}

TEST_F(ValidatorTest, DelegateNotSupported) {
  proto::ComputeSettings settings_proto;
  settings_proto.mutable_tflite_settings()->set_delegate(proto::CORE_ML);
  flatbuffers::FlatBufferBuilder fbb;
  const ComputeSettings* settings = ConvertFromProto(settings_proto, &fbb);

  Validator validator(std::move(validation_model_loader_), settings);
  Validator::Results results;
  EXPECT_EQ(validator.RunValidation(&results),
            kMinibenchmarkDelegateNotSupported);
}

TEST_F(ValidatorTest, NoValidationSubgraph) {
  Validator validator(std::move(plain_model_loader_),
                      default_compute_settings_);
  Validator::Results results;
  EXPECT_EQ(validator.RunValidation(&results),
            kMinibenchmarkValidationSubgraphNotFound);
}

TEST_F(ValidatorTest, NoValidationInputData) {
  ASSERT_EQ(plain_model_loader_->Init(), kMinibenchmarkSuccess);
  FlatBufferBuilder model_with_input;
  CustomValidationEmbedder embedder(
      *plain_model_loader_->GetModel()->GetModel(), 1, {{}});
  EXPECT_EQ(embedder.BuildModel(model_with_input), kMinibenchmarkSuccess);
  std::string model_path = MiniBenchmarkTestHelper::DumpToTempFile(
      "mobilenet_quant_with_input.tflite", model_with_input.GetBufferPointer(),
      model_with_input.GetSize());
  ASSERT_TRUE(!model_path.empty());
  auto model_loader = std::make_unique<PathModelLoader>(model_path);

  Validator validator(std::move(model_loader), default_compute_settings_);
  Validator::Results results;
  EXPECT_EQ(validator.RunValidation(&results),
            kMinibenchmarkValidationInputMissing);
}

TEST_F(ValidatorTest, InvalidModel) {
  // Drop 12k to introduce a truncated model. Last ~11k are associated files.
  const std::string dump_path = MiniBenchmarkTestHelper::DumpToTempFile(
      "foo.tflite", g_tflite_acceleration_embedded_mobilenet_validation_model,
      g_tflite_acceleration_embedded_mobilenet_validation_model_len - 12000);
  ASSERT_TRUE(!dump_path.empty());

  Validator validator(std::make_unique<PathModelLoader>(dump_path),
                      default_compute_settings_);
  Validator::Results results;
  EXPECT_EQ(validator.RunValidation(&results), kMinibenchmarkModelBuildFailed);
}

TEST_F(ValidatorTest, EmptyModelLoader) {
  Validator validator(nullptr, default_compute_settings_);
  Validator::Results results;
  EXPECT_EQ(validator.RunValidation(&results), kMinibenchmarkModelReadFailed);
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
