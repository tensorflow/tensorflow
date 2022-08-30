/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_modifier/custom_validation_embedder.h"

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "flatbuffers/flatbuffers.h"  // from @flatbuffers
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/embedded_mobilenet_model.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/mini_benchmark_test_helper.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_loader.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/schema/schema_generated.h"

namespace tflite {
namespace acceleration {
namespace {
using ::flatbuffers::FlatBufferBuilder;

class CustomValidationEmbedderTest : public ::testing::Test {
 protected:
  void SetUp() override {
    std::string plain_model_path = MiniBenchmarkTestHelper::DumpToTempFile(
        "mobilenet_quant.tflite",
        g_tflite_acceleration_embedded_mobilenet_model,
        g_tflite_acceleration_embedded_mobilenet_model_len);
    ASSERT_TRUE(!plain_model_path.empty());
    plain_model_loader_ = std::make_unique<PathModelLoader>(plain_model_path);
    ASSERT_EQ(plain_model_loader_->Init(), kMinibenchmarkSuccess);
  }
  std::unique_ptr<ModelLoader> plain_model_loader_;
};

TEST_F(CustomValidationEmbedderTest, EmbedInputSucceed) {
  std::vector<float> input_buffer(224 * 224 * 3 / sizeof(float), 1.0);
  uint8_t* ptr = reinterpret_cast<uint8_t*>(input_buffer.data());
  std::vector<uint8_t> input_buffer_uint8{ptr, ptr + input_buffer.size()};

  FlatBufferBuilder model_with_input;
  EXPECT_EQ(GenerateModelWithInput(*plain_model_loader_->GetModel()->GetModel(),
                                   {input_buffer_uint8}, model_with_input),
            kMinibenchmarkSuccess);

  EXPECT_NE(FlatBufferModel::BuildFromModel(
                GetModel(model_with_input.GetBufferPointer()))
                ->GetModel(),
            nullptr);
}

TEST_F(CustomValidationEmbedderTest, TooManyInput) {
  FlatBufferBuilder model_with_input;
  EXPECT_EQ(GenerateModelWithInput(*plain_model_loader_->GetModel()->GetModel(),
                                   {{}, {}}, model_with_input),
            kMinibenchmarkPreconditionNotMet);
}

}  // namespace
}  // namespace acceleration
}  // namespace tflite
