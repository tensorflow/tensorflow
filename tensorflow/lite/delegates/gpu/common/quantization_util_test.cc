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

#include "tensorflow/lite/delegates/gpu/common/quantization_util.h"

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "tensorflow/lite/micro/testing/test_utils.h"
#include "tensorflow/lite/util.h"

using ::testing::Eq;
using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace {

std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> BuildTfLiteIntArray(
    const std::vector<int>& data) {
  std::unique_ptr<TfLiteIntArray, TfLiteIntArrayDeleter> result(
      TfLiteIntArrayCreate(data.size()));
  std::copy(data.begin(), data.end(), result->data);
  return result;
}

TEST(DequantizeInputs, Int8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteIntArray({1, 3, 2, 1});
  std::vector<int8_t> data = {-3, -2, -1, 1, 2, 3};
  std::vector<float> dequantized_data(data.size());

  TfLiteTensor input = tflite::testing::CreateQuantizedTensor(
      data.data(), input_dims.get(), "input",
      /*min=*/-12.8f, /*max=*/12.7f, /*is_variable=*/false);
  TfLiteTensor dequantized_input = tflite::testing::CreateFloatTensor(
      dequantized_data.data(), input_dims.get(), "input_dequant",
      /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{input, dequantized_input};
  tflite::testing::PopulateContext(tensors.data(), tensors.size(),
                                   /*error_reporter=*/nullptr, &context);

  std::vector<uint32_t> input_indices = {1};
  std::unordered_map<int, int> quant_conversion_map = {{1, 0}};

  auto status = DequantizeInputs(&context, input_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(dequantized_data,
              Pointwise(FloatNear(1e-6), {-0.3, -0.2, -0.1, 0.1, 0.2, 0.3}));
}

TEST(DequantizeInputs, UInt8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteIntArray({1, 3, 2, 1});
  std::vector<uint8_t> data = {0, 1, 2, 3, 4, 5};
  std::vector<float> dequantized_data(data.size());

  TfLiteTensor input = tflite::testing::CreateQuantizedTensor(
      data.data(), input_dims.get(), "input",
      /*min=*/0.0f, /*max=*/25.5f, /*is_variable=*/false);
  TfLiteTensor dequantized_input = tflite::testing::CreateFloatTensor(
      dequantized_data.data(), input_dims.get(), "input_dequant",
      /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{input, dequantized_input};
  tflite::testing::PopulateContext(tensors.data(), tensors.size(),
                                   /*error_reporter=*/nullptr, &context);

  std::vector<int64_t> input_indices = {1};
  std::unordered_map<int, int> quant_conversion_map = {{1, 0}};

  auto status = DequantizeInputs(&context, input_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(dequantized_data,
              Pointwise(FloatNear(1e-6), {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}));
}

TEST(QuantizeOutputs, Int8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteIntArray({1, 3, 2, 1});
  std::vector<float> data = {-0.3, -0.2, -0.1, 0.1, 0.2, 0.3};
  std::vector<int8_t> quantized_data(data.size());
  TfLiteTensor output = tflite::testing::CreateFloatTensor(
      data.data(), input_dims.get(), "output", /*is_variable=*/false);
  TfLiteTensor quantized_output = tflite::testing::CreateQuantizedTensor(
      quantized_data.data(), input_dims.get(), "output_quant",
      /*min=*/-12.8f, /*max=*/12.7f, /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{output, quantized_output};
  tflite::testing::PopulateContext(tensors.data(), tensors.size(),
                                   /*error_reporter=*/nullptr, &context);

  std::vector<uint32_t> output_indices = {0};
  std::unordered_map<int, int> quant_conversion_map = {{0, 1}};

  auto status = QuantizeOutputs(&context, output_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(quantized_data, Pointwise(Eq(), {-3, -2, -1, 1, 2, 3}));
}

TEST(QuantizeOutputs, UInt8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteIntArray({1, 3, 2, 1});
  std::vector<float> data = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  std::vector<uint8_t> quantized_data(data.size());
  TfLiteTensor output = tflite::testing::CreateFloatTensor(
      data.data(), input_dims.get(), "output", /*is_variable=*/false);
  TfLiteTensor quantized_output = tflite::testing::CreateQuantizedTensor(
      quantized_data.data(), input_dims.get(), "output_quant",
      /*min=*/0.0f, /*max=*/25.5f, /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{output, quantized_output};
  tflite::testing::PopulateContext(tensors.data(), tensors.size(),
                                   /*error_reporter=*/nullptr, &context);

  std::vector<int64_t> output_indices = {0};
  std::unordered_map<int, int> quant_conversion_map = {{0, 1}};

  auto status = QuantizeOutputs(&context, output_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(quantized_data, Pointwise(Eq(), {0, 1, 2, 3, 4, 5}));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
