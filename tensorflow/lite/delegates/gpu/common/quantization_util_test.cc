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

#include <stdint.h>

#include <limits>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "tensorflow/lite/array.h"
#include "tensorflow/lite/core/c/common.h"

using ::testing::Eq;
using ::testing::FloatNear;
using ::testing::Pointwise;

namespace tflite {
namespace gpu {
namespace {

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
void PopulateContext(std::vector<TfLiteTensor>& tensors,
                     TfLiteContext& context) {
  context.tensors_size = tensors.size();
  context.tensors = tensors.data();
  context.recommended_num_threads = 1;
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
int ElementCount(const TfLiteIntArray& dims) {
  int result = 1;
  for (int i = 0; i < dims.size; ++i) {
    result *= dims.data[i];
  }
  return result;
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
template <typename T>
inline float ScaleFromMinMax(const float min, const float max) {
  return (max - min) / ((std::numeric_limits<T>::max() * 1.0) -
                        std::numeric_limits<T>::min());
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
template <typename T>
inline int ZeroPointFromMinMax(const float min, const float max) {
  return static_cast<int>(std::numeric_limits<T>::min()) +
         static_cast<int>(-min / ScaleFromMinMax<T>(min, max) + 0.5f);
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
TfLiteTensor CreateQuantizedTensor(const int8_t* data, TfLiteIntArray* dims,
                                   const char* name, float min, float max,
                                   bool is_variable) {
  TfLiteTensor result;
  result.type = kTfLiteInt8;
  result.data.int8 = const_cast<int8_t*>(data);
  result.dims = dims;
  result.params = {ScaleFromMinMax<int8_t>(min, max),
                   ZeroPointFromMinMax<int8_t>(min, max)};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(int8_t);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = is_variable;
  return result;
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
TfLiteTensor CreateQuantizedTensor(const uint8_t* data, TfLiteIntArray* dims,
                                   const char* name, float min, float max,
                                   bool is_variable) {
  TfLiteTensor result;
  result.type = kTfLiteUInt8;
  result.data.uint8 = const_cast<uint8_t*>(data);
  result.dims = dims;
  result.params = {ScaleFromMinMax<uint8_t>(min, max),
                   ZeroPointFromMinMax<uint8_t>(min, max)};
  result.allocation_type = kTfLiteMemNone;
  result.bytes = ElementCount(*dims) * sizeof(uint8_t);
  result.allocation = nullptr;
  result.name = name;
  result.is_variable = false;
  return result;
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
TfLiteTensor CreateTensor(TfLiteIntArray* dims, const char* name,
                          bool is_variable) {
  TfLiteTensor result;
  result.dims = dims;
  result.name = name;
  result.params = {};
  result.quantization = {kTfLiteNoQuantization, nullptr};
  result.is_variable = is_variable;
  result.allocation_type = kTfLiteMemNone;
  result.allocation = nullptr;
  return result;
}

// TODO(b/158578883): this function is copied from the Micro codebase. Consider
// moving to a shared location.
TfLiteTensor CreateFloatTensor(const float* data, TfLiteIntArray* dims,
                               const char* name, bool is_variable) {
  TfLiteTensor result = CreateTensor(dims, name, is_variable);
  result.type = kTfLiteFloat32;
  result.data.f = const_cast<float*>(data);
  result.bytes = ElementCount(*dims) * sizeof(float);
  return result;
}

TEST(DequantizeInputs, Int8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteArray({1, 3, 2, 1});
  std::vector<int8_t> data = {-3, -2, -1, 1, 2, 3};
  std::vector<float> dequantized_data(data.size());

  TfLiteTensor input = CreateQuantizedTensor(
      data.data(), input_dims.get(), "input",
      /*min=*/-12.8f, /*max=*/12.7f, /*is_variable=*/false);
  TfLiteTensor dequantized_input = CreateFloatTensor(
      dequantized_data.data(), input_dims.get(), "input_dequant",
      /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{input, dequantized_input};
  PopulateContext(tensors, context);

  std::vector<uint32_t> input_indices = {1};
  absl::flat_hash_map<int, int> quant_conversion_map = {{1, 0}};

  auto status = DequantizeInputs(&context, input_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(dequantized_data,
              Pointwise(FloatNear(1e-6), {-0.3, -0.2, -0.1, 0.1, 0.2, 0.3}));
}

TEST(DequantizeInputs, UInt8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteArray({1, 3, 2, 1});
  std::vector<uint8_t> data = {0, 1, 2, 3, 4, 5};
  std::vector<float> dequantized_data(data.size());

  TfLiteTensor input =
      CreateQuantizedTensor(data.data(), input_dims.get(), "input",
                            /*min=*/0.0f, /*max=*/25.5f, /*is_variable=*/false);
  TfLiteTensor dequantized_input = CreateFloatTensor(
      dequantized_data.data(), input_dims.get(), "input_dequant",
      /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{input, dequantized_input};
  PopulateContext(tensors, context);

  std::vector<int64_t> input_indices = {1};
  absl::flat_hash_map<int, int> quant_conversion_map = {{1, 0}};

  auto status = DequantizeInputs(&context, input_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(dequantized_data,
              Pointwise(FloatNear(1e-6), {0.0, 0.1, 0.2, 0.3, 0.4, 0.5}));
}

TEST(QuantizeOutputs, Int8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteArray({1, 3, 2, 1});
  std::vector<float> data = {-0.3, -0.2, -0.1, 0.1, 0.2, 0.3};
  std::vector<int8_t> quantized_data(data.size());
  TfLiteTensor output = CreateFloatTensor(data.data(), input_dims.get(),
                                          "output", /*is_variable=*/false);
  TfLiteTensor quantized_output = CreateQuantizedTensor(
      quantized_data.data(), input_dims.get(), "output_quant",
      /*min=*/-12.8f, /*max=*/12.7f, /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{output, quantized_output};
  PopulateContext(tensors, context);

  std::vector<uint32_t> output_indices = {0};
  absl::flat_hash_map<int, int> quant_conversion_map = {{0, 1}};

  auto status = QuantizeOutputs(&context, output_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(quantized_data, Pointwise(Eq(), {-3, -2, -1, 1, 2, 3}));
}

TEST(QuantizeOutputs, UInt8) {
  TfLiteContext context;
  auto input_dims = BuildTfLiteArray({1, 3, 2, 1});
  std::vector<float> data = {0.0, 0.1, 0.2, 0.3, 0.4, 0.5};
  std::vector<uint8_t> quantized_data(data.size());
  TfLiteTensor output = CreateFloatTensor(data.data(), input_dims.get(),
                                          "output", /*is_variable=*/false);
  TfLiteTensor quantized_output = CreateQuantizedTensor(
      quantized_data.data(), input_dims.get(), "output_quant",
      /*min=*/0.0f, /*max=*/25.5f, /*is_variable=*/true);

  std::vector<TfLiteTensor> tensors{output, quantized_output};
  PopulateContext(tensors, context);

  std::vector<int64_t> output_indices = {0};
  absl::flat_hash_map<int, int> quant_conversion_map = {{0, 1}};

  auto status = QuantizeOutputs(&context, output_indices, quant_conversion_map);
  EXPECT_TRUE(status.ok());
  EXPECT_THAT(quantized_data, Pointwise(Eq(), {0, 1, 2, 3, 4, 5}));
}

}  // namespace
}  // namespace gpu
}  // namespace tflite
