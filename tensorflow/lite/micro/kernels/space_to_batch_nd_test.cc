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

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kBasicInputOutputSize = 16;
const int basic_input_dims[] = {4, 1, 4, 4, 1};
const float basic_input[kBasicInputOutputSize] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16};
const int basic_block_shape_dims[] = {1, 2};
const int32_t basic_block_shape[] = {2, 2};
const int basic_crops_dims[] = {1, 4};
const int32_t basic_crops[] = {0, 0, 0, 0};
const int basic_output_dims[] = {4, 4, 2, 2, 1};
const float basic_golden[kBasicInputOutputSize] = {1, 3, 9,  11, 2, 4, 10, 12,
                                                   5, 7, 13, 15, 6, 8, 14, 16};

template <typename T>
TfLiteStatus ValidateSpaceToBatchNdGoldens(TfLiteTensor* tensors,
                                           int tensors_size, const T* golden,
                                           T* output, int output_size) {
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_SPACE_TO_BATCH_ND();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, nullptr);

  TF_LITE_ENSURE_STATUS(runner.InitAndPrepare());
  TF_LITE_ENSURE_STATUS(runner.Invoke());

  for (int i = 0; i < output_size; ++i) {
    // TODO(b/158102673): workaround for not having fatal test assertions.
    TF_LITE_MICRO_EXPECT_EQ(golden[i], output[i]);
    if (golden[i] != output[i]) {
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

TfLiteStatus TestSpaceToBatchNdFloat(
    const int* input_dims_data, const float* input_data,
    const int* block_shape_dims_data, const int32_t* block_shape_data,
    const int* crops_dims_data, const int32_t* crops_data,
    const int* output_dims_data, const float* golden, float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* block_shape_dims = IntArrayFromInts(block_shape_dims_data);
  TfLiteIntArray* crops_dims = IntArrayFromInts(crops_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(block_shape_data, block_shape_dims),
      CreateTensor(crops_data, crops_dims),
      CreateTensor(output_data, output_dims),
  };

  return ValidateSpaceToBatchNdGoldens(tensors, tensors_size, golden,
                                       output_data, ElementCount(*output_dims));
}

template <typename T>
TfLiteStatus TestSpaceToBatchNdQuantized(
    const int* input_dims_data, const float* input_data, T* input_quantized,
    float input_scale, int input_zero_point, const int* block_shape_dims_data,
    const int32_t* block_shape_data, const int* crops_dims_data,
    const int32_t* crops_data, const int* output_dims_data, const float* golden,
    T* golden_quantized, float output_scale, int output_zero_point,
    T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* block_shape_dims = IntArrayFromInts(block_shape_dims_data);
  TfLiteIntArray* crops_dims = IntArrayFromInts(crops_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateQuantizedTensor(input_data, input_quantized,
                                             input_dims, input_scale,
                                             input_zero_point),
      tflite::testing::CreateTensor(block_shape_data, block_shape_dims),
      tflite::testing::CreateTensor(crops_data, crops_dims),
      tflite::testing::CreateQuantizedTensor(output_data, output_dims,
                                             output_scale, output_zero_point),
  };
  tflite::Quantize(golden, golden_quantized, ElementCount(*output_dims),
                   output_scale, output_zero_point);

  return ValidateSpaceToBatchNdGoldens(tensors, tensors_size, golden_quantized,
                                       output_data, ElementCount(*output_dims));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SpaceToBatchBasicFloat) {
  float output[tflite::testing::kBasicInputOutputSize];
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestSpaceToBatchNdFloat(
          tflite::testing::basic_input_dims, tflite::testing::basic_input,
          tflite::testing::basic_block_shape_dims,
          tflite::testing::basic_block_shape, tflite::testing::basic_crops_dims,
          tflite::testing::basic_crops, tflite::testing::basic_output_dims,
          tflite::testing::basic_golden, output));
}

TF_LITE_MICRO_TEST(SpaceToBatchBasicInt8) {
  int8_t output[tflite::testing::kBasicInputOutputSize];
  int8_t input_quantized[tflite::testing::kBasicInputOutputSize];
  int8_t golden_quantized[tflite::testing::kBasicInputOutputSize];
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      tflite::testing::TestSpaceToBatchNdQuantized(
          tflite::testing::basic_input_dims, tflite::testing::basic_input,
          input_quantized, 1.0f, 0, tflite::testing::basic_block_shape_dims,
          tflite::testing::basic_block_shape, tflite::testing::basic_crops_dims,
          tflite::testing::basic_crops, tflite::testing::basic_output_dims,
          tflite::testing::basic_golden, golden_quantized, 1.0f, 0, output));
}

TF_LITE_MICRO_TESTS_END
