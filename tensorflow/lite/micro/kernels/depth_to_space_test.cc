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
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kOutputDimsCount = 4;

struct DepthToSpaceTestParams {
  int block_size;
  //  output_dims_data is a TfLiteIntArray
  int output_dims_data[kOutputDimsCount + 1] = {kOutputDimsCount, 0, 0, 0, 0};
};

void ExecuteDepthToSpaceTest(const DepthToSpaceTestParams& params,
                             TfLiteTensor* tensors, int tensors_count) {
  int kInputArrayData[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  int kOutputArrayData[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  TfLiteDepthToSpaceParams op_params = {};
  op_params.block_size = params.block_size;

  const TfLiteRegistration registration = tflite::Register_DEPTH_TO_SPACE();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, static_cast<void*>(&op_params));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void TestDepthToSpace(DepthToSpaceTestParams& params, int* input_dims_data,
                      const T* input_data, int* expected_dims_data,
                      const T* expected_data, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* expected_dims = IntArrayFromInts(expected_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(params.output_dims_data);
  const int expected_count = ElementCount(*expected_dims);

  TfLiteTensor tensors[] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  ExecuteDepthToSpaceTest(params, tensors, tensors_count);

  constexpr float kTolerance = 1e-5;
  for (int i = 0; i < expected_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTolerance);
  }
  for (int i = 0; i < expected_dims->size; i++) {
    // output dims will have been relocated during prepare phase,
    // so use the tensor dims pointer.
    TF_LITE_MICRO_EXPECT_EQ(expected_dims->data[i], tensors[1].dims->data[i]);
  }
}

// min/max are used to compute scale, zero-point, compare tolerance
template <typename T, int kOutputSize>
struct TestQuantParams {
  float data_min;              // input and output data minimum value
  float data_max;              // input and output data maximum value
  T input_data[kOutputSize];   // quantized input storage
  T output_data[kOutputSize];  // quantized output storage
};

// for quantized, the error shouldn't exceed step
template <typename T>
float GetTolerance(float min, float max) {
  float kQuantizedStep =
      2.0f * (max - min) /
      (std::numeric_limits<T>::max() - std::numeric_limits<T>::min());
  return kQuantizedStep;
}

template <typename T, int kOutputSize>
void TestDepthToSpaceQuantized(DepthToSpaceTestParams& params,
                               TestQuantParams<T, kOutputSize>* quant_params,
                               int* input_dims_data, const float* input_data,
                               int* expected_dims_data,
                               const float* expected_data, float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* expected_dims = IntArrayFromInts(expected_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(params.output_dims_data);

  const float scale =
      ScaleFromMinMax<T>(quant_params->data_min, quant_params->data_max);
  const int zero_point =
      ZeroPointFromMinMax<T>(quant_params->data_min, quant_params->data_max);

  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input_data, quant_params->input_data, input_dims,
                            scale, zero_point),
      CreateQuantizedTensor(quant_params->output_data, output_dims, scale,
                            zero_point),
  };
  constexpr int kTensorsCount = std::extent<decltype(tensors)>::value;

  ExecuteDepthToSpaceTest(params, tensors, kTensorsCount);

  Dequantize(quant_params->output_data, kOutputSize, scale, zero_point,
             output_data);
  const float kTolerance =
      GetTolerance<T>(quant_params->data_min, quant_params->data_max);
  for (int i = 0; i < kOutputSize; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTolerance);
  }
  for (int i = 0; i < expected_dims->size; i++) {
    // output dims will have been relocated during prepare phase,
    // so use the tensor dims pointer.
    TF_LITE_MICRO_EXPECT_EQ(expected_dims->data[i], tensors[1].dims->data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(DepthToSpaceOpModelFloat32_1114_2) {
  int kInputDims[] = {4, 1, 1, 1, 4};
  constexpr float kInput[] = {1.4, 2.3, 3.2, 4.1};
  int kExpectDims[] = {4, 1, 2, 2, 1};
  constexpr float kExpect[] = {1.4, 2.3, 3.2, 4.1};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::DepthToSpaceTestParams params;
  params.block_size = 2;

  tflite::testing::TestDepthToSpace(params, kInputDims, kInput, kExpectDims,
                                    kExpect, output_data);
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelFloat32_1124_2) {
  int kInputDims[] = {4, 1, 1, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int kExpectDims[] = {4, 1, 2, 4, 1};
  constexpr float kExpect[] = {1, 2, 5, 6, 3, 4, 7, 8};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::DepthToSpaceTestParams params;
  params.block_size = 2;

  tflite::testing::TestDepthToSpace(params, kInputDims, kInput, kExpectDims,
                                    kExpect, output_data);
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelFloat32_1214_2) {
  int kInputDims[] = {4, 1, 2, 1, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int kExpectDims[] = {4, 1, 4, 2, 1};
  constexpr float kExpect[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::DepthToSpaceTestParams params;
  params.block_size = 2;

  tflite::testing::TestDepthToSpace(params, kInputDims, kInput, kExpectDims,
                                    kExpect, output_data);
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelFloat32_1224_2) {
  int kInputDims[] = {4, 1, 2, 2, 4};
  constexpr float kInput[] = {1, 2,  3,  4,  5,  6,  7,  8,
                              9, 10, 11, 12, 13, 14, 15, 16};
  int kExpectDims[] = {4, 1, 4, 4, 1};
  constexpr float kExpect[] = {1, 2,  5,  6,  3,  4,  7,  8,
                               9, 10, 13, 14, 11, 12, 15, 16};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::DepthToSpaceTestParams params;
  params.block_size = 2;

  tflite::testing::TestDepthToSpace(params, kInputDims, kInput, kExpectDims,
                                    kExpect, output_data);
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelFloat32_1111_1) {
  int kInputDims[] = {4, 1, 1, 1, 1};
  constexpr float kInput[] = {4};
  int kExpectDims[] = {4, 1, 1, 1, 1};
  constexpr float kExpect[] = {4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::DepthToSpaceTestParams params;
  params.block_size = 1;

  tflite::testing::TestDepthToSpace(params, kInputDims, kInput, kExpectDims,
                                    kExpect, output_data);
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelInt8_1114_2) {
  int kInputDims[] = {4, 1, 1, 1, 4};
  constexpr float kInput[] = {1.4, 2.3, 3.2, 4.1};
  int kExpectDims[] = {4, 1, 2, 2, 1};
  constexpr float kExpect[] = {1.4, 2.3, 3.2, 4.1};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::DepthToSpaceTestParams params;
  params.block_size = 2;
  tflite::testing::TestQuantParams<int8_t, kOutputCount> quant_params = {};
  quant_params.data_min = 0.0;
  quant_params.data_max = 5.0;

  tflite::testing::TestDepthToSpaceQuantized<int8_t, kOutputCount>(
      params, &quant_params, kInputDims, kInput, kExpectDims, kExpect,
      output_data);
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelInt8_1124_2) {
  int kInputDims[] = {4, 1, 1, 2, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int kExpectDims[] = {4, 1, 2, 4, 1};
  constexpr float kExpect[] = {1, 2, 5, 6, 3, 4, 7, 8};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::DepthToSpaceTestParams params;
  params.block_size = 2;
  tflite::testing::TestQuantParams<int8_t, kOutputCount> quant_params = {};
  quant_params.data_min = 0.0;
  quant_params.data_max = 9.0;

  tflite::testing::TestDepthToSpaceQuantized<int8_t, kOutputCount>(
      params, &quant_params, kInputDims, kInput, kExpectDims, kExpect,
      output_data);
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelInt8_1214_2) {
  int kInputDims[] = {4, 1, 2, 1, 4};
  constexpr float kInput[] = {1, 2, 3, 4, 5, 6, 7, 8};
  int kExpectDims[] = {4, 1, 4, 2, 1};
  constexpr float kExpect[] = {1, 2, 3, 4, 5, 6, 7, 8};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::DepthToSpaceTestParams params;
  params.block_size = 2;
  tflite::testing::TestQuantParams<int8_t, kOutputCount> quant_params = {};
  quant_params.data_min = 0.0;
  quant_params.data_max = 9.0;

  tflite::testing::TestDepthToSpaceQuantized<int8_t, kOutputCount>(
      params, &quant_params, kInputDims, kInput, kExpectDims, kExpect,
      output_data);
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelInt8_1224_2) {
  int kInputDims[] = {4, 1, 2, 2, 4};
  constexpr float kInput[] = {1, 2,  3,  4,  5,  6,  7,  8,
                              9, 10, 11, 12, 13, 14, 15, 16};
  int kExpectDims[] = {4, 1, 4, 4, 1};
  constexpr float kExpect[] = {1, 2,  5,  6,  3,  4,  7,  8,
                               9, 10, 13, 14, 11, 12, 15, 16};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::DepthToSpaceTestParams params;
  params.block_size = 2;
  tflite::testing::TestQuantParams<int8_t, kOutputCount> quant_params = {};
  quant_params.data_min = 0.0;
  quant_params.data_max = 17.0;

  tflite::testing::TestDepthToSpaceQuantized<int8_t, kOutputCount>(
      params, &quant_params, kInputDims, kInput, kExpectDims, kExpect,
      output_data);
}

TF_LITE_MICRO_TEST(DepthToSpaceOpModelInt8_1111_1) {
  int kInputDims[] = {4, 1, 1, 1, 1};
  constexpr float kInput[] = {4};
  int kExpectDims[] = {4, 1, 1, 1, 1};
  constexpr float kExpect[] = {4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::DepthToSpaceTestParams params;
  params.block_size = 1;
  tflite::testing::TestQuantParams<int8_t, kOutputCount> quant_params = {};
  quant_params.data_min = 3.0;
  quant_params.data_max = 5.0;

  tflite::testing::TestDepthToSpaceQuantized<int8_t, kOutputCount>(
      params, &quant_params, kInputDims, kInput, kExpectDims, kExpect,
      output_data);
}

TF_LITE_MICRO_TESTS_END
