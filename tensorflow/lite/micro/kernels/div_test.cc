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

#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

void ExecuteDivTest(TfLiteTensor* tensors, int tensors_count,
                    TfLiteFusedActivation activation) {
  TfLiteDivParams builtin_data = {};
  builtin_data.activation = activation;

  constexpr int kInputArrayData[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  constexpr int kOutputArrayData[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  const TfLiteRegistration registration = tflite::ops::micro::Register_DIV();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, static_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void TestDiv(const int* input1_dims_data, const T* input1_data,
             const int* input2_dims_data, const T* input2_data,
             const int* expected_dims, const T* expected_data, T* output_data,
             TfLiteFusedActivation activation) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  TfLiteTensor tensors[] = {
      CreateTensor(input1_data, input1_dims),
      CreateTensor(input2_data, input2_dims),
      CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;

  ExecuteDivTest(tensors, tensors_count, activation);

  constexpr float kTolerance = 1e-5;
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTolerance);
  }
}

// For quantized Div, the error shouldn't exceed (2*step + step^2).
inline float GetTolerance(int min, int max) {
  const float kQuantizedStep = (max - min) / 255.0f;
  const float kQuantizedTolerance =
      2.0f * kQuantizedStep + kQuantizedStep * kQuantizedStep;
  return kQuantizedTolerance;
}

// min/max are used to compute scale, zero-point, compare tolerance
template <typename T>
struct TestQuantParams {
  float data_min;  // input and output data minimum value
  float data_max;  // input and output data maximum value
  T* input1_data;  // quantized input1 storage
  T* input2_data;  // quantized input2 storage
  T* output_data;  // quantized output storage
};

template <typename T>
void TestDivQuantized(const int* input1_dims_data, const float* input1_data,
                      const int* input2_dims_data, const float* input2_data,
                      const int* expected_dims, const float* expected_data,
                      float* output_data, TfLiteFusedActivation activation,
                      const TestQuantParams<T>* params) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(expected_dims);
  const int output_count = ElementCount(*output_dims);

  const float scale = ScaleFromMinMax<T>(params->data_min, params->data_max);
  const int zero_point =
      ZeroPointFromMinMax<T>(params->data_min, params->data_max);

  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input1_data, params->input1_data, input1_dims,
                            scale, zero_point),
      CreateQuantizedTensor(input2_data, params->input2_data, input2_dims,
                            scale, zero_point),
      CreateQuantizedTensor(params->output_data, output_dims, scale,
                            zero_point),
  };
  constexpr int kTensorsCount = std::extent<decltype(tensors)>::value;

  ExecuteDivTest(tensors, kTensorsCount, activation);

  Dequantize(params->output_data, output_count, scale, zero_point, output_data);
  const float kTolerance = GetTolerance(params->data_min, params->data_max);
  for (int i = 0; i < output_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i], kTolerance);
  }
}

template <typename T>
void TestDivMultiShape(const int** shapes, const int shapes_count,
                       const T* input1_data, const T* input2_data,
                       const T* expected_data, T* output_data,
                       TfLiteFusedActivation activation) {
  for (int i = 0; i < shapes_count; i++) {
    TestDiv(shapes[i], input1_data, shapes[i], input2_data, shapes[i],
            expected_data, output_data, activation);
  }
}

template <typename T>
void TestDivMultiShapeQuant(const int** shapes, const int shapes_count,
                            const float* input1_data, const float* input2_data,
                            const float* expected_data, float* output_data,
                            TfLiteFusedActivation activation,
                            const TestQuantParams<T>* params) {
  for (int i = 0; i < shapes_count; i++) {
    TestDivQuantized(shapes[i], input1_data, shapes[i], input2_data, shapes[i],
                     expected_data, output_data, activation, params);
  }
}

// when broadcasting input2 is a scaler
template <typename T>
void TestDivMultiBroadcast(const int** shapes, const int shapes_count,
                           const T* input1_data, const T* input2_data,
                           const T* expected_data, T* output_data,
                           TfLiteFusedActivation activation) {
  constexpr int kDimScaler[] = {1, 1};
  for (int i = 0; i < shapes_count; i++) {
    TestDiv(shapes[i], input1_data, kDimScaler, input2_data, shapes[i],
            expected_data, output_data, activation);
  }
}

// when broadcasting input2 is a scaler
template <typename T>
void TestDivMultiBroadcastQuant(const int** shapes, const int shapes_count,
                                const float* input1_data,
                                const float* input2_data,
                                const float* expected_data, float* output_data,
                                TfLiteFusedActivation activation,
                                const TestQuantParams<T>* params) {
  constexpr int kDimScaler[] = {1, 1};
  for (int i = 0; i < shapes_count; i++) {
    TestDivQuantized(shapes[i], input1_data, kDimScaler, input2_data, shapes[i],
                     expected_data, output_data, activation, params);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatDivOpTestActNone) {
  constexpr int kDims[] = {4, 1, 2, 2, 1};
  constexpr float kInput1[] = {-0.2, 0.2, -1.2, 0.8};
  constexpr float kInput2[] = {0.5, 0.2, -1.5, 0.5};
  constexpr float kExpect[] = {-0.4, 1.0, 0.8, 1.6};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestDiv(kDims, kInput1, kDims, kInput2, kDims, kExpect,
                           output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(FloatDivOpTestActReluN1To1) {
  constexpr int kDims[] = {4, 1, 2, 2, 1};
  constexpr float kInput1[] = {-0.2, 0.2, -1.2, 0.8};
  constexpr float kInput2[] = {0.1, 0.2, -1.5, 0.5};
  constexpr float kExpect[] = {-1.0, 1.0, 0.8, 1.0};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestDiv(kDims, kInput1, kDims, kInput2, kDims, kExpect,
                           output_data, kTfLiteActReluN1To1);
}

TF_LITE_MICRO_TEST(FloatDivOpTestMultiShape) {
  constexpr int kShape1[] = {1, 6};
  constexpr int kShape2[] = {2, 2, 3};
  constexpr int kShape3[] = {3, 2, 1, 3};
  constexpr int kShape4[] = {4, 1, 3, 1, 2};
  const int* kDims[] = {kShape1, kShape2, kShape3, kShape4};
  constexpr int kDimsCount = std::extent<decltype(kDims)>::value;

  constexpr float kInput1[] = {-2.0, 0.2, 0.3, 0.8, 1.1, -2.0};
  constexpr float kInput2[] = {0.1, 0.2, 0.6, 0.5, -1.1, -0.1};
  constexpr float kExpect[] = {-20.0, 1.0, 0.5, 1.6, -1.0, 20.0};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestDivMultiShape(kDims, kDimsCount, kInput1, kInput2,
                                     kExpect, output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(FloatDivOpTestBroadcast) {
  constexpr int kShape1[] = {1, 8};
  constexpr int kShape2[] = {2, 2, 4};
  constexpr int kShape3[] = {3, 2, 1, 4};
  constexpr int kShape4[] = {4, 1, 2, 2, 2};
  const int* kDims[] = {kShape1, kShape2, kShape3, kShape4};
  constexpr int kDimsCount = std::extent<decltype(kDims)>::value;

  constexpr float kInput1[] = {-0.2, 0.2,    0.07,  0.08,
                               0.11, -0.123, -0.32, 0.54};
  constexpr float kInput2[] = {0.1};
  constexpr float kExpect[] = {-2.0, 2.0, 0.7, 0.8, 1.1, -1.23, -3.2, 5.4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestDivMultiBroadcast(kDims, kDimsCount, kInput1, kInput2,
                                         kExpect, output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(FloatDivOpTestBroadcast5D) {
  constexpr int kShape1[] = {5, 1, 2, 1, 2, 2};
  const int* kDims[] = {kShape1};
  constexpr int kDimsCount = std::extent<decltype(kDims)>::value;

  constexpr float kInput1[] = {-0.2, 0.2,    0.07,  0.08,
                               0.11, -0.123, -0.32, 0.54};
  constexpr float kInput2[] = {0.1};
  constexpr float kExpect[] = {-2.0, 2.0, 0.7, 0.8, 1.1, -1.23, -3.2, 5.4};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  tflite::testing::TestDivMultiBroadcast(kDims, kDimsCount, kInput1, kInput2,
                                         kExpect, output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(IntegerDivOpTestActNone) {
  constexpr int kDims[] = {4, 1, 2, 2, 1};
  constexpr int32_t kInput1[] = {-2, 2, -15, 8};
  constexpr int32_t kInput2[] = {5, -2, -3, 5};
  constexpr int32_t kExpect[] = {0, -1, 5, 1};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestDiv(kDims, kInput1, kDims, kInput2, kDims, kExpect,
                           output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(IntegerDivOpTestActReluN1To1) {
  constexpr int kDims[] = {4, 1, 2, 2, 1};
  constexpr int32_t kInput1[] = {-2, 2, -12, 8};
  constexpr int32_t kInput2[] = {1, 2, -15, 5};
  constexpr int32_t kExpect[] = {-1, 1, 0, 1};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestDiv(kDims, kInput1, kDims, kInput2, kDims, kExpect,
                           output_data, kTfLiteActReluN1To1);
}

TF_LITE_MICRO_TEST(IntegerDivOpTestMultiShape) {
  constexpr int kShape1[] = {1, 6};
  constexpr int kShape2[] = {2, 2, 3};
  constexpr int kShape3[] = {3, 2, 1, 3};
  constexpr int kShape4[] = {4, 1, 3, 1, 2};
  const int* kDims[] = {kShape1, kShape2, kShape3, kShape4};
  constexpr int kDimsCount = std::extent<decltype(kDims)>::value;

  constexpr int32_t kInput1[] = {-20, 2, 3, 8, 11, -20};
  constexpr int32_t kInput2[] = {1, 2, 6, 5, -11, -1};
  constexpr int32_t kExpect[] = {-20, 1, 0, 1, -1, 20};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestDivMultiShape(kDims, kDimsCount, kInput1, kInput2,
                                     kExpect, output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(IntegerDivOpTestBroadcast) {
  constexpr int kShape1[] = {1, 8};
  constexpr int kShape2[] = {2, 2, 4};
  constexpr int kShape3[] = {3, 2, 1, 4};
  constexpr int kShape4[] = {4, 1, 4, 1, 2};
  constexpr int kShape5[] = {5, 1, 2, 1, 2, 2};
  const int* kDims[] = {kShape1, kShape2, kShape3, kShape4, kShape5};
  constexpr int kDimsCount = std::extent<decltype(kDims)>::value;

  constexpr int32_t kInput1[] = {-20, 21, 7, 8, 11, -123, -42, -48};
  constexpr int32_t kInput2[] = {3};
  constexpr int32_t kExpect[] = {-6, 7, 2, 2, 3, -41, -14, -16};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  int32_t output_data[kOutputCount];

  tflite::testing::TestDivMultiBroadcast(kDims, kDimsCount, kInput1, kInput2,
                                         kExpect, output_data, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(QuantizedDivOpTestActNone) {
  constexpr int kDims[] = {4, 1, 2, 2, 1};
  constexpr float kInput1[] = {-0.8, -0.2, 0.3, 0.7};
  constexpr float kInput2[] = {-0.8, 0.4, 0.8, 1.0};
  constexpr float kExpect[] = {1.0, -0.5, 0.375, 0.7};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  // setup quantization storage and parameters
  uint8_t q_output_data[kOutputCount];
  uint8_t q_input1_data[kOutputCount];
  uint8_t q_input2_data[kOutputCount];
  tflite::testing::TestQuantParams<uint8_t> params = {};
  params.data_min = -1.0;
  params.data_max = 1.0;
  params.input1_data = q_input1_data;
  params.input2_data = q_input2_data;
  params.output_data = q_output_data;

  tflite::testing::TestDivQuantized(kDims, kInput1, kDims, kInput2, kDims,
                                    kExpect, output_data, kTfLiteActNone,
                                    &params);
}

TF_LITE_MICRO_TEST(QuantizedDivOpTestActReluN1To1) {
  constexpr int kDims[] = {4, 1, 2, 2, 1};
  constexpr float kInput1[] = {-0.8, 0.2, 0.9, 0.7};
  constexpr float kInput2[] = {0.6, 0.4, 0.9, -0.8};
  constexpr float kExpect1[] = {-1.0, 0.5, 1.0, -0.875};
  constexpr int kOutputCount = std::extent<decltype(kExpect1)>::value;
  float output_data[kOutputCount];

  // setup quantization storage and parameters
  uint8_t q_output_data[kOutputCount];
  uint8_t q_input1_data[kOutputCount];
  uint8_t q_input2_data[kOutputCount];
  tflite::testing::TestQuantParams<uint8_t> params = {};
  params.data_min = -1.0;
  params.data_max = 1.0;
  params.input1_data = q_input1_data;
  params.input2_data = q_input2_data;
  params.output_data = q_output_data;

  tflite::testing::TestDivQuantized(kDims, kInput1, kDims, kInput2, kDims,
                                    kExpect1, output_data, kTfLiteActReluN1To1,
                                    &params);

  constexpr float kInput3[] = {-0.5, 0.2, 0.6, 0.3};
  constexpr float kInput4[] = {0.6, 0.5, -0.8, 0.5};
  constexpr float kExpect2[] = {-0.833, 0.4, -0.75, 0.6};

  tflite::testing::TestDivQuantized(kDims, kInput3, kDims, kInput4, kDims,
                                    kExpect2, output_data, kTfLiteActReluN1To1,
                                    &params);
}

TF_LITE_MICRO_TEST(QuantizedDivOpTestMultiShape) {
  constexpr int kShape1[] = {1, 6};
  constexpr int kShape2[] = {2, 2, 3};
  constexpr int kShape3[] = {3, 2, 1, 3};
  constexpr int kShape4[] = {4, 1, 3, 1, 2};
  const int* kDims[] = {kShape1, kShape2, kShape3, kShape4};
  constexpr int kDimsCount = std::extent<decltype(kDims)>::value;

  constexpr float kInput1[] = {-2.0, 0.2, 1.7, 0.9, 0.4, 2.0};
  constexpr float kInput2[] = {1.3, 0.3, 1.1, 0.4, -1.1, 1.9};
  constexpr float kExpect[] = {-1.538, 0.667, 1.545, 2.25, -0.364, 1.053};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  // setup quantization storage and parameters
  uint8_t q_output_data[kOutputCount];
  uint8_t q_input1_data[kOutputCount];
  uint8_t q_input2_data[kOutputCount];
  tflite::testing::TestQuantParams<uint8_t> params = {};
  params.data_min = -3.0;
  params.data_max = 3.0;
  params.input1_data = q_input1_data;
  params.input2_data = q_input2_data;
  params.output_data = q_output_data;

  tflite::testing::TestDivMultiShapeQuant(kDims, kDimsCount, kInput1, kInput2,
                                          kExpect, output_data, kTfLiteActNone,
                                          &params);
}

TF_LITE_MICRO_TEST(QuantizedDivOpTestBroadcast) {
  constexpr int kShape1[] = {1, 8};
  constexpr int kShape2[] = {2, 2, 4};
  constexpr int kShape3[] = {3, 2, 1, 4};
  constexpr int kShape4[] = {4, 1, 4, 1, 2};
  constexpr int kShape5[] = {5, 1, 2, 1, 2, 2};
  const int* kDims[] = {kShape1, kShape2, kShape3, kShape4, kShape5};
  constexpr int kDimsCount = std::extent<decltype(kDims)>::value;

  constexpr float kInput1[] = {-2.0, 0.2, 0.7, 0.8, -0.5, 1.1, -1.3, 1.2};
  constexpr float kInput2[] = {0.7};
  constexpr float kExpect[] = {-2.857, 0.286, 1.0,    1.143,
                               -0.714, 1.571, -1.857, 1.714};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];

  // setup quantization storage and parameters
  uint8_t q_output_data[kOutputCount];
  uint8_t q_input1_data[kOutputCount];
  uint8_t q_input2_data[kOutputCount];
  tflite::testing::TestQuantParams<uint8_t> params = {};
  params.data_min = -3.0;
  params.data_max = 3.0;
  params.input1_data = q_input1_data;
  params.input2_data = q_input2_data;
  params.output_data = q_output_data;

  tflite::testing::TestDivMultiBroadcastQuant(kDims, kDimsCount, kInput1,
                                              kInput2, kExpect, output_data,
                                              kTfLiteActNone, &params);
}

TF_LITE_MICRO_TESTS_END
