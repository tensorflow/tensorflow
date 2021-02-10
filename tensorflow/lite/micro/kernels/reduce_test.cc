/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// Common 2D inputs, outputs and axis.
static const int kInputElements2D = 8;
static const int kInputShape2D[] = {2, 2, 4};
static const float kInputData2D[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

static const int kAxisShape2D[] = {1, 1};
static const int32_t kAxisData2D[] = {1};

static const int kOutputElements2D = 2;
static const int kOutputShape2D[] = {2, 1, 2};
static const float kGoldenData2D[] = {2.5, 6.5};

// Common 3D inputs, outputs and axis.
static const int kInputElements3D = 8;
static const int kInputShape3D[] = {3, 2, 2, 2};
static const float kInputData3D[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};

static const int kAxisShape3D[] = {1, 2};
static const int32_t kAxisData3D[] = {1, 2};

static const int kOutputElements3D = 2;
static const int kOutputShape3D[] = {2, 1, 2};
static const float kGoldenData3D[] = {2.5, 6.5};

// Common 4D inputs, outputs and axis.
static const int kInputElements4D = 24;
static const int kInputShape4D[] = {4, 2, 2, 3, 2};
static const float kInputData4D[] = {
    1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

static const int kAxisShape4D[] = {1, 2};
static const int32_t kAxisData4D[] = {1, 2};

static const int kOutputElements4D = 4;
static const int kOutputShape4D[] = {4, 2, 1, 1, 2};
static const float kGoldenData4D[] = {6, 7, 18, 19};

// Axis shape and contents are independent of input / output dimensions.

template <typename T>
TfLiteStatus ValidateReduceGoldens(TfLiteTensor* tensors, int tensors_size,
                                   const T* expected_output_data,
                                   T* output_data, int output_length,
                                   const TfLiteRegistration& registration,
                                   TfLiteReducerParams* params,
                                   float tolerance = 1e-5) {
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, params);

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
  return kTfLiteOk;
}

void TestMeanFloatInput4D(const int* input_dims_data, const float* input_data,
                          const int* axis_dims_data, const int32_t* axis_data,
                          const int* output_dims_data,
                          const float* expected_output_data, float* output_data,
                          TfLiteReducerParams* params, float tolerance = 1e-5) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  const TfLiteRegistration registration = tflite::ops::micro::Register_MEAN();

  constexpr int num_of_inputs = 2;   // input and axis
  constexpr int num_of_outputs = 1;  // output

  constexpr int tensors_size = num_of_inputs + num_of_outputs;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(axis_data, axis_dims),
      CreateTensor(output_data, output_dims),
  };

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, ValidateReduceGoldens(
                     tensors, tensors_size, expected_output_data, output_data,
                     output_dims_count, registration, params, tolerance));
}

void TestReduceOpFloat(const int* input_dims_data, const float* input_data,
                       const int* axis_dims_data, const int32_t* axis_data,
                       const int* output_dims_data, float* output_data,
                       const float* expected_output_data,
                       const TfLiteRegistration& registration,
                       TfLiteReducerParams* params, float tolerance = 1e-5) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int num_of_inputs = 2;   // input and axis
  constexpr int num_of_outputs = 1;  // output

  constexpr int tensors_size = num_of_inputs + num_of_outputs;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(axis_data, axis_dims),
      CreateTensor(output_data, output_dims),
  };

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, ValidateReduceGoldens(
                     tensors, tensors_size, expected_output_data, output_data,
                     output_dims_count, registration, params, tolerance));
}

template <typename T>
void TestReduceOpQuantized(
    const int* input_dims_data, const float* input_data, T* input_data_quant,
    float input_scale, int input_zero_point, const int* axis_dims_data,
    const int32_t* axis_data, const int* output_dims_data,
    const float* expected_output_data, T* output_data_quant,
    T* expected_output_data_quant, float output_scale, int output_zero_point,
    const TfLiteRegistration& registration, TfLiteReducerParams* params) {
  // Convert dimesion arguments to TfLiteArrays
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  // Get number of elements in input and output tensors
  const int output_dims_count = ElementCount(*output_dims);

  // Initialize tensors
  constexpr int tensors_size = 3;
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input_data, input_data_quant, input_dims,
                            input_scale, input_zero_point),
      CreateTensor(axis_data, axis_dims),
      CreateQuantizedTensor(output_data_quant, output_dims, output_scale,
                            output_zero_point),
  };

  // Quantize expected output
  tflite::Quantize(expected_output_data, expected_output_data_quant,
                   output_dims_count, output_scale, output_zero_point);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateReduceGoldens(tensors, tensors_size, expected_output_data_quant,
                            output_data_quant, output_dims_count, registration,
                            params, 0.01));
}

template <typename T>
void TestMeanOpQuantized(const int* input_dims_data, const float* input_data,
                         T* input_data_quant, float input_scale,
                         int input_zero_point, const int* axis_dims_data,
                         const int32_t* axis_data, const int* output_dims_data,
                         const float* expected_output_data,
                         T* output_data_quant, T* expected_output_data_quant,
                         float output_scale, int output_zero_point,
                         TfLiteReducerParams* params) {
  // Convert dimesion arguments to TfLiteArrays
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  // Get number of elements in input and output tensors
  const int output_dims_count = ElementCount(*output_dims);

  // Initialize tensors
  constexpr int tensors_size = 3;
  TfLiteTensor tensors[] = {
      CreateQuantizedTensor(input_data, input_data_quant, input_dims,
                            input_scale, input_zero_point),
      CreateTensor(axis_data, axis_dims),
      CreateQuantizedTensor(output_data_quant, output_dims, output_scale,
                            output_zero_point),
  };

  // Quantize expected output
  tflite::Quantize(expected_output_data, expected_output_data_quant,
                   output_dims_count, output_scale, output_zero_point);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateReduceGoldens(tensors, tensors_size, expected_output_data_quant,
                            output_data_quant, output_dims_count,
                            tflite::ops::micro::Register_MEAN(), params, 1.0));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(MeanFloat2DKeepDims) {
  float output_data[tflite::testing::kOutputElements2D];

  TfLiteReducerParams params = {true};

  tflite::testing::TestMeanFloatInput4D(
      tflite::testing::kInputShape2D, tflite::testing::kInputData2D,
      tflite::testing::kAxisShape2D, tflite::testing::kAxisData2D,
      tflite::testing::kOutputShape2D, tflite::testing::kGoldenData2D,
      output_data, &params);
}

TF_LITE_MICRO_TEST(MeanInt82DKeepDims) {
  int8_t expected_output_data_quant[tflite::testing::kOutputElements2D];
  int8_t output_data_quant[tflite::testing::kOutputElements2D];
  int8_t input_data_quant[tflite::testing::kInputElements2D];

  float input_scale = 0.5f;
  int input_zero_point = 0;
  float output_scale = 0.5f;
  int output_zero_point = 0;

  TfLiteReducerParams params = {
      true  // keep_dims
  };

  tflite::testing::TestMeanOpQuantized<int8_t>(
      tflite::testing::kInputShape2D, tflite::testing::kInputData2D,
      input_data_quant, input_scale, input_zero_point,
      tflite::testing::kAxisShape2D, tflite::testing::kAxisData2D,
      tflite::testing::kOutputShape2D, tflite::testing::kGoldenData2D,
      output_data_quant, expected_output_data_quant, output_scale,
      output_zero_point, &params);
}

TF_LITE_MICRO_TEST(MeanUInt82DKeepDims) {
  uint8_t expected_output_data_quant[tflite::testing::kOutputElements2D];
  uint8_t output_data_quant[tflite::testing::kOutputElements2D];
  uint8_t input_data_quant[tflite::testing::kInputElements2D];

  float input_scale = 0.5f;
  int input_zero_point = 128;
  float output_scale = 0.5f;
  int output_zero_point = 128;

  TfLiteReducerParams params = {
      true  // keep_dims
  };

  tflite::testing::TestMeanOpQuantized<uint8_t>(
      tflite::testing::kInputShape2D, tflite::testing::kInputData2D,
      input_data_quant, input_scale, input_zero_point,
      tflite::testing::kAxisShape2D, tflite::testing::kAxisData2D,
      tflite::testing::kOutputShape2D, tflite::testing::kGoldenData2D,
      output_data_quant, expected_output_data_quant, output_scale,
      output_zero_point, &params);
}

TF_LITE_MICRO_TEST(MeanFloat3DKeepDims) {
  float output_data[tflite::testing::kOutputElements3D];

  TfLiteReducerParams params = {true};

  tflite::testing::TestMeanFloatInput4D(
      tflite::testing::kInputShape3D, tflite::testing::kInputData3D,
      tflite::testing::kAxisShape3D, tflite::testing::kAxisData3D,
      tflite::testing::kOutputShape3D, tflite::testing::kGoldenData3D,
      output_data, &params);
}

TF_LITE_MICRO_TEST(MeanInt83DKeepDims) {
  int8_t expected_output_data_quant[tflite::testing::kOutputElements3D];
  int8_t output_data_quant[tflite::testing::kOutputElements3D];
  int8_t input_data_quant[tflite::testing::kInputElements3D];

  float input_scale = 0.5f;
  int input_zero_point = 0;
  float output_scale = 0.5f;
  int output_zero_point = 0;

  TfLiteReducerParams params = {
      true  // keep_dims
  };

  tflite::testing::TestMeanOpQuantized<int8_t>(
      tflite::testing::kInputShape3D, tflite::testing::kInputData3D,
      input_data_quant, input_scale, input_zero_point,
      tflite::testing::kAxisShape3D, tflite::testing::kAxisData3D,
      tflite::testing::kOutputShape3D, tflite::testing::kGoldenData3D,
      output_data_quant, expected_output_data_quant, output_scale,
      output_zero_point, &params);
}

TF_LITE_MICRO_TEST(MeanUInt83DKeepDims) {
  uint8_t expected_output_data_quant[tflite::testing::kOutputElements3D];
  uint8_t output_data_quant[tflite::testing::kOutputElements3D];
  uint8_t input_data_quant[tflite::testing::kInputElements3D];

  float input_scale = 0.5f;
  int input_zero_point = 138;
  float output_scale = 0.5f;
  int output_zero_point = 138;

  TfLiteReducerParams params = {
      true  // keep_dims
  };

  tflite::testing::TestMeanOpQuantized<uint8_t>(
      tflite::testing::kInputShape3D, tflite::testing::kInputData3D,
      input_data_quant, input_scale, input_zero_point,
      tflite::testing::kAxisShape3D, tflite::testing::kAxisData3D,
      tflite::testing::kOutputShape3D, tflite::testing::kGoldenData3D,
      output_data_quant, expected_output_data_quant, output_scale,
      output_zero_point, &params);
}

TF_LITE_MICRO_TEST(MeanFloat4DKeepDims) {
  float output_data[tflite::testing::kOutputElements4D];

  TfLiteReducerParams params = {
      true  // keep_dims
  };

  tflite::testing::TestMeanFloatInput4D(
      tflite::testing::kInputShape4D, tflite::testing::kInputData4D,
      tflite::testing::kAxisShape4D, tflite::testing::kAxisData4D,
      tflite::testing::kOutputShape4D, tflite::testing::kGoldenData4D,
      output_data, &params);
}

TF_LITE_MICRO_TEST(MeanInt84DKeepDims) {
  int8_t expected_output_data_quant[tflite::testing::kOutputElements4D];
  int8_t output_data_quant[tflite::testing::kOutputElements4D];
  int8_t input_data_quant[tflite::testing::kInputElements4D];

  float input_scale = 0.5f;
  int input_zero_point = 0;
  float output_scale = 0.5f;
  int output_zero_point = 0;

  TfLiteReducerParams params = {
      true  // keep_dims
  };

  tflite::testing::TestMeanOpQuantized<int8_t>(
      tflite::testing::kInputShape4D, tflite::testing::kInputData4D,
      input_data_quant, input_scale, input_zero_point,
      tflite::testing::kAxisShape4D, tflite::testing::kAxisData4D,
      tflite::testing::kOutputShape4D, tflite::testing::kGoldenData4D,
      output_data_quant, expected_output_data_quant, output_scale,
      output_zero_point, &params);
}

TF_LITE_MICRO_TEST(MeanUInt84DKeepDims) {
  uint8_t expected_output_data_quant[tflite::testing::kOutputElements4D];
  uint8_t output_data_quant[tflite::testing::kOutputElements4D];
  uint8_t input_data_quant[tflite::testing::kInputElements4D];

  float input_scale = 0.5f;
  int input_zero_point = 128;
  float output_scale = 0.5f;
  int output_zero_point = 128;

  TfLiteReducerParams params = {
      true  // keep_dims
  };

  tflite::testing::TestMeanOpQuantized<uint8_t>(
      tflite::testing::kInputShape4D, tflite::testing::kInputData4D,
      input_data_quant, input_scale, input_zero_point,
      tflite::testing::kAxisShape4D, tflite::testing::kAxisData4D,
      tflite::testing::kOutputShape4D, tflite::testing::kGoldenData4D,
      output_data_quant, expected_output_data_quant, output_scale,
      output_zero_point, &params);
}

TF_LITE_MICRO_TEST(MeanFloat4DWithoutKeepDims) {
  const int kOutputShape4D[] = {2, 2, 2};
  float output_data[tflite::testing::kOutputElements4D];
  TfLiteReducerParams params = {
      false  // keep_dims
  };

  tflite::testing::TestMeanFloatInput4D(
      tflite::testing::kInputShape4D, tflite::testing::kInputData4D,
      tflite::testing::kAxisShape4D, tflite::testing::kAxisData4D,
      kOutputShape4D, tflite::testing::kGoldenData4D, output_data, &params);
}

TF_LITE_MICRO_TEST(MeanInt84DWithoutKeepDims) {
  int8_t expected_output_data_quant[tflite::testing::kOutputElements4D];
  int8_t output_data_quant[tflite::testing::kOutputElements4D];
  int8_t input_data_quant[tflite::testing::kInputElements4D];

  const int kOutputShape4D[] = {2, 2, 2};
  TfLiteReducerParams params = {
      false  // keep_dims
  };
  float input_scale = 0.5f;
  int input_zero_point = 0;
  float output_scale = 0.5f;
  int output_zero_point = 0;

  tflite::testing::TestMeanOpQuantized<int8_t>(
      tflite::testing::kInputShape4D, tflite::testing::kInputData4D,
      input_data_quant, input_scale, input_zero_point,
      tflite::testing::kAxisShape4D, tflite::testing::kAxisData4D,
      kOutputShape4D, tflite::testing::kGoldenData4D, output_data_quant,
      expected_output_data_quant, output_scale, output_zero_point, &params);
}

TF_LITE_MICRO_TEST(MeanUInt84DWithoutKeepDims) {
  uint8_t expected_output_data_quant[tflite::testing::kOutputElements4D];
  uint8_t output_data_quant[tflite::testing::kOutputElements4D];
  uint8_t input_data_quant[tflite::testing::kInputElements4D];

  const int kOutputShape4D[] = {2, 2, 2};
  TfLiteReducerParams params = {
      false  // keep_dims
  };
  float input_scale = 0.5f;
  int input_zero_point = 128;
  float output_scale = 0.5f;
  int output_zero_point = 128;

  tflite::testing::TestMeanOpQuantized<uint8_t>(
      tflite::testing::kInputShape4D, tflite::testing::kInputData4D,
      input_data_quant, input_scale, input_zero_point,
      tflite::testing::kAxisShape4D, tflite::testing::kAxisData4D,
      kOutputShape4D, tflite::testing::kGoldenData4D, output_data_quant,
      expected_output_data_quant, output_scale, output_zero_point, &params);
}

TF_LITE_MICRO_TEST(MeanFloat4DWithoutKeepDimsWithPrecision) {
  const int kInputShape4D[] = {4, 2, 2, 3, 1};
  const float kInputData4D[] = {1.0,  24.0, 13.0, 3.0,  9.0,  17.0,
                                11.0, 36.0, 14.0, 19.0, 17.0, 22.0};
  const int kOutputElements4D = 2;
  const int kOutputShape4D[] = {2, 2, 1};
  const float kGoldenData4D[] = {11.166667, 19.833334};
  float output_data[kOutputElements4D];
  TfLiteReducerParams params = {
      false  // keep_dims
  };

  tflite::testing::TestMeanFloatInput4D(
      kInputShape4D, kInputData4D, tflite::testing::kAxisShape4D,
      tflite::testing::kAxisData4D, kOutputShape4D, kGoldenData4D, output_data,
      &params);
}

TF_LITE_MICRO_TEST(FloatMaxOpTestNotKeepDims) {
  const int input_shape[] = {3, 4, 3, 2};
  const float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                              9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                              17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  const int axis_shape[] = {1, 4};
  const int32_t axis_data[] = {1, 0, -3, -3};
  const int output_shape[] = {1, 2};
  const float expected_output_data[] = {23, 24};
  float output_data[2];

  TfLiteReducerParams params = {false};

  tflite::testing::TestReduceOpFloat(
      input_shape, input_data, axis_shape, axis_data, output_shape, output_data,
      expected_output_data, tflite::ops::micro::Register_REDUCE_MAX(), &params);
}

TF_LITE_MICRO_TEST(FloatMaxOpTestKeepDims) {
  const int input_shape[] = {3, 4, 3, 2};
  const float input_data[] = {1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,
                              9.0,  10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0,
                              17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};
  const int axis_shape[] = {1, 2};
  const int32_t axis_data[] = {0, 2};
  const int output_shape[] = {1, 3};
  const float expected_output_data[] = {20, 22, 24};
  float output_data[3];

  TfLiteReducerParams params = {true};

  tflite::testing::TestReduceOpFloat(
      input_shape, input_data, axis_shape, axis_data, output_shape, output_data,
      expected_output_data, tflite::ops::micro::Register_REDUCE_MAX(), &params);
}

TF_LITE_MICRO_TEST(Int8MaxOpTestKeepDims) {
  const int input_shape[] = {3, 1, 3, 2};
  const float input_data[] = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  const int axis_shape[] = {1, 1};
  const int32_t axis_data[] = {1, 1};
  const int output_shape[] = {1, 2};
  const float expected_output_data[] = {0.5, 0.6};

  float input_scale = 2 / 255.0;
  int input_zp = 0;

  TfLiteReducerParams params = {true};

  int8_t input_data_quant[6];
  int8_t output_data_quant[2];
  int8_t expected_output_data_quant[2];

  tflite::testing::TestReduceOpQuantized<int8_t>(
      input_shape, input_data, input_data_quant, input_scale, input_zp,
      axis_shape, axis_data, output_shape, expected_output_data,
      output_data_quant, expected_output_data_quant, input_scale, input_zp,
      tflite::ops::micro::Register_REDUCE_MAX(), &params);
}

TF_LITE_MICRO_TEST(Int8MaxOpTestWithoutKeepDims) {
  const int input_shape[] = {3, 1, 3, 2};
  const float input_data[] = {0.4, 0.2, 0.3, 0.4, 0.5, 0.6};
  const int axis_shape[] = {1, 1};
  const int32_t axis_data[] = {1, 1};
  const int output_shape[] = {1, 2};
  const float expected_output_data[] = {0.5, 0.6};

  float input_scale = 2 / 255.0;
  int input_zp = 0;
  float output_scale = 2 / 255.0;
  int output_zp = 0;

  TfLiteReducerParams params = {false};

  int8_t input_data_quant[6];
  int8_t output_data_quant[2];
  int8_t expected_output_data_quant[2];

  tflite::testing::TestReduceOpQuantized<int8_t>(
      input_shape, input_data, input_data_quant, input_scale, input_zp,
      axis_shape, axis_data, output_shape, expected_output_data,
      output_data_quant, expected_output_data_quant, output_scale, output_zp,
      tflite::ops::micro::Register_REDUCE_MAX(), &params);
}

TF_LITE_MICRO_TEST(MeanInt84DWithoutKeepDimsWithPrecision) {
  const int kInputShape4D[] = {4, 2, 2, 3, 1};
  const float kInputData4D[] = {1.0,  24.0, 13.0, 3.0,  9.0,  17.0,
                                11.0, 36.0, 14.0, 19.0, 17.0, 22.0};
  const int kOutputShape4D[] = {2, 2, 1};
  const float kGoldenData4D[] = {11.166667, 19.833334};
  TfLiteReducerParams params = {
      false  // keep_dims
  };
  float input_scale = 0.5f;
  int input_zero_point = 0;
  float output_scale = 0.5f;
  int output_zero_point = 0;

  int8_t output_data_quant[2];
  int8_t expected_output_data_quant[2];
  int8_t input_data_quant[12];

  tflite::testing::TestMeanOpQuantized<int8_t>(
      kInputShape4D, kInputData4D, input_data_quant, input_scale,
      input_zero_point, tflite::testing::kAxisShape4D,
      tflite::testing::kAxisData4D, kOutputShape4D, kGoldenData4D,
      output_data_quant, expected_output_data_quant, output_scale,
      output_zero_point, &params);
}

TF_LITE_MICRO_TEST(MeanUInt84DWithoutKeepDimsWithPrecision) {
  const int kInputShape4D[] = {4, 2, 2, 3, 1};
  const float kInputData4D[] = {1.0,  24.0, 13.0, 3.0,  9.0,  17.0,
                                11.0, 36.0, 14.0, 19.0, 17.0, 22.0};
  const int kOutputShape4D[] = {2, 2, 1};
  const float kGoldenData4D[] = {11.166667, 19.833334};
  TfLiteReducerParams params = {
      false  // keep_dims
  };

  float input_scale = 0.5f;
  int input_zero_point = 128;
  float output_scale = 0.5f;
  int output_zero_point = 128;

  uint8_t output_data_quant[2];
  uint8_t expected_output_data_quant[2];
  uint8_t input_data_quant[12];

  tflite::testing::TestMeanOpQuantized<uint8_t>(
      kInputShape4D, kInputData4D, input_data_quant, input_scale,
      input_zero_point, tflite::testing::kAxisShape4D,
      tflite::testing::kAxisData4D, kOutputShape4D, kGoldenData4D,
      output_data_quant, expected_output_data_quant, output_scale,
      output_zero_point, &params);
}
TF_LITE_MICRO_TESTS_END
