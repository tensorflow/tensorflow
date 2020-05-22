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
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

// Common inputs and outputs.
// static const int kInputElements4D = 24;
static const int kInputShape4D[] = {4, 2, 2, 3, 2};
static const float kInputData4D[] = {
    1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0, 12.0,
    13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0, 24.0};

// static const int kAxisElements = 3;
static const int kAxisShape[] = {1, 2};
static const int32_t kAxisData[] = {1, 2};

static const int kOutputElements = 4;
static const int kOutputShape[] = {4, 2, 1, 1, 2};
static const float kGoldenData[] = {6, 7, 18, 19};

static TfLiteReducerParams params = {
    true  // keep_dims
};

template <typename T>
TfLiteStatus ValidateReduceGoldens(TfLiteTensor* tensors, int tensors_size,
                                   const T* expected_output_data,
                                   T* output_data, int output_length,
                                   TfLiteReducerParams* params,
                                   float tolerance = 1e-5) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;

  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_MEAN, 1);

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = nullptr;
  size_t init_data_size = 0;
  void* user_data = nullptr;

  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = nullptr;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(params);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TfLiteStatus return_val = registration->invoke(&context, &node);
  if (return_val != kTfLiteOk) {
    return return_val;
  }

  if (registration->free) {
    registration->free(&context, user_data);
  }

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

  constexpr int num_of_inputs = 2;   // input and axis
  constexpr int num_of_outputs = 1;  // output

  constexpr int tensors_size = num_of_inputs + num_of_outputs;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateInt32Tensor(axis_data, axis_dims, "axis_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateReduceGoldens(tensors, tensors_size, expected_output_data,
                            output_data, output_dims_count, params, tolerance));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(MeanFloat4DKeepDims) {
  float output_data[tflite::testing::kOutputElements];

  tflite::testing::TestMeanFloatInput4D(
      tflite::testing::kInputShape4D, tflite::testing::kInputData4D,
      tflite::testing::kAxisShape, tflite::testing::kAxisData,
      tflite::testing::kOutputShape, tflite::testing::kGoldenData, output_data,
      &tflite::testing::params);
}

TF_LITE_MICRO_TEST(MeanFloat4DWithoutKeepDims) {
  const int kOutputShape[] = {2, 2, 2};
  float output_data[tflite::testing::kOutputElements];

  TfLiteReducerParams params = {
      false  // keep_dims
  };

  tflite::testing::TestMeanFloatInput4D(
      tflite::testing::kInputShape4D, tflite::testing::kInputData4D,
      tflite::testing::kAxisShape, tflite::testing::kAxisData, kOutputShape,
      tflite::testing::kGoldenData, output_data, &params);
}

TF_LITE_MICRO_TEST(MeanFloat4DWithoutKeepDimsWithPrecision) {
  const int kInputShape4D[] = {4, 2, 2, 3, 1};
  const float kInputData4D[] = {1.0,  24.0, 13.0, 3.0,  9.0,  17.0,
                                11.0, 36.0, 14.0, 19.0, 17.0, 22.0};
  const int kOutputElements = 2;
  const int kOutputShape[] = {2, 2, 1};
  const float kGoldenData[] = {11.166667, 19.833334};
  float output_data[kOutputElements];

  TfLiteReducerParams params = {
      false  // keep_dims
  };

  tflite::testing::TestMeanFloatInput4D(
      kInputShape4D, kInputData4D, tflite::testing::kAxisShape,
      tflite::testing::kAxisData, kOutputShape, kGoldenData, output_data,
      &params);
}

TF_LITE_MICRO_TESTS_END
