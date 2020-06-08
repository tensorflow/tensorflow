/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void ValidateArgMinMaxGoldens(TfLiteTensor* tensors, int tensors_size,
                              const int32_t* golden, int32_t* output,
                              int output_size, bool using_min) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration;
  if (using_min) {
    registration = resolver.FindOp(tflite::BuiltinOperator_ARG_MIN);
  } else {
    registration = resolver.FindOp(tflite::BuiltinOperator_ARG_MAX);
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, nullptr, init_data_size);
  }
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);
  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  if (registration->free) {
    registration->free(&context, user_data);
  }
  for (int i = 0; i < output_size; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden[i], output[i]);
  }
}

void TestArgMinMaxFloat(const int* input_dims_data, const float* input_values,
                        const int* axis_dims_data, const int32_t* axis_values,
                        const int* output_dims_data, int32_t* output,
                        const int32_t* goldens, bool using_min) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_values, input_dims, "input_tensor"),
      CreateInt32Tensor(axis_values, axis_dims, "axis_tensor"),
      CreateInt32Tensor(output, output_dims, "output_tensor"),
  };

  ValidateArgMinMaxGoldens(tensors, tensors_size, goldens, output,
                           output_dims_count, using_min);
}

template <typename T>
void TestArgMinMaxQuantized(const int* input_dims_data,
                            const float* input_values, T* input_quantized,
                            float input_scale, int input_zero_point,
                            const int* axis_dims_data,
                            const int32_t* axis_values,
                            const int* output_dims_data, int32_t* output,
                            const int32_t* goldens, bool using_min) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* axis_dims = IntArrayFromInts(axis_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_values, input_quantized, input_dims,
                            input_scale, input_zero_point, "input_tensor"),
      CreateInt32Tensor(axis_values, axis_dims, "axis_tensor"),
      CreateInt32Tensor(output, output_dims, "output_tensor"),
  };

  ValidateArgMinMaxGoldens(tensors, tensors_size, goldens, output,
                           output_dims_count, using_min);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(GetMaxArgFloat) {
  int32_t output_data[1];
  const int input_dims[] = {4, 1, 1, 1, 4};
  const float input_values[] = {0.1, 0.9, 0.7, 0.3};
  const int axis_dims[] = {3, 1, 1, 1};
  const int32_t axis_values[] = {3};
  const int output_dims[] = {3, 1, 1, 1};
  const int32_t goldens[] = {1};

  tflite::testing::TestArgMinMaxFloat(input_dims, input_values, axis_dims,
                                      axis_values, output_dims, output_data,
                                      goldens, false);
}

TF_LITE_MICRO_TEST(GetMinArgFloat) {
  int32_t output_data[1];
  const int input_dims[] = {4, 1, 1, 1, 4};
  const float input_values[] = {0.1, 0.9, 0.7, 0.3};
  const int axis_dims[] = {3, 1, 1, 1};
  const int32_t axis_values[] = {3};
  const int output_dims[] = {3, 1, 1, 1};
  const int32_t goldens[] = {0};

  tflite::testing::TestArgMinMaxFloat(input_dims, input_values, axis_dims,
                                      axis_values, output_dims, output_data,
                                      goldens, true);
}

TF_LITE_MICRO_TEST(GetMaxArgUInt8) {
  int32_t output_data[1];
  const int input_size = 4;
  const int input_dims[] = {4, 1, 1, 1, input_size};
  const float input_values[] = {1, 9, 7, 3};
  const int axis_dims[] = {3, 1, 1, 1};
  const int32_t axis_values[] = {3};
  const int output_dims[] = {3, 1, 1, 1};
  const int32_t goldens[] = {1};

  float input_scale = 0.5;
  int input_zero_point = 124;
  uint8_t input_quantized[input_size];

  tflite::testing::TestArgMinMaxQuantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      axis_dims, axis_values, output_dims, output_data, goldens, false);
}

TF_LITE_MICRO_TEST(GetMinArgUInt8) {
  int32_t output_data[1];
  const int input_size = 4;
  const int input_dims[] = {4, 1, 1, 1, input_size};
  const float input_values[] = {1, 9, 7, 3};
  const int axis_dims[] = {3, 1, 1, 1};
  const int32_t axis_values[] = {3};
  const int output_dims[] = {3, 1, 1, 1};
  const int32_t goldens[] = {0};

  float input_scale = 0.5;
  int input_zero_point = 124;
  uint8_t input_quantized[input_size];

  tflite::testing::TestArgMinMaxQuantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      axis_dims, axis_values, output_dims, output_data, goldens, true);
}

TF_LITE_MICRO_TEST(GetMaxArgInt8) {
  int32_t output_data[1];
  const int input_size = 4;
  const int input_dims[] = {4, 1, 1, 1, input_size};
  const float input_values[] = {1, 9, 7, 3};
  const int axis_dims[] = {3, 1, 1, 1};
  const int32_t axis_values[] = {3};
  const int output_dims[] = {3, 1, 1, 1};
  const int32_t goldens[] = {1};

  float input_scale = 0.5;
  int input_zero_point = -9;
  int8_t input_quantized[input_size];

  tflite::testing::TestArgMinMaxQuantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      axis_dims, axis_values, output_dims, output_data, goldens, false);
}

TF_LITE_MICRO_TEST(GetMinArgInt8) {
  int32_t output_data[1];
  const int input_size = 4;
  const int input_dims[] = {4, 1, 1, 1, input_size};
  const float input_values[] = {1, 9, 7, 3};
  const int axis_dims[] = {3, 1, 1, 1};
  const int32_t axis_values[] = {3};
  const int output_dims[] = {3, 1, 1, 1};
  const int32_t goldens[] = {0};

  float input_scale = 0.5;
  int input_zero_point = -9;
  int8_t input_quantized[input_size];

  tflite::testing::TestArgMinMaxQuantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      axis_dims, axis_values, output_dims, output_data, goldens, true);
}

TF_LITE_MICRO_TEST(GetMaxArgMulDimensions) {
  int32_t output_data[2];
  const int input_size = 8;
  const int input_dims[] = {4, 1, 1, 2, 4};
  const float input_values[] = {1, 2, 7, 8, 1, 9, 7, 3};
  const int axis_dims[] = {3, 1, 1, 1};
  const int32_t axis_values[] = {3};
  const int output_dims[] = {3, 1, 1, 2};
  const int32_t goldens[] = {3, 1};

  float input_scale = 0.5;
  int input_zero_point = -9;
  int8_t input_quantized[input_size];

  tflite::testing::TestArgMinMaxQuantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      axis_dims, axis_values, output_dims, output_data, goldens, false);
}

TF_LITE_MICRO_TEST(GetMinArgMulDimensions) {
  int32_t output_data[2];
  const int input_size = 8;
  const int input_dims[] = {4, 1, 1, 2, 4};
  const float input_values[] = {1, 2, 7, 8, 1, 9, 7, 3};
  const int axis_dims[] = {3, 1, 1, 1};
  const int32_t axis_values[] = {3};
  const int output_dims[] = {3, 1, 1, 2};
  const int32_t goldens[] = {0, 0};

  float input_scale = 0.5;
  int input_zero_point = -9;
  int8_t input_quantized[input_size];

  tflite::testing::TestArgMinMaxQuantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      axis_dims, axis_values, output_dims, output_data, goldens, true);
}

TF_LITE_MICRO_TEST(GetMaxArgNegativeAxis) {
  const int input_size = 8;
  const int output_size = 4;
  const int input_dims[] = {4, 1, 1, 2, 4};
  const float input_values[] = {1, 2, 7, 8, 1, 9, 7, 3};
  const int axis_dims[] = {3, 1, 1, 1};
  const int32_t axis_values[] = {-2};
  const int output_dims[] = {3, 1, 1, 4};
  const int32_t goldens[] = {0, 1, 0, 0};

  float input_scale = 0.5;
  int input_zero_point = -9;
  int32_t output_data[output_size];
  int8_t input_quantized[input_size];

  tflite::testing::TestArgMinMaxQuantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      axis_dims, axis_values, output_dims, output_data, goldens, false);
}

TF_LITE_MICRO_TESTS_END
