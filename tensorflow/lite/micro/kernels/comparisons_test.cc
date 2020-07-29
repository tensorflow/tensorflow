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
#include <cstdint>
#include <initializer_list>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

constexpr int inputs_size = 2;
constexpr int outputs_size = 1;
constexpr int tensors_size = inputs_size + outputs_size;

void TestComparison(tflite::BuiltinOperator op, TfLiteTensor* tensors,
                    bool* expected_output_data, bool* output_data) {
  const int output_dims_count = ElementCount(*tensors[inputs_size].dims);

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration = resolver.FindOp(op);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  const int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  const int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = nullptr;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

void TestComparisonFloat(tflite::BuiltinOperator op, int* input1_dims_data,
                         float* input1_data, int* input2_dims_data,
                         float* input2_data, bool* expected_output_data,
                         int* output_dims_data, bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input1_data, input1_dims),
      CreateFloatTensor(input2_data, input2_dims),
      CreateBoolTensor(output_data, output_dims),
  };

  TestComparison(op, tensors, expected_output_data, output_data);
}

void TestComparisonBool(tflite::BuiltinOperator op, int* input1_dims_data,
                        bool* input1_data, int* input2_dims_data,
                        bool* input2_data, bool* expected_output_data,
                        int* output_dims_data, bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateBoolTensor(input1_data, input1_dims),
      CreateBoolTensor(input2_data, input2_dims),
      CreateBoolTensor(output_data, output_dims),
  };

  TestComparison(op, tensors, expected_output_data, output_data);
}

void TestComparisonInt(tflite::BuiltinOperator op, int* input1_dims_data,
                       int32_t* input1_data, int* input2_dims_data,
                       int32_t* input2_data, bool* expected_output_data,
                       int* output_dims_data, bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateInt32Tensor(input1_data, input1_dims),
      CreateInt32Tensor(input2_data, input2_dims),
      CreateBoolTensor(output_data, output_dims),
  };

  TestComparison(op, tensors, expected_output_data, output_data);
}

void TestComparisonQuantizedUInt8(tflite::BuiltinOperator op,
                                  int* input1_dims_data, float* input1_data,
                                  uint8_t* input1_quantized, float input1_scale,
                                  int input1_zero_point, int* input2_dims_data,
                                  float* input2_data, uint8_t* input2_quantized,
                                  float input2_scale, int input2_zero_point,
                                  bool* expected_output_data,
                                  int* output_dims_data, bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_quantized, input1_dims,
                            input1_scale, input1_zero_point),
      CreateQuantizedTensor(input2_data, input2_quantized, input2_dims,
                            input2_scale, input2_zero_point),
      CreateBoolTensor(output_data, output_dims),
  };

  TestComparison(op, tensors, expected_output_data, output_data);
}

void TestComparisonQuantizedInt8(tflite::BuiltinOperator op,
                                 int* input1_dims_data, float* input1_data,
                                 int8_t* input1_quantized, float input1_scale,
                                 int input1_zero_point, int* input2_dims_data,
                                 float* input2_data, int8_t* input2_quantized,
                                 float input2_scale, int input2_zero_point,
                                 bool* expected_output_data,
                                 int* output_dims_data, bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInts(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInts(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_quantized, input1_dims,
                            input1_scale, input1_zero_point),
      CreateQuantizedTensor(input2_data, input2_quantized, input2_dims,
                            input2_scale, input2_zero_point),
      CreateBoolTensor(output_data, output_dims),
  };

  TestComparison(op, tensors, expected_output_data, output_data);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(EqualBool) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  bool input1_data[] = {true, false, true, false};
  bool input2_data[] = {true, true, false, false};

  bool expected_data[] = {true, false, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonBool(tflite::BuiltinOperator_EQUAL, input1_dim,
                                      input1_data, input2_dim, input2_data,
                                      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {true, false, false, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {false, false, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};
  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_EQUAL, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {false, false, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_EQUAL, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 2, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {false, false, false, false,
                          false, false, true,  false};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_EQUAL, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualBool) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  bool input1_data[] = {true, false, true, false};
  bool input2_data[] = {true, true, false, false};

  bool expected_data[] = {false, true, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonBool(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {false, true, true, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {true, true, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {true, true, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 2, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {true, true, true, true, true, true, false, true};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {false, true, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {false, true, false, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {false, true, false, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 2, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {false, true, true, false, false, true, false, true};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {true, true, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data,
      input2_dim, input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {false, true, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data,
      input2_dim, input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {false, true, true, false};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data,
      input2_dim, input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 2, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {false, true, true, false, false, true, true, true};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data,
      input2_dim, input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {false, false, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_LESS, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 6, 5};

  bool expected_data[] = {true, false, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_LESS, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {true, false, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_LESS, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 6, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {true, false, false, true, true, false, false, false};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_LESS, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualFloat) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  float input1_data[] = {0.1, 0.9, 0.7, 0.3};
  float input2_data[] = {0.1, 0.2, 0.6, 0.5};

  bool expected_data[] = {true, false, false, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualInt) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {true, false, true, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualBroadcast) {
  int input1_dim[] = {4, 1, 1, 1, 4};
  int input2_dim[] = {4, 1, 1, 1, 1};

  int32_t input1_data[] = {-1, 9, 7, 3};
  int32_t input2_data[] = {7};

  bool expected_data[] = {true, false, true, true};
  int expected_dim[] = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualBroadcastTwoD) {
  int input1_dim[] = {4, 1, 1, 2, 4};
  int input2_dim[] = {4, 1, 1, 1, 4};

  int32_t input1_data[] = {-1, 9, 7, 3, 2, 4, 2, 8};
  int32_t input2_data[] = {7, 1, 2, 4};

  bool expected_data[] = {true, false, false, true, true, false, true, false};
  int expected_dim[] = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualQuantizedUInt8) {
  int input1_dim[] = {4, 1, 2, 2, 1};
  int input2_dim[] = {4, 1, 2, 2, 1};
  float input1_data[] = {1, 9, 7, 3};
  float input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {true, false, true, false};
  int expected_dim[] = {4, 1, 2, 2, 1};

  const float input1_scale = 0.5;
  const int input1_zero_point = 128;
  const float input2_scale = 0.25;
  const int input2_zero_point = 125;
  uint8_t input1_quantized[4];
  uint8_t input2_quantized[4];

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_EQUAL, input1_dim, input1_data, input1_quantized,
      input1_scale, input1_zero_point, input2_dim, input2_data,
      input2_quantized, input2_scale, input2_zero_point, expected_data,
      expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualQuantizedInt8) {
  int input1_dim[] = {4, 1, 2, 2, 1};
  int input2_dim[] = {4, 1, 2, 2, 1};

  float input1_data[] = {1, -9, 7, 3};
  float input2_data[] = {-1, 2, 7, 5};

  bool expected_data[] = {false, false, true, false};
  int expected_dim[] = {4, 1, 2, 2, 1};

  const float input1_scale = 0.5;
  const int input1_zero_point = -5;
  const float input2_scale = 0.25;
  const int input2_zero_point = 5;
  int8_t input1_quantized[4];
  int8_t input2_quantized[4];

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedInt8(
      tflite::BuiltinOperator_EQUAL, input1_dim, input1_data, input1_quantized,
      input1_scale, input1_zero_point, input2_dim, input2_data,
      input2_quantized, input2_scale, input2_zero_point, expected_data,
      expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualQuantizedUInt8) {
  int input1_dim[] = {4, 1, 2, 2, 1};
  int input2_dim[] = {4, 1, 2, 2, 1};
  float input1_data[] = {1, 9, 7, 3};
  float input2_data[] = {1, 2, 7, 0};

  bool expected_data[] = {false, true, false, true};
  int expected_dim[] = {4, 1, 2, 2, 1};

  const float input1_scale = 0.5;
  const int input1_zero_point = 128;
  const float input2_scale = 0.25;
  const int input2_zero_point = 125;
  uint8_t input1_quantized[4];
  uint8_t input2_quantized[4];

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data,
      input1_quantized, input1_scale, input1_zero_point, input2_dim,
      input2_data, input2_quantized, input2_scale, input2_zero_point,
      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualQuantizedInt8) {
  int input1_dim[] = {4, 1, 2, 2, 1};
  int input2_dim[] = {4, 1, 2, 2, 1};

  float input1_data[] = {1, -9, 7, 3};
  float input2_data[] = {1, 2, 7, 5};

  bool expected_data[] = {false, true, false, true};
  int expected_dim[] = {4, 1, 2, 2, 1};

  const float input1_scale = 0.5;
  const int input1_zero_point = -5;
  const float input2_scale = 0.25;
  const int input2_zero_point = 5;
  int8_t input1_quantized[4];
  int8_t input2_quantized[4];

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedInt8(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data,
      input1_quantized, input1_scale, input1_zero_point, input2_dim,
      input2_data, input2_quantized, input2_scale, input2_zero_point,
      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterQuantizedUInt8) {
  int input1_dim[] = {4, 1, 2, 2, 1};
  int input2_dim[] = {4, 1, 2, 2, 1};
  float input1_data[] = {1, 9, 7, 3};
  float input2_data[] = {1, 2, 6, 5};

  bool expected_data[] = {false, true, true, false};
  int expected_dim[] = {4, 1, 2, 2, 1};

  const float input1_scale = 0.5;
  const int input1_zero_point = 128;
  const float input2_scale = 0.25;
  const int input2_zero_point = 125;
  uint8_t input1_quantized[4];
  uint8_t input2_quantized[4];

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data,
      input1_quantized, input1_scale, input1_zero_point, input2_dim,
      input2_data, input2_quantized, input2_scale, input2_zero_point,
      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterQuantizedUInt8SmallRange) {
  int input1_dim[] = {4, 1, 2, 2, 1};
  int input2_dim[] = {4, 1, 2, 2, 1};
  float input1_data[] = {1, 0.5, 0.35, 0.1};
  float input2_data[] = {1.01, 0.25, 0.3, 0.4};

  bool expected_data[] = {false, true, true, false};
  int expected_dim[] = {4, 1, 2, 2, 1};

  const float input1_scale = 0.5;
  const int input1_zero_point = 128;
  const float input2_scale = 0.25;
  const int input2_zero_point = 125;
  uint8_t input1_quantized[4];
  uint8_t input2_quantized[4];

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data,
      input1_quantized, input1_scale, input1_zero_point, input2_dim,
      input2_data, input2_quantized, input2_scale, input2_zero_point,
      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterUInt8EqualQuantized) {
  int input1_dim[] = {4, 1, 2, 2, 1};
  int input2_dim[] = {4, 1, 2, 2, 1};

  float input1_data[] = {1, 9, 7, 3};
  float input2_data[] = {1, 2, 6, 5};

  bool expected_data[] = {true, true, true, false};
  int expected_dim[] = {4, 1, 2, 2, 1};

  const float input1_scale = 0.5;
  const int input1_zero_point = 128;
  const float input2_scale = 0.25;
  const int input2_zero_point = 125;
  uint8_t input1_quantized[4];
  uint8_t input2_quantized[4];

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data,
      input1_quantized, input1_scale, input1_zero_point, input2_dim,
      input2_data, input2_quantized, input1_scale, input1_zero_point,
      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessQuantizedUInt8) {
  int input1_dim[] = {4, 1, 2, 2, 1};
  int input2_dim[] = {4, 1, 2, 2, 1};

  float input1_data[] = {1, 9, 7, 3};
  float input2_data[] = {1, 2, 6, 5};

  bool expected_data[] = {false, false, false, true};
  int expected_dim[] = {4, 1, 2, 2, 1};

  const float input1_scale = 0.5;
  const int input1_zero_point = 128;
  const float input2_scale = 0.25;
  const int input2_zero_point = 125;
  uint8_t input1_quantized[4];
  uint8_t input2_quantized[4];

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_LESS, input1_dim, input1_data, input1_quantized,
      input1_scale, input1_zero_point, input2_dim, input2_data,
      input2_quantized, input1_scale, input1_zero_point, expected_data,
      expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualQuantizedUInt8) {
  int input1_dim[] = {4, 1, 2, 2, 1};
  int input2_dim[] = {4, 1, 2, 2, 1};

  float input1_data[] = {1, 9, 7, 3};
  float input2_data[] = {1, 2, 6, 5};

  bool expected_data[] = {true, false, false, true};
  int expected_dim[] = {4, 1, 2, 2, 1};

  const float input1_scale = 0.5;
  const int input1_zero_point = 128;
  const float input2_scale = 0.25;
  const int input2_zero_point = 125;
  uint8_t input1_quantized[4];
  uint8_t input2_quantized[4];

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data,
      input1_quantized, input1_scale, input1_zero_point, input2_dim,
      input2_data, input2_quantized, input1_scale, input1_zero_point,
      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualQuantizedUInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, 2, 7, 8, 11, 20, 2};
    float input2_data[] = {2};

    bool expected_data[] = {false, true, false, false, false, false};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = 128;
    const float input2_scale = 0.25;
    const int input2_zero_point = 125;
    uint8_t input1_quantized[6];
    uint8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_EQUAL, input1_dim, input1_data,
        input1_quantized, input1_scale, input1_zero_point, input2_dim,
        input2_data, input2_quantized, input1_scale, input1_zero_point,
        expected_data, expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(NotEqualQuantizedUInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, 2, 7, 8, 11, 20};
    float input2_data[] = {2};

    bool expected_data[] = {true, false, true, true, true, true};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = 128;
    const float input2_scale = 0.25;
    const int input2_zero_point = 125;
    uint8_t input1_quantized[6];
    uint8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data,
        input1_quantized, input1_scale, input1_zero_point, input2_dim,
        input2_data, input2_quantized, input1_scale, input1_zero_point,
        expected_data, expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(NotEqualQuantizedInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, -2, -71, 8, 11, 20};
    float input2_data[] = {8};

    bool expected_data[] = {true, true, true, false, true, true};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = -9;
    const float input2_scale = 0.25;
    const int input2_zero_point = 9;
    int8_t input1_quantized[6];
    int8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data,
        input1_quantized, input1_scale, input1_zero_point, input2_dim,
        input2_data, input2_quantized, input1_scale, input1_zero_point,
        expected_data, expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(GreaterQuantizedUInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, 2, 7, 8, 11, 20};
    float input2_data[] = {2};

    bool expected_data[] = {true, false, true, true, true, true};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = 128;
    const float input2_scale = 0.25;
    const int input2_zero_point = 125;
    uint8_t input1_quantized[6];
    uint8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_GREATER, input1_dim, input1_data,
        input1_quantized, input1_scale, input1_zero_point, input2_dim,
        input2_data, input2_quantized, input1_scale, input1_zero_point,
        expected_data, expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(GreaterQuantizedInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, -2, -71, 8, 11, 20};
    float input2_data[] = {8};

    bool expected_data[] = {true, false, false, false, true, true};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = -9;
    const float input2_scale = 0.25;
    const int input2_zero_point = 9;
    int8_t input1_quantized[6];
    int8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::BuiltinOperator_GREATER, input1_dim, input1_data,
        input1_quantized, input1_scale, input1_zero_point, input2_dim,
        input2_data, input2_quantized, input1_scale, input1_zero_point,
        expected_data, expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(GreaterEqualQuantizedUInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, 2, 7, 8, 11, 20};
    float input2_data[] = {2};

    bool expected_data[] = {true, true, true, true, true, true};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = 128;
    const float input2_scale = 0.25;
    const int input2_zero_point = 125;
    uint8_t input1_quantized[6];
    uint8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data,
        input1_quantized, input1_scale, input1_zero_point, input2_dim,
        input2_data, input2_quantized, input1_scale, input1_zero_point,
        expected_data, expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(GreaterEqualQuantizedInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, -2, -71, 8, 11, 20};
    float input2_data[] = {8};

    bool expected_data[] = {true, false, false, true, true, true};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = -9;
    const float input2_scale = 0.25;
    const int input2_zero_point = 9;
    int8_t input1_quantized[6];
    int8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data,
        input1_quantized, input1_scale, input1_zero_point, input2_dim,
        input2_data, input2_quantized, input1_scale, input1_zero_point,
        expected_data, expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(LessQuantizedUInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, 2, -1, 8, 11, 20};
    float input2_data[] = {2};

    bool expected_data[] = {false, false, true, false, false, false};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = 128;
    const float input2_scale = 0.25;
    const int input2_zero_point = 125;
    uint8_t input1_quantized[6];
    uint8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_LESS, input1_dim, input1_data, input1_quantized,
        input1_scale, input1_zero_point, input2_dim, input2_data,
        input2_quantized, input1_scale, input1_zero_point, expected_data,
        expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(LessQuantizedInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, -2, -71, 8, 11, 20};
    float input2_data[] = {8};

    bool expected_data[] = {false, true, true, false, false, false};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = -9;
    const float input2_scale = 0.25;
    const int input2_zero_point = 9;
    int8_t input1_quantized[6];
    int8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::BuiltinOperator_LESS, input1_dim, input1_data, input1_quantized,
        input1_scale, input1_zero_point, input2_dim, input2_data,
        input2_quantized, input1_scale, input1_zero_point, expected_data,
        expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(LessEqualQuantizedUInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, 2, -1, 8, 11, 20};
    float input2_data[] = {2};

    bool expected_data[] = {false, true, true, false, false, false};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = 128;
    const float input2_scale = 0.25;
    const int input2_zero_point = 125;
    uint8_t input1_quantized[6];
    uint8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data,
        input1_quantized, input1_scale, input1_zero_point, input2_dim,
        input2_data, input2_quantized, input1_scale, input1_zero_point,
        expected_data, expected_dim, output_data);
  }
}

TF_LITE_MICRO_TEST(LessEqualQuantizedInt8WithBroadcast) {
  const int num_shapes = 4;
  const int max_shape_size = 5;
  int test_shapes[num_shapes][max_shape_size] = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < num_shapes; ++i) {
    int* input1_dim = test_shapes[i];
    int input2_dim[] = {1, 1};
    float input1_data[] = {20, -2, -71, 8, 11, 20};
    float input2_data[] = {8};

    bool expected_data[] = {false, true, true, true, false, false};
    int* expected_dim = input1_dim;

    const float input1_scale = 0.5;
    const int input1_zero_point = -9;
    const float input2_scale = 0.25;
    const int input2_zero_point = 9;
    int8_t input1_quantized[6];
    int8_t input2_quantized[6];

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data,
        input1_quantized, input1_scale, input1_zero_point, input2_dim,
        input2_data, input2_quantized, input1_scale, input1_zero_point,
        expected_data, expected_dim, output_data);
  }
}

TF_LITE_MICRO_TESTS_END
