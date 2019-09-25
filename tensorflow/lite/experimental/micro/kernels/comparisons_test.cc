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
#include <initializer_list>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

constexpr int inputs_size = 2;
constexpr int outputs_size = 1;
constexpr int tensors_size = inputs_size + outputs_size;

void TestComparison(tflite::BuiltinOperator op, TfLiteTensor* tensors,
                    std::initializer_list<bool> expected_output_data,
                    bool* output_data) {
  const int output_dims_count = ElementCount(*tensors[inputs_size].dims);

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration = resolver.FindOp(op, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteIntArray* inputs_array = IntArrayFromInitializer({2, 0, 1});
  TfLiteIntArray* outputs_array = IntArrayFromInitializer({1, 2});
  TfLiteIntArray* temporaries_array = IntArrayFromInitializer({0});

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
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data.begin()[i], output_data[i]);
  }
}

void TestComparisonFloat(tflite::BuiltinOperator op,
                         std::initializer_list<int> input1_dims_data,
                         std::initializer_list<float> input1_data,
                         std::initializer_list<int> input2_dims_data,
                         std::initializer_list<float> input2_data,
                         std::initializer_list<bool> expected_output_data,
                         std::initializer_list<int> output_dims_data,
                         bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input1_data, input1_dims, "input1_tensor"),
      CreateFloatTensor(input2_data, input2_dims, "input2_tensor"),
      CreateBoolTensor(output_data, output_dims, "output_tensor"),
  };

  TestComparison(op, tensors, expected_output_data, output_data);
}

void TestComparisonBool(tflite::BuiltinOperator op,
                        std::initializer_list<int> input1_dims_data,
                        std::initializer_list<bool> input1_data,
                        std::initializer_list<int> input2_dims_data,
                        std::initializer_list<bool> input2_data,
                        std::initializer_list<bool> expected_output_data,
                        std::initializer_list<int> output_dims_data,
                        bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateBoolTensor(input1_data, input1_dims, "input1_tensor"),
      CreateBoolTensor(input2_data, input2_dims, "input2_tensor"),
      CreateBoolTensor(output_data, output_dims, "output_tensor"),
  };

  TestComparison(op, tensors, expected_output_data, output_data);
}

void TestComparisonInt(tflite::BuiltinOperator op,
                       std::initializer_list<int> input1_dims_data,
                       std::initializer_list<int32_t> input1_data,
                       std::initializer_list<int> input2_dims_data,
                       std::initializer_list<int32_t> input2_data,
                       std::initializer_list<bool> expected_output_data,
                       std::initializer_list<int> output_dims_data,
                       bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);

  TfLiteTensor tensors[tensors_size] = {
      CreateQuantized32Tensor(input1_data, input1_dims, "input1_tensor", 1.0),
      CreateQuantized32Tensor(input2_data, input2_dims, "input2_tensor", 1.0),
      CreateBoolTensor(output_data, output_dims, "output_tensor"),
  };

  TestComparison(op, tensors, expected_output_data, output_data);
}

void TestComparisonQuantizedUInt8(
    tflite::BuiltinOperator op, std::initializer_list<int> input1_dims_data,
    std::initializer_list<uint8_t> input1_data, float input1_min,
    float input1_max, std::initializer_list<int> input2_dims_data,
    std::initializer_list<uint8_t> input2_data, float input2_min,
    float input2_max, std::initializer_list<bool> expected_output_data,
    std::initializer_list<int> output_dims_data, bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, "input1_tensor",
                            input1_min, input1_max),
      CreateQuantizedTensor(input2_data, input2_dims, "input2_tensor",
                            input2_min, input2_max),
      CreateBoolTensor(output_data, output_dims, "output_tensor"),
  };

  TestComparison(op, tensors, expected_output_data, output_data);
}

void TestComparisonQuantizedInt8(
    tflite::BuiltinOperator op, std::initializer_list<int> input1_dims_data,
    std::initializer_list<int8_t> input1_data, float input1_min,
    float input1_max, std::initializer_list<int> input2_dims_data,
    std::initializer_list<int8_t> input2_data, float input2_min,
    float input2_max, std::initializer_list<bool> expected_output_data,
    std::initializer_list<int> output_dims_data, bool* output_data) {
  TfLiteIntArray* input1_dims = IntArrayFromInitializer(input1_dims_data);
  TfLiteIntArray* input2_dims = IntArrayFromInitializer(input2_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input1_data, input1_dims, "input1_tensor",
                            input1_min, input1_max),
      CreateQuantizedTensor(input2_data, input2_dims, "input2_tensor",
                            input2_min, input2_max),
      CreateBoolTensor(output_data, output_dims, "output_tensor"),
  };

  TestComparison(op, tensors, expected_output_data, output_data);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN
using ::tflite::testing::F2Q;
using ::tflite::testing::F2QS;

TF_LITE_MICRO_TEST(EqualBool) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<bool> input1_data = {true, false, true, false};
  std::initializer_list<bool> input2_data = {true, true, false, false};

  std::initializer_list<bool> expected_data = {true, false, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonBool(tflite::BuiltinOperator_EQUAL, input1_dim,
                                      input1_data, input2_dim, input2_data,
                                      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualFloat) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<float> input1_data = {0.1, 0.9, 0.7, 0.3};
  std::initializer_list<float> input2_data = {0.1, 0.2, 0.6, 0.5};

  std::initializer_list<bool> expected_data = {true, false, false, false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualInt) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {1, 2, 7, 5};

  std::initializer_list<bool> expected_data = {false, false, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};
  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_EQUAL, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualBroadcast) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 1};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {7};

  std::initializer_list<bool> expected_data = {false, false, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_EQUAL, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualBroadcastTwoD) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 2, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3, 2, 4, 2, 8};
  std::initializer_list<int32_t> input2_data = {7, 1, 2, 4};

  std::initializer_list<bool> expected_data = {false, false, false, false,
                                               false, false, true,  false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_EQUAL, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualBool) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<bool> input1_data = {true, false, true, false};
  std::initializer_list<bool> input2_data = {true, true, false, false};

  std::initializer_list<bool> expected_data = {false, true, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonBool(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualFloat) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<float> input1_data = {0.1, 0.9, 0.7, 0.3};
  std::initializer_list<float> input2_data = {0.1, 0.2, 0.6, 0.5};

  std::initializer_list<bool> expected_data = {false, true, true, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualInt) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {1, 2, 7, 5};

  std::initializer_list<bool> expected_data = {true, true, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualBroadcast) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 1};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {7};

  std::initializer_list<bool> expected_data = {true, true, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(NotEqualBroadcastTwoD) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 2, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3, 2, 4, 2, 8};
  std::initializer_list<int32_t> input2_data = {7, 1, 2, 4};

  std::initializer_list<bool> expected_data = {true, true, true,  true,
                                               true, true, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterFloat) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<float> input1_data = {0.1, 0.9, 0.7, 0.3};
  std::initializer_list<float> input2_data = {0.1, 0.2, 0.6, 0.5};

  std::initializer_list<bool> expected_data = {false, true, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterInt) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {1, 2, 7, 5};

  std::initializer_list<bool> expected_data = {false, true, false, false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterBroadcast) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 1};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {7};

  std::initializer_list<bool> expected_data = {false, true, false, false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterBroadcastTwoD) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 2, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3, 2, 4, 2, 8};
  std::initializer_list<int32_t> input2_data = {7, 1, 2, 4};

  std::initializer_list<bool> expected_data = {false, true, true,  false,
                                               false, true, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualFloat) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<float> input1_data = {0.1, 0.9, 0.7, 0.3};
  std::initializer_list<float> input2_data = {0.1, 0.2, 0.6, 0.5};

  std::initializer_list<bool> expected_data = {true, true, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data,
      input2_dim, input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualInt) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {1, 2, 7, 5};

  std::initializer_list<bool> expected_data = {false, true, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data,
      input2_dim, input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualBroadcast) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 1};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {7};

  std::initializer_list<bool> expected_data = {false, true, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data,
      input2_dim, input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterEqualBroadcastTwoD) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 2, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3, 2, 4, 2, 8};
  std::initializer_list<int32_t> input2_data = {7, 1, 2, 4};

  std::initializer_list<bool> expected_data = {false, true, true, false,
                                               false, true, true, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data,
      input2_dim, input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessFloat) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<float> input1_data = {0.1, 0.9, 0.7, 0.3};
  std::initializer_list<float> input2_data = {0.1, 0.2, 0.6, 0.5};

  std::initializer_list<bool> expected_data = {false, false, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_LESS, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessInt) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {1, 2, 6, 5};

  std::initializer_list<bool> expected_data = {true, false, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_LESS, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessBroadcast) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 1};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {7};

  std::initializer_list<bool> expected_data = {true, false, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_LESS, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessBroadcastTwoD) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 2, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3, 2, 4, 6, 8};
  std::initializer_list<int32_t> input2_data = {7, 1, 2, 4};

  std::initializer_list<bool> expected_data = {true, false, false, true,
                                               true, false, false, false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(tflite::BuiltinOperator_LESS, input1_dim,
                                     input1_data, input2_dim, input2_data,
                                     expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualFloat) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<float> input1_data = {0.1, 0.9, 0.7, 0.3};
  std::initializer_list<float> input2_data = {0.1, 0.2, 0.6, 0.5};

  std::initializer_list<bool> expected_data = {true, false, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonFloat(
      tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualInt) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {1, 2, 7, 5};

  std::initializer_list<bool> expected_data = {true, false, true, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualBroadcast) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 1, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 1};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3};
  std::initializer_list<int32_t> input2_data = {7};

  std::initializer_list<bool> expected_data = {true, false, true, true};
  std::initializer_list<int> expected_dim = {4, 1, 1, 1, 4};

  bool output_data[4];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(LessEqualBroadcastTwoD) {
  std::initializer_list<int> input1_dim = {4, 1, 1, 2, 4};
  std::initializer_list<int> input2_dim = {4, 1, 1, 1, 4};

  std::initializer_list<int32_t> input1_data = {-1, 9, 7, 3, 2, 4, 2, 8};
  std::initializer_list<int32_t> input2_data = {7, 1, 2, 4};

  std::initializer_list<bool> expected_data = {true, false, false, true,
                                               true, false, true,  false};
  std::initializer_list<int> expected_dim = {4, 1, 1, 2, 4};

  bool output_data[8];
  tflite::testing::TestComparisonInt(
      tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data, input2_dim,
      input2_data, expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(EqualQuantizedUInt8) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<int> input1_dim = {4, 1, 2, 2, 1};
  std::initializer_list<int> input2_dim = {4, 1, 2, 2, 1};
  std::initializer_list<uint8_t> input1_data = {
      F2Q(1, kMin, kMax), F2Q(9, kMin, kMax), F2Q(7, kMin, kMax),
      F2Q(3, kMin, kMax)};
  std::initializer_list<uint8_t> input2_data = {
      F2Q(1, kMin, kMax), F2Q(2, kMin, kMax), F2Q(7, kMin, kMax),
      F2Q(5, kMin, kMax)};

  std::initializer_list<bool> expected_data = {true, false, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 2, 2, 1};

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_EQUAL, input1_dim, input1_data, kMin, kMax,
      input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
      output_data);
}

TF_LITE_MICRO_TEST(EqualQuantizedInt8) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::initializer_list<int> input1_dim = {4, 1, 2, 2, 1};
  std::initializer_list<int> input2_dim = {4, 1, 2, 2, 1};

  std::initializer_list<int8_t> input1_data = {
      F2QS(1, kMin, kMax), F2QS(-9, kMin, kMax), F2QS(7, kMin, kMax),
      F2QS(3, kMin, kMax)};
  std::initializer_list<int8_t> input2_data = {
      F2QS(-1, kMin, kMax), F2QS(2, kMin, kMax), F2QS(7, kMin, kMax),
      F2QS(5, kMin, kMax)};

  std::initializer_list<bool> expected_data = {false, false, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 2, 2, 1};

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedInt8(
      tflite::BuiltinOperator_EQUAL, input1_dim, input1_data, kMin, kMax,
      input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
      output_data);
}

TF_LITE_MICRO_TEST(NotEqualQuantizedUInt8) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<int> input1_dim = {4, 1, 2, 2, 1};
  std::initializer_list<int> input2_dim = {4, 1, 2, 2, 1};
  std::initializer_list<uint8_t> input1_data = {
      F2Q(1, kMin, kMax), F2Q(9, kMin, kMax), F2Q(7, kMin, kMax),
      F2Q(3, kMin, kMax)};
  std::initializer_list<uint8_t> input2_data = {
      F2Q(1, kMin, kMax), F2Q(2, kMin, kMax), F2Q(7, kMin, kMax),
      F2Q(0, kMin, kMax)};

  std::initializer_list<bool> expected_data = {false, true, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 2, 2, 1};

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, kMin, kMax,
      input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
      output_data);
}

TF_LITE_MICRO_TEST(NotEqualQuantizedInt8) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::initializer_list<int> input1_dim = {4, 1, 2, 2, 1};
  std::initializer_list<int> input2_dim = {4, 1, 2, 2, 1};

  std::initializer_list<int8_t> input1_data = {
      F2QS(1, kMin, kMax), F2QS(-9, kMin, kMax), F2QS(7, kMin, kMax),
      F2QS(3, kMin, kMax)};
  std::initializer_list<int8_t> input2_data = {
      F2QS(1, kMin, kMax), F2QS(2, kMin, kMax), F2QS(7, kMin, kMax),
      F2QS(5, kMin, kMax)};

  std::initializer_list<bool> expected_data = {false, true, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 2, 2, 1};

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedInt8(
      tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, kMin, kMax,
      input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
      output_data);
}

TF_LITE_MICRO_TEST(GreaterQuantizedUInt8) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<int> input1_dim = {4, 1, 2, 2, 1};
  std::initializer_list<int> input2_dim = {4, 1, 2, 2, 1};
  std::initializer_list<uint8_t> input1_data = {
      F2Q(1, kMin, kMax), F2Q(9, kMin, kMax), F2Q(7, kMin, kMax),
      F2Q(3, kMin, kMax)};
  std::initializer_list<uint8_t> input2_data = {
      F2Q(1, kMin, kMax), F2Q(2, kMin, kMax), F2Q(6, kMin, kMax),
      F2Q(5, kMin, kMax)};

  std::initializer_list<bool> expected_data = {false, true, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 2, 2, 1};

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data, kMin, kMax,
      input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
      output_data);
}

TF_LITE_MICRO_TEST(GreaterQuantizedUInt8SmallRange) {
  const float input1_min = 0.f;
  const float input1_max = 1.f;
  const float input2_min = 0.f;
  const float input2_max = 2.f;
  std::initializer_list<int> input1_dim = {4, 1, 2, 2, 1};
  std::initializer_list<int> input2_dim = {4, 1, 2, 2, 1};
  std::initializer_list<uint8_t> input1_data = {
      F2Q(1.0, input1_min, input1_max), F2Q(0.5, input1_min, input1_max),
      F2Q(0.35, input1_min, input1_max), F2Q(0.1, input1_min, input1_max)};
  std::initializer_list<uint8_t> input2_data = {
      F2Q(1.01, input2_min, input2_max), F2Q(0.25, input2_min, input2_max),
      F2Q(0.3, input2_min, input2_max), F2Q(0.4, input2_min, input2_max)};

  std::initializer_list<bool> expected_data = {false, true, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 2, 2, 1};

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_GREATER, input1_dim, input1_data, input1_min,
      input1_max, input2_dim, input2_data, input2_min, input2_max,
      expected_data, expected_dim, output_data);
}

TF_LITE_MICRO_TEST(GreaterUInt8EqualQuantized) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<int> input1_dim = {4, 1, 2, 2, 1};
  std::initializer_list<int> input2_dim = {4, 1, 2, 2, 1};

  std::initializer_list<uint8_t> input1_data = {
      F2Q(1, kMin, kMax), F2Q(9, kMin, kMax), F2Q(7, kMin, kMax),
      F2Q(3, kMin, kMax)};
  std::initializer_list<uint8_t> input2_data = {
      F2Q(1, kMin, kMax), F2Q(2, kMin, kMax), F2Q(6, kMin, kMax),
      F2Q(5, kMin, kMax)};

  std::initializer_list<bool> expected_data = {true, true, true, false};
  std::initializer_list<int> expected_dim = {4, 1, 2, 2, 1};

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data, kMin,
      kMax, input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
      output_data);
}

TF_LITE_MICRO_TEST(LessQuantizedUInt8) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<int> input1_dim = {4, 1, 2, 2, 1};
  std::initializer_list<int> input2_dim = {4, 1, 2, 2, 1};

  std::initializer_list<uint8_t> input1_data = {
      F2Q(1, kMin, kMax), F2Q(9, kMin, kMax), F2Q(7, kMin, kMax),
      F2Q(3, kMin, kMax)};
  std::initializer_list<uint8_t> input2_data = {
      F2Q(1, kMin, kMax), F2Q(2, kMin, kMax), F2Q(6, kMin, kMax),
      F2Q(5, kMin, kMax)};

  std::initializer_list<bool> expected_data = {false, false, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 2, 2, 1};

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_LESS, input1_dim, input1_data, kMin, kMax,
      input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
      output_data);
}

TF_LITE_MICRO_TEST(LessEqualQuantizedUInt8) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<int> input1_dim = {4, 1, 2, 2, 1};
  std::initializer_list<int> input2_dim = {4, 1, 2, 2, 1};

  std::initializer_list<uint8_t> input1_data = {
      F2Q(1, kMin, kMax), F2Q(9, kMin, kMax), F2Q(7, kMin, kMax),
      F2Q(3, kMin, kMax)};
  std::initializer_list<uint8_t> input2_data = {
      F2Q(1, kMin, kMax), F2Q(2, kMin, kMax), F2Q(6, kMin, kMax),
      F2Q(5, kMin, kMax)};

  std::initializer_list<bool> expected_data = {true, false, false, true};
  std::initializer_list<int> expected_dim = {4, 1, 2, 2, 1};

  bool output_data[4];
  tflite::testing::TestComparisonQuantizedUInt8(
      tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data, kMin, kMax,
      input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
      output_data);
}

TF_LITE_MICRO_TEST(EqualQuantizedUInt8WithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<std::initializer_list<int>> test_shapes = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};

  for (int i = 0; i < test_shapes.size(); ++i) {
    std::initializer_list<int> input1_dim = test_shapes.begin()[i];
    std::initializer_list<int> input2_dim = {1, 1};

    std::initializer_list<uint8_t> input1_data = {
        F2Q(20, kMin, kMax), F2Q(2, kMin, kMax),  F2Q(7, kMin, kMax),
        F2Q(8, kMin, kMax),  F2Q(11, kMin, kMax), F2Q(20, kMin, kMax)};
    std::initializer_list<uint8_t> input2_data = {F2Q(2, kMin, kMax)};

    std::initializer_list<bool> expected_data = {false, true,  false,
                                                 false, false, false};
    std::initializer_list<int> expected_dim = input1_dim;

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_EQUAL, input1_dim, input1_data, kMin, kMax,
        input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
        output_data);
  }
}

TF_LITE_MICRO_TEST(NotEqualQuantizedUInt8WithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<std::initializer_list<int>> test_shapes = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    std::initializer_list<int> input1_dim = test_shapes.begin()[i];
    std::initializer_list<int> input2_dim = {1, 1};

    std::initializer_list<uint8_t> input1_data = {
        F2Q(20, kMin, kMax), F2Q(2, kMin, kMax),  F2Q(7, kMin, kMax),
        F2Q(8, kMin, kMax),  F2Q(11, kMin, kMax), F2Q(20, kMin, kMax)};
    std::initializer_list<uint8_t> input2_data = {F2Q(2, kMin, kMax)};

    std::initializer_list<bool> expected_data = {true, false, true,
                                                 true, true,  true};
    std::initializer_list<int> expected_dim = input1_dim;

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, kMin, kMax,
        input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
        output_data);
  }
}

TF_LITE_MICRO_TEST(NotEqualQuantizedInt8WithBroadcast) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::initializer_list<std::initializer_list<int>> test_shapes = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    std::initializer_list<int> input1_dim = test_shapes.begin()[i];
    std::initializer_list<int> input2_dim = {1, 1};

    std::initializer_list<int8_t> input1_data = {
        F2QS(20, kMin, kMax), F2QS(-2, kMin, kMax), F2QS(-71, kMin, kMax),
        F2QS(8, kMin, kMax),  F2QS(11, kMin, kMax), F2QS(20, kMin, kMax)};
    std::initializer_list<int8_t> input2_data = {F2QS(8, kMin, kMax)};

    std::initializer_list<bool> expected_data = {true,  true, true,
                                                 false, true, true};
    std::initializer_list<int> expected_dim = input1_dim;

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::BuiltinOperator_NOT_EQUAL, input1_dim, input1_data, kMin, kMax,
        input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
        output_data);
  }
}

TF_LITE_MICRO_TEST(GreaterQuantizedUInt8WithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<std::initializer_list<int>> test_shapes = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    std::initializer_list<int> input1_dim = test_shapes.begin()[i];
    std::initializer_list<int> input2_dim = {1, 1};

    std::initializer_list<uint8_t> input1_data = {
        F2Q(20, kMin, kMax), F2Q(2, kMin, kMax),  F2Q(7, kMin, kMax),
        F2Q(8, kMin, kMax),  F2Q(11, kMin, kMax), F2Q(20, kMin, kMax)};
    std::initializer_list<uint8_t> input2_data = {F2Q(8, kMin, kMax)};

    std::initializer_list<bool> expected_data = {true,  false, false,
                                                 false, true,  true};
    std::initializer_list<int> expected_dim = input1_dim;

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_GREATER, input1_dim, input1_data, kMin, kMax,
        input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
        output_data);
  }
}

TF_LITE_MICRO_TEST(GreaterQuantizedInt8WithBroadcast) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::initializer_list<std::initializer_list<int>> test_shapes = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    std::initializer_list<int> input1_dim = test_shapes.begin()[i];
    std::initializer_list<int> input2_dim = {1, 1};

    std::initializer_list<int8_t> input1_data = {
        F2QS(20, kMin, kMax), F2QS(-2, kMin, kMax), F2QS(-71, kMin, kMax),
        F2QS(8, kMin, kMax),  F2QS(11, kMin, kMax), F2QS(20, kMin, kMax)};
    std::initializer_list<int8_t> input2_data = {F2QS(8, kMin, kMax)};

    std::initializer_list<bool> expected_data = {true,  false, false,
                                                 false, true,  true};
    std::initializer_list<int> expected_dim = input1_dim;

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::BuiltinOperator_GREATER, input1_dim, input1_data, kMin, kMax,
        input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
        output_data);
  }
}

TF_LITE_MICRO_TEST(GreaterEqualQuantizedUInt8WithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<std::initializer_list<int>> test_shapes = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    std::initializer_list<int> input1_dim = test_shapes.begin()[i];
    std::initializer_list<int> input2_dim = {1, 1};

    std::initializer_list<uint8_t> input1_data = {
        F2Q(20, kMin, kMax), F2Q(2, kMin, kMax),  F2Q(7, kMin, kMax),
        F2Q(8, kMin, kMax),  F2Q(11, kMin, kMax), F2Q(20, kMin, kMax)};
    std::initializer_list<uint8_t> input2_data = {F2Q(8, kMin, kMax)};

    std::initializer_list<bool> expected_data = {true, false, false,
                                                 true, true,  true};
    std::initializer_list<int> expected_dim = input1_dim;

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data, kMin,
        kMax, input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
        output_data);
  }
}

TF_LITE_MICRO_TEST(GreaterEqualQuantizedInt8WithBroadcast) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::initializer_list<std::initializer_list<int>> test_shapes = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    std::initializer_list<int> input1_dim = test_shapes.begin()[i];
    std::initializer_list<int> input2_dim = {1, 1};

    std::initializer_list<int8_t> input1_data = {
        F2QS(20, kMin, kMax), F2QS(-2, kMin, kMax), F2QS(-71, kMin, kMax),
        F2QS(8, kMin, kMax),  F2QS(11, kMin, kMax), F2QS(20, kMin, kMax)};
    std::initializer_list<int8_t> input2_data = {F2QS(8, kMin, kMax)};

    std::initializer_list<bool> expected_data = {true, false, false,
                                                 true, true,  true};
    std::initializer_list<int> expected_dim = input1_dim;

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::BuiltinOperator_GREATER_EQUAL, input1_dim, input1_data, kMin,
        kMax, input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
        output_data);
  }
}

TF_LITE_MICRO_TEST(LessQuantizedUInt8WithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<std::initializer_list<int>> test_shapes = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    std::initializer_list<int> input1_dim = test_shapes.begin()[i];
    std::initializer_list<int> input2_dim = {1, 1};

    std::initializer_list<uint8_t> input1_data = {
        F2Q(20, kMin, kMax), F2Q(2, kMin, kMax),  F2Q(7, kMin, kMax),
        F2Q(8, kMin, kMax),  F2Q(11, kMin, kMax), F2Q(20, kMin, kMax)};
    std::initializer_list<uint8_t> input2_data = {F2Q(8, kMin, kMax)};

    std::initializer_list<bool> expected_data = {false, true,  true,
                                                 false, false, false};
    std::initializer_list<int> expected_dim = input1_dim;

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_LESS, input1_dim, input1_data, kMin, kMax,
        input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
        output_data);
  }
}

TF_LITE_MICRO_TEST(LessQuantizedInt8WithBroadcast) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::initializer_list<std::initializer_list<int>> test_shapes = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    std::initializer_list<int> input1_dim = test_shapes.begin()[i];
    std::initializer_list<int> input2_dim = {1, 1};

    std::initializer_list<int8_t> input1_data = {
        F2QS(20, kMin, kMax), F2QS(-2, kMin, kMax), F2QS(-71, kMin, kMax),
        F2QS(8, kMin, kMax),  F2QS(11, kMin, kMax), F2QS(20, kMin, kMax)};
    std::initializer_list<int8_t> input2_data = {F2QS(8, kMin, kMax)};

    std::initializer_list<bool> expected_data = {false, true,  true,
                                                 false, false, false};
    std::initializer_list<int> expected_dim = input1_dim;

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::BuiltinOperator_LESS, input1_dim, input1_data, kMin, kMax,
        input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
        output_data);
  }
}

TF_LITE_MICRO_TEST(LessEqualQuantizedUInt8WithBroadcast) {
  const float kMin = -1.f;
  const float kMax = 128.f;
  std::initializer_list<std::initializer_list<int>> test_shapes = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    std::initializer_list<int> input1_dim = test_shapes.begin()[i];
    std::initializer_list<int> input2_dim = {1, 1};

    std::initializer_list<uint8_t> input1_data = {
        F2Q(20, kMin, kMax), F2Q(2, kMin, kMax),  F2Q(7, kMin, kMax),
        F2Q(8, kMin, kMax),  F2Q(11, kMin, kMax), F2Q(20, kMin, kMax)};
    std::initializer_list<uint8_t> input2_data = {F2Q(8, kMin, kMax)};

    std::initializer_list<bool> expected_data = {false, true,  true,
                                                 true,  false, false};
    std::initializer_list<int> expected_dim = input1_dim;

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedUInt8(
        tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data, kMin, kMax,
        input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
        output_data);
  }
}

TF_LITE_MICRO_TEST(LessEqualQuantizedInt8WithBroadcast) {
  const float kMin = -127.f;
  const float kMax = 127.f;
  std::initializer_list<std::initializer_list<int>> test_shapes = {
      {1, 6}, {2, 2, 3}, {3, 2, 1, 3}, {4, 1, 3, 1, 2}};
  for (int i = 0; i < test_shapes.size(); ++i) {
    std::initializer_list<int> input1_dim = test_shapes.begin()[i];
    std::initializer_list<int> input2_dim = {1, 1};

    std::initializer_list<int8_t> input1_data = {
        F2QS(20, kMin, kMax), F2QS(-2, kMin, kMax), F2QS(-71, kMin, kMax),
        F2QS(8, kMin, kMax),  F2QS(11, kMin, kMax), F2QS(20, kMin, kMax)};
    std::initializer_list<int8_t> input2_data = {F2QS(8, kMin, kMax)};

    std::initializer_list<bool> expected_data = {false, true,  true,
                                                 true,  false, false};
    std::initializer_list<int> expected_dim = input1_dim;

    bool output_data[6];
    tflite::testing::TestComparisonQuantizedInt8(
        tflite::BuiltinOperator_LESS_EQUAL, input1_dim, input1_data, kMin, kMax,
        input2_dim, input2_data, kMin, kMax, expected_data, expected_dim,
        output_data);
  }
}

TF_LITE_MICRO_TESTS_END
