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
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestFloor(const int* input_dims_data, const float* input_data,
               const float* expected_output_data, const int* output_dims_data,
               float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 1;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_FLOOR);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int intermediates_array_data[] = {0};
  TfLiteIntArray* temporaries_array =
      IntArrayFromInts(intermediates_array_data);
  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = nullptr;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i], 1e-5f);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloorOpSingleDimFloat32) {
  const int dims[] = {1, 2};
  const float input[] = {8.5f, 0.0f};
  const float golden[] = {8, 0};
  float output_data[2];
  tflite::testing::TestFloor(dims, input, golden, dims, output_data);
}

TF_LITE_MICRO_TEST(FloorOpMultiDimFloat32) {
  const int dims[] = {4, 2, 1, 1, 5};
  const float input[] = {0.0001f,  8.0001f,  0.9999f,  9.9999f,  0.5f,
                         -0.0001f, -8.0001f, -0.9999f, -9.9999f, -0.5f};
  const float golden[] = {0.0f,  8.0f,  0.0f,  9.0f,   0.0f,
                          -1.0f, -9.0f, -1.0f, -10.0f, -1.0f};
  float output_data[10];
  tflite::testing::TestFloor(dims, input, golden, dims, output_data);
}

TF_LITE_MICRO_TESTS_END
