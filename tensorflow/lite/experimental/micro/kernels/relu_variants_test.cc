/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/simple_tensor_allocator.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

void TestReluFloat(BuiltinOperator op,
                   std::initializer_list<int> input_dims_data,
                   std::initializer_list<float> input_data,
                   std::initializer_list<float> expected_output_data,
                   std::initializer_list<int> output_dims_data,
                   float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration = nullptr;
  if (op == tflite::BuiltinOperator_RELU) {
    registration = resolver.FindOp(tflite::BuiltinOperator_RELU, 1);
  } else if (op == tflite::BuiltinOperator_RELU6) {
    registration = resolver.FindOp(tflite::BuiltinOperator_RELU6, 1);
  } else if (op == BuiltinOperator_ELU) {
    registration = resolver.FindOp(tflite::BuiltinOperator_ELU, 1);
  } else if (op == BuiltinOperator_LEAKY_RELU) {
    registration = resolver.FindOp(tflite::BuiltinOperator_LEAKY_RELU, 1);
  } else if (op == BuiltinOperator_RELU_N1_TO_1, 1) {
    registration = resolver.FindOp(tflite::BuiltinOperator_RELU_N1_TO_1, 1);
  }
  TfLiteLeakyReluParams builtin_data = {0.5f};
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->invoke(&context, &node));
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data.begin()[i], output_data[i],
                              1e-5f);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleReluTest) {
  const int output_dims_count = 10;
  float output_data[output_dims_count];
  tflite::testing::TestReluFloat(               //
      tflite::BuiltinOperator_RELU, {2, 2, 5},  // Input shape.
      {
          1.0, 2.0, 3.0, 8.0, 3.0,       // b = 0
          -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 0
      },
      {
          // Expected results.
          1.0,
          2.0,
          3.0,
          8.0,
          3.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
      },
      {2, 2, 5},  // Output shape.
      output_data);
}

TF_LITE_MICRO_TEST(SimpleRelu6Test) {
  const int output_dims_count = 10;
  float output_data[output_dims_count];
  tflite::testing::TestReluFloat(                //
      tflite::BuiltinOperator_RELU6, {2, 2, 5},  // Input shape.
      {
          1.0, 2.0, 4.0, 8.0, 9.0,       // b = 0
          -1.0, -2.0, -3.0, -4.0, -5.0,  // b = 0
      },
      {
          // Expected results.
          1.0,
          2.0,
          4.0,
          6.0,
          6.0,
          0.0,
          0.0,
          0.0,
          0.0,
          0.0,
      },
      {2, 2, 5},  // Output shape.
      output_data);
}
TF_LITE_MICRO_TEST(SimpleRelu1Test) {
  const int output_dims_count = 10;
  float output_data[output_dims_count];
  tflite::testing::TestReluFloat(                       //
      tflite::BuiltinOperator_RELU_N1_TO_1, {2, 2, 5},  // Input shape.
      {
          0.0, -0.6, 0.2, -0.4, 0.0,  // b = 0
          0.3, -2.0, 1.1, -0.1, 0.0,  // b = 0
      },
      {
          // Expected results.
          0.0,
          -0.6,
          0.2,
          -0.4,
          0.0,
          0.3,
          -1.0,
          1.0,
          -0.1,
          0.0,
      },
      {2, 2, 5},  // Output shape.
      output_data);
}
TF_LITE_MICRO_TEST(SimpleEluTest) {
  const int output_dims_count = 10;
  float output_data[output_dims_count];
  tflite::testing::TestReluFloat(              //
      tflite::BuiltinOperator_ELU, {2, 2, 5},  // Input shape.
      {
          0.0, 2.0, -6.0, 3.0, -4.0,  // b = 0
          3.0, -2.0, 5.0, 6.0, 1.0,   // b = 0
      },
      {
          // Expected results.
          0.0,
          2.0,
          -0.997521,
          3.0,
          -0.981684,
          3.0,
          -0.864665,
          5.0,
          6.0,
          1.0,
      },
      {2, 2, 5},  // Output shape.
      output_data);
}

TF_LITE_MICRO_TEST(SimpleLeakyReluTest) {
  const int output_dims_count = 10;
  float output_data[output_dims_count];
  tflite::testing::TestReluFloat(                     //
      tflite::BuiltinOperator_LEAKY_RELU, {2, 2, 5},  // Input shape.
      {
          0.0, 1.0, 3.0, 0.0, -2.0,   // b = 0
          1.0, -1.0, -2.0, 3.0, 0.0,  // b = 0
      },
      {
          // Expected results.
          0.0,
          1.0,
          3.0,
          0.0,
          -1.0,
          1.0,
          -0.5,
          -1.0,
          3.0,
          0.0,
      },
      {2, 2, 5},  // Output shape.
      output_data);
}
TF_LITE_MICRO_TESTS_END
