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
#include <string>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

template <typename Scalar, int NumInputs>
void TestConcatenationQuantized(
    std::initializer_list<std::initializer_list<int>> input_dims_data,
    std::initializer_list<std::initializer_list<Scalar>> input_data, int axis,
    std::initializer_list<int> output_dims_data,
    std::initializer_list<Scalar> expected_output_data, Scalar* output_data) {
  TF_LITE_MICRO_EXPECT(input_dims_data.size() == input_data.size());
  TF_LITE_MICRO_EXPECT(NumInputs == input_data.size());

  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = NumInputs;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;

  TfLiteTensor tensors[tensors_size];

  typename std::initializer_list<std::initializer_list<int>>::iterator it_dims =
      input_dims_data.begin();
  typename std::initializer_list<std::initializer_list<Scalar>>::iterator
      it_data = input_data.begin();

  for (int i = 0; i < inputs_size; ++i) {
    TfLiteIntArray* input_dims = IntArrayFromInitializer(*it_dims++);
    tensors[i] = CreateQuantizedTensor(
        *it_data++, input_dims,
        std::string("input_tensor" + std::to_string(i)).c_str(), -127, 128);
  }

  tensors[tensors_size - 1] = CreateQuantizedTensor(
      output_data, output_dims, "output_tensor", 0, 15.9375);

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_CONCATENATION, 1);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  TfLiteConcatenationParams builtin_data = {axis, kTfLiteActNone};

  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  constexpr int inputs_array_data_sz = NumInputs + 1;
  int inputs_array_data[inputs_array_data_sz];

  inputs_array_data[0] = NumInputs;
  for (int i = 1; i < inputs_array_data_sz; ++i) inputs_array_data[i] = i - 1;

  int outputs_array_data[] = {1, NumInputs};
  int temporaries_array_data[] = {0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
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
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data.begin()[i], output_data[i],
                              1e-5f);
  }
}
}  // namespace

}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(ConcatTestOneInputFourDimensionalUInt8) {
  uint8_t output_data[6];
  tflite::testing::TestConcatenationQuantized<uint8_t, 1>(
      {{4, 1, 2, 1, 3}},     // Input shapes
      {{1, 3, 0, 6, 4, 2}},  // Input values
      0,                     // axis
      {4, 1, 2, 1, 3},       // Output shape
      {1, 3, 0, 6, 4, 2},    // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsFourDimensionalAxes0UInt8) {
  uint8_t output_data[18];
  tflite::testing::TestConcatenationQuantized<uint8_t, 2>(
      {{4, 1, 2, 1, 3}, {4, 2, 2, 1, 3}},  // Input shapes
      {{15, 16, 17, 0, 1, 2},
       {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}},  // Input values
      0,                                            // axis
      {4, 3, 2, 1, 3},                              // Output shape
      {15, 16, 17, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
       14},  // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsFourDimensionalAxes1UInt8) {
  uint8_t output_data[9];
  tflite::testing::TestConcatenationQuantized<uint8_t, 2>(
      {{4, 1, 2, 1, 3}, {4, 1, 1, 1, 3}},  // Input shapes
      {{6, 7, 8, 0, 1, 2}, {3, 4, 5}},     // Input values
      1,                                   // axis
      {4, 1, 3, 1, 3},                     // Output shape
      {6, 7, 8, 0, 1, 2, 3, 4, 5},         // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsFourDimensionalAxes2UInt8) {
  uint8_t output_data[18];
  tflite::testing::TestConcatenationQuantized<uint8_t, 2>(
      {{4, 1, 2, 1, 3}, {4, 1, 2, 2, 3}},  // Input shapes
      {{15, 16, 17, 0, 1, 2},
       {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}},  // Input values
      2,                                            // axis
      {4, 1, 2, 3, 3},                              // Output shape
      {15, 16, 17, 3, 4, 5, 6, 7, 8, 0, 1, 2, 9, 10, 11, 12, 13,
       14},  // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsFourDimensionalAxes3UInt8) {
  uint8_t output_data[8];
  tflite::testing::TestConcatenationQuantized<uint8_t, 2>(
      {{4, 1, 2, 1, 3}, {4, 1, 2, 1, 1}},  // Input shapes
      {{5, 6, 7, 0, 1, 2}, {3, 4}},        // Input values
      3,                                   // axis
      {4, 1, 2, 1, 4},                     // Output shape
      {5, 6, 7, 3, 0, 1, 2, 4},            // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsFourDimensionalAxesNegativeUInt8) {
  uint8_t output_data[8];
  tflite::testing::TestConcatenationQuantized<uint8_t, 2>(
      {{4, 1, 2, 1, 3}, {4, 1, 2, 1, 1}},  // Input shapes
      {{5, 6, 7, 0, 1, 2}, {3, 4}},        // Input values
      -1,                                  // axis
      {4, 1, 2, 1, 4},                     // Output shape
      {5, 6, 7, 3, 0, 1, 2, 4},            // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestOneInputThreeDimensionalUInt8) {
  uint8_t output_data[4];
  tflite::testing::TestConcatenationQuantized<uint8_t, 1>(
      {{3, 1, 2, 2}},  // Input shapes
      {{1, 3, 0, 6}},  // Input values
      0,               // axis
      {3, 1, 2, 2},    // Output shape
      {1, 3, 0, 6},    // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsThreeDimensionalAxes0UInt8) {
  uint8_t output_data[12];
  tflite::testing::TestConcatenationQuantized<uint8_t, 2>(
      {{3, 1, 2, 2}, {3, 2, 2, 2}},                // Input shapes
      {{9, 10, 11, 0}, {1, 2, 3, 4, 5, 6, 7, 8}},  // Input values
      0,                                           // axis
      {3, 3, 2, 2},                                // Output shape
      {9, 10, 11, 0, 1, 2, 3, 4, 5, 6, 7, 8},      // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsThreeDimensionalAxes1UInt8) {
  uint8_t output_data[3];
  tflite::testing::TestConcatenationQuantized<uint8_t, 2>(
      {{3, 1, 2, 1}, {3, 1, 1, 1}},  // Input shapes
      {{0, 1}, {2}},                 // Input values
      1,                             // axis
      {3, 1, 3, 1},                  // Output shape
      {0, 1, 2},                     // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsThreeDimensionalAxes2UInt8) {
  uint8_t output_data[6];
  tflite::testing::TestConcatenationQuantized<uint8_t, 2>(
      {{3, 1, 2, 1}, {3, 1, 2, 2}},  // Input shapes
      {{4, 5}, {6, 0, 1, 2}},        // Input values
      2,                             // axis
      {3, 1, 2, 3},                  // Output shape
      {4, 6, 0, 5, 1, 2},            // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestThreeInputsThreeDimensionalAxes2UInt8) {
  uint8_t output_data[12];
  tflite::testing::TestConcatenationQuantized<uint8_t, 3>(
      {{3, 1, 2, 2}, {3, 1, 2, 1}, {3, 1, 2, 3}},  // Input shapes
      {
          {9, 10, 11, 0},
          {1, 2},
          {3, 4, 5, 6, 7, 8},
      },                                       // Input values
      2,                                       // axis
      {3, 1, 2, 6},                            // Output shape
      {9, 10, 1, 3, 4, 5, 11, 0, 2, 6, 7, 8},  // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestOneInputFourDimensionalInt8) {
  int8_t output_data[6];
  tflite::testing::TestConcatenationQuantized<int8_t, 1>(
      {{4, 1, 2, 1, 3}},       // Input shapes
      {{1, 3, 0, -6, -2, 2}},  // Input values
      0,                       // axis
      {4, 1, 2, 1, 3},         // Output shape
      {1, 3, 0, -6, -2, 2},    // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsFourDimensionalAxes0Int8) {
  int8_t output_data[18];
  tflite::testing::TestConcatenationQuantized<int8_t, 2>(
      {{4, 1, 2, 1, 3}, {4, 2, 2, 1, 3}},  // Input shapes
      {{-3, -2, -1, 0, 1, 2},
       {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}},  // Input values
      0,                                            // axis
      {4, 3, 2, 1, 3},                              // Output shape
      {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13,
       14},  // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsFourDimensionalAxes1Int8) {
  int8_t output_data[9];
  tflite::testing::TestConcatenationQuantized<int8_t, 2>(
      {{4, 1, 2, 1, 3}, {4, 1, 1, 1, 3}},  // Input shapes
      {{-3, -2, -1, 0, 1, 2}, {3, 4, 5}},  // Input values
      1,                                   // axis
      {4, 1, 3, 1, 3},                     // Output shape
      {-3, -2, -1, 0, 1, 2, 3, 4, 5},      // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsFourDimensionalAxes2Int8) {
  int8_t output_data[18];
  tflite::testing::TestConcatenationQuantized<int8_t, 2>(
      {{4, 1, 2, 1, 3}, {4, 1, 2, 2, 3}},  // Input shapes
      {{-3, -2, -1, 0, 1, 2},
       {3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14}},  // Input values
      2,                                            // axis
      {4, 1, 2, 3, 3},                              // Output shape
      {-3, -2, -1, 3, 4, 5, 6, 7, 8, 0, 1, 2, 9, 10, 11, 12, 13,
       14},  // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsFourDimensionalAxes3Int8) {
  int8_t output_data[8];
  tflite::testing::TestConcatenationQuantized<int8_t, 2>(
      {{4, 1, 2, 1, 3}, {4, 1, 2, 1, 1}},  // Input shapes
      {{-3, -2, -1, 0, 1, 2}, {3, 4}},     // Input values
      3,                                   // axis
      {4, 1, 2, 1, 4},                     // Output shape
      {-3, -2, -1, 3, 0, 1, 2, 4},         // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsFourDimensionalAxesNegativeInt8) {
  int8_t output_data[8];
  tflite::testing::TestConcatenationQuantized<int8_t, 2>(
      {{4, 1, 2, 1, 3}, {4, 1, 2, 1, 1}},  // Input shapes
      {{-3, -2, -1, 0, 1, 2}, {3, 4}},     // Input values
      -1,                                  // axis
      {4, 1, 2, 1, 4},                     // Output shape
      {-3, -2, -1, 3, 0, 1, 2, 4},         // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestOneInputThreeDimensionalInt8) {
  int8_t output_data[4];
  tflite::testing::TestConcatenationQuantized<int8_t, 1>(
      {{3, 1, 2, 2}},   // Input shapes
      {{1, 3, 0, -6}},  // Input values
      0,                // axis
      {3, 1, 2, 2},     // Output shape
      {1, 3, 0, -6},    // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsThreeDimensionalAxes0Int8) {
  int8_t output_data[12];
  tflite::testing::TestConcatenationQuantized<int8_t, 2>(
      {{3, 1, 2, 2}, {3, 2, 2, 2}},                 // Input shapes
      {{-3, -2, -1, 0}, {1, 2, 3, 4, 5, 6, 7, 8}},  // Input values
      0,                                            // axis
      {3, 3, 2, 2},                                 // Output shape
      {-3, -2, -1, 0, 1, 2, 3, 4, 5, 6, 7, 8},      // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsThreeDimensionalAxes1Int8) {
  int8_t output_data[3];
  tflite::testing::TestConcatenationQuantized<int8_t, 2>(
      {{3, 1, 2, 1}, {3, 1, 1, 1}},  // Input shapes
      {{-3, -2}, {-1}},              // Input values
      1,                             // axis
      {3, 1, 3, 1},                  // Output shape
      {-3, -2, -1},                  // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestTwoInputsThreeDimensionalAxes2Int8) {
  int8_t output_data[6];
  tflite::testing::TestConcatenationQuantized<int8_t, 2>(
      {{3, 1, 2, 1}, {3, 1, 2, 2}},  // Input shapes
      {{-3, -2}, {-1, 0, 1, 2}},     // Input values
      2,                             // axis
      {3, 1, 2, 3},                  // Output shape
      {-3, -1, 0, -2, 1, 2},         // Output values
      output_data);
}

TF_LITE_MICRO_TEST(ConcatTestThreeInputsThreeDimensionalAxes2Int8) {
  int8_t output_data[12];
  tflite::testing::TestConcatenationQuantized<int8_t, 3>(
      {{3, 1, 2, 2}, {3, 1, 2, 1}, {3, 1, 2, 3}},  // Input shapes
      {
          {-3, -2, -1, 0},
          {1, 2},
          {3, 4, 5, 6, 7, 8},
      },                                        // Input values
      2,                                        // axis
      {3, 1, 2, 6},                             // Output shape
      {-3, -2, 1, 3, 4, 5, -1, 0, 2, 6, 7, 8},  // Output values
      output_data);
}

TF_LITE_MICRO_TESTS_END
