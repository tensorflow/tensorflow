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

#include "tensorflow/lite/kernels/internal/reference/transpose.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/portable_tensor.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void RunTestPermutation(int num_dims, const int32_t* shape,
                        const int32_t* perms, T* input, T* input_transposed) {
  // Count elements and allocate output.
  int count = 1;
  for (int i = 0; i < num_dims; i++) {
    count *= shape[i];
  }

  // Create the dummy data
  for (int i = 0; i < count; i++) {
    input[i] = i;
  }

  // Make input and output shapes.
  const RuntimeShape input_shape = RuntimeShape(num_dims, shape);
  RuntimeShape output_shape(num_dims);

  for (int i = 0; i < num_dims; i++) {
    output_shape.SetDim(i, shape[perms[i]]);
  }

  TransposeParams params;
  params.perm_count = num_dims;
  for (int i = 0; i < num_dims; ++i) {
    params.perm[i] = perms[i];
  }

  reference_ops::Transpose<T>(params, input_shape, input, output_shape,
                              input_transposed);
}

template <typename T>
TfLiteStatus InvokeTranspose(TfLiteTensor* tensors, int tensors_size,
                             T* output_data, int output_length,
                             TransposeParams* params) {
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_TRANSPOSE();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, reinterpret_cast<void*>(params));

  const char* init_data = reinterpret_cast<const char*>(params);
  TfLiteStatus status = runner.InitAndPrepare(init_data);
  if (status != kTfLiteOk) {
    return status;
  }
  return runner.Invoke();
}

template <typename T>
TfLiteStatus ValidateTranspose(TfLiteTensor* tensors, int tensors_size,
                               const T* expected_output_data, T* output_data,
                               int output_length,
                               tflite::TransposeParams* params,
                               float tolerance = 1e-5) {
  TfLiteStatus status = InvokeTranspose(tensors, tensors_size, output_data,
                                        output_length, params);
  if (status != kTfLiteOk) {
    return status;
  }

  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
  return kTfLiteOk;
}

template <typename T>
void TestTranspose(int* input_dims_data, T* input_data, int* output_dims_data,
                   const T* expected_output_data, T* output_data,
                   TransposeParams* params) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int input_size = ElementCount(*input_dims);
  for (int i = 0; i < input_size; i++) {
    input_data[i] = i;
  }

  for (int i = 0; i < input_dims->size; i++) {
    output_dims->data[i] = input_dims->data[params->perm[i]];
  }

  int perm_dims_data[] = {1, params->perm_count};
  TfLiteIntArray* perm_dims = IntArrayFromInts(perm_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(params->perm, perm_dims),
      CreateTensor(output_data, output_dims),
  };

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, ValidateTranspose(tensors, tensors_size, expected_output_data,
                                   output_data, output_dims_count, params));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(1D) {
  int input_dims_data[] = {1, 3};
  int output_dims_data[] = {1, 3};

  int8_t input_data[3];
  int8_t output_data[3];
  const int8_t expected_output_data[] = {0, 1, 2};

  tflite::TransposeParams params = {1, {0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(2DPerm1) {
  int input_dims_data[] = {2, 3, 2};
  int output_dims_data[] = {2, 3, 2};

  int8_t input_data[6];
  int8_t output_data[6];
  const int8_t expected_output_data[] = {0, 2, 4, 1, 3, 5};

  tflite::TransposeParams params = {2, {1, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(2D4x4KernelLeftOverRightSide) {
  int input_dims_data[] = {2, 4, 6};
  int output_dims_data[] = {2, 4, 6};

  int8_t input_data[24];
  int8_t output_data[24];
  const int8_t expected_output_data[] = {0, 6,  12, 18, 1, 7,  13, 19,
                                         2, 8,  14, 20, 3, 9,  15, 21,
                                         4, 10, 16, 22, 5, 11, 17, 23};

  tflite::TransposeParams params = {2, {1, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(2D4x4KernelLeftOverBottomSide) {
  int input_dims_data[] = {2, 6, 4};
  int output_dims_data[] = {2, 4, 6};

  int8_t input_data[24];
  int8_t output_data[24];
  const int8_t expected_output_data[] = {0,  4,  8,  12, 16, 20, 1,  5,
                                         9,  13, 17, 21, 2,  6,  10, 14,
                                         18, 22, 3,  7,  11, 15, 19, 23};

  tflite::TransposeParams params = {2, {1, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(3D) {
  int input_dims_data[] = {3, 2, 3, 4};
  int output_dims_data[] = {3, 2, 3, 4};

  int8_t input_data[24];
  int8_t output_data[24];
  const int8_t expected_output_data[] = {0,  4,  8,  12, 16, 20, 1,  5,
                                         9,  13, 17, 21, 2,  6,  10, 14,
                                         18, 22, 3,  7,  11, 15, 19, 23};

  tflite::TransposeParams params = {3, {2, 0, 1}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(1DNotShrinked) {
  int input_dims_data[] = {1, 1};
  int output_dims_data[] = {1, 1};

  float input_data[1];
  float output_data[1];
  const float expected_output_data[] = {0};

  tflite::TransposeParams params = {1, {0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(2DShrinkedOneTime) {
  int input_dims_data[] = {2, 2, 1};
  int output_dims_data[] = {2, 2, 1};

  float input_data[2];
  float output_data[2];
  const float expected_output_data[] = {0, 1};

  tflite::TransposeParams params = {2, {1, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(2DShrinkedTwoTimes) {
  int input_dims_data[] = {2, 1, 1};
  int output_dims_data[] = {2, 1, 1};

  float input_data[1];
  float output_data[1];
  const float expected_output_data[] = {0};

  tflite::TransposeParams params = {2, {1, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(3DShrinkedOneTime) {
  int input_dims_data[] = {3, 2, 1, 3};
  int output_dims_data[] = {3, 2, 1, 3};

  float input_data[6];
  float output_data[6];
  const float expected_output_data[] = {0, 1, 2, 3, 4, 5};

  tflite::TransposeParams params = {3, {0, 2, 1}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(3DShrinkedTwoTimes) {
  int input_dims_data[] = {3, 1, 1, 3};
  int output_dims_data[] = {3, 1, 1, 3};

  float input_data[3];
  float output_data[3];
  const float expected_output_data[] = {0, 1, 2};

  tflite::TransposeParams params = {3, {1, 2, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(3DShrinkedAll) {
  int input_dims_data[] = {3, 1, 1, 1};
  int output_dims_data[] = {3, 1, 1, 1};

  float input_data[1];
  float output_data[1];
  const float expected_output_data[] = {0};

  tflite::TransposeParams params = {3, {1, 2, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(4DShrinkedOneTimes) {
  int input_dims_data[] = {4, 2, 2, 3, 1};
  int output_dims_data[] = {4, 2, 2, 3, 1};

  float input_data[12];
  float output_data[12];
  const float expected_output_data[] = {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11};

  tflite::TransposeParams params = {4, {3, 0, 1, 2}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(4DShrinkedTwoTimes) {
  int input_dims_data[] = {4, 2, 1, 3, 1};
  int output_dims_data[] = {4, 2, 1, 3, 1};

  float input_data[6];
  float output_data[6];
  const float expected_output_data[] = {0, 1, 2, 3, 4, 5};

  tflite::TransposeParams params = {4, {0, 3, 1, 2}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(4DShrinkedThreeTimes) {
  int input_dims_data[] = {4, 2, 1, 1, 1};
  int output_dims_data[] = {4, 2, 1, 1, 1};

  float input_data[2];
  float output_data[2];
  const float expected_output_data[] = {0, 1};

  tflite::TransposeParams params = {4, {3, 2, 1, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(4DShrinkedFourTimes) {
  int input_dims_data[] = {4, 1, 1, 1, 1};
  int output_dims_data[] = {4, 1, 1, 1, 1};

  float input_data[1];
  float output_data[1];
  const float expected_output_data[] = {0};

  tflite::TransposeParams params = {4, {2, 3, 1, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(3DFlatten) {
  int input_dims_data[] = {3, 2, 2, 3};
  int output_dims_data[] = {3, 2, 2, 3};

  float input_data[12];
  float output_data[12];
  const float expected_output_data[] = {0, 3, 1, 4, 2, 5, 6, 9, 7, 10, 8, 11};

  tflite::TransposeParams params = {3, {0, 2, 1}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(4DFlatten) {
  int input_dims_data[] = {4, 2, 2, 2, 2};
  int output_dims_data[] = {4, 2, 2, 2, 2};

  float input_data[16];
  float output_data[16];
  const float expected_output_data[] = {0, 2,  1, 3,  4,  6,  5,  7,
                                        8, 10, 9, 11, 12, 14, 13, 15};

  tflite::TransposeParams params = {4, {0, 1, 3, 2}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(4DFlattenTwo) {
  int input_dims_data[] = {4, 2, 2, 2, 2};
  int output_dims_data[] = {4, 2, 2, 2, 2};

  float input_data[16];
  float output_data[16];
  const float expected_output_data[] = {0, 4,  1, 5,  2,  6,  3,  7,
                                        8, 12, 9, 13, 10, 14, 11, 15};

  tflite::TransposeParams params = {4, {0, 2, 3, 1}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(3DDividedIntoTwo2DsOne) {
  float input_data[24];
  float expected_output_data[24];
  int32_t shape[] = {2, 3, 4};
  int32_t perms[] = {1, 2, 0};
  tflite::testing::RunTestPermutation(3, shape, perms, input_data,
                                      expected_output_data);
  int input_dims_data[] = {3, 2, 3, 4};
  int output_dims_data[] = {3, 2, 3, 4};

  float output_data[24];

  tflite::TransposeParams params = {3, {1, 2, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(3DDividedIntoTwo2DsTwo) {
  float input_data[24];
  float expected_output_data[24];
  int32_t shape[] = {2, 3, 4};
  int32_t perms[] = {2, 0, 1};
  tflite::testing::RunTestPermutation(3, shape, perms, input_data,
                                      expected_output_data);
  int input_dims_data[] = {3, 2, 3, 4};
  int output_dims_data[] = {3, 2, 3, 4};

  float output_data[24];

  tflite::TransposeParams params = {3, {2, 0, 1}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(4DDividedIntoTwo2DsOne) {
  int32_t shape[] = {2, 3, 4, 2};
  int32_t perms[] = {1, 2, 3, 0};
  float input_data[48];
  float expected_output_data[48];
  tflite::testing::RunTestPermutation(4, shape, perms, input_data,
                                      expected_output_data);
  int input_dims_data[] = {4, 2, 3, 4, 2};
  int output_dims_data[] = {4, 2, 3, 4, 2};

  float output_data[48];

  tflite::TransposeParams params = {4, {1, 2, 3, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}
TF_LITE_MICRO_TEST(4DDividedIntoTwo2DsTwo) {
  int32_t shape[] = {2, 3, 4, 2};
  int32_t perms[] = {2, 3, 0, 1};
  float input_data[48];
  float expected_output_data[48];
  tflite::testing::RunTestPermutation(4, shape, perms, input_data,
                                      expected_output_data);
  int input_dims_data[] = {4, 2, 3, 4, 2};
  int output_dims_data[] = {4, 2, 3, 4, 2};

  float output_data[48];

  tflite::TransposeParams params = {4, {2, 3, 0, 1}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(4DDividedIntoTwo2DsThree) {
  int32_t shape[] = {2, 3, 4, 2};
  int32_t perms[] = {3, 0, 1, 2};
  float input_data[48];
  float expected_output_data[48];
  tflite::testing::RunTestPermutation(4, shape, perms, input_data,
                                      expected_output_data);
  int input_dims_data[] = {4, 2, 3, 4, 2};
  int output_dims_data[] = {4, 2, 3, 4, 2};

  float output_data[48];

  tflite::TransposeParams params = {4, {3, 0, 1, 2}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(5DDividedIntoTwo2DsOne) {
  int32_t shape[] = {2, 3, 2, 2, 2};
  int32_t perms[] = {1, 4, 2, 3, 0};
  float input_data[48];
  float expected_output_data[48];
  tflite::testing::RunTestPermutation(5, shape, perms, input_data,
                                      expected_output_data);
  int input_dims_data[] = {5, 2, 3, 2, 2, 2};
  int output_dims_data[] = {5, 2, 3, 2, 2, 2};

  float output_data[48];

  tflite::TransposeParams params = {5, {1, 4, 2, 3, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(5DDividedIntoTwo2DsTwo) {
  int32_t shape[] = {2, 3, 2, 2, 2};
  int32_t perms[] = {2, 3, 0, 4, 1};
  float input_data[48];
  float expected_output_data[48];
  tflite::testing::RunTestPermutation(5, shape, perms, input_data,
                                      expected_output_data);
  int input_dims_data[] = {5, 2, 3, 2, 2, 2};
  int output_dims_data[] = {5, 2, 3, 2, 2, 2};

  float output_data[48];

  tflite::TransposeParams params = {5, {2, 3, 0, 4, 1}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(5DDividedIntoTwo2DsThree) {
  int32_t shape[] = {2, 3, 2, 2, 2};
  int32_t perms[] = {3, 0, 4, 1, 2};
  float input_data[48];
  float expected_output_data[48];
  tflite::testing::RunTestPermutation(5, shape, perms, input_data,
                                      expected_output_data);
  int input_dims_data[] = {5, 2, 3, 2, 2, 2};
  int output_dims_data[] = {5, 2, 3, 2, 2, 2};

  float output_data[48];

  tflite::TransposeParams params = {5, {3, 0, 4, 1, 2}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(SimpleTestNoReorder) {
  int input_dims_data[] = {4, 1, 2, 3, 1};
  int output_dims_data[] = {4, 1, 2, 3, 1};

  float input_data[6];
  float output_data[6];
  const float expected_output_data[] = {0, 1, 2, 3, 4, 5};

  tflite::TransposeParams params = {4, {0, 1, 2, 3}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(SimpleTestWithReorder) {
  int input_dims_data[] = {4, 1, 2, 3, 1};
  int output_dims_data[] = {4, 1, 2, 3, 1};

  float input_data[6];
  float output_data[6];
  const float expected_output_data[] = {0, 3, 1, 4, 2, 5};

  tflite::TransposeParams params = {4, {2, 1, 3, 0}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(ComplexTestWithReorder) {
  int input_dims_data[] = {4, 2, 3, 4, 5};
  int output_dims_data[] = {4, 2, 3, 4, 5};

  float input_data[120];
  float output_data[120];
  const float expected_output_data[] = {
      0,  1,  2,  3,  4,  20, 21, 22, 23, 24, 40,  41,  42,  43,  44,
      60, 61, 62, 63, 64, 80, 81, 82, 83, 84, 100, 101, 102, 103, 104,
      5,  6,  7,  8,  9,  25, 26, 27, 28, 29, 45,  46,  47,  48,  49,
      65, 66, 67, 68, 69, 85, 86, 87, 88, 89, 105, 106, 107, 108, 109,
      10, 11, 12, 13, 14, 30, 31, 32, 33, 34, 50,  51,  52,  53,  54,
      70, 71, 72, 73, 74, 90, 91, 92, 93, 94, 110, 111, 112, 113, 114,
      15, 16, 17, 18, 19, 35, 36, 37, 38, 39, 55,  56,  57,  58,  59,
      75, 76, 77, 78, 79, 95, 96, 97, 98, 99, 115, 116, 117, 118, 119};

  tflite::TransposeParams params = {4, {2, 0, 1, 3}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TEST(Complex5DTestWithReorder) {
  int input_dims_data[] = {5, 2, 3, 2, 2, 5};
  int output_dims_data[] = {5, 2, 3, 2, 2, 5};

  float input_data[120];
  float output_data[120];
  const float expected_output_data[] = {
      0,  5,  1,  6,  2,  7,   3,   8,   4,   9,   20,  25,  21,  26,  22,
      27, 23, 28, 24, 29, 40,  45,  41,  46,  42,  47,  43,  48,  44,  49,
      60, 65, 61, 66, 62, 67,  63,  68,  64,  69,  80,  85,  81,  86,  82,
      87, 83, 88, 84, 89, 100, 105, 101, 106, 102, 107, 103, 108, 104, 109,
      10, 15, 11, 16, 12, 17,  13,  18,  14,  19,  30,  35,  31,  36,  32,
      37, 33, 38, 34, 39, 50,  55,  51,  56,  52,  57,  53,  58,  54,  59,
      70, 75, 71, 76, 72, 77,  73,  78,  74,  79,  90,  95,  91,  96,  92,
      97, 93, 98, 94, 99, 110, 115, 111, 116, 112, 117, 113, 118, 114, 119};

  tflite::TransposeParams params = {5, {2, 0, 1, 4, 3}};

  tflite::testing::TestTranspose(input_dims_data, input_data, output_dims_data,
                                 expected_output_data, output_data, &params);
}

TF_LITE_MICRO_TESTS_END
