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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

TfLiteTensor TestCreateTensor(const float* data, TfLiteIntArray* dims) {
  return CreateTensor(data, dims);
}

TfLiteTensor TestCreateTensor(const int8_t* data, TfLiteIntArray* dims) {
  return CreateQuantizedTensor(data, dims, -128, 127);
}

template <typename T>
TfLiteStatus ValidateGoldens(TfLiteTensor* tensors, int tensors_size,
                             const T* expected_output_data, T* output_data,
                             int output_length,
                             TfLiteResizeBilinearParams* params,
                             float tolerance = 1e-5) {
  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_RESIZE_BILINEAR();
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

template <typename T>
void TestResizeBilinear(int* input_dims_data, const T* input_data,
                        const int32_t* expected_size_data,
                        const T* expected_output_data, int* output_dims_data,
                        T* output_data, TfLiteResizeBilinearParams* params,
                        float tolerance = 1e-5) {
  int expected_size_dims_data[] = {1, 2};
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* expected_size_dims =
      IntArrayFromInts(expected_size_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);

  const int output_dims_count = ElementCount(*output_dims);

  // Hack to pass ConstantTensor check in prepare
  TfLiteTensor t = CreateTensor(expected_size_data, expected_size_dims);
  t.allocation_type = kTfLiteMmapRo;

  constexpr int tensors_size = 3;
  TfLiteTensor tensors[tensors_size]{
      TestCreateTensor(input_data, input_dims),
      t,
      TestCreateTensor(output_data, output_dims),
  };

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateGoldens(tensors, tensors_size, expected_output_data, output_data,
                      output_dims_count, params, tolerance));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(HorizontalResize) {
  int input_dims[] = {4, 1, 1, 2, 1};
  const float input_data[] = {3, 6};
  const int32_t expected_size_data[] = {1, 3};
  const float expected_output_data[] = {3, 5, 6};
  int output_dims[] = {4, 1, 1, 3, 1};
  float output_data[3];

  TfLiteResizeBilinearParams params = {
      false, /*align_corners*/
      false  /*half pixel centers*/
  };

  tflite::testing::TestResizeBilinear(input_dims, input_data,
                                      expected_size_data, expected_output_data,
                                      output_dims, output_data, &params);
}

TF_LITE_MICRO_TEST(HorizontalResizeInt8) {
  int input_dims[] = {4, 1, 1, 2, 1};
  const int8_t input_data[] = {3, 6};
  const int32_t expected_size_data[] = {1, 3};
  const int8_t expected_output_data[] = {3, 5, 6};
  int output_dims[] = {4, 1, 1, 3, 1};
  int8_t output_data[3];

  TfLiteResizeBilinearParams params = {
      false, /*align_corners*/
      false  /*half pixel centers*/
  };

  tflite::testing::TestResizeBilinear<int8_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data, &params);
}

TF_LITE_MICRO_TEST(VerticalResize) {
  int input_dims[] = {4, 1, 2, 1, 1};
  const float input_data[] = {3, 9};
  const int32_t expected_size_data[] = {3, 1};
  const float expected_output_data[] = {3, 7, 9};
  int output_dims[] = {4, 1, 3, 1, 1};
  float output_data[3];

  TfLiteResizeBilinearParams params = {
      false, /*align_corners*/
      false  /*half pixel centers*/
  };

  tflite::testing::TestResizeBilinear(input_dims, input_data,
                                      expected_size_data, expected_output_data,
                                      output_dims, output_data, &params);
}

TF_LITE_MICRO_TEST(VerticalResizeInt8) {
  int input_dims[] = {4, 1, 2, 1, 1};
  const int8_t input_data[] = {3, 9};
  const int32_t expected_size_data[] = {3, 1};
  const int8_t expected_output_data[] = {3, 7, 9};
  int output_dims[] = {4, 1, 3, 1, 1};
  int8_t output_data[3];

  TfLiteResizeBilinearParams params = {
      false, /*align_corners*/
      false  /*half pixel centers*/
  };

  tflite::testing::TestResizeBilinear<int8_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data, &params);
}

TF_LITE_MICRO_TEST(TwoDimensionalResize) {
  int input_dims[] = {4, 1, 2, 2, 1};
  const float input_data[] = {
      3, 6,   //
      9, 12,  //
  };
  const int32_t expected_size_data[] = {3, 3};
  const float expected_output_data[] = {
      3, 5,  6,   //
      7, 9,  10,  //
      9, 11, 12   //
  };

  int output_dims[] = {4, 1, 3, 3, 1};
  float output_data[9];

  TfLiteResizeBilinearParams params = {
      false, /*align_corners*/
      false  /*half pixel centers*/
  };

  tflite::testing::TestResizeBilinear(input_dims, input_data,
                                      expected_size_data, expected_output_data,
                                      output_dims, output_data, &params);
}

TF_LITE_MICRO_TEST(TwoDimensionalResizeInt8) {
  int input_dims[] = {4, 1, 2, 2, 1};
  const int8_t input_data[] = {
      3, 6,   //
      9, 12,  //
  };
  const int32_t expected_size_data[] = {3, 3};
  const int8_t expected_output_data[] = {
      3, 5,  6,   //
      7, 9,  10,  //
      9, 11, 12,  //
  };
  int output_dims[] = {4, 1, 3, 3, 1};
  int8_t output_data[9];

  TfLiteResizeBilinearParams params = {
      false, /*align_corners*/
      false  /*half pixel centers*/
  };

  tflite::testing::TestResizeBilinear<int8_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data, &params);
}

TF_LITE_MICRO_TEST(TwoDimensionalResizeWithTwoBatches) {
  int input_dims[] = {4, 2, 2, 2, 1};
  const float input_data[] = {
      3,  6,   //
      9,  12,  //
      4,  10,  //
      10, 16   //
  };
  const int32_t expected_size_data[] = {3, 3};
  const float expected_output_data[] = {
      3,  5,  6,   //
      7,  9,  10,  //
      9,  11, 12,  //
      4,  8,  10,  //
      8,  12, 14,  //
      10, 14, 16,  //
  };
  int output_dims[] = {4, 2, 3, 3, 1};
  float output_data[18];

  TfLiteResizeBilinearParams params = {
      false, /*align_corners*/
      false  /*half pixel centers*/
  };

  tflite::testing::TestResizeBilinear(input_dims, input_data,
                                      expected_size_data, expected_output_data,
                                      output_dims, output_data, &params);
}

TF_LITE_MICRO_TEST(TwoDimensionalResizeWithTwoBatchesInt8) {
  int input_dims[] = {4, 2, 2, 2, 1};
  const int8_t input_data[] = {
      3,  6,   //
      9,  12,  //
      4,  10,  //
      12, 16   //
  };
  const int32_t expected_size_data[] = {3, 3};
  const int8_t expected_output_data[] = {
      3,  5,  6,   //
      7,  9,  10,  //
      9,  11, 12,  //
      4,  8,  10,  //
      9,  12, 13,  //
      12, 14, 16,  //
  };
  int output_dims[] = {4, 2, 3, 3, 1};
  int8_t output_data[18];

  TfLiteResizeBilinearParams params = {
      false, /*align_corners*/
      false  /*half pixel centers*/
  };

  tflite::testing::TestResizeBilinear<int8_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data, &params, /*tolerance=*/1);
}

TF_LITE_MICRO_TEST(ThreeDimensionalResize) {
  int input_dims[] = {4, 1, 2, 2, 2};
  const float input_data[] = {
      3, 4,  6,  10,  //
      9, 10, 12, 16,  //
  };
  const int32_t expected_size_data[] = {3, 3};
  const float expected_output_data[] = {
      3, 4,  5,  8,  6,  10,  //
      7, 8,  9,  12, 10, 14,  //
      9, 10, 11, 14, 12, 16,  //
  };
  int output_dims[] = {4, 1, 3, 3, 2};
  float output_data[18];

  TfLiteResizeBilinearParams params = {
      false, /*align_corners*/
      false  /*half pixel centers*/
  };

  tflite::testing::TestResizeBilinear(input_dims, input_data,
                                      expected_size_data, expected_output_data,
                                      output_dims, output_data, &params);
}

TF_LITE_MICRO_TEST(ThreeDimensionalResizeInt8) {
  int input_dims[] = {4, 1, 2, 2, 2};
  const int8_t input_data[] = {
      3,  4,  6,  10,  //
      10, 12, 14, 16,  //
  };
  const int32_t expected_size_data[] = {3, 3};
  const int8_t expected_output_data[] = {
      3,  4,  5,  8,  6,  10,  //
      7,  9,  10, 12, 11, 13,  //
      10, 12, 12, 14, 14, 16,  //
  };
  int output_dims[] = {4, 1, 3, 3, 2};
  int8_t output_data[18];

  TfLiteResizeBilinearParams params = {
      false, /*align_corners*/
      false  /*half pixel centers*/
  };

  tflite::testing::TestResizeBilinear<int8_t>(
      input_dims, input_data, expected_size_data, expected_output_data,
      output_dims, output_data, &params, /*tolerance=*/1);
}

TF_LITE_MICRO_TESTS_END
