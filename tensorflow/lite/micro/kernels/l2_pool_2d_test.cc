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
#include <type_traits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

constexpr float kTolerance = 1e-5;

constexpr int kOutputDimsCount = 4;

struct L2Pool2DTestParams {
  TfLitePadding padding = kTfLitePaddingValid;
  int stride_width = 2;
  int stride_height = 2;
  int filter_width = 2;
  int filter_height = 2;
  TfLiteFusedActivation activation = kTfLiteActNone;
  float compare_tolerance = kTolerance;
  //  output_dims_data is a TfLiteIntArray
  int output_dims_data[kOutputDimsCount + 1] = {kOutputDimsCount, 0, 0, 0, 0};
};

void ExecuteL2Pool2DTest(const L2Pool2DTestParams& params,
                         TfLiteTensor* tensors, int tensors_count) {
  int kInputArrayData[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(kInputArrayData);
  int kOutputArrayData[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(kOutputArrayData);

  TfLitePoolParams op_params = {};
  op_params.activation = params.activation;
  op_params.filter_height = params.filter_height;
  op_params.filter_width = params.filter_width;
  op_params.padding = params.padding;
  op_params.stride_height = params.stride_height;
  op_params.stride_width = params.stride_width;

  const TfLiteRegistration registration = tflite::Register_L2_POOL_2D();
  micro::KernelRunner runner(registration, tensors, tensors_count, inputs_array,
                             outputs_array, static_cast<void*>(&op_params));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
}

template <typename T>
void TestL2Pool2D(L2Pool2DTestParams& params, int* input_dims_data,
                  const T* input_data, int* expected_dims_data,
                  const T* expected_data, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* expected_dims = IntArrayFromInts(expected_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(params.output_dims_data);
  const int expected_count = ElementCount(*expected_dims);

  TfLiteTensor tensors[] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(output_data, output_dims),
  };
  constexpr int tensors_count = std::extent<decltype(tensors)>::value;
  ExecuteL2Pool2DTest(params, tensors, tensors_count);

  for (int i = 0; i < expected_count; i++) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_data[i], output_data[i],
                              params.compare_tolerance);
  }
  for (int i = 0; i < expected_dims->size; i++) {
    // output dims will have been relocated during prepare phase,
    // so use the tensor dims pointer.
    TF_LITE_MICRO_EXPECT_EQ(expected_dims->data[i], tensors[1].dims->data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2Pool) {
  int kInputDims[] = {4, 1, 2, 4, 1};
  constexpr float kInput[] = {
      0, 6, 2,  4,  //
      3, 2, 10, 7,  //
  };
  int kExpectDims[] = {4, 1, 1, 2, 1};
  constexpr float kExpect[] = {3.5, 6.5};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::L2Pool2DTestParams params;

  tflite::testing::TestL2Pool2D(params, kInputDims, kInput, kExpectDims,
                                kExpect, output_data);
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolActivationRelu) {
  int kInputDims[] = {4, 1, 2, 4, 1};
  constexpr float kInput[] = {
      -1, -6, 2,  4,  //
      -3, -2, 10, 7,  //
  };
  int kExpectDims[] = {4, 1, 1, 2, 1};
  constexpr float kExpect[] = {3.53553, 6.5};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::L2Pool2DTestParams params;
  params.activation = kTfLiteActRelu;

  tflite::testing::TestL2Pool2D(params, kInputDims, kInput, kExpectDims,
                                kExpect, output_data);
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolActivationRelu1) {
  int kInputDims[] = {4, 1, 2, 4, 1};
  constexpr float kInput[] = {
      -0.1, -0.6, 2,  4,  //
      -0.3, -0.2, 10, 7,  //
  };
  int kExpectDims[] = {4, 1, 1, 2, 1};
  constexpr float kExpect[] = {0.353553, 1.0};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::L2Pool2DTestParams params;
  params.activation = kTfLiteActReluN1To1;

  tflite::testing::TestL2Pool2D(params, kInputDims, kInput, kExpectDims,
                                kExpect, output_data);
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolActivationRelu6) {
  int kInputDims[] = {4, 1, 2, 4, 1};
  constexpr float kInput[] = {
      -0.1, -0.6, 2,  4,  //
      -0.3, -0.2, 10, 7,  //
  };
  int kExpectDims[] = {4, 1, 1, 2, 1};
  constexpr float kExpect[] = {0.353553, 6.0};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::L2Pool2DTestParams params;
  params.activation = kTfLiteActRelu6;

  tflite::testing::TestL2Pool2D(params, kInputDims, kInput, kExpectDims,
                                kExpect, output_data);
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolPaddingSame) {
  int kInputDims[] = {4, 1, 2, 4, 1};
  constexpr float kInput[] = {
      0, 6, 2,  4,  //
      3, 2, 10, 7,  //
  };
  int kExpectDims[] = {4, 1, 1, 2, 1};
  constexpr float kExpect[] = {3.5, 6.5};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::L2Pool2DTestParams params;
  params.padding = kTfLitePaddingSame;

  tflite::testing::TestL2Pool2D(params, kInputDims, kInput, kExpectDims,
                                kExpect, output_data);
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolPaddingSameStride1) {
  int kInputDims[] = {4, 1, 2, 4, 1};
  constexpr float kInput[] = {
      0, 6, 2,  4,  //
      3, 2, 10, 7,  //
  };
  int kExpectDims[] = {4, 1, 2, 4, 1};
  constexpr float kExpect[] = {3.5,     6.0,    6.5,     5.70088,
                               2.54951, 7.2111, 8.63134, 7.0};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::L2Pool2DTestParams params;
  params.padding = kTfLitePaddingSame;
  params.compare_tolerance = 1e-4;
  params.stride_width = 1;
  params.stride_height = 1;

  tflite::testing::TestL2Pool2D(params, kInputDims, kInput, kExpectDims,
                                kExpect, output_data);
}

TF_LITE_MICRO_TEST(FloatPoolingOpTestL2PoolPaddingValidStride1) {
  int kInputDims[] = {4, 1, 2, 4, 1};
  constexpr float kInput[] = {
      0, 6, 2,  4,  //
      3, 2, 10, 7,  //
  };
  int kExpectDims[] = {4, 1, 1, 3, 1};
  constexpr float kExpect[] = {3.5, 6.0, 6.5};
  constexpr int kOutputCount = std::extent<decltype(kExpect)>::value;
  float output_data[kOutputCount];
  tflite::testing::L2Pool2DTestParams params;
  params.stride_width = 1;
  params.stride_height = 1;

  tflite::testing::TestL2Pool2D(params, kInputDims, kInput, kExpectDims,
                                kExpect, output_data);
}

TF_LITE_MICRO_TESTS_END
