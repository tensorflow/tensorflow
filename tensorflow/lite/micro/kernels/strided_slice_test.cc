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
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void ValidateStridedSliceGoldens(TfLiteTensor* tensors, int tensors_size,
                                 const T* golden, T* output, int output_len,
                                 TfLiteStridedSliceParams* params,
                                 const bool expect_prepare_err, int num_invoke,
                                 float tolerance = 1e-5) {
  int inputs_array_data[] = {4, 0, 1, 2, 3};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 4};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration =
      tflite::ops::micro::Register_STRIDED_SLICE();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array, reinterpret_cast<void*>(params));
  if (expect_prepare_err) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteError, runner.InitAndPrepare());
    return;
  } else {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  }

  for (int i = 0; i < num_invoke; i++) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());
  }

  for (int i = 0; i < output_len; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(golden[i], output[i], 1e-5f);
  }
}

void TestStridedSliceFloat(int* input_shape, int* begin_shape, int* end_shape,
                           int* strides_shape,
                           TfLiteStridedSliceParams* builtin_data,
                           float* input_data, const int32_t* begin_data,
                           const int32_t* end_data, const int32_t* strides_data,
                           int* output_shape, float* output_data,
                           const float* expected_output,
                           bool expect_prepare_err, int num_invoke = 1) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_shape);
  TfLiteIntArray* begin_dims = IntArrayFromInts(begin_shape);
  TfLiteIntArray* end_dims = IntArrayFromInts(end_shape);
  TfLiteIntArray* strides_dims = IntArrayFromInts(strides_shape);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_shape);
  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateTensor(input_data, input_dims),
      CreateTensor(begin_data, begin_dims),
      CreateTensor(end_data, end_dims),
      CreateTensor(strides_data, strides_dims),
      CreateTensor(output_data, output_dims),
  };

  ValidateStridedSliceGoldens(tensors, tensors_size, expected_output,
                              output_data, ElementCount(*output_dims),
                              builtin_data, expect_prepare_err, num_invoke,
                              1.0);
}

template <typename T>
void TestStridedSliceQuantized(int* input_shape, int* begin_shape,
                               int* end_shape, int* strides_shape,
                               TfLiteStridedSliceParams* builtin_data,
                               const T* input_data, const int32_t* begin_data,
                               const int32_t* end_data,
                               const int32_t* strides_data, int* output_shape,
                               T* output_data, const T* expected_output,
                               bool expect_prepare_err, int num_invoke = 1) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_shape);
  TfLiteIntArray* begin_dims = IntArrayFromInts(begin_shape);
  TfLiteIntArray* end_dims = IntArrayFromInts(end_shape);
  TfLiteIntArray* strides_dims = IntArrayFromInts(strides_shape);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_shape);
  constexpr int inputs_size = 4;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  int zero_point =
      std::numeric_limits<T>::max() + std::numeric_limits<T>::min() / 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_dims, 1.0, zero_point),
      CreateTensor(begin_data, begin_dims),
      CreateTensor(end_data, end_dims),
      CreateTensor(strides_data, strides_dims),
      CreateQuantizedTensor(output_data, output_dims, 1.0, zero_point),
  };

  ValidateStridedSliceGoldens(tensors, tensors_size, expected_output,
                              output_data, ElementCount(*output_dims),
                              builtin_data, expect_prepare_err, num_invoke,
                              1.0);
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(UnsupportedInputSize) {
  int input_shape[] = {5, 2, 2, 2, 2, 2};
  int begin_shape[] = {1, 5};
  int end_shape[] = {1, 5};
  int strides_shape[] = {1, 5};
  int output_shape[] = {0};
  float input_data[] = {};
  int32_t begin_data[] = {};
  int32_t end_data[] = {};
  int32_t strides_data[] = {};
  float golden[] = {};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, true);
}

TF_LITE_MICRO_TEST(In1D) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 2};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {1};
  int32_t end_data[] = {3};
  int32_t strides_data[] = {1};
  float golden[] = {2, 3};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_EmptyOutput) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 0};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {10};
  int32_t end_data[] = {3};
  int32_t strides_data[] = {1};
  float golden[] = {};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_NegativeBegin) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 2};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {-3};
  int32_t end_data[] = {3};
  int32_t strides_data[] = {1};
  float golden[] = {2, 3};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_OutOfRangeBegin) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 3};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {-5};
  int32_t end_data[] = {3};
  int32_t strides_data[] = {1};
  float golden[] = {1, 2, 3};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_NegativeEnd) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 1};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {1};
  int32_t end_data[] = {-2};
  int32_t strides_data[] = {1};
  float golden[] = {2};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_OutOfRangeEnd) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 3};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {-3};
  int32_t end_data[] = {5};
  int32_t strides_data[] = {1};
  float golden[] = {2, 3, 4};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_BeginMask) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 3};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {1};
  int32_t end_data[] = {3};
  int32_t strides_data[] = {1};
  float golden[] = {1, 2, 3};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {1, 0, 0, 0, 0};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_NegativeBeginNegativeStride) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 1};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {-2};
  int32_t end_data[] = {-3};
  int32_t strides_data[] = {-1};
  float golden[] = {3};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_OutOfRangeBeginNegativeStride) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 1};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {5};
  int32_t end_data[] = {2};
  int32_t strides_data[] = {-1};
  float golden[] = {4};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_NegativeEndNegativeStride) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 2};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {2};
  int32_t end_data[] = {-4};
  int32_t strides_data[] = {-1};
  float golden[] = {3, 2};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_OutOfRangeEndNegativeStride) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 2};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {-3};
  int32_t end_data[] = {-5};
  int32_t strides_data[] = {-1};
  float golden[] = {2, 1};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_EndMask) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 3};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {1};
  int32_t end_data[] = {3};
  int32_t strides_data[] = {1};
  float golden[] = {2, 3, 4};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {0, 1, 0, 0, 0};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_NegStride) {
  int input_shape[] = {1, 3};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 3};
  float input_data[] = {1, 2, 3};
  int32_t begin_data[] = {-1};
  int32_t end_data[] = {-4};
  int32_t strides_data[] = {-1};
  float golden[] = {3, 2, 1};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_EvenLenStride2) {
  int input_shape[] = {1, 2};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 1};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {0};
  int32_t end_data[] = {4};
  int32_t strides_data[] = {2};
  float golden[] = {1};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_OddLenStride2) {
  int input_shape[] = {1, 3};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {1, 2};
  float input_data[] = {1, 2, 3, 4};
  int32_t begin_data[] = {0};
  int32_t end_data[] = {3};
  int32_t strides_data[] = {2};
  float golden[] = {1, 3};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_Identity) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {2, 2, 3};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {0, 0};
  int32_t end_data[] = {2, 3};
  int32_t strides_data[] = {1, 1};
  float golden[] = {1, 2, 3, 4, 5, 6};
  float output_data[8];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {2, 1, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {1, 0};
  int32_t end_data[] = {2, 2};
  int32_t strides_data[] = {1, 1};
  float golden[] = {4, 5};
  float output_data[8];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_Stride2) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {2, 1, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {0, 0};
  int32_t end_data[] = {2, 3};
  int32_t strides_data[] = {2, 2};
  float golden[] = {1, 3};
  float output_data[8];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_NegStride) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {2, 1, 3};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {1, -1};
  int32_t end_data[] = {2, -4};
  int32_t strides_data[] = {2, -1};
  float golden[] = {6, 5, 4};
  float output_data[8];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_BeginMask) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {2, 2, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {1, 0};
  int32_t end_data[] = {2, 2};
  int32_t strides_data[] = {1, 1};
  float golden[] = {1, 2, 4, 5};
  float output_data[8];

  TfLiteStridedSliceParams builtin_data = {1, 0, 0, 0, 0};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_EndMask) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {2, 1, 3};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {1, 0};
  int32_t end_data[] = {2, 2};
  int32_t strides_data[] = {1, 1};
  float golden[] = {4, 5, 6};
  float output_data[8];

  TfLiteStridedSliceParams builtin_data = {0, 2, 0, 0, 0};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_NegStrideBeginMask) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {2, 1, 3};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {1, -2};
  int32_t end_data[] = {2, -4};
  int32_t strides_data[] = {1, -1};
  float golden[] = {6, 5, 4};
  float output_data[8];

  TfLiteStridedSliceParams builtin_data = {2, 0, 0, 0, 0};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_NegStrideEndMask) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {2, 1, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {1, -2};
  int32_t end_data[] = {2, -3};
  int32_t strides_data[] = {1, -1};
  float golden[] = {5, 4};
  float output_data[8];

  TfLiteStridedSliceParams builtin_data = {0, 2, 0, 0, 0};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In3D_Identity) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {3, 2, 3, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {2, 3, 2};
  int32_t strides_data[] = {1, 1, 1};
  float golden[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In3D_NegStride) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {3, 2, 3, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {2, 3, 2};
  int32_t strides_data[] = {1, 1, 1};
  float golden[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In3D_Strided2) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {3, 1, 2, 1};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {2, 3, 2};
  int32_t strides_data[] = {2, 2, 2};
  float golden[] = {1, 5};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_ShrinkAxisMask1) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {3, 2, 3, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {2, 3, 2};
  int32_t strides_data[] = {1, 1, 1};
  float golden[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_ShrinkAxisMask1_NegativeSlice) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {0};
  float input_data[] = {0, 1, 2, 3};
  int32_t begin_data[] = {-1};
  int32_t end_data[] = {0};
  int32_t strides_data[] = {1};
  float golden[] = {3};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 1};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_ShrinkAxis3_NegativeSlice) {
  int input_shape[] = {2, 4, 1};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {0};
  float input_data[] = {0, 1, 2, 3};
  int32_t begin_data[] = {-2, -1};
  int32_t end_data[] = {-1, 0};
  int32_t strides_data[] = {1, 1};
  float golden[] = {2};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 3};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_ShrinkAxis2_BeginEndAxis1_NegativeSlice) {
  int input_shape[] = {2, 4, 1};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {1, 4};
  float input_data[] = {0, 1, 2, 3};
  int32_t begin_data[] = {0, -1};
  int32_t end_data[] = {0, 0};
  int32_t strides_data[] = {1, 1};
  float golden[] = {0, 1, 2, 3};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {1, 1, 0, 0, 2};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In1D_BeginMaskShrinkAxisMask1) {
  int input_shape[] = {1, 4};
  int begin_shape[] = {1, 1};
  int end_shape[] = {1, 1};
  int strides_shape[] = {1, 1};
  int output_shape[] = {0};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {1};
  int32_t end_data[] = {1};
  int32_t strides_data[] = {1};
  float golden[] = {1};
  float output_data[4];

  TfLiteStridedSliceParams builtin_data = {1, 0, 0, 0, 1};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_ShrinkAxisMask1) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {1, 3};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {0, 0};
  int32_t end_data[] = {1, 3};
  int32_t strides_data[] = {1, 1};
  float golden[] = {1, 2, 3};
  float output_data[6];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 1};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_ShrinkAxisMask2) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {1, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {0, 0};
  int32_t end_data[] = {2, 1};
  int32_t strides_data[] = {1, 1};
  float golden[] = {1, 4};
  float output_data[6];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 2};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In2D_ShrinkAxisMask3) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {0};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {0, 0};
  int32_t end_data[] = {1, 1};
  int32_t strides_data[] = {1, 1};
  float golden[] = {1};
  float output_data[6];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 3};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis1) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {2, 3, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {1, 3, 2};
  int32_t strides_data[] = {1, 1, 1};
  float golden[] = {1, 2, 3, 4, 5, 6};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 1};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis2) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {2, 2, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {2, 1, 2};
  int32_t strides_data[] = {1, 1, 1};
  float golden[] = {1, 2, 7, 8};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 2};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis3) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {1, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {1, 1, 2};
  int32_t strides_data[] = {1, 1, 1};
  float golden[] = {1, 2};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 3};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis4) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {2, 2, 3};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {2, 3, 2};
  int32_t strides_data[] = {1, 1, 1};
  float golden[] = {1, 3, 5, 7, 9, 11};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 4};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis5) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {1, 3};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {1, 3, 1};
  int32_t strides_data[] = {1, 1, 1};
  float golden[] = {1, 3, 5};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 5};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis6) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {1, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {2, 1, 1};
  int32_t strides_data[] = {1, 1, 1};
  float golden[] = {1, 7};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 6};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis7) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {0};
  float input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {1, 1, 1};
  int32_t strides_data[] = {1, 1, 1};
  float golden[] = {1};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 7};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

// This tests catches a very subtle bug that was fixed by cl/188403234.
TF_LITE_MICRO_TEST(RunTwice) {
  int input_shape[] = {2, 2, 3};
  int begin_shape[] = {1, 2};
  int end_shape[] = {1, 2};
  int strides_shape[] = {1, 2};
  int output_shape[] = {2, 2, 2};
  float input_data[] = {1, 2, 3, 4, 5, 6};
  int32_t begin_data[] = {1, 0};
  int32_t end_data[] = {2, 2};
  int32_t strides_data[] = {1, 1};
  float golden[] = {1, 2, 4, 5};
  float output_data[16];

  TfLiteStridedSliceParams builtin_data = {1, 0, 0, 0, 0};

  tflite::testing::TestStridedSliceFloat(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false, 2);
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis1Uint8) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {2, 3, 2};
  uint8_t input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {1, 3, 2};
  int32_t strides_data[] = {1, 1, 1};
  uint8_t golden[] = {1, 2, 3, 4, 5, 6};
  uint8_t output_data[12];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 1};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TEST(In3D_IdentityShrinkAxis1int8) {
  int input_shape[] = {3, 2, 3, 2};
  int begin_shape[] = {1, 3};
  int end_shape[] = {1, 3};
  int strides_shape[] = {1, 3};
  int output_shape[] = {2, 3, 2};
  int8_t input_data[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
  int32_t begin_data[] = {0, 0, 0};
  int32_t end_data[] = {1, 3, 2};
  int32_t strides_data[] = {1, 1, 1};
  int8_t golden[] = {1, 2, 3, 4, 5, 6};
  int8_t output_data[12];

  TfLiteStridedSliceParams builtin_data = {0, 0, 0, 0, 1};

  tflite::testing::TestStridedSliceQuantized(
      input_shape, begin_shape, end_shape, strides_shape, &builtin_data,
      input_data, begin_data, end_data, strides_data, output_shape, output_data,
      golden, false);
}

TF_LITE_MICRO_TESTS_END
