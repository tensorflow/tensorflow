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
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

// used to set the quantization parameters for the int8_t and uint8_t tests
constexpr float kInputMin = -2.0;
constexpr float kInputMax = 2.0;
constexpr float kOutputMin = -1.0;
constexpr float kOutputMax = 127.0 / 128.0;

TfLiteTensor CreateL2NormTensor(const float* data, TfLiteIntArray* dims,
                                bool is_input) {
  return CreateTensor(data, dims);
}

template <typename T>
TfLiteTensor CreateL2NormTensor(const T* data, TfLiteIntArray* dims,
                                bool is_input) {
  float kInputScale = ScaleFromMinMax<T>(kInputMin, kInputMax);
  int kInputZeroPoint = ZeroPointFromMinMax<T>(kInputMin, kInputMax);
  float kOutputScale = ScaleFromMinMax<T>(kOutputMin, kOutputMax);
  int kOutputZeroPoint = ZeroPointFromMinMax<T>(kOutputMin, kOutputMax);
  TfLiteTensor tensor;
  if (is_input) {
    tensor = CreateQuantizedTensor(data, dims, kInputScale, kInputZeroPoint);
  } else {
    tensor = CreateQuantizedTensor(data, dims, kOutputScale, kOutputZeroPoint);
  }

  tensor.quantization.type = kTfLiteAffineQuantization;
  return tensor;
}

template <typename T>
void TestL2Normalization(int* input_dims_data, const T* input_data,
                         const T* expected_output_data, T* output_data) {
  TfLiteIntArray* dims = IntArrayFromInts(input_dims_data);

  const int output_dims_count = ElementCount(*dims);

  constexpr int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateL2NormTensor(input_data, dims, true),
      CreateL2NormTensor(output_data, dims, false),
  };

  int inputs_array_data[] = {1, 0};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 1};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteL2NormParams builtin_data = {
      .activation = kTfLiteActNone,
  };

  const TfLiteRegistration registration =
      ops::micro::Register_L2_NORMALIZATION();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(&builtin_data));

  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.InitAndPrepare());
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleFloatTest) {
  int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const float input_data[data_length] = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1};
  const float expected_output_data[data_length] = {-0.55, 0.3,   0.35,
                                                   0.6,   -0.35, 0.05};
  float output_data[data_length];

  tflite::testing::TestL2Normalization<float>(
      input_dims, input_data, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(ZerosVectorFloatTest) {
  int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const float input_data[data_length] = {0, 0, 0, 0, 0, 0};
  const float expected_output_data[data_length] = {0, 0, 0, 0, 0, 0};
  float output_data[data_length];

  tflite::testing::TestL2Normalization<float>(
      input_dims, input_data, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(SimpleFloatWithRankLessThanFourTest) {
  int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const float input_data[data_length] = {-1.1, 0.6, 0.7, 1.2, -0.7, 0.1};
  const float expected_output_data[data_length] = {-0.55, 0.3,   0.35,
                                                   0.6,   -0.35, 0.05};
  float output_data[data_length];

  tflite::testing::TestL2Normalization<float>(
      input_dims, input_data, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(MultipleBatchFloatTest) {
  int input_dims[] = {4, 3, 1, 1, 6};
  constexpr int data_length = 18;
  const float input_data[data_length] = {
      -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 1
      -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 2
      -1.1, 0.6, 0.7, 1.2, -0.7, 0.1,  // batch 3
  };
  const float expected_output_data[data_length] = {
      -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 1
      -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 2
      -0.55, 0.3, 0.35, 0.6, -0.35, 0.05,  // batch 3
  };
  float output_data[data_length];

  tflite::testing::TestL2Normalization<float>(
      input_dims, input_data, expected_output_data, output_data);
}

TF_LITE_MICRO_TEST(ZerosVectorUint8Test) {
  int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const uint8_t input_data[data_length] = {127, 127, 127, 127, 127, 127};
  const uint8_t expected_output[data_length] = {128, 128, 128, 128, 128, 128};
  uint8_t output_data[data_length];

  tflite::testing::TestL2Normalization<uint8_t>(input_dims, input_data,
                                                expected_output, output_data);
}

TF_LITE_MICRO_TEST(SimpleUint8Test) {
  int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const uint8_t input_data[data_length] = {57, 165, 172, 204, 82, 133};
  const uint8_t expected_output[data_length] = {
      58, 166, 173, 205, 83, 134,
  };
  uint8_t output_data[data_length];

  tflite::testing::TestL2Normalization<uint8_t>(input_dims, input_data,
                                                expected_output, output_data);
}

TF_LITE_MICRO_TEST(SimpleInt8Test) {
  int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const int8_t input_data[data_length] = {-71, 37, 44, 76, -46, 5};
  const int8_t expected_output[data_length] = {-70, 38, 45, 77, -45, 6};
  int8_t output_data[data_length];

  tflite::testing::TestL2Normalization<int8_t>(input_dims, input_data,
                                               expected_output, output_data);
}

TF_LITE_MICRO_TEST(ZerosVectorInt8Test) {
  int input_dims[] = {4, 1, 1, 1, 6};
  constexpr int data_length = 6;
  const int8_t input_data[data_length] = {-1, -1, -1, -1, -1, -1};
  const int8_t expected_output[data_length] = {0, 0, 0, 0, 0, 0};
  int8_t output_data[data_length];

  tflite::testing::TestL2Normalization<int8_t>(input_dims, input_data,
                                               expected_output, output_data);
}

TF_LITE_MICRO_TEST(MultipleBatchUint8Test) {
  int input_dims[] = {2, 3, 6};
  constexpr int data_length = 18;
  const uint8_t input_data[data_length] = {
      57, 165, 172, 204, 82, 133,  // batch 1
      57, 165, 172, 204, 82, 133,  // batch 2
      57, 165, 172, 204, 82, 133,  // batch 3
  };
  const uint8_t expected_output[data_length] = {
      58, 166, 173, 205, 83, 134,  // batch 1
      58, 166, 173, 205, 83, 134,  // batch 2
      58, 166, 173, 205, 83, 134,  // batch 3
  };
  uint8_t output_data[data_length];

  tflite::testing::TestL2Normalization<uint8_t>(input_dims, input_data,
                                                expected_output, output_data);
}

TF_LITE_MICRO_TEST(MultipleBatchInt8Test) {
  int input_dims[] = {2, 3, 6};
  constexpr int data_length = 18;
  const int8_t input_data[data_length] = {
      -71, 37, 44, 76, -46, 5,  // batch 1
      -71, 37, 44, 76, -46, 5,  // batch 2
      -71, 37, 44, 76, -46, 5,  // batch 3
  };
  const int8_t expected_output[data_length] = {
      -70, 38, 45, 77, -45, 6,  // batch 1
      -70, 38, 45, 77, -45, 6,  // batch 2
      -70, 38, 45, 77, -45, 6,  // batch 3
  };
  int8_t output_data[data_length];

  tflite::testing::TestL2Normalization<int8_t>(input_dims, input_data,
                                               expected_output, output_data);
}

TF_LITE_MICRO_TESTS_END
