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
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/experimental/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void TestQuantize(std::initializer_list<int> input_dims_data,
                  std::initializer_list<float> input_data,
                  std::initializer_list<int> output_dims_data,
                  std::initializer_list<T> expected_output_data, float min,
                  float max, T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInitializer(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInitializer(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  ::tflite::ops::micro::AllOpsResolver resolver;

  float scale = ScaleFromMinMax<T>(min, max);
  int32_t zero_point = ZeroPointFromMinMax<T>(min, max);

  // TFLite float array takes int size followed by a variable size float array.
  struct {
    TfLiteFloatArray arr;
    float data[1];
  } scale_array = {{1}, {scale}};

  TfLiteAffineQuantization builtin_data = {
      .scale = reinterpret_cast<TfLiteFloatArray*>(&scale_array),
      .zero_point = IntArrayFromInitializer({1, static_cast<int>(zero_point)}),
  };

  TfLiteTensor output_tensor = CreateQuantizedTensor(output_data, output_dims,
                                                     "output_tensor", min, max);
  output_tensor.quantization.type = kTfLiteAffineQuantization;
  output_tensor.quantization.params = &builtin_data;

  // 1 input, 1 output.
  constexpr int tensors_size = 2;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      output_tensor,
  };

  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  // Version 4 ops support int8 quantization.
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_QUANTIZE, 4);

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
  size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

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
    TF_LITE_MICRO_EXPECT_EQ(expected_output_data.begin()[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(QuantizeOpTestUint8) {
  // [-63.5, 64] -> scale=0.5 zero_point=127 for UINT8
  uint8_t output[10];
  tflite::testing::TestQuantize(
      {2, 2, 5}, {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64},
      {2, 2, 5},
      std::initializer_list<uint8_t>{0, 1, 2, 3, 4, 251, 252, 253, 254, 255},
      -63.5, 64, output);
}

TF_LITE_MICRO_TEST(QuantizeOpTestUint8NoScale) {
  // [-127, 128] -> scale=1.0 zero_point=128 for UINT8
  uint8_t output[10];
  tflite::testing::TestQuantize(
      {2, 2, 5}, {-127, -126, -125, -124, -123, 124, 125, 126, 127, 128},
      {2, 2, 5},
      std::initializer_list<uint8_t>{0, 1, 2, 3, 4, 251, 252, 253, 254, 255},
      -127, 128, output);
}

TF_LITE_MICRO_TEST(QuantizeOpTestInt8) {
  // [-63.5, 64] -> scale=0.5, zero_point=-1 for INT8
  int8_t output[10];
  tflite::testing::TestQuantize(
      {2, 2, 5}, {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64},
      {2, 2, 5},
      std::initializer_list<int8_t>{-128, -127, -126, -125, -124, 123, 124, 125,
                                    126, 127},
      -63.5, 64, output);
}

TF_LITE_MICRO_TEST(QuantizeOpTestInt8NoScale) {
  // [-128, 127] -> scale=1.0, zero_point=0 for INT8
  int8_t output[10];
  tflite::testing::TestQuantize(
      {2, 2, 5}, {-63.5, -63, -62.5, -62, -61.5, 62, 62.5, 63, 63.5, 64},
      {2, 2, 5},
      std::initializer_list<int8_t>{-64, -63, -63, -62, -62, 62, 63, 63, 64,
                                    64},
      -128, 127, output);
}

TF_LITE_MICRO_TESTS_END
