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
#include "tensorflow/lite/experimental/micro/test_helpers.h"
#include "tensorflow/lite/experimental/micro/testing/micro_test.h"
#include "tensorflow/lite/experimental/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
void TestQuantize(const int* input_dims_data, const float* input_data,
                  const int* output_dims_data, const float* golden,
                  T* golden_quantized, float scale, int zero_point,
                  T* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  ::tflite::ops::micro::AllOpsResolver resolver;

  TfLiteTensor output_tensor = CreateQuantizedTensor(
      output_data, output_dims, scale, zero_point, "output_tensor");

  TfLiteAffineQuantization quant;
  float scales[] = {1, scale};
  int zero_points[] = {1, zero_point};
  quant.scale = FloatArrayFromFloats(scales);
  quant.zero_point = IntArrayFromInts(zero_points);
  output_tensor.quantization = {kTfLiteAffineQuantization, &quant};

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

  const char* init_data = nullptr;
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
  node.builtin_data = nullptr;
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

  // Use reference quantization from test utils to compare against op output.
  AsymmetricQuantize(golden, golden_quantized, output_dims_count, scale,
                     zero_point);
  for (int i = 0; i < output_dims_count; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden_quantized[i], output_data[i]);
  }
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(QuantizeOpTestUint8) {
  const int length = 10;
  const int dims[] = {2, 2, 5};
  const float values[] = {-63.5, -63,  -62.5, -62,  -61.5,
                          62,    62.5, 63,    63.5, 64};
  const float scale = 0.5;
  const int zero_point = 127;
  uint8_t output[length];
  uint8_t values_quantized[length];
  tflite::testing::TestQuantize(dims, values, dims, values, values_quantized,
                                scale, zero_point, output);
}

TF_LITE_MICRO_TEST(QuantizeOpTestUint8NoScale) {
  const int length = 10;
  const int dims[] = {2, 2, 5};
  const float values[] = {-127, -126, -125, -124, -123,
                          124,  125,  126,  127,  128};
  const float scale = 1.0;
  const int zero_point = 127;
  uint8_t output[length];
  uint8_t values_quantized[length];
  tflite::testing::TestQuantize(dims, values, dims, values, values_quantized,
                                scale, zero_point, output);
}

TF_LITE_MICRO_TEST(QuantizeOpTestInt8) {
  const int length = 10;
  const int dims[] = {2, 2, 5};
  const float values[] = {-63.5, -63,  -62.5, -62,  -61.5,
                          62,    62.5, 63,    63.5, 64};
  const float scale = 0.5;
  const int zero_point = -1;
  uint8_t output[length];
  uint8_t values_quantized[length];
  tflite::testing::TestQuantize(dims, values, dims, values, values_quantized,
                                scale, zero_point, output);
}

TF_LITE_MICRO_TEST(QuantizeOpTestInt8NoScale) {
  const int length = 10;
  const int dims[] = {2, 2, 5};
  const float values[] = {-128, -127, -126, -125, -124,
                          123,  124,  125,  126,  127};
  const float scale = 1.0;
  const int zero_point = 0;
  uint8_t output[length];
  uint8_t values_quantized[length];
  tflite::testing::TestQuantize(dims, values, dims, values, values_quantized,
                                scale, zero_point, output);
}

TF_LITE_MICRO_TESTS_END
