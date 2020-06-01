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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

template <typename T>
TfLiteStatus ValidatePadGoldens(TfLiteTensor* tensors, int tensors_size,
                                const T* golden, T* output_data,
                                int output_length) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_PAD);
  TF_LITE_ENSURE(&context, registration != nullptr);

  int inputs_array_data[] = {2, 0, 1};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 2};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);
  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = nullptr;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->prepare);
  TF_LITE_ENSURE_EQ(&context, kTfLiteOk,
                    registration->prepare(&context, &node));
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_ENSURE_EQ(&context, kTfLiteOk, registration->invoke(&context, &node));
  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden[i], output_data[i]);
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus ValidatePadV2Goldens(TfLiteTensor* tensors, int tensors_size,
                                  const T* golden, T* output_data,
                                  int output_length) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);
  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_PADV2);
  TF_LITE_ENSURE(&context, registration != nullptr);

  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);
  int temporaries_array_data[] = {0};
  TfLiteIntArray* temporaries_array = IntArrayFromInts(temporaries_array_data);
  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.temporaries = temporaries_array;
  node.user_data = nullptr;
  node.builtin_data = nullptr;
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->prepare);
  // Prepare should catch dimension mismatches.
  TfLiteStatus prepare_status = registration->prepare(&context, &node);
  if (prepare_status != kTfLiteOk) {
    return prepare_status;
  }

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  // Eval should catch quantization mismatches.
  TfLiteStatus invoke_status = registration->invoke(&context, &node);
  if (invoke_status != kTfLiteOk) {
    return invoke_status;
  }

  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_EQ(golden[i], output_data[i]);
  }
  return kTfLiteOk;
}

// output data and golden must be shaped correctly
void TestPadFloat(const int* input_dims_data, const float* input_data,
                  const int* pad_dims_data, const int32_t* pad_data,
                  const int* output_dims_data, const float* golden,
                  float* output_data,
                  TfLiteStatus expected_status = kTfLiteOk) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* pad_dims = IntArrayFromInts(pad_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateInt32Tensor(pad_data, pad_dims, "padding tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor")};

  // Pad tensor must be constant.
  tensors[1].allocation_type = kTfLiteMmapRo;

  TF_LITE_MICRO_EXPECT_EQ(expected_status,
                          ValidatePadGoldens(tensors, tensors_size, golden,
                                             output_data, output_dims_count));
}

// output data and golden must be shaped correctly
void TestPadV2Float(const int* input_dims_data, const float* input_data,
                    const int* pad_dims_data, const int32_t* pad_data,
                    const float pad_value, const int* output_dims_data,
                    const float* golden, float* output_data,
                    TfLiteStatus expected_status = kTfLiteOk) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* pad_dims = IntArrayFromInts(pad_dims_data);
  const int pad_value_dims_data[] = {1, 1};  // Only one padding value allowed.
  TfLiteIntArray* pad_value_dims = IntArrayFromInts(pad_value_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateInt32Tensor(pad_data, pad_dims, "padding tensor"),
      CreateFloatTensor(&pad_value, pad_value_dims, "pad value tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor")};

  // Pad tensor must be constant.
  tensors[1].allocation_type = kTfLiteMmapRo;

  TF_LITE_MICRO_EXPECT_EQ(expected_status,
                          ValidatePadV2Goldens(tensors, tensors_size, golden,
                                               output_data, output_dims_count));
}

template <typename T>
void TestPadQuantized(const int* input_dims_data, const float* input_data,
                      T* input_quantized, float input_scale,
                      int input_zero_point, const int* pad_dims_data,
                      const int32_t* pad_data, const int* output_dims_data,
                      const float* golden, T* golden_quantized,
                      float output_scale, int output_zero_point, T* output_data,
                      TfLiteStatus expected_status = kTfLiteOk) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* pad_dims = IntArrayFromInts(pad_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 2;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point, "input_tensor"),
      CreateInt32Tensor(pad_data, pad_dims, "padding tensor"),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point, "output_tensor")};

  // Pad tensor must be constant.
  tensors[1].allocation_type = kTfLiteMmapRo;

  tflite::AsymmetricQuantize(golden, golden_quantized, output_dims_count,
                             output_scale, output_zero_point);
  TF_LITE_MICRO_EXPECT_EQ(
      expected_status,
      ValidatePadGoldens(tensors, tensors_size, golden_quantized, output_data,
                         output_dims_count));
}

template <typename T>
void TestPadV2Quantized(const int* input_dims_data, const float* input_data,
                        T* input_quantized, float input_scale,
                        int input_zero_point, const int* pad_dims_data,
                        const int32_t* pad_data, const float pad_value,
                        const float pad_value_scale,
                        const int pad_value_zero_point,
                        const int* output_dims_data, const float* golden,
                        T* golden_quantized, float output_scale,
                        int output_zero_point, T* output_data,
                        TfLiteStatus expected_status = kTfLiteOk) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* pad_dims = IntArrayFromInts(pad_dims_data);
  const int pad_value_dims_data[] = {1, 1};  // Only one padding value allowed.
  TfLiteIntArray* pad_value_dims = IntArrayFromInts(pad_value_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  T pad_value_quantized;
  const int output_dims_count = ElementCount(*output_dims);
  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point, "input_tensor"),
      CreateInt32Tensor(pad_data, pad_dims, "padding tensor"),
      CreateQuantizedTensor(&pad_value, &pad_value_quantized, pad_value_dims,
                            pad_value_scale, pad_value_zero_point,
                            "pad value tensor"),
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point, "output_tensor")};

  // Pad tensor must be constant.
  tensors[1].allocation_type = kTfLiteMmapRo;
  tensors[2].params.scale = pad_value_scale;
  tensors[3].params.scale = output_scale;

  tflite::AsymmetricQuantize(golden, golden_quantized, output_dims_count,
                             output_scale, output_zero_point);
  TF_LITE_MICRO_EXPECT_EQ(
      expected_status,
      ValidatePadV2Goldens(tensors, tensors_size, golden_quantized, output_data,
                           output_dims_count));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(Test2DFloat) {
  const int input_dims[] = {4, 1, 2, 2, 1};
  const float input_values[] = {1, 2, 3, 4};
  const int pad_dims[] = {2, 4, 2};
  const int32_t pad_values[] = {1, 1, 0, 0, 1, 1, 0, 0};
  const int output_dims[] = {4, 3, 2, 4, 1};
  const float golden[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
                          0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  float output_data[24];

  tflite::testing::TestPadFloat(input_dims, input_values, pad_dims, pad_values,
                                output_dims, golden, output_data);
}

TF_LITE_MICRO_TEST(Test4DFloat) {
  const int input_dims[] = {4, 1, 1, 1, 1};
  const float input_values[] = {42};
  const int pad_dims[] = {2, 4, 2};
  const int32_t pad_values[] = {1, 1, 1, 1, 1, 1, 1, 1};
  const int output_dims[] = {4, 3, 3, 3, 3};
  const int kOutputLen = 81;  // 3 * 3 * 3 * 3
  float golden[kOutputLen];
  for (int i = 0; i < kOutputLen; i++) {
    golden[i] = 0;
  }
  golden[40] = 42;
  float output_data[kOutputLen];

  tflite::testing::TestPadFloat(input_dims, input_values, pad_dims, pad_values,
                                output_dims, const_cast<const float*>(golden),
                                output_data);
}

TF_LITE_MICRO_TEST(Test2DFloatV2) {
  const int input_dims[] = {4, 1, 2, 2, 1};
  const float input_values[] = {1, 2, 3, 4};
  const int pad_dims[] = {2, 4, 2};
  const int32_t pad_values[] = {1, 1, 0, 0, 1, 1, 0, 0};
  const float pad_value = 42;
  const int output_dims[] = {4, 3, 2, 4, 1};
  const float golden[] = {42, 42, 42, 42, 42, 42, 42, 42, 42, 1,  2,  42,
                          42, 3,  4,  42, 42, 42, 42, 42, 42, 42, 42, 42};
  float output_data[24];

  tflite::testing::TestPadV2Float(input_dims, input_values, pad_dims,
                                  pad_values, pad_value, output_dims, golden,
                                  output_data);
}

TF_LITE_MICRO_TEST(Test2DUInt8) {
  const int input_dims[] = {4, 1, 2, 2, 1};
  const float input_values[] = {1, 2, 3, 4};
  const float input_scale = 1.0f;
  const int input_zero_point = 127;
  const int pad_dims[] = {2, 4, 2};
  const int32_t pad_values[] = {1, 1, 0, 0, 1, 1, 0, 0};
  const int output_dims[] = {4, 3, 2, 4, 1};
  const float golden[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
                          0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const float output_scale = 1.0f;
  const int output_zero_point = 127;
  uint8_t output_data[24];
  uint8_t input_quantized[4];
  uint8_t golden_quantized[24];

  tflite::testing::TestPadQuantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      pad_dims, pad_values, output_dims, golden, golden_quantized, output_scale,
      output_zero_point, output_data);
}

TF_LITE_MICRO_TEST(Test2DUInt8V2) {
  const int input_dims[] = {4, 1, 2, 2, 1};
  const float input_values[] = {1, 2, 3, 4};
  const float input_scale = 1.0f;
  const int input_zero_point = 127;
  const int pad_dims[] = {2, 4, 2};
  const int32_t pad_values[] = {1, 1, 0, 0, 1, 1, 0, 0};
  const float pad_value = 42;
  const float pad_value_scale = 1.0;
  const float pad_value_zero_point = 127;
  const int output_dims[] = {4, 3, 2, 4, 1};
  const float golden[] = {42, 42, 42, 42, 42, 42, 42, 42, 42, 1,  2,  42,
                          42, 3,  4,  42, 42, 42, 42, 42, 42, 42, 42, 42};
  const float output_scale = 1.0f;
  const int output_zero_point = 127;
  uint8_t output_data[24];
  uint8_t input_quantized[4];
  uint8_t golden_quantized[24];

  tflite::testing::TestPadV2Quantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      pad_dims, pad_values, pad_value, pad_value_scale, pad_value_zero_point,
      output_dims, golden, golden_quantized, output_scale, output_zero_point,
      output_data);
}

TF_LITE_MICRO_TEST(Test2DInt8) {
  const int input_dims[] = {4, 1, 2, 2, 1};
  const float input_values[] = {1, 2, 3, 4};
  const float input_scale = 1.0f;
  const int input_zero_point = 0;
  const int pad_dims[] = {2, 4, 2};
  const int32_t pad_values[] = {1, 1, 0, 0, 1, 1, 0, 0};
  const int output_dims[] = {4, 3, 2, 4, 1};
  const float golden[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 0,
                          0, 3, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const float output_scale = 1.0f;
  const int output_zero_point = 0;
  int8_t output_data[24];
  int8_t input_quantized[4];
  int8_t golden_quantized[24];

  tflite::testing::TestPadQuantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      pad_dims, pad_values, output_dims, golden, golden_quantized, output_scale,
      output_zero_point, output_data);
}

TF_LITE_MICRO_TEST(Test2DInt8V2) {
  const int input_dims[] = {4, 1, 2, 2, 1};
  const float input_values[] = {1, 2, 3, 4};
  const float input_scale = 1.0f;
  const int input_zero_point = 0;
  const int pad_dims[] = {2, 4, 2};
  const int32_t pad_values[] = {1, 1, 0, 0, 1, 1, 0, 0};
  const float pad_value = 42;
  const float pad_value_scale = 1.0;
  const float pad_value_zero_point = 0;
  const int output_dims[] = {4, 3, 2, 4, 1};
  const float golden[] = {42, 42, 42, 42, 42, 42, 42, 42, 42, 1,  2,  42,
                          42, 3,  4,  42, 42, 42, 42, 42, 42, 42, 42, 42};
  const float output_scale = 1.0f;
  const int output_zero_point = 0;
  int8_t output_data[24];
  int8_t input_quantized[4];
  int8_t golden_quantized[24];

  tflite::testing::TestPadV2Quantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      pad_dims, pad_values, pad_value, pad_value_scale, pad_value_zero_point,
      output_dims, golden, golden_quantized, output_scale, output_zero_point,
      output_data);
}

TF_LITE_MICRO_TEST(Test2DInt8V2ExpectFailurePadValueQuantizationMismatch) {
  const int input_dims[] = {4, 1, 2, 2, 1};
  const float input_values[] = {1, 2, 3, 4};
  const float input_scale = 1.0f;
  const int input_zero_point = 0;
  const int pad_dims[] = {2, 4, 2};
  const int32_t pad_values[] = {1, 1, 0, 0, 1, 1, 0, 0};
  const float pad_value = 42;
  // Causes failure since this is in a different quantization space than input.
  const float pad_value_scale = .5;
  const float pad_value_zero_point = 0;
  const int output_dims[] = {4, 3, 2, 4, 1};
  const float golden[] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                          0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  const float output_scale = 1.0f;
  const int output_zero_point = 0;
  int8_t output_data[24] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                            0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
  int8_t input_quantized[4];
  int8_t golden_quantized[24];

  tflite::testing::TestPadV2Quantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      pad_dims, pad_values, pad_value, pad_value_scale, pad_value_zero_point,
      output_dims, golden, golden_quantized, output_scale, output_zero_point,
      output_data, kTfLiteError);
}

TF_LITE_MICRO_TEST(Test2DInt8ExpectFailureQuantizationRangeExcludesZero) {
  const int input_dims[] = {4, 1, 2, 2, 1};
  const float input_values[] = {1, 2, 3, 4};
  const float input_scale = 1.0f;
  const int input_zero_point = 0;
  const int pad_dims[] = {2, 4, 2};
  const int32_t pad_values[] = {1, 1, 0, 0, 1, 1, 0, 0};
  const int output_dims[] = {4, 3, 2, 4, 1};
  const float golden[] = {42, 42, 42, 42, 42, 42, 42, 42, 42, 1,  2,  42,
                          42, 3,  4,  42, 42, 42, 42, 42, 42, 42, 42, 42};
  // Causes failure since this quantization zero point excludes zero.
  const float output_scale = 1.0f;
  const int output_zero_point = 129;
  int8_t output_data[24];
  int8_t input_quantized[4];
  int8_t golden_quantized[24];

  tflite::testing::TestPadQuantized(
      input_dims, input_values, input_quantized, input_scale, input_zero_point,
      pad_dims, pad_values, output_dims, golden, golden_quantized, output_scale,
      output_zero_point, output_data, kTfLiteError);
}

TF_LITE_MICRO_TESTS_END
