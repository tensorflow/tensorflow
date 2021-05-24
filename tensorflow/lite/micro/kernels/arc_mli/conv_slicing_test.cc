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

// This test checks that slicing logic doesn`t affect result of convolution
// kernel
//
// This test doesn`t replace default convolution test
// (tensorflow/lite/micro/kernels/conv_test.cc). It is added to the whole
// testset only in case MLI for ARC platform is used during generation (which is
// handled in arc_mli.inc). So such tests won`t be generated for other
// platforms.

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_utils.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

// Common inputs and outputs 1.
static const int kInput1Elements = 20;
static const int kInput1Shape[] = {4, 1, 5, 2, 2};
static const float kInput1Data[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                    2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
static const int kFilter1Elements = 36;
static const int kFilter1Shape[] = {4, 2, 3, 3, 2};
static const float kFilter1Data[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
static const int kBias1Elements = 2;
static const int kBias1Shape[] = {1, 2};
static const float kBias1Data[] = {2, 2};
static const int kOutput1Elements = 20;
static const int kOutput1Shape[] = {4, 1, 5, 2, 2};
static const float kGolden1Data[] = {34, 34, 34, 34, 50, 50, 50, 50, 50, 50,
                                     50, 50, 50, 50, 50, 50, 34, 34, 34, 34};

// Common inputs and outputs 2.
static const int kInput2Elements = 80;
static const int kInput2Shape[] = {4, 1, 20, 2, 2};
static const float kInput2Data[] = {
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
static const int kFilter2Elements = 36;
static const int kFilter2Shape[] = {4, 2, 3, 3, 2};
static const float kFilter2Data[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                     2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
static const int kBias2Elements = 2;
static const int kBias2Shape[] = {1, 2};
static const float kBias2Data[] = {2, 2};
static const int kOutput2Elements = 80;
static const int kOutput2Shape[] = {4, 1, 20, 2, 2};
static const float kGolden2Data[] = {
    34, 34, 34, 34, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
    50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 34, 34, 34, 34};

// Common inputs and outputs 3.
static const int kInput3Elements = 40;
static const int kInput3Shape[] = {4, 1, 2, 2, 10};
static const float kInput3Data[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static const int kFilter3Elements = 90;
static const int kFilter3Shape[] = {4, 1, 3, 3, 10};  // 1 3 3 10
static const float kFilter3Data[] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static const int kBias3Elements = 1;
static const int kBias3Shape[] = {1, 1};
static const float kBias3Data[] = {1};
static const int kOutput3Elements = 4;
static const int kOutput3Shape[] = {4, 1, 2, 2, 1};  // 2 2 1
static const float kGolden3Data[] = {41, 41, 41, 41};

// Common inputs and outputs 4.
static const int kInput4Elements = 80;
static const int kInput4Shape[] = {4, 1, 4, 2, 10};
static const float kInput4Data[] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static const int kFilter4Elements = 90;
static const int kFilter4Shape[] = {4, 1, 3, 3, 10};
static const float kFilter4Data[] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
static const int kBias4Elements = 1;
static const int kBias4Shape[] = {1, 1};
static const float kBias4Data[] = {1};
static const int kOutput4Elements = 8;
static const int kOutput4Shape[] = {4, 1, 4, 2, 1};
static const float kGolden4Data[] = {41, 41, 61, 61, 61, 61, 41, 41};

static TfLiteConvParams common_conv_params = {
    kTfLitePaddingSame,  // padding
    1,                   // stride_width
    1,                   // stride_height
    kTfLiteActNone,      // activation
    1,                   // dilation_width_factor
    1,                   // dilation_height_factor
};

template <typename T>
TfLiteStatus ValidateConvGoldens(TfLiteTensor* tensors, int tensors_size,
                                 const T* expected_output_data, T* output_data,
                                 int output_length,
                                 TfLiteConvParams* conv_params,
                                 float tolerance = 1e-5) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::AllOpsResolver resolver;

  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_CONV_2D);

  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  const char* init_data = reinterpret_cast<const char*>(conv_params);
  size_t init_data_size = 0;
  void* user_data = nullptr;

  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

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
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(conv_params);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;

  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TfLiteStatus return_val = registration->invoke(&context, &node);
  if (return_val != kTfLiteOk) {
    return return_val;
  }

  if (registration->free) {
    registration->free(&context, user_data);
  }

  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
  return kTfLiteOk;
}

void TestConvQuantizedPerChannel(
    int* input_dims_data, const float* input_data, int8_t* input_quantized,
    float input_scale, int input_zero_point, int* filter_dims_data,
    const float* filter_data, int8_t* filter_data_quantized,
    int* bias_dims_data, const float* bias_data, int32_t* bias_data_quantized,
    float* bias_scales, int* bias_zero_points, int* output_dims_data,
    const float* expected_output_data, int8_t* expected_output_data_quantized,
    int8_t* output_data, float output_scale, int output_zero_point,
    TfLiteConvParams* conv_params) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  int filter_zero_points[5];
  float filter_scales[5];
  TfLiteAffineQuantization filter_quant;
  TfLiteAffineQuantization bias_quant;
  TfLiteTensor input_tensor =
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point, "input_tensor");
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      filter_data, filter_data_quantized, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, 0 /* quantized dimension */,
      "filter_tensor");

  // DN: to replace scales and quantized data to avoid second quantization
  int channel_count = filter_dims->data[0];
  float true_filter_scales[5] = {1.0, 1.0, 1.0, 1.0, 1.0};
  true_filter_scales[0] = static_cast<float>(channel_count);
  TfLiteAffineQuantization* to_change =
      (TfLiteAffineQuantization*)filter_tensor.quantization.params;
  to_change->scale = FloatArrayFromFloats(true_filter_scales);

  int filter_size = filter_tensor.bytes;
  for (int i = 0; i < filter_size; ++i) {
    filter_tensor.data.int8[i] = filter_data[i];
  }

  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensor(
      bias_data, bias_data_quantized, bias_dims, input_scale, &filter_scales[1],
      bias_scales, bias_zero_points, &bias_quant, 0 /* quantized dimension */,
      "bias_tensor");
  TfLiteTensor output_tensor =
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            output_zero_point, "output_tensor");

  float input_scales[] = {1, input_scale};
  int input_zero_points[] = {1, input_zero_point};
  TfLiteAffineQuantization input_quant = {FloatArrayFromFloats(input_scales),
                                          IntArrayFromInts(input_zero_points)};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  float output_scales[] = {1, output_scale};
  int output_zero_points[] = {1, output_zero_point};
  TfLiteAffineQuantization output_quant = {
      FloatArrayFromFloats(output_scales),
      IntArrayFromInts(output_zero_points)};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  tflite::AsymmetricQuantize(expected_output_data,
                             expected_output_data_quantized, output_dims_count,
                             output_scale, output_zero_point);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateConvGoldens(tensors, tensors_size, expected_output_data_quantized,
                          output_data, output_dims_count, conv_params,
                          1.0 /* tolerance */));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// Test group 1
TF_LITE_MICRO_TEST(SystemTestQuantizedPerChannel1) {
  const int output_dims_count = 20;
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[tflite::testing::kInput1Elements];
  int8_t filter_quantized[tflite::testing::kFilter1Elements];
  int32_t bias_quantized[tflite::testing::kBias1Elements];
  int8_t golden_quantized[tflite::testing::kOutput1Elements];
  int8_t output_data[output_dims_count];

  int zero_points[tflite::testing::kBias1Elements + 1];
  float scales[tflite::testing::kBias1Elements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInput1Shape, tflite::testing::kInput1Data,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilter1Shape, tflite::testing::kFilter1Data,
      filter_quantized, tflite::testing::kBias1Shape,
      tflite::testing::kBias1Data, bias_quantized, scales, zero_points,
      tflite::testing::kOutput1Shape, tflite::testing::kGolden1Data,
      golden_quantized, output_data, output_scale, output_zero_point,
      &tflite::testing::common_conv_params);
}

TF_LITE_MICRO_TEST(LocalTestQuantizedPerChannel1) {
  const int output_dims_count = 20;
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

#pragma Bss(".Xdata")
  static int8_t input_quantized[tflite::testing::kInput1Elements];
  static int8_t filter_quantized[tflite::testing::kFilter1Elements];
  static int32_t bias_quantized[tflite::testing::kBias1Elements];
  static int8_t output_data[output_dims_count];
#pragma Bss()

  int8_t golden_quantized[tflite::testing::kOutput1Elements];
  int zero_points[tflite::testing::kBias1Elements + 1];
  float scales[tflite::testing::kBias1Elements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInput1Shape, tflite::testing::kInput1Data,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilter1Shape, tflite::testing::kFilter1Data,
      filter_quantized, tflite::testing::kBias1Shape,
      tflite::testing::kBias1Data, bias_quantized, scales, zero_points,
      tflite::testing::kOutput1Shape, tflite::testing::kGolden1Data,
      golden_quantized, output_data, output_scale, output_zero_point,
      &tflite::testing::common_conv_params);
}

// Test group 2
TF_LITE_MICRO_TEST(SystemTestQuantizedPerChannel2) {
  const int output_dims_count = 80;
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[tflite::testing::kInput2Elements];
  int8_t filter_quantized[tflite::testing::kFilter2Elements];
  int32_t bias_quantized[tflite::testing::kBias2Elements];
  int8_t golden_quantized[tflite::testing::kOutput2Elements];
  int8_t output_data[output_dims_count];

  int zero_points[tflite::testing::kBias2Elements + 1];
  float scales[tflite::testing::kBias2Elements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInput2Shape, tflite::testing::kInput2Data,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilter2Shape, tflite::testing::kFilter2Data,
      filter_quantized, tflite::testing::kBias2Shape,
      tflite::testing::kBias2Data, bias_quantized, scales, zero_points,
      tflite::testing::kOutput2Shape, tflite::testing::kGolden2Data,
      golden_quantized, output_data, output_scale, output_zero_point,
      &tflite::testing::common_conv_params);
}

TF_LITE_MICRO_TEST(LocalTestQuantizedPerChannel2) {
  const int output_dims_count = 80;
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

#pragma Bss(".Xdata")
  static int8_t input_quantized[tflite::testing::kInput2Elements];
  static int8_t filter_quantized[tflite::testing::kFilter2Elements];
  static int32_t bias_quantized[tflite::testing::kBias2Elements];
  static int8_t output_data[output_dims_count];
#pragma Bss()

  int8_t golden_quantized[tflite::testing::kOutput2Elements];
  int zero_points[tflite::testing::kBias2Elements + 1];
  float scales[tflite::testing::kBias2Elements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInput2Shape, tflite::testing::kInput2Data,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilter2Shape, tflite::testing::kFilter2Data,
      filter_quantized, tflite::testing::kBias2Shape,
      tflite::testing::kBias2Data, bias_quantized, scales, zero_points,
      tflite::testing::kOutput2Shape, tflite::testing::kGolden2Data,
      golden_quantized, output_data, output_scale, output_zero_point,
      &tflite::testing::common_conv_params);
}

// Test group 3
TF_LITE_MICRO_TEST(SystemTestQuantizedPerChannel3) {
  const int output_dims_count = 4;
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[tflite::testing::kInput3Elements];
  int8_t filter_quantized[tflite::testing::kFilter3Elements];
  int32_t bias_quantized[tflite::testing::kBias3Elements];
  int8_t golden_quantized[tflite::testing::kOutput3Elements];
  int8_t output_data[output_dims_count];

  int zero_points[tflite::testing::kBias3Elements + 1];
  float scales[tflite::testing::kBias3Elements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInput3Shape, tflite::testing::kInput3Data,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilter3Shape, tflite::testing::kFilter3Data,
      filter_quantized, tflite::testing::kBias3Shape,
      tflite::testing::kBias3Data, bias_quantized, scales, zero_points,
      tflite::testing::kOutput3Shape, tflite::testing::kGolden3Data,
      golden_quantized, output_data, output_scale, output_zero_point,
      &tflite::testing::common_conv_params);
}

TF_LITE_MICRO_TEST(LocalTestQuantizedPerChannel3) {
  const int output_dims_count = 4;
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

#pragma Bss(".Xdata")
  static int8_t input_quantized[tflite::testing::kInput3Elements];
  static int8_t filter_quantized[tflite::testing::kFilter3Elements];
  static int32_t bias_quantized[tflite::testing::kBias3Elements];
  static int8_t output_data[output_dims_count];
#pragma Bss()

  int8_t golden_quantized[tflite::testing::kOutput3Elements];
  int zero_points[tflite::testing::kBias3Elements + 1];
  float scales[tflite::testing::kBias3Elements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInput3Shape, tflite::testing::kInput3Data,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilter3Shape, tflite::testing::kFilter3Data,
      filter_quantized, tflite::testing::kBias3Shape,
      tflite::testing::kBias3Data, bias_quantized, scales, zero_points,
      tflite::testing::kOutput3Shape, tflite::testing::kGolden3Data,
      golden_quantized, output_data, output_scale, output_zero_point,
      &tflite::testing::common_conv_params);
}

// Test group 4
TF_LITE_MICRO_TEST(SystemTestQuantizedPerChannel4) {
  const int output_dims_count = 8;
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[tflite::testing::kInput4Elements];
  int8_t filter_quantized[tflite::testing::kFilter4Elements];
  int32_t bias_quantized[tflite::testing::kBias4Elements];
  int8_t golden_quantized[tflite::testing::kOutput4Elements];
  int8_t output_data[output_dims_count];

  int zero_points[tflite::testing::kBias4Elements + 1];
  float scales[tflite::testing::kBias4Elements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInput4Shape, tflite::testing::kInput4Data,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilter4Shape, tflite::testing::kFilter4Data,
      filter_quantized, tflite::testing::kBias4Shape,
      tflite::testing::kBias4Data, bias_quantized, scales, zero_points,
      tflite::testing::kOutput4Shape, tflite::testing::kGolden4Data,
      golden_quantized, output_data, output_scale, output_zero_point,
      &tflite::testing::common_conv_params);
}

TF_LITE_MICRO_TEST(LocalTestQuantizedPerChannel4) {
  const int output_dims_count = 8;
  const float input_scale = 1.0f;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

#pragma Bss(".Xdata")
  static int8_t input_quantized[tflite::testing::kInput4Elements];
  static int8_t filter_quantized[tflite::testing::kFilter4Elements];
  static int32_t bias_quantized[tflite::testing::kBias4Elements];
  static int8_t output_data[output_dims_count];
#pragma Bss()

  int8_t golden_quantized[tflite::testing::kOutput4Elements];
  int zero_points[tflite::testing::kBias4Elements + 1];
  float scales[tflite::testing::kBias4Elements + 1];

  tflite::testing::TestConvQuantizedPerChannel(
      tflite::testing::kInput4Shape, tflite::testing::kInput4Data,
      input_quantized, input_scale, input_zero_point,
      tflite::testing::kFilter4Shape, tflite::testing::kFilter4Data,
      filter_quantized, tflite::testing::kBias4Shape,
      tflite::testing::kBias4Data, bias_quantized, scales, zero_points,
      tflite::testing::kOutput4Shape, tflite::testing::kGolden4Data,
      golden_quantized, output_data, output_scale, output_zero_point,
      &tflite::testing::common_conv_params);
}
TF_LITE_MICRO_TESTS_END
