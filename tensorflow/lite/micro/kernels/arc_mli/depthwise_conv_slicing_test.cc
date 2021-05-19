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

// This test checks that slicing logic doesn`t affect result of depthwise
// convolution kernel
//
// This test doesn`t replace default depthwise convolution test
// (tensorflow/lite/micro/kernels/depthwise_conv_test.cc). It is added to the
// whole testset only in case MLI for ARC platform is used during generation
// (which is handled in arc_mli.inc). So such tests won`t be generated for other
// platforms.

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

constexpr int kMaxFilterChannels = 64;
constexpr int kMaxBiasChannels = 64;

// Index of the output tensor in context->tensors, specific to
// DepthwiseConv.
constexpr int kOutputTensorIndex = 3;

// Creates a DepthwiseConv opeerator, calls it with the provided input tensors
// and some defaults parameters, and compares the output with
// expected_output_data.
//
// The tensors parameter contains both the input tensors as well as a
// preallocated output tensor into which the output is stored.
template <typename T>
TfLiteStatus ValidateDepthwiseConvGoldens(const T* expected_output_data,
                                          int output_length,
                                          TfLiteFusedActivation activation,
                                          float tolerance, int tensors_size,
                                          TfLiteTensor* tensors) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_DEPTHWISE_CONV_2D);
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration);

  int input_depth = tensors[0].dims->data[3];
  int output_depth = tensors[1].dims->data[3];
  int depth_mul = output_depth / input_depth;
  TfLiteDepthwiseConvParams builtin_data;
  builtin_data.padding = kTfLitePaddingValid;
  builtin_data.activation = activation;
  builtin_data.stride_height = 1;
  builtin_data.stride_width = 1;
  builtin_data.dilation_height_factor = 1;
  builtin_data.dilation_width_factor = 1;
  builtin_data.depth_multiplier = depth_mul;

  const char* init_data = reinterpret_cast<const char*>(&builtin_data);
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
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;
  node.delegate = nullptr;
  if (registration->prepare) {
    TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TfLiteStatus invoke_status = registration->invoke(&context, &node);
  if (invoke_status != kTfLiteOk) {
    return invoke_status;
  }

  if (registration->free) {
    registration->free(&context, user_data);
  }

  const T* output_data = tflite::GetTensorData<T>(&tensors[kOutputTensorIndex]);
  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
  return kTfLiteOk;
}

void TestDepthwiseConvQuantizedPerChannel(
    const int* input_dims_data, const float* input_data,
    int8_t* input_quantized, float input_scale, int input_zero_point,
    const int* filter_dims_data, const float* filter_data,
    int8_t* filter_data_quantized, const int* bias_dims_data,
    const float* bias_data, int32_t* bias_data_quantized,
    const int* output_dims_data, const float* expected_output_data,
    int8_t* expected_output_data_quantized, int8_t* output_data,
    float output_scale, int output_zero_point,
    TfLiteFusedActivation activation) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  int filter_zero_points[kMaxFilterChannels];
  float filter_scales[kMaxFilterChannels];
  int bias_zero_points[kMaxBiasChannels];
  float bias_scales[kMaxBiasChannels];
  TfLiteAffineQuantization filter_quant;
  TfLiteAffineQuantization bias_quant;
  TfLiteTensor input_tensor =
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, input_zero_point, "input_tensor");
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      filter_data, filter_data_quantized, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, 3 /* quantized dimension */,
      "filter_tensor");
  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensor(
      bias_data, bias_data_quantized, bias_dims, input_scale, &filter_scales[1],
      bias_scales, bias_zero_points, &bias_quant, 3 /* quantized dimension */,
      "bias_tensor");
  TfLiteTensor output_tensor =
      CreateQuantizedTensor(output_data, output_dims, output_scale,
                            input_zero_point, "output_tensor");

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

  AsymmetricQuantize(expected_output_data, expected_output_data_quantized,
                     output_dims_count, output_scale, output_zero_point);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, ValidateDepthwiseConvGoldens(expected_output_data_quantized,
                                              output_dims_count, activation,
                                              1.0, tensors_size, tensors));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

// Test group 1
TF_LITE_MICRO_TEST(SystemTestQuantizedPerChannel1) {
  const int input_elements = 20;
  const int input_shape[] = {4, 1, 5, 2, 2};
  const float input_values[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  const int filter_elements = 36;
  const int filter_shape[] = {4, 2, 3, 3, 2};
  const float filter_values[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  const int bias_elements = 2;
  const int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 20;
  const float bias_values[] = {2, 2};
  const float golden[] = {34, 34, 34, 34, 50, 50, 50, 50, 50, 50,
                          50, 50, 50, 50, 50, 50, 34, 34, 34, 34};
  const int output_shape[] = {4, 1, 5, 2, 2};
  const int output_dims_count = 20;
  int8_t output_data[output_dims_count];

  const float input_scale = 1.0;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(LocalTestQuantizedPerChannel1) {
  const int input_elements = 20;
  const int input_shape[] = {4, 1, 5, 2, 2};
  const int filter_elements = 36;
  const int filter_shape[] = {4, 2, 3, 3, 2};
  const int bias_elements = 2;
  const int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 20;
  const int output_shape[] = {4, 1, 5, 2, 2};
  const int output_dims_count = 20;

#pragma Bss(".Zdata")
  const float input_values[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  const float filter_values[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  const float bias_values[] = {2, 2};
  int8_t output_data[output_dims_count];
#pragma Bss()

  const float golden[] = {34, 34, 34, 34, 50, 50, 50, 50, 50, 50,
                          50, 50, 50, 50, 50, 50, 34, 34, 34, 34};

  const float input_scale = 1.0;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, kTfLiteActNone);
}

// Test group 2
TF_LITE_MICRO_TEST(SystemTestQuantizedPerChannel2) {
  const int input_elements = 80;
  const int input_shape[] = {4, 1, 20, 2, 2};
  const float input_values[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  const int filter_elements = 36;
  const int filter_shape[] = {4, 2, 3, 3, 2};
  const float filter_values[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                                 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  const int bias_elements = 2;
  const int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 80;
  const float bias_values[] = {2, 2};
  const float golden[] = {
      34, 34, 34, 34, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 34, 34, 34, 34};
  const int output_shape[] = {4, 1, 20, 2, 2};
  const int output_dims_count = 80;
  int8_t output_data[output_dims_count];

  const float input_scale = 1.0;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(LocalTestQuantizedPerChannel2) {
  const int input_elements = 80;
  const int input_shape[] = {4, 1, 20, 2, 2};
  const int filter_elements = 36;
  const int filter_shape[] = {4, 2, 3, 3, 2};
  const int bias_elements = 2;
  const int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 80;
  const int output_shape[] = {4, 1, 20, 2, 2};
  const int output_dims_count = 80;

#pragma Bss(".Zdata")
  float input_values[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                          2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  float filter_values[] = {2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
                           2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2};
  float bias_values[] = {2, 2};
  int8_t output_data[output_dims_count];
#pragma Bss()

  const float golden[] = {
      34, 34, 34, 34, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50,
      50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 50, 34, 34, 34, 34};

  const float input_scale = 1.0;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, kTfLiteActNone);
}

// Test group 3
TF_LITE_MICRO_TEST(SystemTestQuantizedPerChannel3) {
  const int input_elements = 40;
  const int input_shape[] = {4, 1, 2, 2, 10};
  const float input_values[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int filter_elements = 90;
  const int filter_shape[] = {4, 1, 3, 3, 10};
  const float filter_values[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int bias_elements = 1;
  const int bias_shape[] = {4, 1, 1, 1, 1};
  const int output_elements = 4;
  const float bias_values[] = {1};
  const float golden[] = {41, 41, 41, 41};
  const int output_shape[] = {4, 1, 2, 2, 1};
  const int output_dims_count = 4;
  int8_t output_data[output_dims_count];

  const float input_scale = 1.0;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(LocalTestQuantizedPerChannel3) {
  const int input_elements = 40;
  const int input_shape[] = {4, 1, 2, 2, 10};
  const int filter_elements = 90;
  const int filter_shape[] = {4, 1, 3, 3, 10};
  const int bias_elements = 1;
  const int bias_shape[] = {4, 1, 1, 1, 1};
  const int output_elements = 4;
  const int output_shape[] = {4, 1, 2, 2, 1};
  const int output_dims_count = 4;

#pragma Bss(".Zdata")
  float input_values[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float filter_values[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float bias_values[] = {1};
  int8_t output_data[output_dims_count];
#pragma Bss()

  const float golden[] = {41, 41, 41, 41};

  const float input_scale = 1.0;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, kTfLiteActNone);
}

// Test group 4
TF_LITE_MICRO_TEST(SystemTestQuantizedPerChannel4) {
  const int input_elements = 80;
  const int input_shape[] = {4, 1, 4, 2, 10};
  const float input_values[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int filter_elements = 90;
  const int filter_shape[] = {4, 1, 3, 3, 10};
  const float filter_values[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int bias_elements = 1;
  const int bias_shape[] = {4, 1, 1, 1, 1};
  const int output_elements = 8;
  const float bias_values[] = {1};
  const float golden[] = {41, 41, 61, 61, 61, 61, 41, 41};
  const int output_shape[] = {4, 1, 4, 2, 1};
  const int output_dims_count = 8;
  int8_t output_data[output_dims_count];

  const float input_scale = 1.0;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(LocalTestQuantizedPerChannel4) {
  const int input_elements = 80;
  const int input_shape[] = {4, 1, 4, 2, 10};
  const int filter_elements = 90;
  const int filter_shape[] = {4, 1, 3, 3, 10};
  const int bias_elements = 1;
  const int bias_shape[] = {4, 1, 1, 1, 1};
  const int output_elements = 8;
  const int output_shape[] = {4, 1, 4, 2, 1};
  const int output_dims_count = 8;

#pragma Bss(".Zdata")
  float input_values[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                          1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float filter_values[] = {
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
      1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  float bias_values[] = {1};
  int8_t output_data[output_dims_count];
#pragma Bss()

  const float golden[] = {41, 41, 61, 61, 61, 61, 41, 41};

  const float input_scale = 1.0;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, kTfLiteActNone);
}
TF_LITE_MICRO_TESTS_END
