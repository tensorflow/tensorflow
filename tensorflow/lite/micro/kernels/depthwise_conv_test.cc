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
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
#include "tensorflow/lite/micro/testing/micro_test.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

static const int kMaxFilterChannels = 64;
static const int kMaxBiasChannels = 64;

template <typename T>
TfLiteStatus ValidateDepthwiseConvGoldens(TfLiteTensor* tensors,
                                          int tensors_size,
                                          const T* expected_output_data,
                                          T* output_data, int output_length,
                                          TfLiteFusedActivation activation,
                                          float tolerance = 1e-5) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
  const TfLiteRegistration* registration =
      resolver.FindOp(tflite::BuiltinOperator_DEPTHWISE_CONV_2D, 1);
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
  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
  return kTfLiteOk;
}

void TestDepthwiseConvFloat(const int* input_dims_data, const float* input_data,
                            const int* filter_dims_data,
                            const float* filter_data, const int* bias_dims_data,
                            const float* bias_data,
                            const float* expected_output_data,
                            const int* output_dims_data,
                            TfLiteFusedActivation activation,
                            float* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateFloatTensor(input_data, input_dims, "input_tensor"),
      CreateFloatTensor(filter_data, filter_dims, "filter_tensor"),
      CreateFloatTensor(bias_data, bias_dims, "bias_tensor"),
      CreateFloatTensor(output_data, output_dims, "output_tensor"),
  };

  ValidateDepthwiseConvGoldens(tensors, tensors_size, expected_output_data,
                               output_data, output_dims_count, activation);
}

void TestDepthwiseConvQuantizedPerLayer(
    const int* input_dims_data, const float* input_data,
    uint8_t* input_quantized, float input_scale, int input_zero_point,
    const int* filter_dims_data, const float* filter_data,
    uint8_t* filter_quantized, float filter_scale, int filter_zero_point,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const float* golden, uint8_t* golden_quantized, const int* output_dims_data,
    uint8_t* output_data, float output_scale, int output_zero_point,
    TfLiteFusedActivation activation) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      tflite::testing::CreateQuantizedTensor(input_data, input_quantized,
                                             input_dims, input_scale,
                                             input_zero_point, "input_tensor"),
      tflite::testing::CreateQuantizedTensor(
          filter_data, filter_quantized, filter_dims, filter_scale,
          filter_zero_point, "filter_tensor"),
      tflite::testing::CreateQuantizedBiasTensor(bias_data, bias_quantized,
                                                 bias_dims, input_scale,
                                                 filter_scale, "bias_tensor"),
      tflite::testing::CreateQuantizedTensor(output_data, output_dims,
                                             output_scale, output_zero_point,
                                             "output_tensor"),
  };

  // TODO(njeff): Affine Quantization Params should be set on tensor creation.
  float filter_scales[] = {1, filter_scale};
  int filter_zero_points[] = {1, 128};
  TfLiteAffineQuantization filter_quant = {
      FloatArrayFromFloats(filter_scales),
      IntArrayFromInts(filter_zero_points)};
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant};

  AsymmetricQuantize(golden, golden_quantized, output_dims_count, output_scale,
                     output_zero_point);
  ValidateDepthwiseConvGoldens(tensors, tensors_size, golden_quantized,
                               output_data, output_dims_count, activation, 1.0);
}

void TestDepthwiseConvQuantizedPerChannel(
    const int* input_dims_data, const float* input_data,
    int8_t* input_quantized, float input_scale, const int* filter_dims_data,
    const float* filter_data, int8_t* filter_data_quantized,
    const int* bias_dims_data, const float* bias_data,
    int32_t* bias_data_quantized, const int* output_dims_data,
    const float* expected_output_data, int8_t* expected_output_data_quantized,
    int8_t* output_data, float output_scale, TfLiteFusedActivation activation) {
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
  TfLiteTensor input_tensor = CreateQuantizedTensor(
      input_data, input_quantized, input_dims, input_scale, 0, "input_tensor");
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
                            0 /* zero point */, "output_tensor");

  // TODO(njeff): Affine Quantization Params should be set on tensor creation.
  float input_scales[] = {1, input_scale};
  int input_zero_points[] = {1, 0};
  TfLiteAffineQuantization input_quant = {FloatArrayFromFloats(input_scales),
                                          IntArrayFromInts(input_zero_points)};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  float output_scales[] = {1, output_scale};
  int output_zero_points[] = {1, 0};
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
                     output_dims_count, output_scale, 0);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, ValidateDepthwiseConvGoldens(
                     tensors, tensors_size, expected_output_data_quantized,
                     output_data, output_dims_count, activation, 1.0));
}

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

TF_LITE_MICRO_TEST(SimpleTest) {
  const int input_elements = 12;
  const int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_elements = 16;
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 8;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      71, -34, 99, -20, 91, -26, 127, -4,
  };
  const int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  float output_data[output_dims_count];
  tflite::testing::TestDepthwiseConvFloat(
      input_shape, input_values, filter_shape, filter_values, bias_shape,
      bias_values, golden, output_shape, kTfLiteActNone, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestQuantized) {
  const int input_elements = 12;
  const int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_elements = 16;
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 8;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      71, -34, 99, -20, 91, -26, 127, -4,
  };
  const int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float filter_scale = 0.5f;
  const int filter_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;

  uint8_t input_quantized[input_elements];
  uint8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  uint8_t golden_quantized[output_elements];
  uint8_t output_data[output_elements];

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, golden,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(SimpleTestRelu) {
  const int input_elements = 12;
  const int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_elements = 16;
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 8;
  const float bias_values[] = {1, 2, 3, 4};
  const int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  const float golden_relu[] = {71, 0, 99, 0, 91, 0, 127, 0};
  float output_data[output_dims_count];

  tflite::testing::TestDepthwiseConvFloat(
      input_shape, input_values, filter_shape, filter_values, bias_shape,
      bias_values, golden_relu, output_shape, kTfLiteActRelu, output_data);
}

TF_LITE_MICRO_TEST(SimpleTestReluQuantized) {
  const int input_elements = 12;
  const int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_elements = 16;
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 8;
  const float bias_values[] = {1, 2, 3, 4};
  const int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  const float golden_relu[] = {71, 0, 99, 0, 91, 0, 127, 0};

  const float input_scale = 0.5f;
  const int input_zero_point = 128;
  const float filter_scale = 0.5f;
  const int filter_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;

  uint8_t input_quantized[input_elements];
  uint8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  uint8_t golden_quantized[output_elements];
  uint8_t output_data[output_elements];

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, golden_relu,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, kTfLiteActRelu);
}

TF_LITE_MICRO_TEST(SimpleTestOptimizedFilterWidth) {
  const int input_elements = 12;
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_elements = 16;
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  const float bias_values[] = {1, 2, 3, 4};
  const int output_dims_count = 9;
  const int input_shape[] = {4, 1, 1, 9, 1};
  const int filter_shape[] = {4, 2, 1, 8, 1};
  const int bias_shape[] = {1, 1};
  const float goldens[] = {
      92, 56, 12, 22, 33, 72, 44, 20, 5,
  };
  const int output_shape[] = {4, 1, 1, 9, 1};

  const float input_scale = 1.0f;
  const int input_zero_point = 128;
  const float filter_scale = 0.5f;
  const int filter_zero_point = 128;
  const float output_scale = 1.0f;
  const int output_zero_point = 128;

  uint8_t input_quantized[input_elements];
  uint8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  uint8_t golden_quantized[output_dims_count];
  uint8_t output_data[output_dims_count];

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, goldens,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, kTfLiteActNone);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannel) {
  const int input_elements = 12;
  const int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_elements = 16;
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 8;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      71, -34, 99, -20, 91, -26, 127, -4,
  };
  const int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  int8_t output_data[output_dims_count];

  const float input_scale = 0.5;
  const float output_scale = 1.0f;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, filter_shape,
      filter_values, filter_quantized, bias_shape, bias_values, bias_quantized,
      output_shape, golden, golden_quantized, output_data, output_scale,
      kTfLiteActNone);
}

TF_LITE_MICRO_TEST(TestQuantizedPerChannelCompareWithFloat) {
  const int input_dims[] = {4, 1, 2, 3, 2};
  const float input_data[] = {3, 2, 1, -1, -2, -3, 4, 3, 2, -2, -3, -4};
  const int filter_dims[] = {4, 1, 2, 2, 4};
  const float filter_data[] = {1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 5, 6, 3, 4, 1, 2};
  const int bias_dims[] = {4, 1, 1, 1, 4};
  const float bias_data[] = {3, -2, 4, 6};
  const int output_dims[] = {4, 1, 1, 2, 4};
  const float golden[] = {43, 48, 18, 22, 3, -4, -28, -36};

  const int input_size = 12;
  const int filter_size = 16;
  const int output_size = 8;
  const int bias_size = 4;
  int8_t input_quantized[input_size];
  int8_t filter_quantized[filter_size];
  int32_t bias_quantized[bias_size];
  int8_t golden_quantized[output_size];
  int zero_points[bias_size + 1];
  float scales[bias_size + 1];
  int8_t output_data[output_size];
  float output_float[output_size];

  float input_scale = 0.5;
  float output_scale = 1.0;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_dims, input_data, input_quantized, input_scale, filter_dims,
      filter_data, filter_quantized, bias_dims, bias_data, bias_quantized,
      output_dims, golden, golden_quantized, output_data, output_scale,
      kTfLiteActNone);

  tflite::testing::TestDepthwiseConvFloat(
      input_dims, input_data, filter_dims, filter_data, bias_dims, bias_data,
      golden, output_dims, kTfLiteActNone, output_float);
}

TF_LITE_MICRO_TEST(FilterDimsNotMatchingAffineQuantization) {
  const int input_shape[] = {4, 1, 2, 3, 2};
  const float input_data[] = {3, 2, 1, -1, -2, -3, 4, 3, 2, -2, -3, -4};
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_data[] = {1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 5, 6, 3, 4, 1, 2};
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const float bias_data[] = {3, -2, 4, 6};
  const int output_shape[] = {4, 1, 1, 2, 4};
  const float golden[] = {43, 48, 18, 22, 3, -4, -28, -36};

  const int input_size = 12;
  const int filter_size = 16;
  const int output_size = 8;
  const int bias_size = 4;
  int8_t input_quantized[input_size];
  int8_t filter_quantized[filter_size];
  int32_t bias_quantized[bias_size];
  int8_t golden_quantized[output_size];
  int zero_points[bias_size + 1];
  float scales[bias_size + 1];
  int8_t output_data[output_size];
  float output_float[output_size];

  float input_scale = 0.5;
  float output_scale = 1.0;

  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = tflite::testing::IntArrayFromInts(filter_shape);
  TfLiteIntArray* bias_dims = tflite::testing::IntArrayFromInts(bias_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);

  int filter_zero_points[5];
  float filter_scales[5];
  TfLiteAffineQuantization filter_quant;
  TfLiteAffineQuantization bias_quant;
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_quantized, input_dims, input_scale, 0, "input_tensor");
  TfLiteTensor filter_tensor =
      tflite::testing::CreateSymmetricPerChannelQuantizedTensor(
          filter_data, filter_quantized, filter_dims, filter_scales,
          filter_zero_points, &filter_quant, 0 /* quantized dimension */,
          "filter_tensor");
  TfLiteTensor bias_tensor =
      tflite::testing::CreatePerChannelQuantizedBiasTensor(
          bias_data, bias_quantized, bias_dims, input_scale, &filter_scales[1],
          scales, zero_points, &bias_quant, 0, "bias_tensor");
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_data, output_dims, output_scale, 0 /* quantized dimension */,
      "output_tensor");

  float input_scales[] = {1, input_scale};
  int input_zero_points[] = {1, 128};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points)};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  // Set filter quant to mismatched dimension.
  TfLiteAffineQuantization* quant = reinterpret_cast<TfLiteAffineQuantization*>(
      filter_tensor.quantization.params);
  quant->scale->size = 1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, tflite::testing::ValidateDepthwiseConvGoldens(
                        tensors, tensors_size, golden_quantized, output_data,
                        output_size, kTfLiteActNone));

  // Set scale back to correct dimension, and make zero point array too short.
  quant->scale->size = filter_shape[0];
  quant->zero_point->size = 1;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, tflite::testing::ValidateDepthwiseConvGoldens(
                        tensors, tensors_size, golden_quantized, output_data,
                        output_size, kTfLiteActNone));
}

TF_LITE_MICRO_TESTS_END
