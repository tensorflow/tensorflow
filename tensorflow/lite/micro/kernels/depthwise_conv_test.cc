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
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/micro/kernels/all_ops_resolver.h"
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
  PopulateContext(tensors, tensors_size, micro_test::reporter, &context);

  ::tflite::ops::micro::AllOpsResolver resolver;
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
    TF_LITE_ENSURE_OK(context, registration->prepare(&context, &node));
  }
  TF_LITE_MICRO_EXPECT_NE(nullptr, registration->invoke);
  TF_LITE_ENSURE_OK(context, registration->invoke(&context, &node));

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

  ValidateDepthwiseConvGoldens(expected_output_data, output_dims_count,
                               activation, 1e-5, tensors_size, tensors);
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

  float bias_scales[] = {1, filter_scale * input_scale};
  int bias_zero_points[] = {1, 128};
  TfLiteAffineQuantization bias_quant = {FloatArrayFromFloats(bias_scales),
                                         IntArrayFromInts(bias_zero_points)};
  tensors[2].quantization = {kTfLiteAffineQuantization, &bias_quant};

  AsymmetricQuantize(golden, golden_quantized, output_dims_count, output_scale,
                     output_zero_point);
  ValidateDepthwiseConvGoldens(golden_quantized, output_dims_count, activation,
                               1.0, tensors_size, tensors);
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

  // TODO(njeff): Affine Quantization Params should be set on tensor creation.
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

TF_LITE_MICRO_TEST(SimpleTestQuantizedPerChannelDepthMultiplier1) {
  const int input_elements = 12;
  const int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_elements = 8;
  const int filter_shape[] = {4, 1, 2, 2, 2};
  const float filter_values[] = {1, 2, 3, 4, -9, 10, -11, 12};
  const int bias_elements = 2;
  const int bias_shape[] = {4, 1, 1, 1, 2};
  const int output_elements = 4;
  const float bias_values[] = {1, 2};
  const float golden[] = {
      -103,
      127,
      -128,
      127,
  };
  const int output_shape[] = {4, 1, 2, 1, 2};
  const int output_dims_count = 4;
  int8_t output_data[output_dims_count];

  const float input_scale = 1.0f;
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

TF_LITE_MICRO_TEST(TestQuantizedPerChannelDepthMultiplier1Relu6) {
  const int input_elements = 24;
  const int input_shape[] = {4, 1, 3, 2, 4};
  const float input_values[] = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                                1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
  const int filter_elements = 16;
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {0,  1, 8,   -2, -1, 2, -10, 0,
                                 -1, 3, -18, 0,  0,  4, 20,  -3};
  const int bias_elements = 4;
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 8;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      0, 6, 3, 0, 0, 6, 3, 0,
  };
  const int output_shape[] = {4, 1, 2, 1, 4};
  int8_t output_data[output_elements];
  float output_float[output_elements];

  const float input_scale = 0.023529f;
  const float output_scale = 0.023529f;
  const int input_zero_point = -128;
  const int output_zero_point = -128;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];
  int zero_points[bias_elements + 1];
  float scales[bias_elements + 1];

  tflite::testing::TestDepthwiseConvFloat(
      input_shape, input_values, filter_shape, filter_values, bias_shape,
      bias_values, golden, output_shape, kTfLiteActRelu6, output_float);

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, kTfLiteActRelu6);
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

  const float input_scale = 0.5;
  const float output_scale = 1.0;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_dims, input_data, input_quantized, input_scale, input_zero_point,
      filter_dims, filter_data, filter_quantized, bias_dims, bias_data,
      bias_quantized, output_dims, golden, golden_quantized, output_data,
      output_scale, output_zero_point, kTfLiteActNone);

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

  const float input_scale = 0.5;
  const float output_scale = 1.0;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = tflite::testing::IntArrayFromInts(filter_shape);
  TfLiteIntArray* bias_dims = tflite::testing::IntArrayFromInts(bias_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);

  int filter_zero_points[5];
  float filter_scales[5];
  TfLiteAffineQuantization filter_quant;
  TfLiteAffineQuantization bias_quant;
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_data, input_quantized, input_dims, input_scale, input_zero_point,
      "input_tensor");
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
      output_data, output_dims, output_scale, output_zero_point,
      "output_tensor");

  float input_scales[] = {1, input_scale};
  int input_zero_points[] = {1, input_zero_point};
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
  quant->scale->size = 2;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, tflite::testing::ValidateDepthwiseConvGoldens(
                        golden_quantized, output_size, kTfLiteActNone, 1e-5,
                        tensors_size, tensors));

  // Set scale back to correct dimension, and make zero point array too short.
  quant->scale->size = filter_shape[0];
  quant->zero_point->size = 2;
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteError, tflite::testing::ValidateDepthwiseConvGoldens(
                        golden_quantized, output_size, kTfLiteActNone, 1e-5,
                        tensors_size, tensors));
}

TF_LITE_MICRO_TEST(PerChannelBroadcastQuantizationParams) {
  const float input_scale = 1.0f;
  const float filter_scale = 1.0f;
  const float output_scale = 1.0f;

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

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];

  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = tflite::testing::IntArrayFromInts(filter_shape);
  TfLiteIntArray* bias_dims = tflite::testing::IntArrayFromInts(bias_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);

  // Create per-layer quantized int8 input tensor.
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_values, input_quantized, input_dims, input_scale, 0,
      "input_tensor");
  int input_zero_points[2] = {1, 0};
  float input_scales[2] = {1, input_scale};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points)};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Create per-layer quantized int8 filter tensor.
  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      filter_values, filter_quantized, filter_dims, filter_scale, 0,
      "filter_tensor");
  int filter_zero_points[2] = {1, 0};
  float filter_scales[2] = {1, filter_scale};
  TfLiteAffineQuantization filter_quant = {
      tflite::testing::FloatArrayFromFloats(filter_scales),
      tflite::testing::IntArrayFromInts(filter_zero_points)};
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Create per-layer quantized int32 bias tensor.
  tflite::SymmetricQuantize(bias_values, bias_quantized, bias_elements,
                            input_scale * output_scale);
  TfLiteTensor bias_tensor = tflite::testing::CreateInt32Tensor(
      bias_quantized, bias_dims, "bias_tensor");

  int bias_zero_points[2] = {1, 0};
  float bias_scales[2] = {1, input_scale * filter_scale};
  TfLiteAffineQuantization bias_quant = {
      tflite::testing::FloatArrayFromFloats(bias_scales),
      tflite::testing::IntArrayFromInts(bias_zero_points)};
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Create per-layer quantized int8 output tensor.
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_data, output_dims, output_scale, 0, "output_tensor");
  int output_zero_points[2] = {1, 0};
  float output_scales[2] = {1, output_scale};
  TfLiteAffineQuantization output_quant = {
      tflite::testing::FloatArrayFromFloats(output_scales),
      tflite::testing::IntArrayFromInts(output_zero_points)};
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

  tflite::AsymmetricQuantize(golden, golden_quantized, output_dims_count,
                             output_scale, 0);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::ValidateDepthwiseConvGoldens(
                     golden_quantized, output_dims_count, kTfLiteActNone, 1e-5,
                     tensors_size, tensors));
}

TF_LITE_MICRO_TEST(Int8Input32x4Filter32x4ShouldMatchGolden) {
  const int input_elements = 32 * 4;
  const int filter_elements = 32 * 4;
  const int bias_elements = 32;
  const int output_elements = 32;
  const int input_shape[] = {4, 1, 4, 1, 32};
  const int filter_shape[] = {4, 1, 4, 1, 32};
  const int bias_shape[] = {1, 32};
  const int output_shape[] = {4, 1, 1, 1, 32};
  const float input_values[] = {
      11.0589, 10.8824, 11.1766, 11.5295, 10.8236, 9.5295, 9.5295, 10.0001,
      11.2354, 10.8824, 9.1765,  9.0589,  9.6471,  8.9412, 7.9412, 9.0001,
      9.3530,  7.5295,  9.2354,  9.5883,  7.5883,  8.1765, 7.5883, 9.2942,
      9.1177,  8.5883,  8.2354,  8.6471,  8.0589,  8.0001, 7.4118, 7.3530,
      11.0001, 11.1177, 11.0589, 11.2354, 10.5883, 9.2942, 9.2942, 10.1177,
      11.2354, 10.8824, 8.9412,  8.8236,  9.2354,  8.8824, 7.0001, 9.1177,
      9.5883,  8.2354,  9.1765,  9.5295,  7.4118,  8.5883, 8.1177, 9.1765,
      9.0001,  9.0589,  8.9412,  8.2942,  7.8824,  8.4118, 7.2942, 7.2354,
      10.4118, 10.8824, 11.1177, 11.0001, 10.0001, 9.7060, 9.7648, 10.1766,
      11.1766, 10.6471, 8.6471,  8.5295,  9.5295,  9.0001, 7.0001, 9.4118,
      9.8236,  8.0001,  9.2354,  9.5883,  7.5295,  9.0001, 8.5295, 9.0589,
      8.9412,  9.1177,  8.9412,  8.0001,  8.0589,  8.8824, 7.0589, 7.3530,
      11.3530, 11.0589, 10.7060, 10.7648, 9.9413,  9.1177, 9.1177, 9.7648,
      10.7060, 10.2354, 8.5883,  8.8236,  9.7648,  9.2942, 7.5295, 9.2354,
      9.7060,  8.1177,  9.2942,  9.5883,  7.7648,  9.6471, 9.1177, 9.4707,
      9.3530,  8.8236,  8.5295,  8.0589,  8.6471,  9.5883, 7.4118, 7.5883};
  const float filter_values[] = {
      -0.1617, -0.1948, 0.1419,  -0.2311, -0.0891, 0.1551,  0.0033,  0.3037,
      -0.1683, 0.1353,  0.1518,  -0.1683, -0.1386, 0.1452,  0.1816,  0.1716,
      -0.1948, 0.2080,  0.2245,  -0.1981, -0.2410, 0.1849,  0.1981,  0.1584,
      0.2509,  0.1783,  -0.2146, -0.1518, 0.2080,  -0.2872, 0.2014,  0.2476,
      -0.4126, -0.0561, -0.3235, -0.0594, -0.0957, 0.2014,  -0.1056, 0.1386,
      -0.2542, -0.1617, 0.1287,  -0.1816, -0.0363, 0.1419,  -0.0594, 0.2344,
      -0.0099, 0.4192,  0.1287,  -0.2311, -0.2212, -0.0528, -0.2080, 0.1816,
      -0.1452, 0.1221,  0.1254,  -0.1056, -0.0759, 0.1221,  0.1023,  0.1485,
      0.2707,  0.1716,  -0.1882, -0.1783, 0.1650,  -0.2740, 0.1915,  0.2080,
      -0.2971, -0.2575, -0.3169, 0.0198,  -0.0231, 0.2410,  -0.0429, 0.0660,
      -0.1816, 0.1981,  0.2014,  -0.1386, -0.1915, 0.1716,  0.1320,  0.1419,
      0.1320,  0.1353,  -0.1386, -0.1716, 0.1320,  -0.1650, 0.1386,  0.0825,
      -0.1419, -0.1023, 0.1783,  0.0462,  0.2047,  -0.2179, -0.1518, -0.1551,
      0.1518,  0.3334,  0.3103,  -0.2047, -0.2047, -0.0957, -0.1650, 0.1221,
      0.0990,  0.1353,  -0.1617, -0.1485, 0.1650,  -0.1816, 0.1518,  0.1254,
      -0.0363, -0.1254, 0.1386,  0.0429,  0.2113,  -0.2839, -0.1056, -0.2278};
  const float bias_values[] = {
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
      0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000};
  const float golden[] = {
      -5.1194, -2.0075, -2.1751, -4.7958, 1.7073,  -1.2963, -0.4641, 5.0416,
      -6.4424, 0.3836,  2.4684,  -4.7643, -3.8913, 3.8382,  -0.5164, 5.4304,
      -2.7400, 7.7016,  3.6115,  -6.8545, -3.6290, 0.8509,  2.3247,  5.6117,
      1.8215,  2.7645,  -0.7032, -3.2156, 3.9689,  -5.4583, 2.4346,  1.7731};

  // Quantization Parameters.  All scales except output are 1.0, and all zero
  // points are 0. This direct-maps the values to floating point and makes it
  // easy to reson about them.
  const float input_scale = 0.058824;
  const float filter_scale = 0.003301;
  const float output_scale = 0.092596;
  const int input_zero_point = -128;
  const int output_zero_point = 0;

  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = tflite::testing::IntArrayFromInts(filter_shape);
  TfLiteIntArray* bias_dims = tflite::testing::IntArrayFromInts(bias_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);

  // Create per-tensor quantized int8 input tensor.
  int8_t input_quantized[input_elements];
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_values, input_quantized, input_dims, input_scale, input_zero_point,
      "input_tensor");

  // Set zero point and scale arrays with a single element for each.
  int input_zero_points[] = {1, input_zero_point};
  float input_scales[] = {1, input_scale};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points)};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Create per-tensor quantized int8 filter tensor.
  int8_t filter_quantized[filter_elements];
  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      filter_values, filter_quantized, filter_dims, filter_scale, 0,
      "filter_tensor");

  // Set zero point and scale arrays with a single element for each.
  int filter_zero_points[] = {1, 0};
  float filter_scales[] = {1, filter_scale};
  TfLiteAffineQuantization filter_quant = {
      tflite::testing::FloatArrayFromFloats(filter_scales),
      tflite::testing::IntArrayFromInts(filter_zero_points)};
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Create per-tensor quantized int32 bias tensor.
  int32_t bias_quantized[bias_elements];
  // See https://www.tensorflow.org/lite/performance/quantization_spec for a
  // detailed explanation of why bias scale is input_scale * filter_scale.
  tflite::SymmetricQuantize(bias_values, bias_quantized, bias_elements,
                            input_scale * output_scale);
  TfLiteTensor bias_tensor = tflite::testing::CreateInt32Tensor(
      bias_quantized, bias_dims, "bias_tensor");

  // Set zero point and scale arrays with a single element for each.
  int bias_zero_points[] = {1, 0};
  float bias_scales[] = {1, input_scale * filter_scale};
  TfLiteAffineQuantization bias_quant = {
      tflite::testing::FloatArrayFromFloats(bias_scales),
      tflite::testing::IntArrayFromInts(bias_zero_points)};
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Create per-tensor quantized int8 output tensor.
  int8_t output_quantized[output_elements];
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_quantized, output_dims, output_scale, output_zero_point,
      "output_tensor");

  // Set zero point and scale arrays with a single element for each.
  int output_zero_points[] = {1, output_zero_point};
  float output_scales[] = {1, output_scale};
  TfLiteAffineQuantization output_quant = {
      tflite::testing::FloatArrayFromFloats(output_scales),
      tflite::testing::IntArrayFromInts(output_zero_points)};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  // The 3 inputs include the input, filter and bias tensors.
  constexpr int kInputsSize = 3;
  constexpr int kOutputsSize = 1;
  constexpr int kTensorsSize = kInputsSize + kOutputsSize;
  TfLiteTensor tensors[kTensorsSize] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  int8_t golden_quantized[output_elements];
  tflite::AsymmetricQuantize(golden, golden_quantized, output_elements,
                             output_scale, 0);

  // Errors due to quantization should not exceed 1.
  constexpr int kQuantizationTolerance = 1;

  TfLiteStatus status = tflite::testing::ValidateDepthwiseConvGoldens(
      golden_quantized, output_elements, kTfLiteActNone, kQuantizationTolerance,
      kTensorsSize, tensors);
}

TF_LITE_MICRO_TESTS_END
