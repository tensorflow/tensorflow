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
#include "tensorflow/lite/micro/kernels/kernel_runner.h"
#include "tensorflow/lite/micro/test_helpers.h"
#include "tensorflow/lite/micro/testing/micro_test.h"

namespace tflite {
namespace testing {
namespace {

#if !defined(XTENSA)  // Needed to avoid build errors from unused variables.
constexpr int kMaxFilterChannels = 64;
constexpr int kMaxBiasChannels = 64;
#endif  // !defined(XTENSA)

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
TfLiteStatus ValidateDepthwiseConvGoldens(
    const T* expected_output_data, int output_length,
    TfLiteDepthwiseConvParams* conv_params, float tolerance, int tensors_size,
    TfLiteTensor* tensors) {
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  const TfLiteRegistration registration = Register_DEPTHWISE_CONV_2D();
  micro::KernelRunner runner(registration, tensors, tensors_size, inputs_array,
                             outputs_array,
                             reinterpret_cast<void*>(conv_params));

  int input_depth = tensors[0].dims->data[3];
  int output_depth = tensors[1].dims->data[3];
  int depth_mul = output_depth / input_depth;

  conv_params->padding = kTfLitePaddingValid;
  conv_params->stride_height = 1;
  conv_params->stride_width = 1;
  conv_params->depth_multiplier = depth_mul;

  const char* init_data = reinterpret_cast<const char*>(conv_params);

  // TODO(b/154240825): Use a test macro here which fails and returns.
  TfLiteStatus status = runner.InitAndPrepare(init_data);
  if (status != kTfLiteOk) {
    return status;
  }
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteOk, runner.Invoke());

  const T* output_data = tflite::GetTensorData<T>(&tensors[kOutputTensorIndex]);
  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
  return kTfLiteOk;
}

#if !defined(XTENSA)  // Needed to avoid build errors from unsused functions.
void TestDepthwiseConvFloat(const int* input_dims_data, const float* input_data,
                            const int* filter_dims_data,
                            const float* filter_data, const int* bias_dims_data,
                            const float* bias_data,
                            const float* expected_output_data,
                            const int* output_dims_data,
                            TfLiteDepthwiseConvParams* conv_params,
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
      CreateTensor(input_data, input_dims),
      CreateTensor(filter_data, filter_dims),
      CreateTensor(bias_data, bias_dims),
      CreateTensor(output_data, output_dims),
  };

  ValidateDepthwiseConvGoldens(expected_output_data, output_dims_count,
                               conv_params, 1e-5, tensors_size, tensors);
}

void TestDepthwiseConvQuantizedPerLayer(
    const int* input_dims_data, const float* input_data,
    uint8_t* input_quantized, float input_scale, int input_zero_point,
    const int* filter_dims_data, const float* filter_data,
    uint8_t* filter_quantized, float filter_scale, int filter_zero_point,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const float* golden, uint8_t* golden_quantized, const int* output_dims_data,
    uint8_t* output_data, float output_scale, int output_zero_point,
    TfLiteDepthwiseConvParams* conv_params) {
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
                                             input_zero_point),
      tflite::testing::CreateQuantizedTensor(filter_data, filter_quantized,
                                             filter_dims, filter_scale,
                                             filter_zero_point),
      tflite::testing::CreateQuantizedBiasTensor(
          bias_data, bias_quantized, bias_dims, input_scale, filter_scale),
      tflite::testing::CreateQuantizedTensor(output_data, output_dims,
                                             output_scale, output_zero_point),
  };

  // TODO(njeff): Affine Quantization Params should be set on tensor creation.
  float filter_scales[] = {1, filter_scale};
  int filter_zero_points[] = {1, 128};
  TfLiteAffineQuantization filter_quant = {FloatArrayFromFloats(filter_scales),
                                           IntArrayFromInts(filter_zero_points),
                                           0};
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant};

  float bias_scales[] = {1, filter_scale * input_scale};
  int bias_zero_points[] = {1, 128};
  TfLiteAffineQuantization bias_quant = {FloatArrayFromFloats(bias_scales),
                                         IntArrayFromInts(bias_zero_points), 0};
  tensors[2].quantization = {kTfLiteAffineQuantization, &bias_quant};

  Quantize(golden, golden_quantized, output_dims_count, output_scale,
           output_zero_point);
  ValidateDepthwiseConvGoldens(golden_quantized, output_dims_count, conv_params,
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
    TfLiteDepthwiseConvParams* conv_params) {
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
      input_data, input_quantized, input_dims, input_scale, input_zero_point);
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      filter_data, filter_data_quantized, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, 3 /* quantized dimension */
  );
  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensor(
      bias_data, bias_data_quantized, bias_dims, input_scale, &filter_scales[1],
      bias_scales, bias_zero_points, &bias_quant, 3 /* quantized dimension */
  );
  TfLiteTensor output_tensor = CreateQuantizedTensor(
      output_data, output_dims, output_scale, input_zero_point);

  // TODO(njeff): Affine Quantization Params should be set on tensor creation.
  float input_scales[] = {1, input_scale};
  int input_zero_points[] = {1, input_zero_point};
  TfLiteAffineQuantization input_quant = {FloatArrayFromFloats(input_scales),
                                          IntArrayFromInts(input_zero_points),
                                          0};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  float output_scales[] = {1, output_scale};
  int output_zero_points[] = {1, output_zero_point};
  TfLiteAffineQuantization output_quant = {FloatArrayFromFloats(output_scales),
                                           IntArrayFromInts(output_zero_points),
                                           0};
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

  Quantize(expected_output_data, expected_output_data_quantized,
           output_dims_count, output_scale, output_zero_point);

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, ValidateDepthwiseConvGoldens(expected_output_data_quantized,
                                              output_dims_count, conv_params,
                                              1.0, tensors_size, tensors));
}

#endif  // !defined(XTENSA)

}  // namespace
}  // namespace testing
}  // namespace tflite

TF_LITE_MICRO_TESTS_BEGIN

#if !defined(XTENSA)  // TODO(b/170322965): xtensa kernels are less general than
                      // reference kernels and we ifdef out test cases that are
                      // currently known to fail.
TF_LITE_MICRO_TEST(SimpleTest) {
  const int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      71, -34, 99, -20, 91, -26, 127, -4,
  };
  const int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  float output_data[output_dims_count];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  tflite::testing::TestDepthwiseConvFloat(
      input_shape, input_values, filter_shape, filter_values, bias_shape,
      bias_values, golden, output_shape, &conv_params, output_data);
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

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, golden,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, &conv_params);
}

TF_LITE_MICRO_TEST(SimpleTestDilatedQuantized) {
  const int input_elements = 48;
  const int input_shape[] = {4, 1, 4, 6, 2};
  const float input_values[] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,   // h = 0
                                3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,   // h = 1
                                1, 2, 3, 4, 5, 6, 2, 6, 2, 4, 4, 2,   // h = 2
                                3, 2, 6, 5, 1, 4, 1, 2, 1, 4, 6, 3};  // h = 3
  const int filter_elements = 16;
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 24;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      15, 2,  88, -48, 25, 14, 72, 0,  61, -2,  56, 48,  // h = 0
      -4, 52, 12, 48,  11, 70, 63, 40, 51, -30, 41, 48   // h = 1
  };
  const int output_shape[] = {4, 1, 2, 3, 4};

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

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 3;
  conv_params.dilation_height_factor = 2;

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, golden,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, &conv_params);
}

TF_LITE_MICRO_TEST(SimpleTestRelu) {
  const int input_shape[] = {4, 1, 3, 2, 2};
  const float input_values[] = {1, 2, 7, 8, 3, 4, 9, 10, 5, 6, 11, 12};
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const float bias_values[] = {1, 2, 3, 4};
  const int output_shape[] = {4, 1, 2, 1, 4};
  const int output_dims_count = 8;
  const float golden_relu[] = {71, 0, 99, 0, 91, 0, 127, 0};
  float output_data[output_dims_count];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActRelu;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  tflite::testing::TestDepthwiseConvFloat(
      input_shape, input_values, filter_shape, filter_values, bias_shape,
      bias_values, golden_relu, output_shape, &conv_params, output_data);
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

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActRelu;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, golden_relu,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, &conv_params);
}

TF_LITE_MICRO_TEST(SimpleTestQuantizedOptimizedFilterWidth) {
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

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerLayer(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, filter_scale,
      filter_zero_point, bias_shape, bias_values, bias_quantized, goldens,
      golden_quantized, output_shape, output_data, output_scale,
      output_zero_point, &conv_params);
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

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params);
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

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params);
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

  const float input_scale = 0.023529f;
  const float output_scale = 0.023529f;
  const int input_zero_point = -128;
  const int output_zero_point = -128;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActRelu6;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params);
}

TF_LITE_MICRO_TEST(SimpleTestDilatedQuantizedPerChannel) {
  const int input_elements = 48;
  const int input_shape[] = {4, 1, 4, 6, 2};
  const float input_values[] = {1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,   // h = 0
                                3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4,   // h = 1
                                1, 2, 3, 4, 5, 6, 2, 6, 2, 4, 4, 2,   // h = 2
                                3, 2, 6, 5, 1, 4, 1, 2, 1, 4, 6, 3};  // h = 3
  const int filter_elements = 16;
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_values[] = {1, 2, 3, 4, -9, 10,  -11, 12,
                                 5, 6, 7, 8, 13, -14, 15,  -16};
  const int bias_elements = 4;
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const int output_elements = 24;
  const float bias_values[] = {1, 2, 3, 4};
  const float golden[] = {
      15, 2,  88, -48, 25, 14, 72, 0,  61, -2,  56, 48,  // h = 0
      -4, 52, 12, 48,  11, 70, 63, 40, 51, -30, 41, 48   // h = 1
  };
  const int output_shape[] = {4, 1, 2, 3, 4};
  int8_t output_data[output_elements];

  const float input_scale = 0.5;
  const float output_scale = 1.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  int8_t input_quantized[input_elements];
  int8_t filter_quantized[filter_elements];
  int32_t bias_quantized[bias_elements];
  int8_t golden_quantized[output_elements];

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 3;
  conv_params.dilation_height_factor = 2;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_shape, input_values, input_quantized, input_scale, input_zero_point,
      filter_shape, filter_values, filter_quantized, bias_shape, bias_values,
      bias_quantized, output_shape, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params);
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
  int8_t output_data[output_size];
  float output_float[output_size];

  const float input_scale = 0.5;
  const float output_scale = 1.0;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  tflite::testing::TestDepthwiseConvQuantizedPerChannel(
      input_dims, input_data, input_quantized, input_scale, input_zero_point,
      filter_dims, filter_data, filter_quantized, bias_dims, bias_data,
      bias_quantized, output_dims, golden, golden_quantized, output_data,
      output_scale, output_zero_point, &conv_params);

  tflite::testing::TestDepthwiseConvFloat(
      input_dims, input_data, filter_dims, filter_data, bias_dims, bias_data,
      golden, output_dims, &conv_params, output_float);
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

  // Create per-layer quantized int8_t input tensor.
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_values, input_quantized, input_dims, input_scale, 0);
  int input_zero_points[2] = {1, 0};
  float input_scales[2] = {1, input_scale};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points), 0};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Create per-layer quantized int8_t filter tensor.
  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      filter_values, filter_quantized, filter_dims, filter_scale, 0);
  int filter_zero_points[2] = {1, 0};
  float filter_scales[2] = {1, filter_scale};
  TfLiteAffineQuantization filter_quant = {
      tflite::testing::FloatArrayFromFloats(filter_scales),
      tflite::testing::IntArrayFromInts(filter_zero_points), 0};
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Create per-layer quantized int32_t bias tensor.
  tflite::SymmetricQuantize(bias_values, bias_quantized, bias_elements,
                            input_scale * output_scale);
  TfLiteTensor bias_tensor =
      tflite::testing::CreateTensor(bias_quantized, bias_dims);

  int bias_zero_points[2] = {1, 0};
  float bias_scales[2] = {1, input_scale * filter_scale};
  TfLiteAffineQuantization bias_quant = {
      tflite::testing::FloatArrayFromFloats(bias_scales),
      tflite::testing::IntArrayFromInts(bias_zero_points), 0};
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Create per-layer quantized int8_t output tensor.
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_data, output_dims, output_scale, 0);
  int output_zero_points[2] = {1, 0};
  float output_scales[2] = {1, output_scale};
  TfLiteAffineQuantization output_quant = {
      tflite::testing::FloatArrayFromFloats(output_scales),
      tflite::testing::IntArrayFromInts(output_zero_points), 0};
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

  tflite::Quantize(golden, golden_quantized, output_dims_count, output_scale,
                   0);

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, tflite::testing::ValidateDepthwiseConvGoldens(
                     golden_quantized, output_dims_count, &conv_params, 1e-5,
                     tensors_size, tensors));
}

#endif  // !defined(XTENSA)

TF_LITE_MICRO_TEST(FilterDimsNotMatchingAffineQuantization) {
  const int input_shape[] = {4, 1, 2, 3, 2};
  const float input_data[] = {3, 2, 1, -1, -2, -3, 4, 3, 2, -2, -3, -4};
  const int filter_shape[] = {4, 1, 2, 2, 4};
  const float filter_data[] = {1, 2, 3, 4, 3, 4, 5, 6, 7, 8, 5, 6, 3, 4, 1, 2};
  const int bias_shape[] = {4, 1, 1, 1, 4};
  const float bias_data[] = {3, -2, 4, 6};
  const int output_shape[] = {4, 1, 1, 2, 4};

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
      input_data, input_quantized, input_dims, input_scale, input_zero_point);
  TfLiteTensor filter_tensor =
      tflite::testing::CreateSymmetricPerChannelQuantizedTensor(
          filter_data, filter_quantized, filter_dims, filter_scales,
          filter_zero_points, &filter_quant, 0 /* quantized dimension */);
  TfLiteTensor bias_tensor =
      tflite::testing::CreatePerChannelQuantizedBiasTensor(
          bias_data, bias_quantized, bias_dims, input_scale, &filter_scales[1],
          scales, zero_points, &bias_quant, 0);
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_data, output_dims, output_scale, output_zero_point);

  float input_scales[] = {1, input_scale};
  int input_zero_points[] = {1, input_zero_point};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points), 0};
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

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;

  // Set filter quant to mismatched dimension.
  TfLiteAffineQuantization* quant = reinterpret_cast<TfLiteAffineQuantization*>(
      filter_tensor.quantization.params);
  quant->scale->size = 2;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          tflite::testing::ValidateDepthwiseConvGoldens(
                              golden_quantized, output_size, &conv_params, 1e-5,
                              tensors_size, tensors));

  // Set scale back to correct dimension, and make zero point array too short.
  quant->scale->size = filter_shape[0];
  quant->zero_point->size = 2;
  TF_LITE_MICRO_EXPECT_EQ(kTfLiteError,
                          tflite::testing::ValidateDepthwiseConvGoldens(
                              golden_quantized, output_size, &conv_params, 1e-5,
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

  // Create per-tensor quantized int8_t input tensor.
  int8_t input_quantized[input_elements];
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_values, input_quantized, input_dims, input_scale, input_zero_point);

  // Set zero point and scale arrays with a single element for each.
  int input_zero_points[] = {1, input_zero_point};
  float input_scales[] = {1, input_scale};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points), 0};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Create per-tensor quantized int8_t filter tensor.
  int8_t filter_quantized[filter_elements];
  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      filter_values, filter_quantized, filter_dims, filter_scale, 0);

  // Set zero point and scale arrays with a single element for each.
  int filter_zero_points[] = {1, 0};
  float filter_scales[] = {1, filter_scale};
  TfLiteAffineQuantization filter_quant = {
      tflite::testing::FloatArrayFromFloats(filter_scales),
      tflite::testing::IntArrayFromInts(filter_zero_points), 0};
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Create per-tensor quantized int32_t bias tensor.
  int32_t bias_quantized[bias_elements];
  // See https://www.tensorflow.org/lite/performance/quantization_spec for a
  // detailed explanation of why bias scale is input_scale * filter_scale.
  tflite::SymmetricQuantize(bias_values, bias_quantized, bias_elements,
                            input_scale * output_scale);
  TfLiteTensor bias_tensor =
      tflite::testing::CreateTensor(bias_quantized, bias_dims);

  // Set zero point and scale arrays with a single element for each.
  int bias_zero_points[] = {1, 0};
  float bias_scales[] = {1, input_scale * filter_scale};
  TfLiteAffineQuantization bias_quant = {
      tflite::testing::FloatArrayFromFloats(bias_scales),
      tflite::testing::IntArrayFromInts(bias_zero_points), 0};
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Create per-tensor quantized int8_t output tensor.
  int8_t output_quantized[output_elements];
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_quantized, output_dims, output_scale, output_zero_point);

  // Set zero point and scale arrays with a single element for each.
  int output_zero_points[] = {1, output_zero_point};
  float output_scales[] = {1, output_scale};
  TfLiteAffineQuantization output_quant = {
      tflite::testing::FloatArrayFromFloats(output_scales),
      tflite::testing::IntArrayFromInts(output_zero_points), 0};
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
  tflite::Quantize(golden, golden_quantized, output_elements, output_scale, 0);

  // Errors due to quantization should not exceed 1.
  constexpr int kQuantizationTolerance = 1;

  TfLiteDepthwiseConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_width_factor = 1;
  conv_params.dilation_height_factor = 1;
  tflite::testing::ValidateDepthwiseConvGoldens(
      golden_quantized, output_elements, &conv_params, kQuantizationTolerance,
      kTensorsSize, tensors);
}

TF_LITE_MICRO_TESTS_END
