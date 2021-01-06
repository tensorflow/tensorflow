/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/micro/kernels/conv_test.h"

namespace tflite {
namespace testing {

template <typename T>
TfLiteStatus InvokeConv(TfLiteTensor* tensors, int tensors_size,
                        int output_length, TfLiteConvParams* conv_params,
                        TfLiteRegistration registration, T* output_data) {
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  micro::KernelRunner runner(
      registration, tensors, tensors_size, inputs_array, outputs_array,
      reinterpret_cast<void*>(conv_params), micro_test::reporter);

  const char* init_data = reinterpret_cast<const char*>(conv_params);
  TfLiteStatus status = runner.InitAndPrepare(init_data);
  if (status != kTfLiteOk) {
    return status;
  }
  return runner.Invoke();
}

template <typename T>
TfLiteStatus ValidateConvGoldens(TfLiteTensor* tensors, int tensors_size,
                                 const T* expected_output_data,
                                 int output_length,
                                 TfLiteConvParams* conv_params,
                                 TfLiteRegistration registration,
                                 T* output_data, float tolerance) {
  TfLiteStatus status = InvokeConv(tensors, tensors_size, output_length,
                                   conv_params, registration, output_data);
  if (status != kTfLiteOk) {
    return status;
  }
  for (int i = 0; i < output_length; ++i) {
    TF_LITE_MICRO_EXPECT_NEAR(expected_output_data[i], output_data[i],
                              tolerance);
  }
  return kTfLiteOk;
}

TfLiteStatus InvokeConv(TfLiteTensor* tensors, int tensors_size,
                        int output_length, TfLiteConvParams* conv_params,
                        TfLiteRegistration registration, float* output_data) {
  return InvokeConv<float>(tensors, tensors_size, output_length, conv_params,
                           registration, output_data);
}

TfLiteStatus InvokeConv(TfLiteTensor* tensors, int tensors_size,
                        int output_length, TfLiteConvParams* conv_params,
                        TfLiteRegistration registration, int8_t* output_data) {
  return InvokeConv<int8_t>(tensors, tensors_size, output_length, conv_params,
                            registration, output_data);
}

TfLiteStatus InvokeConv(TfLiteTensor* tensors, int tensors_size,
                        int output_length, TfLiteConvParams* conv_params,
                        TfLiteRegistration registration, uint8_t* output_data) {
  return InvokeConv<uint8_t>(tensors, tensors_size, output_length, conv_params,
                             registration, output_data);
}

TfLiteStatus ValidateConvGoldens(TfLiteTensor* tensors, int tensors_size,
                                 const float* expected_output_data,
                                 int output_length,
                                 TfLiteConvParams* conv_params,
                                 TfLiteRegistration registration,
                                 float* output_data, float tolerance) {
  return ValidateConvGoldens<float>(tensors, tensors_size, expected_output_data,
                                    output_length, conv_params, registration,
                                    output_data, tolerance);
}

TfLiteStatus ValidateConvGoldens(TfLiteTensor* tensors, int tensors_size,
                                 const int8_t* expected_output_data,
                                 int output_length,
                                 TfLiteConvParams* conv_params,
                                 TfLiteRegistration registration,
                                 int8_t* output_data, float tolerance) {
  return ValidateConvGoldens<int8_t>(
      tensors, tensors_size, expected_output_data, output_length, conv_params,
      registration, output_data, tolerance);
}

TfLiteStatus ValidateConvGoldens(TfLiteTensor* tensors, int tensors_size,
                                 const uint8_t* expected_output_data,
                                 int output_length,
                                 TfLiteConvParams* conv_params,
                                 TfLiteRegistration registration,
                                 uint8_t* output_data, float tolerance) {
  return ValidateConvGoldens<uint8_t>(
      tensors, tensors_size, expected_output_data, output_length, conv_params,
      registration, output_data, tolerance);
}

void TestConvFloat(const int* input_dims_data, const float* input_data,
                   const int* filter_dims_data, const float* filter_data,
                   const int* bias_dims_data, const float* bias_data,
                   const int* output_dims_data,
                   const float* expected_output_data,
                   TfLiteConvParams* conv_params,
                   TfLiteRegistration registration, float* output_data) {
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

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk, ValidateConvGoldens(tensors, tensors_size,
                                     expected_output_data, output_dims_count,
                                     conv_params, registration, output_data));
}

void TestConvQuantizedPerLayer(
    const int* input_dims_data, const float* input_data,
    uint8_t* input_quantized, float input_scale, const int* filter_dims_data,
    const float* filter_data, uint8_t* filter_quantized, float filter_scale,
    const int* bias_dims_data, const float* bias_data, int32_t* bias_quantized,
    const int* output_dims_data, const float* expected_output_data,
    uint8_t* expected_output_quantized, float output_scale,
    TfLiteConvParams* conv_params, TfLiteRegistration registration,
    uint8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  tflite::Quantize(expected_output_data, expected_output_quantized,
                   output_dims_count, output_scale, 128);

  constexpr int inputs_size = 3;
  constexpr int outputs_size = 1;
  constexpr int tensors_size = inputs_size + outputs_size;
  TfLiteTensor tensors[tensors_size] = {
      CreateQuantizedTensor(input_data, input_quantized, input_dims,
                            input_scale, 128),
      CreateQuantizedTensor(filter_data, filter_quantized, filter_dims,
                            filter_scale, 128),
      CreateQuantizedBiasTensor(bias_data, bias_quantized, bias_dims,
                                input_scale, filter_scale),
      CreateQuantizedTensor(output_data, output_dims, output_scale, 128)};

  float filter_scales[] = {1, filter_scale};
  int filter_zero_points[] = {1, 128};
  TfLiteAffineQuantization filter_quant = {FloatArrayFromFloats(filter_scales),
                                           IntArrayFromInts(filter_zero_points),
                                           0};
  tensors[1].quantization = {kTfLiteAffineQuantization, &filter_quant};

  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateConvGoldens(tensors, tensors_size, expected_output_quantized,
                          output_dims_count, conv_params, registration,
                          output_data));
}

void TestConvQuantizedPerChannel(
    const int* input_dims_data, const float* input_data,
    int8_t* input_quantized, float input_scale, int input_zero_point,
    const int* filter_dims_data, const float* filter_data,
    int8_t* filter_data_quantized, const int* bias_dims_data,
    const float* bias_data, int32_t* bias_data_quantized, float* bias_scales,
    int* bias_zero_points, const int* output_dims_data,
    const float* expected_output_data, int8_t* expected_output_data_quantized,
    float output_scale, int output_zero_point, TfLiteConvParams* conv_params,
    TfLiteRegistration registration, int8_t* output_data) {
  TfLiteIntArray* input_dims = IntArrayFromInts(input_dims_data);
  TfLiteIntArray* filter_dims = IntArrayFromInts(filter_dims_data);
  TfLiteIntArray* bias_dims = IntArrayFromInts(bias_dims_data);
  TfLiteIntArray* output_dims = IntArrayFromInts(output_dims_data);
  const int output_dims_count = ElementCount(*output_dims);

  int filter_zero_points[5];
  float filter_scales[5];
  TfLiteAffineQuantization filter_quant;
  TfLiteAffineQuantization bias_quant;
  TfLiteTensor input_tensor = CreateQuantizedTensor(
      input_data, input_quantized, input_dims, input_scale, input_zero_point);
  TfLiteTensor filter_tensor = CreateSymmetricPerChannelQuantizedTensor(
      filter_data, filter_data_quantized, filter_dims, filter_scales,
      filter_zero_points, &filter_quant, 0 /* quantized dimension */);
  TfLiteTensor bias_tensor = CreatePerChannelQuantizedBiasTensor(
      bias_data, bias_data_quantized, bias_dims, input_scale, &filter_scales[1],
      bias_scales, bias_zero_points, &bias_quant, 0 /* quantized dimension */);
  TfLiteTensor output_tensor = CreateQuantizedTensor(
      output_data, output_dims, output_scale, output_zero_point);

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

  tflite::Quantize(expected_output_data, expected_output_data_quantized,
                   output_dims_count, output_scale, output_zero_point);
  TF_LITE_MICRO_EXPECT_EQ(
      kTfLiteOk,
      ValidateConvGoldens(tensors, tensors_size, expected_output_data_quantized,
                          output_dims_count, conv_params, registration,
                          output_data, 1.0 /* tolerance */));
}

}  // namespace testing
}  // namespace tflite
