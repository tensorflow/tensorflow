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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_time.h"
#include "tensorflow/lite/micro/testing/test_utils.h"

namespace tflite {
namespace testing {
namespace {

// Takes in quantized tensors along with expected outputs, and runs a single
// iteration of the conv op with the supplied parameters. Compares outputs vs
// the expected outputs and logs any differences found. Additionally, logs the
// number of clock ticks taken by the invoke call.
TfLiteStatus ValidateConvGoldens(TfLiteTensor* tensors, int tensors_size,
                                 TfLiteConvParams* conv_params, int tolerance,
                                 int output_length,
                                 const int8_t* expected_output_data,
                                 ErrorReporter* reporter) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, reporter, &context);

  const TfLiteRegistration* registration = ops::micro::Register_CONV_2D();

  const char* init_data = reinterpret_cast<const char*>(conv_params);

  // Init data size is always 0 for builtin ops.
  const size_t init_data_size = 0;
  void* user_data = nullptr;
  if (registration->init) {
    user_data = registration->init(&context, init_data, init_data_size);
  }

  // For an N element array, the raw array will be {N, Element 1, ... Element N}
  // There are 3 inputs at index 0, 1 and 2 in the tensors array.
  int inputs_array_data[] = {3, 0, 1, 2};
  TfLiteIntArray* inputs_array = IntArrayFromInts(inputs_array_data);
  // There is 1 output at index 3 in the tensors array.
  int outputs_array_data[] = {1, 3};
  TfLiteIntArray* outputs_array = IntArrayFromInts(outputs_array_data);

  TfLiteNode node;
  node.inputs = inputs_array;
  node.outputs = outputs_array;
  node.user_data = user_data;
  node.builtin_data = reinterpret_cast<void*>(conv_params);
  node.custom_initial_data = nullptr;
  node.custom_initial_data_size = 0;

  if (registration->prepare) {
    TfLiteStatus prepare_status = registration->prepare(&context, &node);
    if (prepare_status != kTfLiteOk) {
      return prepare_status;
    }
  }

  int32_t start = tflite::GetCurrentTimeTicks();
  TfLiteStatus invoke_status = registration->invoke(&context, &node);
  TF_LITE_REPORT_ERROR(reporter, "invoke took %d cycles\n",
                       tflite::GetCurrentTimeTicks() - start);

  if (registration->free) {
    registration->free(&context, user_data);
  }

  if (invoke_status != kTfLiteOk) {
    return invoke_status;
  }

  int8_t* output_data = tensors[3].data.int8;
  for (int i = 0; i < output_length; ++i) {
    if (std::abs(expected_output_data[i] - output_data[i]) > tolerance) {
      TF_LITE_REPORT_ERROR(reporter, "output[%d] failed, was %d expected %d\n",
                           i, static_cast<int>(output_data[i]),
                           static_cast<int>(expected_output_data[i]));
    }
  }
  return kTfLiteOk;
}

}  // namespace
}  // namespace testing
}  // namespace tflite

int main() {
  tflite::MicroErrorReporter reporter;
  const int input_shape[] = {4, 1, 1, 1, 32};
  const int filter_shape[] = {4, 32, 1, 1, 32};
  const int bias_shape[] = {1, 32};
  const int output_shape[] = {4, 1, 1, 1, 32};
  float filter_values[32 * 32];
  float input_values[32];
  float bias_values[32];

  // Generated these outputs using the floating point reference conv kernel.
  // TODO(b/149942509): Do this comparison automatically on random inputs.
  float expected_output[32] = {
      5168.000000,  3377.000000,  306.000000,   -4045.000000, -4556.000000,
      -1227.000000, 822.000000,   1591.000000,  5176.000000,  3385.000000,
      314.000000,   -4037.000000, -4548.000000, -1219.000000, 830.000000,
      1599.000000,  5184.000000,  3393.000000,  322.000000,   -4029.000000,
      -4540.000000, -1211.000000, 838.000000,   1607.000000,  5192.000000,
      3401.000000,  330.000000,   -4021.000000, -4532.000000, -1203.000000,
      846.000000,   1615.000000};

  for (int i = 0; i < 32; i++) {
    bias_values[i] = i;
    input_values[i] = i - 16;
  }

  for (int i = 0; i < 32 * 32; i++) {
    filter_values[i] = (i * 25) % 256 - 128;
  }

  TfLiteConvParams conv_params;
  conv_params.activation = kTfLiteActNone;
  conv_params.dilation_height_factor = 1;
  conv_params.dilation_width_factor = 1;
  conv_params.stride_height = 1;
  conv_params.stride_width = 1;
  conv_params.padding = kTfLitePaddingValid;

  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = tflite::testing::IntArrayFromInts(filter_shape);
  TfLiteIntArray* bias_dims = tflite::testing::IntArrayFromInts(bias_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);
  const int output_dims_count = tflite::ElementCount(*output_dims);

  // Quantization Parameters.  All scales except output are 1.0, and all zero
  // points are 0. This direct-maps the values to floating point and makes it
  // easy to reson about them.
  int input_zero_point = 0;
  float input_scale = 1.0f;
  int filter_zero_point = 0;
  float filter_scale = 1.0f;
  int output_zero_point = 0;
  // Output scale of 50 is needed to accomodate a float range of [-6400, 6350]
  float output_scale = 50.0f;

  // Create per-tensor quantized int8 input tensor.
  int8_t input_quantized[32];
  TfLiteTensor input_tensor = tflite::testing::CreateQuantizedTensor(
      input_values, input_quantized, input_dims, input_scale, input_zero_point);
  // Set zero point and scale arrays with a single element for each.
  int input_zero_points[] = {1, input_zero_point};
  float input_scales[] = {1, input_scale};
  TfLiteAffineQuantization input_quant = {
      tflite::testing::FloatArrayFromFloats(input_scales),
      tflite::testing::IntArrayFromInts(input_zero_points)};
  input_tensor.quantization = {kTfLiteAffineQuantization, &input_quant};

  // Create per-tensor quantized int8 filter tensor.
  int8_t filter_quantized[32 * 32];
  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      filter_values, filter_quantized, filter_dims, filter_scale,
      filter_zero_point);
  // Set zero point and scale arrays with a single element for each.
  int filter_zero_points[] = {1, filter_zero_point};
  float filter_scales[] = {1, filter_scale};
  TfLiteAffineQuantization filter_quant = {
      tflite::testing::FloatArrayFromFloats(filter_scales),
      tflite::testing::IntArrayFromInts(filter_zero_points)};
  filter_tensor.quantization = {kTfLiteAffineQuantization, &filter_quant};

  // Create per-tensor quantized int32 bias tensor.
  int32_t bias_quantized[32];
  tflite::SymmetricQuantize(bias_values, bias_quantized, 32,
                            input_scale * output_scale);
  TfLiteTensor bias_tensor =
      tflite::testing::CreateInt32Tensor(bias_quantized, bias_dims);

  // There is a single zero point of 0, and a single scale of
  // input_scale * filter_scale.
  int bias_zero_points[] = {1, 0};
  float bias_scales[] = {1, input_scale * filter_scale};
  TfLiteAffineQuantization bias_quant = {
      tflite::testing::FloatArrayFromFloats(bias_scales),
      tflite::testing::IntArrayFromInts(bias_zero_points)};
  bias_tensor.quantization = {kTfLiteAffineQuantization, &bias_quant};

  // Create per-tensor quantized int8 output tensor.
  int8_t output_quantized[32];
  TfLiteTensor output_tensor = tflite::testing::CreateQuantizedTensor(
      output_quantized, output_dims, output_scale, output_zero_point);
  // Set zero point and scale arrays with a single element for each.
  int output_zero_points[] = {1, output_zero_point};
  float output_scales[] = {1, output_scale};
  TfLiteAffineQuantization output_quant = {
      tflite::testing::FloatArrayFromFloats(output_scales),
      tflite::testing::IntArrayFromInts(output_zero_points)};
  output_tensor.quantization = {kTfLiteAffineQuantization, &output_quant};

  // The 3 inputs include the input, filter and bias tensors.
  TfLiteTensor tensors[] = {
      input_tensor,
      filter_tensor,
      bias_tensor,
      output_tensor,
  };

  int8_t golden_quantized[32];
  tflite::AsymmetricQuantize(expected_output, golden_quantized,
                             output_dims_count, output_scale,
                             output_zero_point);

  // Rounding errors due to quantization should not exceed 1.
  constexpr int kQuantizationTolerance = 1;
  const int num_tensors = sizeof(tensors) / sizeof(TfLiteTensor);
  TfLiteStatus status = tflite::testing::ValidateConvGoldens(
      tensors, num_tensors, &conv_params, kQuantizationTolerance,
      output_dims_count, golden_quantized, &reporter);
  if (status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&reporter, "Model invoke failed\n");
  }
  return 0;
}
