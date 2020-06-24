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
// iteration of the depthwise_conv op with the supplied parameters. Compares
// outputs vs the expected outputs and logs any differences found. Additionally,
// logs the number of clock ticks taken by the invoke call.
TfLiteStatus ValidateDepthwiseConvGoldens(TfLiteTensor* tensors,
                                          int tensors_size,
                                          TfLiteFusedActivation activation,
                                          int tolerance, int output_length,
                                          const int8_t* expected_output_data,
                                          ErrorReporter* reporter) {
  TfLiteContext context;
  PopulateContext(tensors, tensors_size, reporter, &context);

  const TfLiteRegistration* registration =
      ops::micro::Register_DEPTHWISE_CONV_2D();

  int input_depth = tensors[0].dims->data[3];
  int output_depth = tensors[1].dims->data[3];
  TF_LITE_ENSURE(&context, input_depth > 0);
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
  node.builtin_data = reinterpret_cast<void*>(&builtin_data);
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
      TF_LITE_REPORT_ERROR(reporter, "outputs[%d] was %d expected %d\n", i,
                           static_cast<int>(output_data[i]),
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
  const int input_elements = 32 * 4;
  const int filter_elements = 32 * 4;
  const int bias_elements = 32;
  const int output_elements = 32;
  const int input_shape[] = {4, 1, 4, 1, 32};
  const int filter_shape[] = {4, 1, 4, 1, 32};
  const int bias_shape[] = {1, 32};
  const int output_shape[] = {4, 1, 1, 1, 32};
  float input_values[input_elements];
  float filter_values[filter_elements];
  float bias_values[bias_elements];
  const float golden[] = {
      10304.000000, 8483.000000, 6862.000000,  5441.000000,   4220.000000,
      3199.000000,  2378.000000, -8227.000000, -10952.000000, -5797.000000,
      -6586.000000, 6393.000000, 5748.000000,  5303.000000,   5058.000000,
      5013.000000,  5168.000000, -7021.000000, -11330.000000, -11087.000000,
      -7572.000000, 3823.000000, 4154.000000,  4685.000000,   5416.000000,
      6347.000000,  7478.000000, -6295.000000, -5020.000000,  -10969.000000,
      -9038.000000, 773.000000};

  for (int i = 0; i < input_elements; i++) {
    input_values[i] = i - 64;
  }

  for (int i = 0; i < filter_elements; i++) {
    filter_values[i] = (i * 25) % 256 - 128;
  }

  for (int i = 0; i < bias_elements; i++) {
    bias_values[i] = 64 - i;
  }

  // Quantization Parameters.  All scales except output are 1.0, and all zero
  // points are 0. This direct-maps the values to floating point and makes it
  // easy to reson about them.
  const float input_scale = 1.0f;
  const float filter_scale = 1.0f;
  const float output_scale = 100.0f;
  const int input_zero_point = 0;
  const int output_zero_point = 0;

  TfLiteIntArray* input_dims = tflite::testing::IntArrayFromInts(input_shape);
  TfLiteIntArray* filter_dims = tflite::testing::IntArrayFromInts(filter_shape);
  TfLiteIntArray* bias_dims = tflite::testing::IntArrayFromInts(bias_shape);
  TfLiteIntArray* output_dims = tflite::testing::IntArrayFromInts(output_shape);

  // Create per-tensor quantized int8 input tensor.
  int8_t input_quantized[input_elements];
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
  int8_t filter_quantized[filter_elements];
  TfLiteTensor filter_tensor = tflite::testing::CreateQuantizedTensor(
      filter_values, filter_quantized, filter_dims, filter_scale, 0);

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
  TfLiteTensor bias_tensor =
      tflite::testing::CreateInt32Tensor(bias_quantized, bias_dims);

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
      output_quantized, output_dims, output_scale, output_zero_point);

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
      tensors, kTensorsSize, kTfLiteActNone, kQuantizationTolerance,
      output_elements, golden_quantized, &reporter);
  if (status != kTfLiteOk) {
    TF_LITE_REPORT_ERROR(&reporter, "Model invoke failed\n");
  }
  return 0;
}
