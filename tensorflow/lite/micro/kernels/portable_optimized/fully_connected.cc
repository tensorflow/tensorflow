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

#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace micro {
namespace fully_connected {
namespace {

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensor where the quantized inputs are cached.
  int input_quantized_index;
  // A buffer containing the sum-of-weights factor
  int32* sum_of_weights_factor;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

TfLiteStatus CalculateOpData(TfLiteContext* context,
                             TfLiteFullyConnectedParams* params,
                             TfLiteType data_type, const TfLiteTensor* input,
                             const TfLiteTensor* weights,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             OpData* data) {
  TfLiteStatus status = kTfLiteOk;
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, weights, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = -exponent;
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }
  return status;
}

}  // namespace

template <typename T>
inline void PrecomputeSumOfWeightsFactor(const int32* bias, const T* weights,
                                         int32_t* sum_of_weights_factor,
                                         int cols, int rows,
                                         int32_t weights_offset,
                                         int32_t input_offset) {
  for (int row = 0; row < rows; row++) {
    int32_t sum_of_weights = 0;
    for (int col = 0; col < cols; col++) {
      sum_of_weights += weights[col];
    }
    weights += cols;
    sum_of_weights_factor[row] =
        (sum_of_weights + cols * weights_offset) * input_offset;
    if (bias) {
      sum_of_weights_factor[row] += bias[row];
    }
  }
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  void* raw;
  context->AllocatePersistentBuffer(context, sizeof(OpData), &raw);
  OpData* data = reinterpret_cast<OpData*>(raw);
  *data = {};
  return raw;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);

  if (weights->type == kTfLiteInt8 || weights->type == kTfLiteUInt8) {
    // Calculate data for quantized operation
    OpData* data = reinterpret_cast<OpData*>(node->user_data);
    auto* params =
        reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
    const TfLiteTensor* input = GetInput(context, node, kInputTensor);
    const TfLiteTensor* bias =
        GetOptionalInputTensor(context, node, kBiasTensor);
    TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
    TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input->type, input,
                                          weights, bias, output, data));
    // Pre-compute factors for quantized operation
    const int32_t weights_offset = -weights->params.zero_point;
    RuntimeShape weights_shape = GetTensorShape(weights);
    TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
    const int rows = weights_shape.Dims(0);
    const int cols = weights_shape.Dims(1);

    void* raw;
    context->AllocatePersistentBuffer(context, sizeof(int32_t) * rows, &raw);
    data->sum_of_weights_factor = reinterpret_cast<int32_t*>(raw);
    const int32_t input_offset = -input->params.zero_point;
    const int32* bias_data = GetTensorData<int32_t>(bias);

    if (weights->type == kTfLiteInt8) {
      PrecomputeSumOfWeightsFactor<int8_t>(bias_data,
                                           GetTensorData<int8_t>(weights),
                                           data->sum_of_weights_factor, cols,
                                           rows, weights_offset, input_offset);
    } else {
      PrecomputeSumOfWeightsFactor<uint8_t>(bias_data,
                                            GetTensorData<uint8_t>(weights),
                                            data->sum_of_weights_factor, cols,
                                            rows, weights_offset, input_offset);
    }
  }
  return kTfLiteOk;
}

template <typename T>
inline void CalculateOutputNodes(T* output, const T* input, const T* weights,
                                 const int32_t* sum_of_weights_factor,
                                 int32_t sum_of_inputs_factor, int accum_depth,
                                 int output_depth, int32_t output_offset,
                                 int32_t output_multiplier, int output_shift,
                                 int32_t activation_min,
                                 int32_t activation_max) {
  for (int out_c = 0; out_c < output_depth; out_c++) {
    // Multiply and accumulate inputs and weights
    int32_t accum = *sum_of_weights_factor + sum_of_inputs_factor;
    for (int d = 0; d < accum_depth; ++d) {
      accum += weights[d] * input[d];
    }
    // Re-quantize and clamp
    accum =
        MultiplyByQuantizedMultiplier(accum, output_multiplier, output_shift);
    accum += output_offset;
    accum = ActivationFunctionWithMinMax(accum, activation_min, activation_max);
    *output = static_cast<T>(accum);
    // Increment pointers
    output++;
    sum_of_weights_factor++;
    weights += accum_depth;
  }
}

template <typename T>
void EvalQuantized(OpData* opData, const TfLiteTensor* input,
                   const TfLiteTensor* weights, const TfLiteTensor* bias,
                   TfLiteTensor* output) {
  // Get input info
  const T* input_data = GetTensorData<T>(input);

  // Get weights info
  const T* weights_data = GetTensorData<T>(weights);
  const int32_t weights_offset = -weights->params.zero_point;
  RuntimeShape weights_shape = GetTensorShape(weights);
  TFLITE_DCHECK_GE(weights_shape.DimensionsCount(), 2);
  const int weights_dim_count = weights_shape.DimensionsCount();
  const int accum_depth = weights_shape.Dims(weights_dim_count - 1);

  // Get output info
  T* output_data = GetTensorData<T>(output);
  const int32_t output_offset = output->params.zero_point;
  RuntimeShape output_shape = GetTensorShape(output);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 2);
  const int32_t output_multiplier = opData->output_multiplier;
  // TODO(b/138810107): Figure out whether output shift should be inverted
  const int output_shift = -opData->output_shift;
  const int32_t output_activation_min = opData->output_activation_min;
  const int32_t output_activation_max = opData->output_activation_max;
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int batches = output_shape.Dims(0);
  const int output_depth = output_shape.Dims(1);
  TFLITE_DCHECK_LE(output_depth, weights_shape.Dims(weights_dim_count - 2));

  // Get factor pre-computed in the Prepare-phase
  const int32_t* sum_of_weights_factor = opData->sum_of_weights_factor;

  for (int b = 0; b < batches; ++b) {
    // Pre-compute factor for this output-batch
    int32_t sum_of_inputs_factor = 0;
    if (weights_offset != 0) {
      for (int d = 0; d < accum_depth; ++d) {
        sum_of_inputs_factor += input_data[d];
      }
      sum_of_inputs_factor *= weights_offset;
    }
    // Calculate output-nodes using pre-computed factors
    CalculateOutputNodes(output_data, input_data, weights_data,
                         sum_of_weights_factor, sum_of_inputs_factor,
                         accum_depth, output_depth, output_offset,
                         output_multiplier, output_shift, output_activation_min,
                         output_activation_max);
    output_data += output_depth;
    input_data += accum_depth;
  }
}

void EvalQuantizedUint8WithOutputInt16(OpData* opData,
                                       const TfLiteTensor* input,
                                       const TfLiteTensor* weights,
                                       const TfLiteTensor* bias,
                                       TfLiteTensor* output) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = -input->params.zero_point;
  op_params.weights_offset = -weights->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier = opData->output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -opData->output_shift;
  op_params.quantized_activation_min = opData->output_activation_min;
  op_params.quantized_activation_max = opData->output_activation_max;
  reference_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
      GetTensorShape(weights), GetTensorData<uint8_t>(weights),
      GetTensorShape(bias), GetTensorData<int32_t>(bias),
      GetTensorShape(output), GetTensorData<int16_t>(output));
}


TfLiteStatus EvalFloat(TfLiteFullyConnectedParams* params, const TfLiteTensor* input,
               const TfLiteTensor* weights, const TfLiteTensor* bias,
               TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
  tflite::reference_ops::FullyConnected(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(weights), GetTensorData<float>(weights),
      GetTensorShape(bias), GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output));
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* opData = reinterpret_cast<OpData*>(node->user_data);

  switch (weights->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      EvalFloat(params, input, weights, bias, output);
      break;
    case kTfLiteInt8:
      switch (output->type) {
        case kTfLiteInt8:
          EvalQuantized<int8_t>(opData, input, weights, bias, output);
          break;
        default:
          TF_LITE_KERNEL_LOG(context, "Quantized int8 expects output int8");
          return kTfLiteError;
      }
      break;
    case kTfLiteUInt8:
      switch (output->type) {
        case kTfLiteUInt8:
          EvalQuantized<uint8_t>(opData, input, weights, bias, output);
          break;
        case kTfLiteInt16:
          EvalQuantizedUint8WithOutputInt16(opData, input, weights, bias,
                                            output);
          break;
        default:
          TF_LITE_KERNEL_LOG(context,
                             "Quantized uint8 expects output uint8 or int16");
          return kTfLiteError;
      }
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Weight type %d not currently supported.",
                         weights->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration Register_FULLY_CONNECTED() {
  return {/*init=*/fully_connected::Init,
          /*free=*/fully_connected::Free,
          /*prepare=*/fully_connected::Prepare,
          /*invoke=*/fully_connected::Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
