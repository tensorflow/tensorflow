/*******************************************************************************
* Copyright (c) 2019-2020 Cadence Design Systems, Inc.
*
* Permission is hereby granted, free of charge, to any person obtaining
* a copy of this software and associated documentation files (the
* "Software"), to use this Software with Cadence processor cores only and
* not with any other processors and platforms, subject to
* the following conditions:
*
* The above copyright notice and this permission notice shall be included
* in all copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
* EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
* MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
* IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
* CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
* TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
* SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

******************************************************************************/
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

#include "tensorflow/lite/micro/kernels/fully_connected.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifi/xtensa_tf_micro_common.h"

namespace tflite {
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
  // Cached zero point values of tensors.
  int32_t input_zero_point;
  int32_t filter_zero_point;
  int32_t output_zero_point;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;

TfLiteStatus CalculateOpData(TfLiteContext* context,
                             TfLiteFusedActivation activation,
                             TfLiteType data_type, const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             OpData* data) {
  TfLiteStatus status = kTfLiteOk;
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = -exponent;
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, activation, output, &data->output_activation_min,
        &data->output_activation_max));

    data->input_zero_point = input->params.zero_point;
    data->filter_zero_point = filter->params.zero_point;
    data->output_zero_point = output->params.zero_point;
  }
  return status;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  OpData* data = static_cast<OpData*>(node->user_data);
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE(context, input != nullptr);
  const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  TF_LITE_ENSURE(context, filter != nullptr);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  return CalculateOpData(context, params->activation, input->type, input,
                         filter, bias, output, data);
}

TfLiteStatus EvalQuantizedInt8(TfLiteContext* context, TfLiteNode* node,
                               const OpData& data,
                               const TfLiteEvalTensor* input,
                               const TfLiteEvalTensor* filter,
                               const TfLiteEvalTensor* bias,
                               TfLiteEvalTensor* output) {
  tflite::FullyConnectedParams op_params;
  op_params.input_offset = -data.input_zero_point;
  op_params.weights_offset = -data.filter_zero_point;
  op_params.output_offset = data.output_zero_point;
  op_params.output_multiplier = data.output_multiplier;
  // TODO(b/138810107): Figure out whether output shift should be inverted
  op_params.output_shift = -data.output_shift;
  op_params.quantized_activation_min = data.output_activation_min;
  op_params.quantized_activation_max = data.output_activation_max;

#ifdef NNLIB_HIFI5
  // TODO(pnikam-cad): remove this condition when all the testcases
  // have symmetric weights
  if (op_params.weights_offset == 0) {
    int ret, b, weight_depth, out_depth, batches;
    int8_t* p_out = tflite::micro::GetTensorData<int8_t>(output);
    weight_depth = tflite::micro::GetTensorShape(filter).Dims(
        tflite::micro::GetTensorShape(filter).DimensionsCount() - 1);
    out_depth = tflite::micro::GetTensorShape(output).Dims(
        tflite::micro::GetTensorShape(output).DimensionsCount() - 1);
    batches = FlatSizeSkipDim(
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorShape(output).DimensionsCount() - 1);

    for (b = 0; b < batches; b++) {
      ret = xa_nn_fully_connected_sym8sxasym8s_asym8s(
          (tflite::micro::GetTensorData<int8_t>(output) + b * out_depth),
          tflite::micro::GetTensorData<int8_t>(filter),
          (tflite::micro::GetTensorData<int8_t>(input) + b * weight_depth),
          tflite::micro::GetTensorData<int32_t>(bias), weight_depth, out_depth,
          op_params.input_offset, op_params.output_multiplier,
          op_params.output_shift, op_params.output_offset);

      CHECK_ERR_HIFI_NNLIB_KER(
          ret, "xa_nn_fully_connected_sym8sxasym8s_asym8s failed");
    }

    ret = xa_nn_vec_activation_min_max_8_8(
        p_out, p_out, data.output_activation_min, data.output_activation_max,
        batches * out_depth);

    CHECK_ERR_HIFI_NNLIB_KER(ret, "xa_nn_vec_activation_min_max_8_8 failed");
    return kTfLiteOk;
  }
#endif
  reference_integer_ops::FullyConnected(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           const OpData& data, const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* filter,
                           const TfLiteEvalTensor* bias,
                           TfLiteEvalTensor* output) {
  const int32_t input_offset = -data.input_zero_point;
  const int32_t filter_offset = -data.filter_zero_point;
  const int32_t output_offset = data.output_zero_point;

  tflite::FullyConnectedParams op_params;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data.output_multiplier;
  // Legacy ops used mixed left and right shifts. Now all are +ve-means-left.
  op_params.output_shift = -data.output_shift;
  op_params.quantized_activation_min = data.output_activation_min;
  op_params.quantized_activation_max = data.output_activation_max;

#define TF_LITE_FULLY_CONNECTED(output_data_type)      \
  reference_ops::FullyConnected(                       \
      op_params, tflite::micro::GetTensorShape(input), \
      tflite::micro::GetTensorData<uint8_t>(input),    \
      tflite::micro::GetTensorShape(filter),           \
      tflite::micro::GetTensorData<uint8_t>(filter),   \
      tflite::micro::GetTensorShape(bias),             \
      tflite::micro::GetTensorData<int32_t>(bias),     \
      tflite::micro::GetTensorShape(output),           \
      tflite::micro::GetTensorData<output_data_type>(output))
  switch (output->type) {
    case kTfLiteUInt8: {
      int ret, b, weight_depth, out_depth, batches;
      uint8_t* p_out = tflite::micro::GetTensorData<uint8_t>(output);
      weight_depth = tflite::micro::GetTensorShape(filter).Dims(
          tflite::micro::GetTensorShape(filter).DimensionsCount() - 1);
      out_depth = tflite::micro::GetTensorShape(output).Dims(
          tflite::micro::GetTensorShape(output).DimensionsCount() - 1);
      batches = FlatSizeSkipDim(
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorShape(output).DimensionsCount() - 1);
      for (b = 0; b < batches; b++) {
        ret = xa_nn_fully_connected_asym8xasym8_asym8(
            (tflite::micro::GetTensorData<uint8_t>(output) + b * out_depth),
            tflite::micro::GetTensorData<uint8_t>(filter),
            (tflite::micro::GetTensorData<uint8_t>(input) + b * weight_depth),
            tflite::micro::GetTensorData<int32_t>(bias), weight_depth,
            out_depth, op_params.input_offset, op_params.weights_offset,
            op_params.output_multiplier, op_params.output_shift,
            op_params.output_offset);
        CHECK_ERR_HIFI_NNLIB_KER(
            ret, "xa_nn_fully_connected_asym8xasym8_asym8 failed");
      }
      ret = xa_nn_vec_activation_min_max_asym8_asym8(
          p_out, p_out, data.output_activation_min, data.output_activation_max,
          batches * out_depth);

      CHECK_ERR_HIFI_NNLIB_KER(
          ret, "xa_nn_vec_activation_min_max_asym8_asym8 failed");
      break;
    }
    case kTfLiteInt16:
      TF_LITE_FULLY_CONNECTED(int16_t);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(output->type), output->type);
      return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFusedActivation activation,
                       const TfLiteEvalTensor* input,
                       const TfLiteEvalTensor* filter,
                       const TfLiteEvalTensor* bias, TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(activation, &output_activation_min,
                           &output_activation_max);
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
#if HIFI_VFPU && !defined NNLIB_HIFI5
  int ret, b, weight_depth, out_depth, batches;
  weight_depth = tflite::micro::GetTensorShape(filter).Dims(
      tflite::micro::GetTensorShape(filter).DimensionsCount() - 1);
  out_depth = tflite::micro::GetTensorShape(output).Dims(
      tflite::micro::GetTensorShape(output).DimensionsCount() - 1);
  batches = FlatSizeSkipDim(
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorShape(output).DimensionsCount() - 1);

  for (b = 0; b < batches; b++) {
    ret = xa_nn_fully_connected_f32(
        (tflite::micro::GetTensorData<float>(output) + b * out_depth),
        tflite::micro::GetTensorData<float>(filter),
        (tflite::micro::GetTensorData<float>(input) + b * weight_depth),
        tflite::micro::GetTensorData<float>(bias), weight_depth, out_depth);
    CHECK_ERR_HIFI_NNLIB_KER(ret, "xa_nn_fully_connected_f32 failed.");
  }
  float* p_out = tflite::micro::GetTensorData<float>(output);
  ret = xa_nn_vec_activation_min_max_f32_f32(
      p_out, p_out, output_activation_min, output_activation_max,
      batches * out_depth);
  CHECK_ERR_HIFI_NNLIB_KER(ret, "xa_nn_vec_activation_min_max_f32_f32 failed");
#else
  tflite::reference_ops::FullyConnected(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<float>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<float>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<float>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<float>(output));
#endif /* HIFI_VFPU */
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpData& data = *(static_cast<const OpData*>(node->user_data));

  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    case kTfLiteFloat32:
      return EvalFloat(context, node, params->activation, input, filter, bias,
                       output);
    case kTfLiteInt8:
      return EvalQuantizedInt8(context, node, data, input, filter, bias,
                               output);

    case kTfLiteUInt8:
      return EvalQuantized(context, node, data, input, filter, bias, output);

    default:
      TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                         TfLiteTypeGetName(input->type), input->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_FULLY_CONNECTED() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
