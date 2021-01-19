/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"

#include "CEVA_TFLM_lib.h"
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#ifdef MCPS_MEASUREMENT
#include "mcps_macros.h"
#endif

#ifndef ORIGINAL_IMPLEMENTATION
extern int32_t* CEVA_TFLM_KERNELS_SCRATCH;
extern int32_t CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL;
#endif  // ORIGINAL_IMPLEMENTATION

namespace tflite {
namespace {

#if defined(CEVA_BX1) || defined(CEVA_SP500)
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
#endif

#if defined(CEVA_BX1) || defined(CEVA_SP500)
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
#endif

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
#if defined(CEVA_BX1) || defined(CEVA_SP500)
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
#else
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataFullyConnected));
#endif
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);
#if defined(CEVA_BX1) || defined(CEVA_SP500)
  OpData* data = static_cast<OpData*>(node->user_data);
#else
  auto* data = static_cast<OpDataFullyConnected*>(node->user_data);
#endif
  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  return CalculateOpData(context, params->activation, input->type, input,
                         filter, bias, output, data);
}

TfLiteStatus EvalQuantizedInt8CEVA(TfLiteContext* context, TfLiteNode* node,
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

#ifdef ORIGINAL_IMPLEMENTATION

  reference_integer_ops::FullyConnected(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<int8_t>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<int8_t>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<int32_t>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<int8_t>(output));
#else  // ORIGINAL_IMPLEMENTATION
  int input_shape_DimensionsCount =
      tflite::micro::GetTensorShape(input).DimensionsCount();
  int weights_shape_DimensionsCount =
      tflite::micro::GetTensorShape(filter).DimensionsCount();
  int* weights_shape_DimsData =
      (int*)tflite::micro::GetTensorShape(filter).DimsData();
  int bias_shape_DimensionsCount =
      tflite::micro::GetTensorShape(bias).DimensionsCount();
  int output_shape_DimensionsCount =
      tflite::micro::GetTensorShape(output).DimensionsCount();
  int* output_shape_DimsData =
      (int*)tflite::micro::GetTensorShape(output).DimsData();

  void* params = (void*)&op_params;
  int8_t* inputp = (int8_t*)tflite::micro::GetTensorData<int8_t>(input);
  int8_t* filterp = (int8_t*)tflite::micro::GetTensorData<int8_t>(filter);
  int32_t* biasp = (int32_t*)tflite::micro::GetTensorData<int32_t>(bias);
  int8_t* outputp = (int8_t*)tflite::micro::GetTensorData<int8_t>(output);
#ifdef MCPS_MEASUREMENT
  int batches = output_shape_DimsData[0];
  int output_depth = weights_shape_DimsData[weights_shape_DimensionsCount - 2];
  int accum_depth = weights_shape_DimsData[weights_shape_DimensionsCount - 1];
  MCPS_START_ONE;
#endif

  int sizeof_scr = output_shape_DimsData[1];
  //  int32_t *scratch = (int32_t *)malloc(sizeof(int32_t)*sizeof_scr);
  if (sizeof_scr > CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL) {
    TF_LITE_KERNEL_LOG(context, "Scratch size (%d) less that required (%d)",
                       CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL, sizeof_scr);
    //	  return kTfLiteError;
  }

  CEVA_TFLM_FullyConnected_int8(
      params,
      input_shape_DimensionsCount,  // GetTensorShape(input),
      inputp,
      weights_shape_DimensionsCount,  // GetTensorShape(filter),
      weights_shape_DimsData, filterp,
      bias_shape_DimensionsCount,  // GetTensorShape(bias),
      biasp,
      output_shape_DimensionsCount,  // GetTensorShape(output),
      output_shape_DimsData, outputp, CEVA_TFLM_KERNELS_SCRATCH);
#ifdef MCPS_MEASUREMENT
  MCPS_STOP_ONE(
      "Test params:Call CEVA_TFLM_FullyConnected_int8 inetrnal loop = %dx%dx%d",
      batches, output_depth, accum_depth);
#endif

#endif  // ORIGINAL_IMPLEMENTATION

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
    case kTfLiteUInt8:
      TF_LITE_FULLY_CONNECTED(uint8_t);
      break;
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

TfLiteStatus EvalFloatCEVA(TfLiteContext* context, TfLiteNode* node,
                           TfLiteFusedActivation activation,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* filter,
                           const TfLiteEvalTensor* bias,
                           TfLiteEvalTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(activation, &output_activation_min,
                           &output_activation_max);
  tflite::FullyConnectedParams op_params;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;
#ifdef ORIGINAL_IMPLEMENTATION
  tflite::reference_ops::FullyConnected(
      op_params, tflite::micro::GetTensorShape(input),
      tflite::micro::GetTensorData<float>(input),
      tflite::micro::GetTensorShape(filter),
      tflite::micro::GetTensorData<float>(filter),
      tflite::micro::GetTensorShape(bias),
      tflite::micro::GetTensorData<float>(bias),
      tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<float>(output));
#else  // ORIGINAL_IMPLEMENTATION
  int input_shape_DimensionsCount =
      tflite::micro::GetTensorShape(input).DimensionsCount();
  int weights_shape_DimensionsCount =
      tflite::micro::GetTensorShape(filter).DimensionsCount();
  int* weights_shape_DimsData =
      (int*)tflite::micro::GetTensorShape(filter).DimsData();
  int bias_shape_DimensionsCount =
      tflite::micro::GetTensorShape(bias).DimensionsCount();
  int output_shape_DimensionsCount =
      tflite::micro::GetTensorShape(output).DimensionsCount();
  int* output_shape_DimsData =
      (int*)tflite::micro::GetTensorShape(output).DimsData();

  void* params = (void*)&op_params;
  float* inputp = (float*)tflite::micro::GetTensorData<float>(input);
  float* filterp = (float*)tflite::micro::GetTensorData<float>(filter);
  float* biasp = (float*)tflite::micro::GetTensorData<float>(bias);
  float* outputp = (float*)tflite::micro::GetTensorData<float>(output);

#ifdef MCPS_MEASUREMENT
  int batches = 1;
  int i;
  for (i = 0; i < (output_shape_DimensionsCount - 1); i++)
    batches *= output_shape_DimsData[i];

  int output_depth = weights_shape_DimsData[weights_shape_DimensionsCount - 2];
  int accum_depth = weights_shape_DimsData[weights_shape_DimensionsCount - 1];
  MCPS_START_ONE;
#endif
  CEVA_TFLM_FullyConnected_Float32(
      params,
      input_shape_DimensionsCount,  // GetTensorShape(input),
      inputp,
      weights_shape_DimensionsCount,  // GetTensorShape(filter),
      weights_shape_DimsData, filterp,
      bias_shape_DimensionsCount,  // GetTensorShape(bias),
      biasp,
      output_shape_DimensionsCount,  // GetTensorShape(output),
      output_shape_DimsData, outputp);
#ifdef MCPS_MEASUREMENT
  MCPS_STOP_ONE(
      "Test params:Call CEVA_TFLM_FullyConnected_Float32 inetrnal loop = "
      "%dx%dx%d",
      batches, output_depth, accum_depth);
#endif

#endif  // ORIGINAL_IMPLEMENTATION

  return kTfLiteOk;
}

TfLiteStatus EvalCEVA(TfLiteContext* context, TfLiteNode* node) {
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
      return EvalFloatCEVA(context, node, params->activation, input, filter,
                           bias, output);
    case kTfLiteInt8:
      return EvalQuantizedInt8CEVA(context, node, data, input, filter, bias,
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
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
#if defined(CEVA_BX1) || defined(CEVA_SP500)
  return EvalCEVA(context, node);
#else
  return EvalQuantizeReference(context, node);
#endif
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
