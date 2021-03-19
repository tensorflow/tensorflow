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

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/fully_connected.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/ceva/ceva_tflm_lib.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
//#define MCPS_MEASUREMENT
#ifdef MCPS_MEASUREMENT
#include "tensorflow/lite/micro/kernels/ceva/mcps_macros.h"
#endif

#if defined(CEVA_BX1) || defined(CEVA_SP500)
extern int32_t* CEVA_TFLM_KERNELS_SCRATCH;
extern int32_t CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL;
#endif  // CEVA platform

namespace tflite {
namespace {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);

  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataFullyConnected));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  TFLITE_DCHECK(node->builtin_data != nullptr);

  auto* data = static_cast<OpDataFullyConnected*>(node->user_data);

  const auto params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteTensor* input =
      GetInput(context, node, kFullyConnectedInputTensor);
  const TfLiteTensor* filter =
      GetInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteTensor* bias =
      GetOptionalInputTensor(context, node, kFullyConnectedBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kFullyConnectedOutputTensor);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);
  TF_LITE_ENSURE_MSG(context, input->type == filter->type,
                     "Hybrid models are not supported on TFLite Micro.");

  return CalculateOpDataFullyConnected(context, params->activation, input->type,
                                       input, filter, bias, output, data);
}

TfLiteStatus EvalQuantizedInt8CEVA(TfLiteContext* context, TfLiteNode* node,
                                   const OpDataFullyConnected& data,
                                   const TfLiteEvalTensor* input,
                                   const TfLiteEvalTensor* filter,
                                   const TfLiteEvalTensor* bias,
                                   TfLiteEvalTensor* output) {
  tflite::FullyConnectedParams op_params = FullyConnectedParamsQuantized(data);

  int input_shape_dimensions_count =
      tflite::micro::GetTensorShape(input).DimensionsCount();
  int weights_shape_dimensions_count =
      tflite::micro::GetTensorShape(filter).DimensionsCount();
  int* weights_shape_dims_data =
      const_cast<int*>(tflite::micro::GetTensorShape(filter).DimsData());
  int bias_shape_dimensions_count =
      tflite::micro::GetTensorShape(bias).DimensionsCount();
  int output_shape_dimensions_count =
      tflite::micro::GetTensorShape(output).DimensionsCount();
  int* output_shape_dims_data =
      const_cast<int*>(tflite::micro::GetTensorShape(output).DimsData());

  void* params = (void*)&op_params;
  int8_t* inputp =
      const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(input));
  int8_t* filterp =
      const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(filter));
  int32_t* biasp =
      const_cast<int32_t*>(tflite::micro::GetTensorData<int32_t>(bias));
  int8_t* outputp =
      const_cast<int8_t*>(tflite::micro::GetTensorData<int8_t>(output));

#ifdef MCPS_MEASUREMENT
  int batches = output_shape_dims_data[0];
  int output_depth =
      weights_shape_dims_data[weights_shape_dimensions_count - 2];
  int accum_depth = weights_shape_dims_data[weights_shape_dimensions_count - 1];
  MCPS_START_ONE;
#endif

  int sizeof_scratch_required = output_shape_dims_data[1];

  if (sizeof_scratch_required > CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL) {
    TF_LITE_KERNEL_LOG(context, "Scratch size (%d) less that required (%d)",
                       CEVA_TFLM_KERNELS_SCRATCH_SIZE_VAL,
                       sizeof_scratch_required);
    return kTfLiteError;
  }

  CEVA_TFLM_FullyConnected_int8(
      params, input_shape_dimensions_count, inputp,
      weights_shape_dimensions_count, weights_shape_dims_data, filterp,
      bias_shape_dimensions_count, biasp, output_shape_dimensions_count,
      output_shape_dims_data, outputp, CEVA_TFLM_KERNELS_SCRATCH);
#ifdef MCPS_MEASUREMENT
  MCPS_STOP_ONE(
      "Test params:Call CEVA_TFLM_FullyConnected_int8 inetrnal loop = %dx%dx%d",
      batches, output_depth, accum_depth);
#endif

  return kTfLiteOk;
}

TfLiteStatus EvalFloatCEVA(TfLiteContext* context, TfLiteNode* node,
                           TfLiteFusedActivation activation,
                           const TfLiteEvalTensor* input,
                           const TfLiteEvalTensor* filter,
                           const TfLiteEvalTensor* bias,
                           TfLiteEvalTensor* output) {
  // float output_activation_min, output_activation_max;
  tflite::FullyConnectedParams op_params;
  CalculateActivationRange(activation, &op_params.float_activation_min,
                           &op_params.float_activation_max);

  // op_params.float_activation_min = output_activation_min;
  // op_params.float_activation_max = output_activation_max;

  int input_shape_dimensions_count =
      tflite::micro::GetTensorShape(input).DimensionsCount();
  int weights_shape_dimensions_count =
      tflite::micro::GetTensorShape(filter).DimensionsCount();
  int* weights_shape_dims_data =
      const_cast<int*>(tflite::micro::GetTensorShape(filter).DimsData());
  int bias_shape_dimensions_count =
      tflite::micro::GetTensorShape(bias).DimensionsCount();
  int output_shape_dimensions_count =
      tflite::micro::GetTensorShape(output).DimensionsCount();
  int* output_shape_dims_data =
      const_cast<int*>(tflite::micro::GetTensorShape(output).DimsData());

  void* params = (void*)&op_params;
  float* inputp =
      const_cast<float*>(tflite::micro::GetTensorData<float>(input));
  float* filterp =
      const_cast<float*>(tflite::micro::GetTensorData<float>(filter));
  float* biasp = const_cast<float*>(tflite::micro::GetTensorData<float>(bias));
  float* outputp =
      const_cast<float*>(tflite::micro::GetTensorData<float>(output));

#ifdef MCPS_MEASUREMENT
  int batches = 1;
  int i;
  for (i = 0; i < (output_shape_dimensions_count - 1); i++)
    batches *= output_shape_dims_data[i];

  int output_depth =
      weights_shape_dims_data[weights_shape_dimensions_count - 2];
  int accum_depth = weights_shape_dims_data[weights_shape_dimensions_count - 1];
  MCPS_START_ONE;
#endif
  CEVA_TFLM_FullyConnected_Float32(
      params,
      input_shape_dimensions_count,  // GetTensorShape(input),
      inputp,
      weights_shape_dimensions_count,  // GetTensorShape(filter),
      weights_shape_dims_data, filterp,
      bias_shape_dimensions_count,  // GetTensorShape(bias),
      biasp,
      output_shape_dimensions_count,  // GetTensorShape(output),
      output_shape_dims_data, outputp);
#ifdef MCPS_MEASUREMENT
  MCPS_STOP_ONE(
      "Test params:Call CEVA_TFLM_FullyConnected_Float32 inetrnal loop = "
      "%dx%dx%d",
      batches, output_depth, accum_depth);
#endif

  return kTfLiteOk;
}

TfLiteStatus EvalCEVA(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->builtin_data != nullptr);
  const auto* params =
      static_cast<const TfLiteFullyConnectedParams*>(node->builtin_data);

  const TfLiteEvalTensor* input =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedInputTensor);
  const TfLiteEvalTensor* filter =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedWeightsTensor);
  const TfLiteEvalTensor* bias =
      tflite::micro::GetEvalInput(context, node, kFullyConnectedBiasTensor);
  TfLiteEvalTensor* output =
      tflite::micro::GetEvalOutput(context, node, kFullyConnectedOutputTensor);

  TFLITE_DCHECK(node->user_data != nullptr);
  const OpDataFullyConnected& data =
      *(static_cast<const OpDataFullyConnected*>(node->user_data));

  // Checks in Prepare ensure input, output and filter types are all the same.
  switch (input->type) {
    case kTfLiteFloat32:
      return EvalFloatCEVA(context, node, params->activation, input, filter,
                           bias, output);
    case kTfLiteInt8:
      return EvalQuantizedInt8CEVA(context, node, data, input, filter, bias,
                                   output);

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
