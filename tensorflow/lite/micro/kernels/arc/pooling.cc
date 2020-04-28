/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/pooling.h"

#include "mli_api.h"  // NOLINT
#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/micro/kernels/arc/mli_tf_utils.h"

namespace tflite {
namespace ops {
namespace micro {
namespace pooling {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

struct OpData {
  TfLitePaddingValues padding;
};

TfLiteStatus CalculateOpData(const TfLiteContext* context,
                             const TfLitePoolParams* params,
                             const TfLiteTensor* input,
                             const TfLiteTensor* output, OpData* data) {
  // input: batch, height, width, channel
  int height = SizeOfDimension(input, 1);
  int width = SizeOfDimension(input, 2);

  int out_height, out_width;

  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width,
      /*dilation_rate_height=*/1,
      /*dilation_rate_width=*/1, height, width, params->filter_height,
      params->filter_width, params->padding, &out_height, &out_width);

  return kTfLiteOk;
}

void AverageEvalFloat(const TfLiteContext* context, const TfLiteNode* node,
                      const TfLitePoolParams* params, const OpData* data,
                      const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = activation_min;
  op_params.float_activation_max = activation_max;
  reference_ops::AveragePool(
      op_params, GetTensorShape(input), GetTensorData<float>(input),
      GetTensorShape(output), GetTensorData<float>(output));
}

void AverageEvalUint8(TfLiteContext* context, const TfLiteNode* node,
                      const TfLitePoolParams* params, const OpData* data,
                      const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min, activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);

  PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = activation_min;
  op_params.quantized_activation_max = activation_max;
  reference_ops::AveragePool(
      op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
      GetTensorShape(output), GetTensorData<uint8_t>(output));
}

void AverageEvalInt8(TfLiteContext* context, const TfLiteNode* node,
                     const TfLitePoolParams* params, const OpData* data,
                     const TfLiteTensor* input, TfLiteTensor* output) {
  // Run Average Pooling MLI kernel
  // MLI optimized version only supports int8 dataype and no fused Relu
  // TODO: subject to add mli_saturate kernel
  if (input->type == kTfLiteInt8 && params->activation == kTfLiteActNone) {
    mli_tensor mli_in = {0};
    mli_tensor mli_out = {0};
    mli_pool_cfg cfg = {0};

    ConvertToMliTensor<int8_t>(input, &mli_in);
    ConvertToMliTensor<int8_t>(output, &mli_out);

    cfg.kernel_width = params->filter_width;
    cfg.kernel_height = params->filter_height;
    cfg.stride_width = params->stride_width;
    cfg.stride_height = params->stride_height;

    if (params->padding == kTfLitePaddingValid) {
      cfg.padding_left = 0;
      cfg.padding_right = 0;
      cfg.padding_top = 0;
      cfg.padding_bottom = 0;
    } else {
      cfg.padding_left = data->padding.width;
      cfg.padding_right = data->padding.width + data->padding.width_offset;
      cfg.padding_top = data->padding.height;
      cfg.padding_bottom = data->padding.height + data->padding.height_offset;
    }

    mli_point_to_subtsr_cfg substr_cfg_in = {
        {0, 0}, 2, static_cast<uint8_t>(mli_in.shape[1])};
    mli_point_to_subtsr_cfg substr_cfg_out = {
        {0, 0}, 2, static_cast<uint8_t>(mli_out.shape[1])};
    mli_tensor sub_mli_in = {0};
    mli_tensor sub_mli_out = {0};

    const int batches =
        MatchingDim(GetTensorShape(input), 0, GetTensorShape(output), 0);

    for (int i = 0; i < batches; i++) {
      substr_cfg_in.start_coord[0] = i;
      substr_cfg_out.start_coord[0] = i;
      mli_hlp_point_to_subtensor(&mli_in, &substr_cfg_in, &sub_mli_in);
      mli_hlp_point_to_subtensor(&mli_out, &substr_cfg_out, &sub_mli_out);

      mli_krn_avepool_hwc_sa8(&sub_mli_in, &cfg, &sub_mli_out);
    }
  } else {
    int32_t activation_min, activation_max;
    (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                            &activation_min, &activation_max);
    PoolParams op_params;
    op_params.stride_height = params->stride_height;
    op_params.stride_width = params->stride_width;
    op_params.filter_height = params->filter_height;
    op_params.filter_width = params->filter_width;
    op_params.padding_values.height = data->padding.height;
    op_params.padding_values.width = data->padding.width;
    op_params.quantized_activation_min = activation_min;
    op_params.quantized_activation_max = activation_max;
    reference_integer_ops::AveragePool(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(output), GetTensorData<int8_t>(output));
  }
}

void MaxEvalFloat(TfLiteContext* context, TfLiteNode* node,
                  TfLitePoolParams* params, OpData* data,
                  const TfLiteTensor* input, TfLiteTensor* output) {
  float activation_min, activation_max;
  CalculateActivationRange(params->activation, &activation_min,
                           &activation_max);

  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.float_activation_min = activation_min;
  op_params.float_activation_max = activation_max;
  reference_ops::MaxPool(op_params, GetTensorShape(input),
                         GetTensorData<float>(input), GetTensorShape(output),
                         GetTensorData<float>(output));
}

void MaxEvalQuantizedUInt8(TfLiteContext* context, TfLiteNode* node,
                           TfLitePoolParams* params, OpData* data,
                           const TfLiteTensor* input, TfLiteTensor* output) {
  int32_t activation_min, activation_max;
  (void)CalculateActivationRangeQuantized(context, params->activation, output,
                                          &activation_min, &activation_max);

  tflite::PoolParams op_params;
  op_params.stride_height = params->stride_height;
  op_params.stride_width = params->stride_width;
  op_params.filter_height = params->filter_height;
  op_params.filter_width = params->filter_width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width = data->padding.width;
  op_params.quantized_activation_min = activation_min;
  op_params.quantized_activation_max = activation_max;
  reference_ops::MaxPool(op_params, GetTensorShape(input),
                         GetTensorData<uint8_t>(input), GetTensorShape(output),
                         GetTensorData<uint8_t>(output));
}

}  // namespace

TfLiteStatus AverageEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData data;

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, &data));

  // Inputs and outputs share the same type, guarenteed by the converter.
  switch (input->type) {
    case kTfLiteFloat32:
      AverageEvalFloat(context, node, params, &data, input, output);
      break;
    case kTfLiteUInt8:
      AverageEvalUint8(context, node, params, &data, input, output);
      break;
    case kTfLiteInt8:
      AverageEvalInt8(context, node, params, &data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Input type %s is not currently supported",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus MaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLitePoolParams*>(node->builtin_data);
  OpData data;

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_STATUS(CalculateOpData(context, params, input, output, &data));

  switch (input->type) {
    case kTfLiteFloat32:
      MaxEvalFloat(context, node, params, &data, input, output);
      break;
    case kTfLiteUInt8:
      MaxEvalQuantizedUInt8(context, node, params, &data, input, output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s not currently supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace pooling

TfLiteRegistration* Register_AVERAGE_POOL_2D() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/nullptr,
                                 /*invoke=*/pooling::AverageEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

TfLiteRegistration* Register_MAX_POOL_2D() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/nullptr,
                                 /*invoke=*/pooling::MaxEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
