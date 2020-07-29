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

#include "tensorflow/lite/kernels/internal/reference/quantize.h"

#include <xtensa/tie/xt_hifi2.h>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa_hifimini_legacy/fixedpoint_utils.h"

namespace tflite {
namespace ops {
namespace micro {

namespace xtensa {
namespace hifimini {

void AffineQuantize(int scale_multiplier,
                    const tflite::QuantizationParams& op_params,
                    const RuntimeShape& input_shape, const int16_t* input_data,
                    const RuntimeShape& output_shape, int8_t* output_data) {
  const int32 zero_point = op_params.zero_point;
  const int flat_size = MatchingFlatSize(input_shape, output_shape);
  ae_q56s min_val_56 = AE_CVTQ48A32S(INT16_MIN);
  ae_q56s max_val_56 = AE_CVTQ48A32S(INT16_MAX);
  ae_q56s zero_point_56 = AE_CVTQ48A32S(zero_point);

  const ae_p16x2s* input_data_ptr = (const ae_p16x2s*)(input_data - 2);

  ae_p24x2s scale_multiplier_24x2 = AE_MOVPA24(scale_multiplier);

  int iters = flat_size / 2;
  for (int i = 0; i < iters; i++) {
    // Load two 16bit pairs into the 2x24bit register PR:
    // Values need to be right shifted 8 bits to align from upper 16bits to a
    // 24bit value:
    ae_p24x2s inputs_24x2;
    AE_LP16X2F_IU(inputs_24x2, input_data_ptr, 4);
    inputs_24x2 = AE_P24X2S_SRAI(inputs_24x2, 8);

    // Q0.23 * Q16.0 == Q16.23
    {
      ae_q56s sum_56 = AE_MULP24S_HH(scale_multiplier_24x2, inputs_24x2);

      // Q16.23 -> Q16.0
      // Shift right only 7 bits (23 - 16). This truncated shift aligns the
      // 16bit value at the truncation line for 32bit in the QR register. The
      // lower 16 bits will be used for rounding in AE_ROUNDSQ32SYM.
      sum_56 = AE_Q56S_SRAI(sum_56, 7);

      // Round and truncate 32 bits
      sum_56 = AE_ROUNDSQ32SYM(sum_56);

      // Add offset (zero_point_56 is already aligned at 32bits.
      sum_56 = AE_ADDQ56(sum_56, zero_point_56);

      // Saturate:
      sum_56 = AE_MINQ56S(sum_56, max_val_56);
      sum_56 = AE_MAXQ56S(sum_56, min_val_56);

      output_data[i * 2] = static_cast<int16_t>(AE_TRUNCA32Q48(sum_56));
    }
    {
      ae_q56s sum_56 = AE_MULP24S_LL(scale_multiplier_24x2, inputs_24x2);

      // Q16.23 -> Q16.0
      // Shift right only 7 bits (23 - 16). This truncated shift aligns the
      // 16bit value at the truncation line for 32bit in the QR register. The
      // lower 16 bits will be used for rounding in AE_ROUNDSQ32SYM.
      sum_56 = AE_Q56S_SRAI(sum_56, 23 - 16);

      // Round and truncate 32 bits
      sum_56 = AE_ROUNDSQ32SYM(sum_56);

      // Add offset (zero_point_56 is already aligned at 32bits.
      sum_56 = AE_ADDQ56(sum_56, zero_point_56);

      // Saturate:
      sum_56 = AE_MINQ56S(sum_56, max_val_56);
      sum_56 = AE_MAXQ56S(sum_56, min_val_56);

      output_data[i * 2 + 1] = static_cast<int16_t>(AE_TRUNCA32Q48(sum_56));
    }
  }
}

}  // namespace hifimini
}  // namespace xtensa

namespace quantize {

struct OpData {
  int scale_multiplier = 0;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  void* data = nullptr;
  if (context->AllocatePersistentBuffer(context, sizeof(OpData), &data) ==
      kTfLiteError) {
    return nullptr;
  }
  return data;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* op_data = static_cast<OpData*>(node->user_data);

  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);

  // TODO(b/155682734): Fix dangerous input/output scale ratio assumptions.
  op_data->scale_multiplier = xtensa::hifimini::CreateQConstantForInt24(
      0, input->params.scale / output->params.scale);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
  auto* op_data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  tflite::QuantizationParams op_params;
  op_params.zero_point = output->params.zero_point;

  if (input->type != kTfLiteInt16 && output->type != kTfLiteInt8) {
    TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                       TfLiteTypeGetName(input->type),
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }

  xtensa::hifimini::AffineQuantize(
      op_data->scale_multiplier, op_params, GetTensorShape(input),
      GetTensorData<int16_t>(input), GetTensorShape(output),
      GetTensorData<int8_t>(output));
  return kTfLiteOk;
}

}  // namespace quantize

// This Op (QUANTIZE) quantizes the input and produces quantized output.
// AffineQuantize takes scale and zero point and quantizes the float value to
// quantized output, in int8 or uint8 format.
TfLiteRegistration* Register_QUANTIZE() {
  static TfLiteRegistration r = {/*init=*/quantize::Init,
                                 /*free=*/nullptr,
                                 /*prepare=*/quantize::Prepare,
                                 /*invoke=*/quantize::Eval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
