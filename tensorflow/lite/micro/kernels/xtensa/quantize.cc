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

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/quantize.h"
#include "tensorflow/lite/kernels/internal/reference/requantize.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/quantize.h"
#include "tensorflow/lite/micro/kernels/xtensa/fixedpoint_utils.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"
#include "tensorflow/lite/micro/micro_utils.h"

namespace tflite {
namespace {

#if defined(HIFIMINI)
struct OpData {
  int32_t zero_point = 0;
  int scale_multiplier = 0;

  // Use 32-bit multiplier and scale for requantize version of this operator
  // to preserve compatibility with reference op.
  int32_t requantize_output_multiplier;
  int requantize_output_shift;
  int32_t input_zero_point = 0;
};

void AffineQuantize(int scale_multiplier, const int32_t zero_point,
                    const RuntimeShape& input_shape, const int16_t* input_data,
                    const RuntimeShape& output_shape, int8_t* output_data) {
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

#endif  // defined(HIFIMINI)

#if defined(HIFIMINI) || defined(FUSION_F1)
TfLiteStatus EvalXtensa(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);
#if defined(HIFIMINI)
  auto* op_data = static_cast<OpData*>(node->user_data);
#elif defined(FUSION_F1)
  auto* op_data = static_cast<OpDataQuantizeReference*>(node->user_data);
#endif

  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);

  if (output->type == kTfLiteInt8 && input->type == kTfLiteInt16) {
#if defined(HIFIMINI)
    AffineQuantize(op_data->scale_multiplier, op_data->zero_point,
                   tflite::micro::GetTensorShape(input),
                   tflite::micro::GetTensorData<int16_t>(input),
                   tflite::micro::GetTensorShape(output),
                   tflite::micro::GetTensorData<int8_t>(output));
#elif defined(FUSION_F1)
    int size = ElementCount(*input->dims);
    TF_LITE_ENSURE_EQ(
        context,
        xa_nn_elm_quantize_asym16s_asym8s(
            tflite::micro::GetTensorData<int8_t>(output),
            tflite::micro::GetTensorData<int16_t>(input),
            op_data->input_zero_point, op_data->quantization_params.zero_point,
            op_data->requantize_output_shift,
            op_data->requantize_output_multiplier, size),
        0);
#else
    static_assert(false, "Unsupported xtensa architecture.");
#endif
  } else if (output->type == kTfLiteInt32 && input->type == kTfLiteInt16) {
    int size = ElementCount(*input->dims);

    // This ifdef is only needed because the hifimini code is not following the
    // convention of the rest of the codebase. Ideally we would be using the
    // same structs as much as possible and reduce the need for such ifdefs.
#if defined(HIFIMINI)
    int32_t zero_point = op_data->zero_point;
#elif defined(FUSION_F1)
    int32_t zero_point = op_data->quantization_params.zero_point;
#endif
    reference_ops::Requantize(tflite::micro::GetTensorData<int16_t>(input),
                              size, op_data->requantize_output_multiplier,
                              op_data->requantize_output_shift,
                              op_data->input_zero_point, zero_point,
                              tflite::micro::GetTensorData<int32_t>(output));
  } else {
    TF_LITE_KERNEL_LOG(context, "Input %s, output %s not supported.",
                       TfLiteTypeGetName(input->type),
                       TfLiteTypeGetName(output->type));
    return kTfLiteError;
  }
  return kTfLiteOk;
}
#endif  // defined(HIFIMINI) || defined(FUSION_F1)

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
#if defined(HIFIMINI)
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
#else
  return context->AllocatePersistentBuffer(context,
                                           sizeof(OpDataQuantizeReference));
#endif
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TFLITE_DCHECK(node->user_data != nullptr);

  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* input = GetInput(context, node, 0);

#if defined(HIFIMINI)
  auto* op_data = static_cast<OpData*>(node->user_data);
  // TODO(b/155682734): Fix dangerous input/output scale ratio assumptions.
  op_data->scale_multiplier =
      CreateQConstantForInt24(0, input->params.scale / output->params.scale);
  op_data->zero_point = output->params.zero_point;
#else
  auto* op_data = static_cast<OpDataQuantizeReference*>(node->user_data);
  op_data->quantization_params.zero_point = output->params.zero_point;
  op_data->quantization_params.scale =
      static_cast<double>(output->params.scale);
#endif

  op_data->input_zero_point = input->params.zero_point;

  double effective_scale = static_cast<double>(input->params.scale) /
                           static_cast<double>(output->params.scale);
  QuantizeMultiplier(effective_scale, &op_data->requantize_output_multiplier,
                     &op_data->requantize_output_shift);

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFIMINI) || defined(FUSION_F1)
  return EvalXtensa(context, node);
#else
  return EvalQuantizeReference(context, node);
#endif
}

}  // namespace

TfLiteRegistration Register_QUANTIZE() {
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
