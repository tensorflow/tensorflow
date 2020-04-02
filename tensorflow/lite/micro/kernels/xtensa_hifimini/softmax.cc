/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/softmax.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {

namespace xtensa {
namespace hifimini {

// Quantized softmax with int8 input and int8/int16 output.
template <typename OutputT = int8_t>
inline void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const int8* input_data,
                    const RuntimeShape& output_shape, OutputT* output_data) {
  const int32_t input_beta_multiplier = params.input_multiplier;
  const int32_t input_beta_left_shift = params.input_left_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32_t, kScaledDiffIntegerBits>;
  using FixedPointAccum =
      gemmlowp::FixedPoint<int32_t, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32_t, 0>;

  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    int8 max_in_row = -128;
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, input_data[i * depth + c]);
    }

    FixedPointAccum sum_of_exps = FixedPointAccum::Zero();
    for (int c = 0; c < depth; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);
        sum_of_exps = sum_of_exps + gemmlowp::Rescale<kAccumulationIntegerBits>(
                                        exp_on_negative_values(scaled_diff_f8));
      }
    }

    int num_bits_over_unit;
    FixedPoint0 shifted_scale = FixedPoint0::FromRaw(GetReciprocal(
        sum_of_exps.raw(), kAccumulationIntegerBits, &num_bits_over_unit));

    for (int c = 0; c < depth; ++c) {
      int32_t input_diff =
          static_cast<int32_t>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32_t input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);

        FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
        const int32_t unsat_output = gemmlowp::RoundingDivideByPOT(
            (shifted_scale * exp_in_0).raw(),
            num_bits_over_unit + 31 - (sizeof(OutputT) * 8));
        // TODO(b/148494470): Handle int32 shifts properly:
        const int32_t shifted_output =
            unsat_output -
            (static_cast<int32_t>(std::numeric_limits<OutputT>::max()) + 1);
        output_data[i * depth + c] = static_cast<OutputT>(std::max(
            std::min(shifted_output,
                     static_cast<int32_t>(std::numeric_limits<OutputT>::max())),
            static_cast<int32_t>(std::numeric_limits<OutputT>::min())));
      } else {
        output_data[i * depth + c] = std::numeric_limits<OutputT>::min();
      }
    }
  }
}

}  // namespace hifimini
}  // namespace xtensa

namespace activations {
namespace {

// This size will work for both the hotword (1) and ambient music (0):
static SoftmaxParams kStaticOpData;

TfLiteStatus CalculateSoftmaxOpData(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    TfLiteTensor* output,
                                    const TfLiteSoftmaxParams* params,
                                    SoftmaxParams* op_data) {
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    } else {
      if (output->type == kTfLiteInt16) {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, -32768);
        // NOTE: Current int16 softmax output does not require symmetric scaling
        // - so no need to verify scale here.
      } else {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
        TF_LITE_ENSURE(context, output->params.scale == 1.f / 256);
      }
    }

    static const int kScaledDiffIntegerBits = 5;

    int input_left_shift;
    tflite::PreprocessSoftmaxScaling(
        params->beta, input->params.scale, kScaledDiffIntegerBits,
        &op_data->input_multiplier, &input_left_shift);
    op_data->input_left_shift = input_left_shift;
    op_data->diff_min =
        -1.0 * tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                            op_data->input_left_shift);
  }
  return kTfLiteOk;
}

}  // namespace

void SoftmaxQuantized(const TfLiteTensor* input, TfLiteTensor* output,
                      const SoftmaxParams& op_params) {
  if (output->type == kTfLiteInt16) {
    xtensa::hifimini::Softmax(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(output), GetTensorData<int16_t>(output));

  } else {
    xtensa::hifimini::Softmax(
        op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(output), GetTensorData<int8_t>(output));
  }
}

TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = static_cast<TfLiteSoftmaxParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  // TODO(b/132070898): Use statically slotted SoftmaxParams structures until a
  // scratch memory API is ready.
  SoftmaxParams* op_params = &kStaticOpData;
  node->user_data = op_params;

  TF_LITE_ENSURE_STATUS(
      CalculateSoftmaxOpData(context, input, output, params, op_params));

  return kTfLiteOk;
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* op_params = static_cast<SoftmaxParams*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  switch (input->type) {
    case kTfLiteInt8: {
      SoftmaxQuantized(input, output, *op_params);
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Only int8_t input supported currently, got %d.",
                         input->type);
      return kTfLiteError;
  }
}
}  // namespace activations

TfLiteRegistration* Register_SOFTMAX() {
  static TfLiteRegistration r = {/*init=*/nullptr,
                                 /*free=*/nullptr,
                                 /*prepare=*/activations::SoftmaxPrepare,
                                 /*invoke=*/activations::SoftmaxEval,
                                 /*profiling_string=*/nullptr,
                                 /*builtin_code=*/0,
                                 /*custom_name=*/nullptr,
                                 /*version=*/0};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
