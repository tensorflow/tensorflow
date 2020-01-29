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
#include "tensorflow/lite/kernels/internal/reference/integer_ops/softmax.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace micro {

namespace xtensa {
namespace hifimini {

// Quantized softmax with int8 input and output.
void Softmax(const SoftmaxParams& params,
                    const RuntimeShape& input_shape, const int8* input_data,
                    const RuntimeShape& output_shape, int8* output_data) {
  const int32 input_beta_multiplier = params.input_multiplier;
  const int32 input_beta_left_shift = params.input_left_shift;
  const int diff_min = params.diff_min;
  // The representation chosen for the input to the exp() function is Q5.26.
  // We need to leave extra space since values that we skip might be as large as
  // -32 before multiplying by input_beta_multiplier, and therefore as large as
  // -16 afterwards.  Note that exp(-8) is definitely not insignificant to
  // accumulation, but exp(-16) definitely is.
  static const int kScaledDiffIntegerBits = 5;
  static const int kAccumulationIntegerBits = 12;
  using FixedPointScaledDiff =
      gemmlowp::FixedPoint<int32, kScaledDiffIntegerBits>;
  using FixedPointAccum = gemmlowp::FixedPoint<int32, kAccumulationIntegerBits>;
  using FixedPoint0 = gemmlowp::FixedPoint<int32, 0>;

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
      int32 input_diff =
          static_cast<int32>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32 input_diff_rescaled =
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
      int32 input_diff =
          static_cast<int32>(input_data[i * depth + c]) - max_in_row;
      if (input_diff >= diff_min) {
        const int32 input_diff_rescaled =
            MultiplyByQuantizedMultiplierGreaterThanOne(
                input_diff, input_beta_multiplier, input_beta_left_shift);
        const FixedPointScaledDiff scaled_diff_f8 =
            FixedPointScaledDiff::FromRaw(input_diff_rescaled);

        FixedPoint0 exp_in_0 = exp_on_negative_values(scaled_diff_f8);
        const int32 unsat_output = gemmlowp::RoundingDivideByPOT(
            (shifted_scale * exp_in_0).raw(), num_bits_over_unit + 31 - 8);
        const int32 shifted_output = unsat_output - 128;

        output_data[i * depth + c] = static_cast<int8>(
            std::max(std::min(shifted_output, static_cast<int32>(127)),
                     static_cast<int32>(-128)));

      } else {
        output_data[i * depth + c] = -128;
      }
    }
  }
}

}  // namespace hifimini
}  // namespace xtensa

namespace activations {
namespace {

struct OpData {
  int32_t input_multiplier = 0;
  int input_left_shift = 0;
  int32_t input_range_radius = 0;
  int diff_min = 0;
};

// This size will work for both the hotword (1) and ambient music (0):
static OpData kStaticOpData;

TfLiteStatus CalculateSoftmaxOpData(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    TfLiteTensor* output,
                                    const TfLiteSoftmaxParams* params,
                                    OpData* data) {
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    } else {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
    }
    TF_LITE_ENSURE(context, output->params.scale == 1.f / 256);

    static const int kScaledDiffIntegerBits = 5;

    tflite::PreprocessSoftmaxScaling(
        params->beta, input->params.scale, kScaledDiffIntegerBits,
        &data->input_multiplier, &data->input_left_shift);
    data->diff_min = -1.0 * tflite::CalculateInputRadius(
                                kScaledDiffIntegerBits, data->input_left_shift);
  }
  return kTfLiteOk;
}

}  // namespace

void Softmax2DQuantized(const TfLiteTensor* input, TfLiteTensor* output,
                        TfLiteSoftmaxParams* params, OpData* data) {
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int32_t shape_data[4] = {batch_size, 1, 1, input_size};
  RuntimeShape shape(4, shape_data);
  SoftmaxParams op_params;
  op_params.input_multiplier = data->input_multiplier;
  op_params.input_left_shift = data->input_left_shift;
  op_params.diff_min = data->diff_min;

  xtensa::hifimini::Softmax(op_params, shape, GetTensorData<int8_t>(input),
                            shape, GetTensorData<int8_t>(output));
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  // TODO(b/132070898): Use statically slotted OpData structures until a
  // scratch memory API is ready.
  OpData* op_data = &kStaticOpData;
  node->user_data = op_data;

  TF_LITE_ENSURE_STATUS(
      CalculateSoftmaxOpData(context, input, output, params, op_data));

  return kTfLiteOk;
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  auto* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  switch (input->type) {
    case kTfLiteInt8: {
      if (NumDimensions(input) == 2) {
        Softmax2DQuantized(input, output, params, op_data);
        return kTfLiteOk;
      }
      context->ReportError(context,
                           "Only 2D tensors supported currently, got %dD.",
                           NumDimensions(input));
      return kTfLiteError;
    }
    default:
      context->ReportError(context, "Only int8_t supported currently, got %d.",
                           input->type);
      return kTfLiteError;
  }
}
}  // namespace activations

TfLiteRegistration* Register_SOFTMAX() {
  static TfLiteRegistration r = {activations::Init, activations::Free,
                                 activations::SoftmaxPrepare,
                                 activations::SoftmaxEval};
  return &r;
}

}  // namespace micro
}  // namespace ops
}  // namespace tflite
