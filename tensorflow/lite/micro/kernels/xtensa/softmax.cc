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

#include "tensorflow/lite/micro/kernels/softmax.h"

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/xtensa/xtensa.h"

namespace tflite {
namespace {

#if defined(HIFIMINI)
struct OpData {
  uint16_t* exp_lut;
};
#elif defined(FUSION_F1)
struct OpData {
  SoftmaxParams params;
  int scratch_tensor_index;
};
#endif

#if defined(HIFIMINI)
// Number of unique int8_t and int16_t values.  Used in exponent lookup table
// computation.
constexpr int kInt8Range =
    std::numeric_limits<int8_t>::max() - std::numeric_limits<int8_t>::min() + 1;
constexpr int kInt16Range = std::numeric_limits<int16_t>::max() -
                            std::numeric_limits<int16_t>::min() + 1;
// Each 16-bit precalculated exponent is expressed as a Q0.16 fixedpoint
// value. We special-case e^0 since 1.0 requires 1 integer bit to
// express.
constexpr int kExpFractionalBits = 16;
// e^0 expressed as Q1.15 exceeds the int16_t range, so it must be handled
// specially.
constexpr int kMaxExponentValue = (1 << kExpFractionalBits);

// Quantized softmax with int8_t input and int16_t output.
// Passing OpData by value does not have much savings in this op, but following
// that as a best practice, at least for the xtensa kernels. See b/155656675 for
// more details.
TfLiteStatus SoftmaxHifimini(OpData op_data, const RuntimeShape& input_shape,
                             const int8_t* input_data,
                             const RuntimeShape& output_shape,
                             int16_t* output_data) {
  // The last dimension is depth.  Outer size is the total input size
  // divided by depth.
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  for (int i = 0; i < outer_size; ++i) {
    int8_t max_in_row = std::numeric_limits<int8_t>::min();
    for (int c = 0; c < depth; ++c) {
      max_in_row = std::max(max_in_row, input_data[i * depth + c]);
    }

    uint32_t sum_of_exps = 0;
    for (int c = 0; c < depth; ++c) {
      TFLITE_DCHECK(max_in_row >= input_data[i * depth + c]);
      uint8_t input_diff = max_in_row - input_data[i * depth + c];

      sum_of_exps +=
          input_diff == 0 ? kMaxExponentValue : op_data.exp_lut[input_diff];
    }

    // Ensure we cannot overflow the full_range_output value.  We need to
    // guarantee that kInt16Range * max(input_data) / sum_of_exps < kInt16Range.
    TFLITE_DCHECK(sum_of_exps >= kMaxExponentValue);

    for (int c = 0; c < depth; ++c) {
      uint8_t input_diff = max_in_row - input_data[i * depth + c];
      // Special case for diff == 0
      uint32_t unscaled_output =
          input_diff == 0 ? kMaxExponentValue : op_data.exp_lut[input_diff];
      int64_t scaled_output = static_cast<int64_t>(unscaled_output) *
                              static_cast<int64_t>(kInt16Range);
      int32_t full_range_output =
          scaled_output / sum_of_exps + std::numeric_limits<int16_t>::min();
      // Round up if remainder exceeds half of the divider value.
      uint32_t remainder = scaled_output % sum_of_exps;
      if (remainder * 2 >= sum_of_exps) {
        full_range_output++;
      }
      output_data[i * depth + c] = static_cast<int16_t>(std::max(
          std::min(full_range_output,
                   static_cast<int32_t>(std::numeric_limits<int16_t>::max())),
          static_cast<int32_t>(std::numeric_limits<int16_t>::min())));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus CalculateSoftmaxOpDataHifimini(TfLiteContext* context,
                                            const TfLiteTensor* input,
                                            TfLiteTensor* output,
                                            const TfLiteSoftmaxParams* params,
                                            OpData* op_data) {
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    } else {
      if (output->type == kTfLiteInt16) {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                          std::numeric_limits<int16_t>::min());
        // NOTE: Current int16_t softmax output does not require symmetric
        // scaling
        // - so no need to verify scale here.
      } else {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                          std::numeric_limits<int8_t>::min());
        TF_LITE_ENSURE(context, output->params.scale == 1.f / 256);
      }
    }

    // Precompute e^(-x * input_scale * beta) for every possible int8_t input.
    // This computation is used for every iteration of Softmax.  We must compute
    // using pre-scaled inputs to avoid introducing additional error, while
    // restricting our input range to the int8_t range. This is valid since beta
    // and input scale are constant for a given op in the graph. Skip index 0
    // since that is a special case which requires 1 integer bit instead of 0.
    for (int i = 1; i <= kInt8Range; i++) {
      float scaled_input = i * input->params.scale;
      float exp_value =
          std::exp((-scaled_input) * static_cast<float>(params->beta));

      float exponent_scaled =
          std::round(exp_value * static_cast<float>(1 << kExpFractionalBits));
      op_data->exp_lut[i] = static_cast<uint16_t>(exponent_scaled);
    }
  }
  return kTfLiteOk;
}

TfLiteStatus PrepareHifimini(TfLiteContext* context, TfLiteNode* node) {
  auto* params = static_cast<TfLiteSoftmaxParams*>(node->builtin_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  TFLITE_DCHECK(node->user_data != nullptr);
  OpData* op_data = static_cast<OpData*>(node->user_data);

  // Allocate an array to precompute exponents over all int8_t inputs, applying
  // the scale and beta before calculating exp. It is mandatory to apply beta
  // and scale here, since each softmax op may have different beta and scale
  // values. Beta and scale will remain constant for a given softmax op.
  op_data->exp_lut = static_cast<uint16_t*>(context->AllocatePersistentBuffer(
      context, (kInt8Range + 1) * sizeof(uint16_t)));
  TF_LITE_ENSURE(context, op_data->exp_lut != nullptr);

  TF_LITE_ENSURE_STATUS(
      CalculateSoftmaxOpDataHifimini(context, input, output, params, op_data));

  return kTfLiteOk;
}
#endif  // defined(HIFIMINI)

#if defined(FUSION_F1)
TfLiteStatus PrepareHifi4(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, SoftmaxPrepare(context, node));

  // Calculate scratch memory requirements and request scratch buffer
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* output = GetOutput(context, node, 0);

  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& output_shape = GetTensorShape(output);
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  if (input->type == kTfLiteInt8) {
    int required_scratch =
        get_softmax_scratch_size(PREC_ASYM8S, PREC_ASYM8S, depth);
    TF_LITE_ENSURE(context, required_scratch > 0);

    auto* data = static_cast<OpData*>(node->user_data);
    TF_LITE_ENSURE_OK(
        context, context->RequestScratchBufferInArena(
                     context, required_scratch, &(data->scratch_tensor_index)));
  }

  return kTfLiteOk;
}

TfLiteStatus EvalHifi4(const OpData* op_data, const TfLiteEvalTensor* input,
                       TfLiteEvalTensor* output, TfLiteContext* context) {
  const RuntimeShape& input_shape = tflite::micro::GetTensorShape(input);
  const int8_t* input_data = tflite::micro::GetTensorData<int8_t>(input);
  const RuntimeShape& output_shape = tflite::micro::GetTensorShape(output);
  int16_t* output_data = tflite::micro::GetTensorData<int16_t>(output);
  const int trailing_dim = input_shape.DimensionsCount() - 1;
  const int outer_size =
      MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
  const int depth =
      MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);

  void* p_scratch = static_cast<void*>(
      context->GetScratchBuffer(context, op_data->scratch_tensor_index));

  for (int i = 0; i < outer_size; ++i) {
    int err = xa_nn_vec_softmax_asym8s_16(
        &output_data[i * depth], &input_data[i * depth],
        op_data->params.diff_min, op_data->params.input_left_shift,
        op_data->params.input_multiplier, depth, p_scratch);
    TF_LITE_ENSURE(context, err == 0);
  }
  return kTfLiteOk;
}

#endif  // defined(FUSION_F1)

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
#if defined(HIFIMINI) || defined(FUSION_F1)
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
#else
  return SoftmaxInit(context, buffer, length);
#endif
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
#if defined(HIFIMINI)
  return PrepareHifimini(context, node);
#elif defined(FUSION_F1)
  return PrepareHifi4(context, node);
#else
  return SoftmaxPrepare(context, node);
#endif
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteEvalTensor* input = tflite::micro::GetEvalInput(context, node, 0);
  TfLiteEvalTensor* output = tflite::micro::GetEvalOutput(context, node, 0);
  TFLITE_DCHECK(node->user_data != nullptr);

  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt16) {
#if defined(HIFIMINI)
    return SoftmaxHifimini(*static_cast<OpData*>(node->user_data),
                           tflite::micro::GetTensorShape(input),
                           tflite::micro::GetTensorData<int8_t>(input),
                           tflite::micro::GetTensorShape(output),
                           tflite::micro::GetTensorData<int16_t>(output));
#elif defined(FUSION_F1)
    return EvalHifi4(static_cast<OpData*>(node->user_data), input, output,
                     context);
#else
    SoftmaxParams op_data = *static_cast<SoftmaxParams*>(node->user_data);
    tflite::reference_ops::Softmax(
        op_data, tflite::micro::GetTensorShape(input),
        tflite::micro::GetTensorData<int8_t>(input),
        tflite::micro::GetTensorShape(output),
        tflite::micro::GetTensorData<int16_t>(output));
#endif
  } else {
    TF_LITE_KERNEL_LOG(context, "Type %s (%d) not supported.",
                       TfLiteTypeGetName(input->type), input->type);
    return kTfLiteError;
  }
}

}  // namespace

TfLiteRegistration Register_SOFTMAX() {
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
