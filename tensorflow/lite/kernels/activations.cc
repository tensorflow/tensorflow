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
#include <cassert>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/softmax.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/log_softmax.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

#if __aarch64__ && __clang__
#include <arm_neon.h>
#endif

namespace tflite {
namespace ops {
namespace builtin {
namespace activations {

// TODO(b/142762739): We should figure out a multi-threading plan for most of
// the activation ops below.

enum KernelType {
  kReference,
  kGenericOptimized,
  kFixedPointOptimized,
};

struct OpData {
  int32_t input_multiplier = 0;
  int input_left_shift = 0;
  int32_t input_range_radius = 0;
  int diff_min = 0;
  uint8_t table[256] = {0};
};

struct SoftmaxOpData {
  struct SoftmaxParams params = {};
  float table[256];
};

struct LogSoftmaxOpData : public OpData {
  int32_t reverse_scaling_divisor = 0;
  int32_t reverse_scaling_right_shift = 0;
  struct SoftmaxParams params = {};
  float f_table[256];
};

struct LeakyReluOpData : public OpData {
  uint8_t q_alpha;
  int32_t output_multiplier = 0;
  int output_shift = 0;
};

struct PreluOpData : public OpData {
  int32_t output_multiplier = 0;
  int output_shift = 0;
};

struct HardSwishData {
  HardSwishParams params;
};

struct ReluOpData : public OpData {
  int32_t output_multiplier = 0;
  int output_shift = 0;
};

namespace {
TfLiteStatus CheckOutputQuantParams(TfLiteContext* context,
                                    const TfLiteTensor* input,
                                    const TfLiteTensor* output) {
  TF_LITE_ENSURE(context, output->params.scale == 1. / 256);
  if (input->type == kTfLiteUInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  } else {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
  }
  return kTfLiteOk;
}

template <typename T>
void PopulateLookupTable(struct OpData* data, const TfLiteTensor* input,
                         TfLiteTensor* output,
                         const std::function<float(float)>& transform) {
  static_assert(sizeof(T) == 1, "Lookup table valid only for 8bit");
  const float inverse_scale = 1 / output->params.scale;
  int32_t maxval = std::numeric_limits<T>::max();
  int32_t minval = std::numeric_limits<T>::min();
  for (int32_t val = minval; val <= maxval; ++val) {
    const float dequantized =
        input->params.scale * (val - input->params.zero_point);
    const float transformed = transform(dequantized);
    const float rescaled = std::round(transformed * inverse_scale);
    const int32_t quantized =
        static_cast<int32_t>(rescaled + output->params.zero_point);
    data->table[static_cast<uint8_t>(static_cast<T>(val))] =
        static_cast<uint8_t>(
            static_cast<T>(std::max(std::min(maxval, quantized), minval)));
  }
}

#if __aarch64__ && __clang__
namespace {
// Looks up each element of <indices> in <table>, returns them in a vector.
// idx_offset must be a int8x16_t vector containing 64 in each lane.
inline uint8x16_t aarch64_lookup_vector(const uint8x16x4_t table[4],
                                        uint8x16_t indices) {
  // Look up in 1st quarter of the table: top 2 bits of indices == 00
  uint8x16_t output1 = vqtbl4q_u8(table[0], indices);
  // Look up in 2nd quarter of the table: top 2 bits of indices == 01
  uint8x16_t output2 =
      vqtbl4q_u8(table[1], veorq_u8(indices, vdupq_n_u8(0x40)));
  // Look up in 3rd quarter of the table: top 2 bits of indices == 10
  uint8x16_t output3 =
      vqtbl4q_u8(table[2], veorq_u8(indices, vdupq_n_u8(0x80)));
  // Look up in 4th quarter of the table: top 2 bits of indices == 11
  uint8x16_t output4 =
      vqtbl4q_u8(table[3], veorq_u8(indices, vdupq_n_u8(0xc0)));

  // Combine result of the 4 lookups.
  return vorrq_u8(vorrq_u8(output1, output2), vorrq_u8(output3, output4));
}
}  // namespace
#endif

// TODO(b/143696793): move this to optimized_ops.
void EvalUsingLookupTable(struct OpData* data, const TfLiteTensor* input,
                          TfLiteTensor* output) {
  const int size =
      MatchingFlatSize(GetTensorShape(input), GetTensorShape(output));
  uint8_t* output_data = GetTensorData<uint8_t>(output);
  const uint8_t* input_data = GetTensorData<uint8_t>(input);
  int i = 0;
#if __aarch64__ && __clang__
  // This code uses ARM64-only instructions.
  // TODO(b/143709993): Port to ARMv7

  // Load the tables into registers. (4*4 128-bit registers)
  uint8x16x4_t table[4];
  table[0] = vld1q_u8_x4(data->table + 16 * 4 * 0);
  table[1] = vld1q_u8_x4(data->table + 16 * 4 * 1);
  table[2] = vld1q_u8_x4(data->table + 16 * 4 * 2);
  table[3] = vld1q_u8_x4(data->table + 16 * 4 * 3);

  // Vectorized loop; process uint8x16_t (16 elements) at a time.
  constexpr int vectorized_16_loop_step = 16;
  const int vectorized_16_loop_end =
      size / vectorized_16_loop_step * vectorized_16_loop_step;
  for (; i < vectorized_16_loop_end; i += vectorized_16_loop_step) {
    uint8x16_t input = vld1q_u8(input_data + i);
    uint8x16_t output = aarch64_lookup_vector(table, input);
    vst1q_u8(output_data + i, output);
  }
  // Postamble and non-ARM64 code: simple for loop.
#endif
  for (; i < size; ++i) {
    output_data[i] = data->table[input_data[i]];
  }
}

template <typename T>
void QuantizedReluX(float act_min, float act_max, const TfLiteTensor* input,
                    TfLiteTensor* output, const ReluOpData* data) {
  ReluParams params;
  params.quantized_activation_min =
      std::max(static_cast<int32_t>(std::numeric_limits<T>::min()),
               output->params.zero_point +
                   static_cast<int32>(roundf(act_min / output->params.scale)));
  params.quantized_activation_max =
      act_max == std::numeric_limits<float>::infinity()
          ? static_cast<int32_t>(std::numeric_limits<T>::max())
          : std::min(
                static_cast<int32_t>(std::numeric_limits<T>::max()),
                output->params.zero_point +
                    static_cast<int32>(roundf(act_max / output->params.scale)));
  params.input_offset = input->params.zero_point;
  params.output_offset = output->params.zero_point;
  params.output_multiplier = data->output_multiplier;
  params.output_shift = data->output_shift;
  optimized_ops::ReluX(params, GetTensorShape(input), GetTensorData<T>(input),
                       GetTensorShape(output), GetTensorData<T>(output));
}

}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  return new OpData;
}

void* SoftmaxInit(TfLiteContext* context, const char* buffer, size_t length) {
  return new SoftmaxOpData;
}

void SoftmaxFree(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<SoftmaxOpData*>(buffer);
}

void* LogSoftmaxInit(TfLiteContext* context, const char* buffer,
                     size_t length) {
  return new LogSoftmaxOpData;
}

void* PreluInit(TfLiteContext* context, const char* buffer, size_t length) {
  return new PreluOpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

void LogSoftmaxFree(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<LogSoftmaxOpData*>(buffer);
}

void PreluFree(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<PreluOpData*>(buffer);
}

void* HardSwishInit(TfLiteContext* context, const char* buffer, size_t length) {
  return new HardSwishData;
}

TfLiteStatus GenericPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

void* ReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  return new ReluOpData;
}

void ReluFree(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<ReluOpData*>(buffer);
}

TfLiteStatus ReluPrepare(TfLiteContext* context, TfLiteNode* node) {
  ReluOpData* data = reinterpret_cast<ReluOpData*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8) {
    double real_multiplier = input->params.scale / output->params.scale;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

void* LeakyReluInit(TfLiteContext* context, const char* buffer, size_t length) {
  return new LeakyReluOpData;
}

void LeakyReluFree(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<LeakyReluOpData*>(buffer);
}

void HardSwishFree(TfLiteContext* context, void* buffer) {
  delete static_cast<HardSwishData*>(buffer);
}

void DownScaleInt32ToInt16Multiplier(int32_t multiplier_int32,
                                     int16_t* multiplier_int16) {
  TFLITE_DCHECK_GE(multiplier_int32, 0);
  static constexpr int32_t kRoundingOffset = 1 << 15;
  if (multiplier_int32 >=
      std::numeric_limits<int32_t>::max() - kRoundingOffset) {
    *multiplier_int16 = std::numeric_limits<int16_t>::max();
    return;
  }
  const int32_t result = (multiplier_int32 + kRoundingOffset) >> 16;
  TFLITE_DCHECK_LE(result << 16, multiplier_int32 + kRoundingOffset);
  TFLITE_DCHECK_GT(result << 16, multiplier_int32 - kRoundingOffset);
  *multiplier_int16 = result;
  TFLITE_DCHECK_EQ(*multiplier_int16, result);
}

TfLiteStatus HardSwishPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(GenericPrepare(context, node));
  TfLiteTensor* output = GetOutput(context, node, 0);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    HardSwishData* data = static_cast<HardSwishData*>(node->user_data);
    HardSwishParams* params = &data->params;
    const TfLiteTensor* input = GetInput(context, node, 0);
    params->input_zero_point = input->params.zero_point;
    params->output_zero_point = output->params.zero_point;
    const float input_scale = input->params.scale;
    const float hires_input_scale = (1.0f / 128.0f) * input_scale;
    const float reluish_scale = 3.0f / 32768.0f;
    const float output_scale = output->params.scale;

    const float output_multiplier = hires_input_scale / output_scale;

    int32_t output_multiplier_fixedpoint_int32;
    QuantizeMultiplier(output_multiplier, &output_multiplier_fixedpoint_int32,
                       &params->output_multiplier_exponent);
    DownScaleInt32ToInt16Multiplier(
        output_multiplier_fixedpoint_int32,
        &params->output_multiplier_fixedpoint_int16);
    TF_LITE_ENSURE(context, params->output_multiplier_exponent <= 0);

    const float reluish_multiplier = hires_input_scale / reluish_scale;
    int32_t reluish_multiplier_fixedpoint_int32;
    QuantizeMultiplier(reluish_multiplier, &reluish_multiplier_fixedpoint_int32,
                       &params->reluish_multiplier_exponent);
    DownScaleInt32ToInt16Multiplier(
        reluish_multiplier_fixedpoint_int32,
        &params->reluish_multiplier_fixedpoint_int16);
  }
  return kTfLiteOk;
}

TfLiteStatus LeakyReluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  LeakyReluOpData* data = reinterpret_cast<LeakyReluOpData*>(node->user_data);

  if (output->type == kTfLiteUInt8) {
    const auto* params =
        reinterpret_cast<TfLiteLeakyReluParams*>(node->builtin_data);
    // Quantize the alpha with same zero-point and scale as of input
    data->q_alpha = static_cast<uint8_t>(std::max<float>(
        std::numeric_limits<uint8_t>::min(),
        std::min<float>(std::numeric_limits<uint8_t>::max(),
                        std::round(input->params.zero_point +
                                   (params->alpha / input->params.scale)))));

    double real_multiplier =
        input->params.scale * input->params.scale / output->params.scale;
    QuantizeMultiplierSmallerThanOneExp(
        real_multiplier, &data->output_multiplier, &data->output_shift);
  }
  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <KernelType kernel_type>
TfLiteStatus TanhPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  if (kernel_type == kFixedPointOptimized) {
    if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
      static constexpr int kInputIntegerBits = 4;

      const double input_real_multiplier =
          input->params.scale *
          static_cast<double>(1 << (15 - kInputIntegerBits));

      const double q =
          std::frexp(input_real_multiplier, &data->input_left_shift);
      auto q_fixed = static_cast<int32_t>(TfLiteRound(q * (1ll << 15)));
      data->input_multiplier = static_cast<int16_t>(q_fixed);

      int16_t input_range_radius =
          CalculateInputRadius(kInputIntegerBits, data->input_left_shift, 15);
      data->input_range_radius = input_range_radius;
    }
  }

  if (kernel_type == kGenericOptimized || kernel_type == kReference) {
    if (input->type == kTfLiteUInt8) {
      PopulateLookupTable<uint8_t>(
          data, input, output, [](float value) { return std::tanh(value); });
    } else if (input->type == kTfLiteInt8) {
      PopulateLookupTable<int8_t>(data, input, output,
                                  [](float value) { return std::tanh(value); });
    }
  }

  if (input->type == kTfLiteInt16) {
    static constexpr int kInputIntegerBits = 3;
    static constexpr int kOutputFractionalBits = 15;

    // These operators are implemented in fixed-point arithmetic,
    // which intrinsically wants symmetric ranges (zero_point==0)
    // and power-of-two scales (power-of-two is abbreviated below as POT).
    // While more general support would be possible by means of rescaling,
    // that would add some overhead and some loss of accuracy and wouldn't
    // be used at the moment as current quantized LSTM applications are
    // happy with symmetric, power-of-two-scales quantization. So we just
    // implement that narrow case only for now.

    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    int input_scale_log2_rounded;
    TF_LITE_ENSURE(context,
                   CheckedLog2(input->params.scale, &input_scale_log2_rounded));

    int output_scale_log2_rounded;
    TF_LITE_ENSURE(
        context, CheckedLog2(output->params.scale, &output_scale_log2_rounded));
    TF_LITE_ENSURE_EQ(context, output_scale_log2_rounded,
                      -kOutputFractionalBits);

    data->input_left_shift =
        (15 - kInputIntegerBits) + input_scale_log2_rounded;
    // Support for shifts is limited until we have a parameterized version of
    // SaturatingRoundingMultiplyByPOT().
    TF_LITE_ENSURE(context, data->input_left_shift >= 0);
    TF_LITE_ENSURE(context, data->input_left_shift <= 1);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <KernelType kernel_type>
TfLiteStatus SigmoidPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  if (kernel_type == kFixedPointOptimized) {
    if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
      if (input->type == kTfLiteUInt8) {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                          std::numeric_limits<uint8_t>::min());
      }
      if (input->type == kTfLiteInt8) {
        TF_LITE_ENSURE_EQ(context, output->params.zero_point,
                          std::numeric_limits<int8_t>::min());
      }
      TF_LITE_ENSURE(context, output->params.scale == 1. / 256);

      static constexpr int kInputIntegerBits = 4;

      const double input_real_multiplier =
          input->params.scale *
          static_cast<double>(1 << (15 - kInputIntegerBits));

      const double q =
          std::frexp(input_real_multiplier, &data->input_left_shift);
      auto q_fixed = static_cast<int32_t>(TfLiteRound(q * (1ll << 15)));
      data->input_multiplier = static_cast<int16_t>(q_fixed);

      int16_t input_range_radius =
          CalculateInputRadius(kInputIntegerBits, data->input_left_shift, 15);
      data->input_range_radius = input_range_radius;
    }
  }

  if (kernel_type == kGenericOptimized || kernel_type == kReference) {
    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE(context, output->params.scale == 1. / 256);
      PopulateLookupTable<uint8_t>(data, input, output, [](float value) {
        return 1.0f / (1.0f + std::exp(-value));
      });
    } else if (input->type == kTfLiteInt8) {
      TF_LITE_ENSURE(context, output->params.scale == 1. / 256);
      PopulateLookupTable<int8_t>(data, input, output, [](float value) {
        return 1.0f / (1.0f + std::exp(-value));
      });
    }
  }

  if (input->type == kTfLiteInt16) {
    static constexpr int kInputIntegerBits = 3;
    static constexpr int kOutputFractionalBits = 15;

    // See comments in TanhPrepare about requiring zero_point==0
    // and a power-of-two ("POT") scale.

    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    int input_scale_log2_rounded;
    TF_LITE_ENSURE(context,
                   CheckedLog2(input->params.scale, &input_scale_log2_rounded));

    int output_scale_log2_rounded;
    TF_LITE_ENSURE(
        context, CheckedLog2(output->params.scale, &output_scale_log2_rounded));
    TF_LITE_ENSURE_EQ(context, output_scale_log2_rounded,
                      -kOutputFractionalBits);

    data->input_left_shift =
        (15 - kInputIntegerBits) + input_scale_log2_rounded;
    // The int16 logistic implementation does not support shifting of the input.
    TF_LITE_ENSURE_EQ(context, data->input_left_shift, 0);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  SoftmaxOpData* data = reinterpret_cast<SoftmaxOpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  const int num_dims = NumDimensions(input);
  TF_LITE_ENSURE(context, num_dims >= 1 && num_dims <= 4);

  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    data->params.table = data->table;
    optimized_ops::PopulateSoftmaxLookupTable(
        &data->params, input->params.scale, params->beta);
    data->params.zero_point = output->params.zero_point;
    data->params.scale = output->params.scale;
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus LogSoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  LogSoftmaxOpData* data = reinterpret_cast<LogSoftmaxOpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  TF_LITE_ENSURE_EQ(context, input->type, output->type);

  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.scale, 16.0 / 256);
    static const double kBeta = 1.0;
    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 255);
      data->params.table = data->f_table;
      optimized_ops::PopulateSoftmaxLookupTable(&data->params,
                                                input->params.scale, kBeta);
      data->params.zero_point = output->params.zero_point;
      data->params.scale = output->params.scale;
    }
    if (input->type == kTfLiteInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 127);
      static const int kScaledDiffIntegerBits = 5;
      tflite::PreprocessLogSoftmaxScalingExp(
          kBeta, input->params.scale, kScaledDiffIntegerBits,
          &data->input_multiplier, &data->input_left_shift,
          &data->reverse_scaling_divisor, &data->reverse_scaling_right_shift);
      data->reverse_scaling_right_shift *= -1;
      data->diff_min =
          -1.0 * tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                              data->input_left_shift);
    }
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus PreluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  const TfLiteTensor* alpha = GetInput(context, node, 1);
  PreluOpData* data = reinterpret_cast<PreluOpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, input->type, alpha->type);
  output->type = input->type;

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt16) {
    double real_multiplier =
        input->params.scale * alpha->params.scale / output->params.scale;
    QuantizeMultiplierSmallerThanOneExp(
        real_multiplier, &data->output_multiplier, &data->output_shift);
  }

  // PRelu (parameteric Relu) shares the same alpha value on "shared axis".
  // This means it's always required to "broadcast" alpha values in PRelu.
  TfLiteIntArray* output_size = nullptr;
  TF_LITE_ENSURE_OK(
      context, CalculateShapeForBroadcast(context, input, alpha, &output_size));

  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size));
  // After broadcasting, the output shape should always be the same as the
  // input shape.
  TF_LITE_ENSURE(context, HaveSameShapes(input, output));

  return kTfLiteOk;
}

TfLiteStatus ReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  const ReluOpData* data = reinterpret_cast<ReluOpData*>(node->user_data);
  switch (input->type) {
    case kTfLiteFloat32: {
      optimized_ops::Relu(GetTensorShape(input), GetTensorData<float>(input),
                          GetTensorShape(output), GetTensorData<float>(output));
    } break;
    // TODO(renjieliu): We may revisit the quantization calculation logic,
    // the unbounded upper limit is actually hard to quantize.
    case kTfLiteUInt8: {
      QuantizedReluX<uint8_t>(0.0f, std::numeric_limits<float>::infinity(),
                              input, output, data);
    } break;
    case kTfLiteInt8: {
      QuantizedReluX<int8_t>(0.0f, std::numeric_limits<float>::infinity(),
                             input, output, data);
    } break;
    default:
      context->ReportError(
          context, "Only float32 & int8/uint8 is supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Relu1Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  const ReluOpData* data = reinterpret_cast<ReluOpData*>(node->user_data);
  switch (input->type) {
    case kTfLiteFloat32: {
      optimized_ops::Relu1(GetTensorShape(input), GetTensorData<float>(input),
                           GetTensorShape(output),
                           GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      QuantizedReluX<uint8_t>(-1.0f, 1.0f, input, output, data);
      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
      QuantizedReluX<int8_t>(-1, 1, input, output, data);
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context,
                           "Only float32, uint8, int8 supported "
                           "currently, got %s.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

template <KernelType kernel_type>
TfLiteStatus HardSwishEval(TfLiteContext* context, TfLiteNode* node) {
  HardSwishData* data = static_cast<HardSwishData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      if (kernel_type == kReference) {
        reference_ops::HardSwish(
            GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
        optimized_ops::HardSwish(
            GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      }
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      HardSwishParams& params = data->params;
      if (kernel_type == kReference) {
        reference_ops::HardSwish(
            params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<uint8_t>(output));
      } else {
        optimized_ops::HardSwish(
            params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<uint8_t>(output));
      }
      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
      HardSwishParams& params = data->params;
      if (kernel_type == kReference) {
        reference_ops::HardSwish(
            params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(output), GetTensorData<int8_t>(output));
      } else {
        optimized_ops::HardSwish(
            params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(output), GetTensorData<int8_t>(output));
      }
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(
          context,
          "Only float32, uint8 and int8 are supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TfLiteStatus Relu6Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  ReluOpData* data = reinterpret_cast<ReluOpData*>(node->user_data);
  switch (input->type) {
    case kTfLiteFloat32: {
      size_t elements = input->bytes / sizeof(float);
      const float* in = GetTensorData<float>(input);
      const float* in_end = in + elements;
      float* out = GetTensorData<float>(output);
      for (; in < in_end; in++, out++) *out = std::min(std::max(0.f, *in), 6.f);
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8:
      QuantizedReluX<uint8_t>(0.0f, 6.0f, input, output, data);
      return kTfLiteOk;
    case kTfLiteInt8: {
      QuantizedReluX<int8_t>(0.0f, 6.0f, input, output, data);
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(
          context,
          "Only float32, uint8 and int8 are supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

template <KernelType kernel_type>
TfLiteStatus TanhEval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      if (kernel_type == kReference) {
        reference_ops::Tanh(GetTensorShape(input), GetTensorData<float>(input),
                            GetTensorShape(output),
                            GetTensorData<float>(output));
      } else {
        optimized_ops::Tanh(GetTensorShape(input), GetTensorData<float>(input),
                            GetTensorShape(output),
                            GetTensorData<float>(output));
      }
      return kTfLiteOk;
    } break;
    case kTfLiteInt16: {
      TanhParams params;
      params.input_left_shift = data->input_left_shift;
      if (kernel_type == kReference) {
        reference_ops::Tanh(
            params, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(output), GetTensorData<int16_t>(output));
      } else {
        optimized_ops::Tanh(
            params, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(output), GetTensorData<int16_t>(output));
      }
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      if (kernel_type == kFixedPointOptimized) {
        TanhParams params;
        params.input_zero_point = input->params.zero_point;
        params.input_range_radius = data->input_range_radius;
        params.input_multiplier = data->input_multiplier;
        params.input_left_shift = data->input_left_shift;
        optimized_ops::Tanh16bitPercision(
            params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<uint8_t>(output));
      } else {
        EvalUsingLookupTable(data, input, output);
      }
      return kTfLiteOk;
    } break;
    case kTfLiteInt8: {
      if (kernel_type == kFixedPointOptimized) {
        TanhParams params;
        params.input_zero_point = input->params.zero_point;
        params.input_range_radius = data->input_range_radius;
        params.input_multiplier = data->input_multiplier;
        params.input_left_shift = data->input_left_shift;
        optimized_ops::Tanh16bitPercision(
            params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(output), GetTensorData<int8_t>(output));
      } else {
        EvalUsingLookupTable(data, input, output);
      }
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context,
                           "Only float32, uint8, int16 and int8 are supported "
                           "currently, got %s.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

// Sigmoid is also know as "Logistic".
template <KernelType kernel_type>
TfLiteStatus SigmoidEval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      if (kernel_type == kReference) {
        reference_ops::Logistic(
            GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
        optimized_ops::Logistic(
            GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      }
      break;
    }
    case kTfLiteInt16: {
      LogisticParams params;
      if (kernel_type == kReference) {
        reference_ops::Logistic(
            params, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(output), GetTensorData<int16_t>(output));
      } else {
        optimized_ops::Logistic(
            params, GetTensorShape(input), GetTensorData<int16_t>(input),
            GetTensorShape(output), GetTensorData<int16_t>(output));
      }
      break;
    }
    case kTfLiteUInt8: {
      if (kernel_type == kFixedPointOptimized) {
        LogisticParams params;
        params.input_zero_point = input->params.zero_point;
        params.input_range_radius = data->input_range_radius;
        params.input_multiplier = data->input_multiplier;
        params.input_left_shift = data->input_left_shift;
        optimized_ops::Logistic16bitPercision(
            params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<uint8_t>(output));
      } else {
        EvalUsingLookupTable(data, input, output);
      }
      break;
    }
    case kTfLiteInt8: {
      if (kernel_type == kFixedPointOptimized) {
        LogisticParams params;
        params.input_zero_point = input->params.zero_point;
        params.input_range_radius = data->input_range_radius;
        params.input_multiplier = data->input_multiplier;
        params.input_left_shift = data->input_left_shift;
        optimized_ops::Logistic16bitPercision(
            params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(output), GetTensorData<int8_t>(output));
      } else {
        EvalUsingLookupTable(data, input, output);
      }
      break;
    }
    default:
      context->ReportError(context,
                           "Only float32, uint8, int16 and int8 are supported "
                           "currently, got %s.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus SoftmaxFloat(TfLiteContext* context, const TfLiteTensor* input,
                          TfLiteTensor* output, TfLiteSoftmaxParams* params) {
  switch (NumDimensions(input)) {
    case 1:
    case 2:
    case 3:
    case 4:
      SoftmaxParams op_params;
      op_params.beta = params->beta;
      optimized_ops::Softmax(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(output), GetTensorData<float>(output),
          CpuBackendContext::GetFromContext(context));
      return kTfLiteOk;
    default:
      context->ReportError(
          context,
          "Only 1D, 2D, 3D and 4D tensors supported currently, got %dD.",
          NumDimensions(input));
      return kTfLiteError;
  }
}

template <typename T>
TfLiteStatus SoftmaxQuantized(TfLiteContext* context, const TfLiteTensor* input,
                              TfLiteTensor* output, SoftmaxOpData* data) {
  if (NumDimensions(input) >= 1 && NumDimensions(input) <= 4) {
    optimized_ops::Softmax(data->params, GetTensorShape(input),
                           GetTensorData<T>(input), GetTensorShape(output),
                           GetTensorData<T>(output));
    return kTfLiteOk;
  } else {
    context->ReportError(
        context, "Only 1D, 2D, 3D and 4D tensors supported currently, got %dD.",
        NumDimensions(input));
    return kTfLiteError;
  }
}

TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  SoftmaxOpData* data = reinterpret_cast<SoftmaxOpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  // TODO(ahentz): consider an implementation that works for many (all?)
  // dimensions.
  switch (input->type) {
    case kTfLiteFloat32: {
      return SoftmaxFloat(context, input, output, params);
    }
    case kTfLiteUInt8: {
      return SoftmaxQuantized<uint8_t>(context, input, output, data);
    }
    case kTfLiteInt8: {
      return SoftmaxQuantized<int8_t>(context, input, output, data);
    }

    default:
      context->ReportError(
          context,
          "Only float32, uint8_t and Int8_t are supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

template <KernelType kernel_type>
TfLiteStatus LogSoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const LogSoftmaxOpData* data =
      reinterpret_cast<LogSoftmaxOpData*>(node->user_data);
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      SoftmaxParams op_params;
      if (kernel_type == kGenericOptimized) {
        optimized_ops::LogSoftmax(
            op_params, GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
        reference_ops::LogSoftmax(
            op_params, GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      }
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      SoftmaxParams op_params = data->params;
      if (kernel_type == kGenericOptimized) {
        optimized_ops::LogSoftmax(
            op_params, input->params.scale, GetTensorShape(input),
            GetTensorData<uint8_t>(input), GetTensorShape(output),
            GetTensorData<uint8_t>(output));
      } else {
        reference_ops::LogSoftmax(
            op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<uint8_t>(output));
      }
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      const auto input_shape = GetTensorShape(input);
      const auto output_shape = GetTensorShape(output);
      const int trailing_dim = input_shape.DimensionsCount() - 1;
      const int outer_size =
          MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
      const int depth =
          MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
      reference_integer_ops::LogSoftmax(
          data->input_multiplier, data->input_left_shift,
          data->reverse_scaling_divisor, data->reverse_scaling_right_shift,
          data->diff_min, outer_size, depth, GetTensorData<int8_t>(input),
          GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    default:
      context->ReportError(
          context,
          "Only float32, uint8 and int8 are supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

template <typename T>
T ApplyPrelu(T input, T alpha) {
  return input >= 0.0 ? input : input * alpha;
}

TfLiteStatus PreluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  const TfLiteTensor* alpha = GetInput(context, node, 1);
  TfLiteTensor* output = GetOutput(context, node, 0);
  const PreluOpData* data = reinterpret_cast<PreluOpData*>(node->user_data);
  switch (input->type) {
    case kTfLiteFloat32: {
      reference_ops::BroadcastBinaryFunction4DSlow<float, float, float>(
          GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(alpha), GetTensorData<float>(alpha),
          GetTensorShape(output), GetTensorData<float>(output),
          ApplyPrelu<float>);
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      PreluParams op_params;
      op_params.input_offset = -input->params.zero_point;
      op_params.alpha_offset = -alpha->params.zero_point;
      op_params.output_offset = output->params.zero_point;
      op_params.output_multiplier = data->output_multiplier;
      op_params.output_shift = data->output_shift;
      reference_ops::BroadcastPrelu4DSlow(
          op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
          GetTensorShape(alpha), GetTensorData<uint8_t>(alpha),
          GetTensorShape(output), GetTensorData<uint8_t>(output));
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(
          context, "Only float32 and uint8 are supported currently, got %d.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

namespace {
template <typename T>
void QLeakyRelu(const TfLiteTensor* input, TfLiteTensor* output, float alpha,
                const LeakyReluOpData* data) {
  LeakyReluParams op_params;
  op_params.input_offset = input->params.zero_point;
  op_params.alpha_offset = input->params.zero_point;
  op_params.output_offset = output->params.zero_point;

  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;

  reference_ops::QuantizeLeakyRelu(
      op_params, data->q_alpha, GetTensorShape(input), GetTensorData<T>(input),
      GetTensorShape(output), GetTensorData<T>(output));
}
}  // namespace

TfLiteStatus LeakyReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  const auto* params =
      reinterpret_cast<TfLiteLeakyReluParams*>(node->builtin_data);
  const LeakyReluOpData* data =
      reinterpret_cast<LeakyReluOpData*>(node->user_data);

  LeakyReluParams op_params;
  op_params.alpha = params->alpha;
  switch (input->type) {
    case kTfLiteFloat32: {
      optimized_ops::LeakyRelu(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    case kTfLiteUInt8: {
      QLeakyRelu<uint8_t>(input, output, params->alpha, data);
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(
          context, "Only float32 and uint8 is supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TfLiteStatus EluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);
  switch (input->type) {
    case kTfLiteFloat32: {
      optimized_ops::Elu(GetTensorShape(input), GetTensorData<float>(input),
                         GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    } break;
    default:
      context->ReportError(context,
                           "Only float32 is supported currently, got %s.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

}  // namespace activations

TfLiteRegistration* Register_ELU() {
  static TfLiteRegistration r = {/*init=*/nullptr, /*free=*/nullptr,
                                 activations::GenericPrepare,
                                 activations::EluEval};
  return &r;
}

TfLiteRegistration* Register_RELU() {
  static TfLiteRegistration r = {activations::ReluInit, activations::ReluFree,
                                 activations::ReluPrepare,
                                 activations::ReluEval};
  return &r;
}

TfLiteRegistration* Register_RELU_N1_TO_1() {
  static TfLiteRegistration r = {activations::ReluInit, activations::ReluFree,
                                 activations::ReluPrepare,
                                 activations::Relu1Eval};
  return &r;
}

TfLiteRegistration* Register_RELU6() {
  static TfLiteRegistration r = {activations::ReluInit, activations::ReluFree,
                                 activations::ReluPrepare,
                                 activations::Relu6Eval};
  return &r;
}

TfLiteRegistration* Register_TANH_REF() {
  static TfLiteRegistration r = {
      activations::Init, activations::Free,
      activations::TanhPrepare<activations::kReference>,
      activations::TanhEval<activations::kReference>};
  return &r;
}

TfLiteRegistration* Register_TANH_GENERIC_OPT() {
  static TfLiteRegistration r = {
      activations::Init, activations::Free,
      activations::TanhPrepare<activations::kGenericOptimized>,
      activations::TanhEval<activations::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_TANH_FIXED_POINT_OPT() {
  static TfLiteRegistration r = {
      activations::Init, activations::Free,
      activations::TanhPrepare<activations::kFixedPointOptimized>,
      activations::TanhEval<activations::kFixedPointOptimized>};
  return &r;
}

TfLiteRegistration* Register_TANH() {
  // TODO(b/134622898): Switch over from the LUT optimized method to the fixed
  // point optimized method when typical Android hardware performs better on
  // the latter one.
  return Register_TANH_GENERIC_OPT();
}

TfLiteRegistration* Register_LOGISTIC_REF() {
  static TfLiteRegistration r = {
      activations::Init, activations::Free,
      activations::SigmoidPrepare<activations::kReference>,
      activations::SigmoidEval<activations::kReference>};
  return &r;
}

TfLiteRegistration* Register_LOGISTIC_GENERIC_OPT() {
  static TfLiteRegistration r = {
      activations::Init, activations::Free,
      activations::SigmoidPrepare<activations::kGenericOptimized>,
      activations::SigmoidEval<activations::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_LOGISTIC_FIXED_POINT_OPT() {
  static TfLiteRegistration r = {
      activations::Init, activations::Free,
      activations::SigmoidPrepare<activations::kFixedPointOptimized>,
      activations::SigmoidEval<activations::kFixedPointOptimized>};
  return &r;
}

TfLiteRegistration* Register_LOGISTIC() {
  // TODO(b/134622898): Switch over from the LUT optimized method to the fixed
  // point optimized method when typical Android hardware performs better on
  // the latter one.
  return Register_LOGISTIC_GENERIC_OPT();
}

TfLiteRegistration* Register_SOFTMAX() {
  static TfLiteRegistration r = {
      activations::SoftmaxInit, activations::SoftmaxFree,
      activations::SoftmaxPrepare, activations::SoftmaxEval};
  return &r;
}

TfLiteRegistration* Register_LOG_SOFTMAX_REF() {
  static TfLiteRegistration r = {
      activations::LogSoftmaxInit, activations::LogSoftmaxFree,
      activations::LogSoftmaxPrepare,
      activations::LogSoftmaxEval<activations::kReference>};
  return &r;
}

TfLiteRegistration* Register_LOG_SOFTMAX() {
  static TfLiteRegistration r = {
      activations::LogSoftmaxInit, activations::LogSoftmaxFree,
      activations::LogSoftmaxPrepare,
      activations::LogSoftmaxEval<activations::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_PRELU() {
  static TfLiteRegistration r = {activations::PreluInit, activations::PreluFree,
                                 activations::PreluPrepare,
                                 activations::PreluEval};
  return &r;
}

TfLiteRegistration* Register_LEAKY_RELU() {
  static TfLiteRegistration r = {
      activations::LeakyReluInit, activations::LeakyReluFree,
      activations::LeakyReluPrepare, activations::LeakyReluEval};
  return &r;
}

TfLiteRegistration* Register_HARD_SWISH() {
  static TfLiteRegistration r = {
      activations::HardSwishInit, activations::HardSwishFree,
      activations::HardSwishPrepare,
      activations::HardSwishEval<activations::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_HARD_SWISH_REF() {
  static TfLiteRegistration r = {
      activations::HardSwishInit, activations::HardSwishFree,
      activations::HardSwishPrepare,
      activations::HardSwishEval<activations::kReference>};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
