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
#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/leaky_relu.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/lut.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/binary_function.h"
#include "tensorflow/lite/kernels/internal/reference/gelu.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/log_softmax.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/lut.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/tanh.h"
#include "tensorflow/lite/kernels/internal/reference/logistic.h"
#include "tensorflow/lite/kernels/internal/reference/prelu.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/reference/softmax.h"
#include "tensorflow/lite/kernels/internal/reference/tanh.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

#ifdef TFLITE_KERNEL_USE_XNNPACK
#include "xnnpack.h"  // from @XNNPACK
#include "tensorflow/lite/logger.h"
#include "tensorflow/lite/minimal_logging.h"
#include "pthreadpool.h"  // from @pthreadpool
#endif  // TFLITE_KERNEL_USE_XNNPACK

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
  union {
    uint8_t lut_uint8[LUTSize<uint8_t>()];
    int8_t lut_int8[LUTSize<int8_t>()];
    int16_t lut_int16[LUTSize<int16_t>()];
  };
};

struct SoftmaxOpData {
  struct SoftmaxParams params = {};
  float table[256];
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
  uint8_t uint8_table1[256];
  uint8_t uint8_table2[256];
#endif
  static constexpr int kInt16LUTArraySize = LUTSize<int16_t>();
  int16_t exp_lut[kInt16LUTArraySize];  // int16 LUT for exp(x), where x uniform
                                        // distributed between [-10.0 , 0.0]
  int16_t one_over_one_plus_x_lut[kInt16LUTArraySize];  // int16 LUT for 1 /
                                                        // (1 + x), where x
                                                        // uniform distributed
                                                        // between [0.0 , 1.0]
};

struct LogSoftmaxOpData : public OpData {
  int32_t reverse_scaling_divisor = 0;
  int32_t reverse_scaling_right_shift = 0;
  struct SoftmaxParams params = {};
  float f_table[256];
};

struct LeakyReluOpData : public OpData {
  int32_t output_multiplier_alpha = 0;
  int32_t output_shift_alpha = 0;
  int32_t output_multiplier_identity = 0;
  int32_t output_shift_identity = 0;
};

struct PreluOpData : public OpData {
  int32_t output_multiplier_1 = 0;
  int32_t output_shift_1 = 0;
  int32_t output_multiplier_2 = 0;
  int32_t output_shift_2 = 0;
  bool requires_broadcast;
};

struct HardSwishData {
  HardSwishParams params;
};

struct ReluOpData : public OpData {
  int32_t output_multiplier = 0;
  int output_shift = 0;
};

namespace {

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
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

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
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8 ||
      input->type == kTfLiteInt16) {
    double real_multiplier = input->params.scale / output->params.scale;
    QuantizeMultiplier(real_multiplier, &data->output_multiplier,
                       &data->output_shift);
  }

  if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
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

TfLiteStatus HardSwishPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_STATUS(GenericPrepare(context, node));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    HardSwishData* data = static_cast<HardSwishData*>(node->user_data);
    HardSwishParams* params = &data->params;
    const TfLiteTensor* input;
    TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
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
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  LeakyReluOpData* data = reinterpret_cast<LeakyReluOpData*>(node->user_data);

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8 ||
      output->type == kTfLiteInt16) {
    const auto* params =
        reinterpret_cast<TfLiteLeakyReluParams*>(node->builtin_data);

    double alpha_multiplier =
        input->params.scale * params->alpha / output->params.scale;
    QuantizeMultiplier(alpha_multiplier, &data->output_multiplier_alpha,
                       &data->output_shift_alpha);
    double identity_multiplier = input->params.scale / output->params.scale;
    QuantizeMultiplier(identity_multiplier, &data->output_multiplier_identity,
                       &data->output_shift_identity);
  }

  if (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <KernelType kernel_type>
TfLiteStatus TanhPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (kernel_type == kFixedPointOptimized) {
    if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
      static constexpr int kInputIntegerBits = 4;

      const double input_real_multiplier =
          input->params.scale *
          static_cast<double>(1 << (15 - kInputIntegerBits));

      const double q =
          std::frexp(input_real_multiplier, &data->input_left_shift);
      auto q_fixed = static_cast<int32_t>(TfLiteRound(q * (1LL << 15)));
      data->input_multiplier = static_cast<int16_t>(q_fixed);

      int16_t input_range_radius =
          CalculateInputRadius(kInputIntegerBits, data->input_left_shift, 15);
      data->input_range_radius = input_range_radius;
    }
  }

  if (kernel_type == kGenericOptimized || kernel_type == kReference) {
    if (input->type == kTfLiteUInt8) {
      LUTPopulate<uint8_t>(
          input->params.scale, input->params.zero_point, output->params.scale,
          output->params.zero_point,
          [](float value) { return std::tanh(value); }, data->lut_uint8);

    } else if (input->type == kTfLiteInt8) {
      LUTPopulate<int8_t>(
          input->params.scale, input->params.zero_point, output->params.scale,
          output->params.zero_point,
          [](float value) { return std::tanh(value); }, data->lut_int8);
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
    bool param_scale_pot =
        CheckedLog2(input->params.scale, &input_scale_log2_rounded);

    data->input_left_shift =
        (15 - kInputIntegerBits) + input_scale_log2_rounded;
    param_scale_pot &=
        (data->input_left_shift == 0 || data->input_left_shift == 1);

    if (!param_scale_pot) {
      // Calculate multiplier to change input scale to 1/(3*4096)
      // as required by the table lookup.
      // The number 3.0 in the multiplier comes from here,
      // because the interval is [-10.7, 10.7] instead of [-8, 8].
      // So, in this scaling +/-2^17 represents +/-10.7.

      double multiplier = input->params.scale * 4096.0 * 3.0;
      data->input_left_shift = 0;

      while (multiplier <= 32767.0 / 2.0 && data->input_left_shift <= 30) {
        data->input_left_shift++;
        multiplier = multiplier * 2.0;
      }

      data->input_multiplier = static_cast<int32_t>(multiplier);
    }

    int output_scale_log2_rounded;
    TF_LITE_ENSURE(
        context, CheckedLog2(output->params.scale, &output_scale_log2_rounded));
    TF_LITE_ENSURE_EQ(context, output_scale_log2_rounded,
                      -kOutputFractionalBits);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <KernelType kernel_type>
TfLiteStatus SigmoidPrepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

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
      auto q_fixed = static_cast<int32_t>(TfLiteRound(q * (1LL << 15)));
      data->input_multiplier = static_cast<int16_t>(q_fixed);

      int16_t input_range_radius =
          CalculateInputRadius(kInputIntegerBits, data->input_left_shift, 15);
      data->input_range_radius = input_range_radius;
    }
  }

  if (kernel_type == kGenericOptimized || kernel_type == kReference) {
    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE(context, output->params.scale == 1. / 256);
      LUTPopulate<uint8_t>(
          input->params.scale, input->params.zero_point, output->params.scale,
          output->params.zero_point,
          [](float value) { return 1.0f / (1.0f + std::exp(-value)); },
          data->lut_uint8);
    } else if (input->type == kTfLiteInt8) {
      TF_LITE_ENSURE(context, output->params.scale == 1. / 256);
      LUTPopulate<int8_t>(
          input->params.scale, input->params.zero_point, output->params.scale,
          output->params.zero_point,
          [](float value) { return 1.0f / (1.0f + std::exp(-value)); },
          data->lut_int8);
    } else if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE(context, output->params.scale == 1. / 32768);
      TF_LITE_ENSURE(context, output->params.zero_point == 0);
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
    bool param_scale_pot =
        CheckedLog2(input->params.scale, &input_scale_log2_rounded);

    data->input_left_shift =
        (15 - kInputIntegerBits) + input_scale_log2_rounded;
    param_scale_pot &= (data->input_left_shift == 0);

    if (!param_scale_pot) {
      // Calculate multiplier to change input scale to 1/(3*4096)
      // as required by the table lookup.
      // In this scaling +/-2^17 represents +/-10.7
      double multiplier = input->params.scale * 4096.0 * 3.0;

      data->input_left_shift = 0;

      while (multiplier <= 32767.0 / 2.0 && data->input_left_shift <= 30) {
        data->input_left_shift++;
        multiplier = multiplier * 2.0;
      }

      data->input_multiplier = static_cast<int32_t>(multiplier);
    }

    int output_scale_log2_rounded;
    TF_LITE_ENSURE(
        context, CheckedLog2(output->params.scale, &output_scale_log2_rounded));
    TF_LITE_ENSURE_EQ(context, output_scale_log2_rounded,
                      -kOutputFractionalBits);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <KernelType kernel_type>
TfLiteStatus SoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  SoftmaxOpData* data = reinterpret_cast<SoftmaxOpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  TF_LITE_ENSURE(context, NumDimensions(input) >= 1);

  if (input->type == kTfLiteInt8 && output->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, -128);
    TF_LITE_ENSURE_NEAR(context, output->params.scale, 1.f / 256,
                        (0.001f * 1.f / 256));
  } else if (input->type == kTfLiteInt16 && output->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
    TF_LITE_ENSURE_NEAR(context, output->params.scale, 1.f / 32768,
                        (0.001f * 1.f / 32768));
  }

  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    if (kernel_type == kReference) {
      const int kScaledDiffIntegerBits = 5;
      int input_left_shift;
      tflite::PreprocessSoftmaxScaling(
          static_cast<double>(params->beta),
          static_cast<double>(input->params.scale), kScaledDiffIntegerBits,
          &data->params.input_multiplier, &input_left_shift);
      data->params.input_left_shift = input_left_shift;
      data->params.diff_min =
          -1.0 * tflite::CalculateInputRadius(kScaledDiffIntegerBits,
                                              input_left_shift);
    } else {
      switch (output->type) {
        case kTfLiteUInt8:
        case kTfLiteInt8:
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
          // Only apply when both input & output are uint8/int8 & build with
          // clang on aarch64.
          // TODO(b/143709993): Port to ARMv7 and other platforms.
          data->params.uint8_table1 = data->uint8_table1;
          data->params.uint8_table2 = data->uint8_table2;
          optimized_ops::PopulateSoftmaxUInt8LookupTable(
              &data->params, input->params.scale, params->beta);
          break;
#endif
        case kTfLiteInt16:
        default:
          data->params.table = data->table;
          optimized_ops::PopulateSoftmaxLookupTable(
              &data->params, input->params.scale, params->beta);
      }

      data->params.zero_point = output->params.zero_point;
      data->params.scale = output->params.scale;
    }
  } else if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    const int32_t range = std::numeric_limits<int16_t>::max() -
                          std::numeric_limits<int16_t>::min();
    // Exp LUT is only used with negative values. Generate the LUT in the input
    // range [-10.0; 0] as we consider resulting values smaller than exp(-10)
    // insignificant. Use a symmetric output range of [-1.0; 1.0] and double as
    // FloatT for backward compatibility.
    data->params.exp_lut = data->exp_lut;
    LUTPopulate<int16_t>(
        10.0 / range, std::numeric_limits<int16_t>::max(), 2.0 / range, 0,
        [](double value) { return std::exp(value); }, data->params.exp_lut);

    // Input is in the [0; 1] range and use a symmetric output range of
    // [-1.0; 1.0] and double as FloatT for backward compatibility.
    data->params.one_over_one_plus_x_lut = data->one_over_one_plus_x_lut;
    LUTPopulate<int16_t>(
        1.0 / range, std::numeric_limits<int16_t>::min(), 2.0 / range, 0,
        [](double value) { return 1.0 / (1.0 + value); },
        data->params.one_over_one_plus_x_lut);

    data->params.zero_point = output->params.zero_point;
    data->params.scale = output->params.scale;

    double input_scale_beta_rescale =
        input->params.scale * params->beta /
        (10.0 / 65535.0);  // scale the input_diff such that [-65535, 0]
                           // correspond to [-10.0, 0.0]
    QuantizeMultiplier(input_scale_beta_rescale, &data->params.input_multiplier,
                       &data->params.input_left_shift);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <KernelType kernel_type>
TfLiteStatus LogSoftmaxPrepare(TfLiteContext* context, TfLiteNode* node) {
  LogSoftmaxOpData* data = reinterpret_cast<LogSoftmaxOpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, output->type);

  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
    TF_LITE_ENSURE_EQ(context, output->params.scale, 16.0 / 256);
    static const double kBeta = 1.0;
    if (input->type == kTfLiteUInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 255);
    }
    if (input->type == kTfLiteInt8) {
      TF_LITE_ENSURE_EQ(context, output->params.zero_point, 127);
    }

    if (kernel_type == kReference) {
      const int kScaledDiffIntegerBits = 5;
      int input_left_shift;
      int reverse_scaling_right_shift;
      tflite::PreprocessLogSoftmaxScalingExp(
          kBeta, static_cast<double>(input->params.scale),
          kScaledDiffIntegerBits, &data->params.input_multiplier,
          &input_left_shift, &data->params.reverse_scaling_divisor,
          &reverse_scaling_right_shift);
      reverse_scaling_right_shift *= -1;
      data->params.input_left_shift = input_left_shift;
      data->params.reverse_scaling_right_shift = reverse_scaling_right_shift;
      data->params.diff_min = -tflite::CalculateInputRadius(
          kScaledDiffIntegerBits, input_left_shift);
    } else {
      data->params.table = data->f_table;
      optimized_ops::PopulateSoftmaxLookupTable(&data->params,
                                                input->params.scale, kBeta);
      data->params.zero_point = output->params.zero_point;
      data->params.scale = output->params.scale;
    }
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

TfLiteStatus PreluPrepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* alpha;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &alpha));
  PreluOpData* data = reinterpret_cast<PreluOpData*>(node->user_data);

  TF_LITE_ENSURE_TYPES_EQ(context, input->type, alpha->type);

  output->type = input->type;

  if (output->type == kTfLiteUInt8 || output->type == kTfLiteInt8) {
    // prelu(x) = x if x >= 0 else x * alpha.
    // So if we translate that for quantized computation:
    //
    // input_float = (input_q - input_zp) * input_scale
    // output_float = (output_q - output_zp) * output_scale
    // alpha_float = (alpha_q - alpha_zp) * alpha_scale
    //
    // When input_q - input_zp >= 0:
    // output_q = (input_q - input_zp) * input_scale / output_scale + output_q
    // else:
    // output_q = (input_q - input_zp) * (alpha_q - alpha_zp) * input_scale
    //            * alpha_scale / output_scale + output_q
    //
    // So for input_q - input_zp >= 0:
    // output real multiplier 1 is input_scale / output_scale;
    // for input_q - input_zp < 0:
    // output real multiplier 2 is input_scale  * alpha_scale/ output_scale.
    double real_multiplier_1 = input->params.scale / output->params.scale;
    double real_multiplier_2 =
        input->params.scale * alpha->params.scale / output->params.scale;
    QuantizeMultiplier(real_multiplier_1, &data->output_multiplier_1,
                       &data->output_shift_1);
    QuantizeMultiplier(real_multiplier_2, &data->output_multiplier_2,
                       &data->output_shift_2);
  }

  data->requires_broadcast = !HaveSameShapes(input, alpha);
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
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const ReluOpData* data = reinterpret_cast<ReluOpData*>(node->user_data);
  switch (input->type) {
    case kTfLiteFloat32: {
#ifdef TFLITE_KERNEL_USE_XNNPACK
      const size_t channel_dim = 1;
      const size_t batch_size = NumElements(input->dims);
      CpuBackendContext* cpu_backend_context =
          CpuBackendContext::GetFromContext(context);
      pthreadpool_t threadpool = cpu_backend_context->get_xnnpack_threadpool();
      xnn_status status = xnn_run_clamp_nc_f32(
          channel_dim, channel_dim, channel_dim, batch_size,
          GetTensorData<float>(input), GetTensorData<float>(output),
          /*min=*/0.0f, /*max=*/std::numeric_limits<float>::infinity(),
          /*flags=*/XNN_FLAG_YIELD_WORKERS, threadpool);
      if (status == xnn_status_success) {
        return kTfLiteOk;
      }
      TFLITE_LOG(TFLITE_LOG_INFO,
                 "Failed to run xnnpack xnn_run_clamp_nc_f32. Error code: %d",
                 status);
#endif
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
    case kTfLiteInt16: {
      QuantizedReluX<int16_t>(0.0f, std::numeric_limits<float>::infinity(),
                              input, output, data);
    } break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Only float32, uint8, int8 and int16 are supported "
                         "currently, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus Relu1Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const ReluOpData* data = reinterpret_cast<ReluOpData*>(node->user_data);
  switch (input->type) {
    case kTfLiteFloat32: {
#ifdef TFLITE_KERNEL_USE_XNNPACK
      const size_t channel_dim = 1;
      const size_t batch_size = NumElements(input->dims);
      CpuBackendContext* cpu_backend_context =
          CpuBackendContext::GetFromContext(context);
      pthreadpool_t threadpool = cpu_backend_context->get_xnnpack_threadpool();
      xnn_status status = xnn_run_clamp_nc_f32(
          channel_dim, channel_dim, channel_dim, batch_size,
          GetTensorData<float>(input), GetTensorData<float>(output),
          /*min=*/-1.0f, /*max=*/1.0f, /*flags=*/XNN_FLAG_YIELD_WORKERS,
          threadpool);
      if (status == xnn_status_success) {
        return kTfLiteOk;
      }
      TFLITE_LOG(TFLITE_LOG_INFO,
                 "Failed to run xnnpack xnn_run_clamp_nc_f32. Error code: %d",
                 status);
#endif
      optimized_ops::Relu1(GetTensorShape(input), GetTensorData<float>(input),
                           GetTensorShape(output),
                           GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      QuantizedReluX<uint8_t>(-1.0f, 1.0f, input, output, data);
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      QuantizedReluX<int8_t>(-1, 1, input, output, data);
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Only float32, uint8, int8 supported "
                         "currently, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

template <KernelType kernel_type>
TfLiteStatus HardSwishEval(TfLiteContext* context, TfLiteNode* node) {
  HardSwishData* data = static_cast<HardSwishData*>(node->user_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  switch (input->type) {
    case kTfLiteFloat32: {
      if (kernel_type == kReference) {
        reference_ops::HardSwish(
            GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
#ifdef TFLITE_KERNEL_USE_XNNPACK
        const size_t channel_dim = 1;
        const size_t batch_size = NumElements(input->dims);
        CpuBackendContext* cpu_backend_context =
            CpuBackendContext::GetFromContext(context);
        pthreadpool_t threadpool =
            cpu_backend_context->get_xnnpack_threadpool();
        xnn_status status = xnn_run_hardswish_nc_f32(
            channel_dim, channel_dim, channel_dim, batch_size,
            GetTensorData<float>(input), GetTensorData<float>(output),
            /*flags=*/XNN_FLAG_YIELD_WORKERS, threadpool);
        if (status == xnn_status_success) {
          return kTfLiteOk;
        }
        TFLITE_LOG(
            TFLITE_LOG_INFO,
            "Failed to run xnnpack xnn_run_hardswish_nc_f32. Error code: %d",
            status);
#endif
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
      TF_LITE_KERNEL_LOG(
          context,
          "Only float32, uint8 and int8 are supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TfLiteStatus Relu0to1Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const ReluOpData* data = reinterpret_cast<ReluOpData*>(node->user_data);
  switch (input->type) {
    case kTfLiteFloat32: {
#ifdef TFLITE_KERNEL_USE_XNNPACK
      const size_t channel_dim = 1;
      const size_t batch_size = NumElements(input->dims);
      CpuBackendContext* cpu_backend_context =
          CpuBackendContext::GetFromContext(context);
      pthreadpool_t threadpool = cpu_backend_context->get_xnnpack_threadpool();
      xnn_status status = xnn_run_clamp_nc_f32(
          channel_dim, channel_dim, channel_dim, batch_size,
          GetTensorData<float>(input), GetTensorData<float>(output),
          /*min=*/0.0f, /*max=*/1.0f, /*flags=*/XNN_FLAG_YIELD_WORKERS,
          threadpool);
      if (status == xnn_status_success) {
        return kTfLiteOk;
      }
      TFLITE_LOG(TFLITE_LOG_INFO,
                 "Failed to run xnnpack xnn_run_clamp_nc_f32. Error code: %d",
                 status);
#endif
      optimized_ops::Relu0To1(
          GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      QuantizedReluX<uint8_t>(0.0f, 1.0f, input, output, data);
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      QuantizedReluX<int8_t>(0, 1, input, output, data);
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Only float32, uint8, int8 supported "
                         "currently, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TfLiteStatus Relu6Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  ReluOpData* data = reinterpret_cast<ReluOpData*>(node->user_data);
  switch (input->type) {
    case kTfLiteFloat32: {
#ifdef TFLITE_KERNEL_USE_XNNPACK
      const size_t channel_dim = 1;
      const size_t batch_size = NumElements(input->dims);
      CpuBackendContext* cpu_backend_context =
          CpuBackendContext::GetFromContext(context);
      pthreadpool_t threadpool = cpu_backend_context->get_xnnpack_threadpool();
      xnn_status status = xnn_run_clamp_nc_f32(
          channel_dim, channel_dim, channel_dim, batch_size,
          GetTensorData<float>(input), GetTensorData<float>(output),
          /*min=*/0.0f, /*max=*/6.0f, /*flags=*/XNN_FLAG_YIELD_WORKERS,
          threadpool);
      if (status == xnn_status_success) {
        return kTfLiteOk;
      }
      TFLITE_LOG(TFLITE_LOG_INFO,
                 "Failed to run xnnpack xnn_run_clamp_nc_f32. Error code: %d",
                 status);
#endif
      size_t elements = input->bytes / sizeof(float);
      const float* in = GetTensorData<float>(input);
      const float* in_end = in + elements;
      float* out = GetTensorData<float>(output);
      for (; in < in_end; in++, out++) *out = std::min(std::max(0.f, *in), 6.f);
      return kTfLiteOk;
    }
    case kTfLiteUInt8:
      QuantizedReluX<uint8_t>(0.0f, 6.0f, input, output, data);
      return kTfLiteOk;
    case kTfLiteInt8: {
      QuantizedReluX<int8_t>(0.0f, 6.0f, input, output, data);
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      QuantizedReluX<int16_t>(0.0f, 6.0f, input, output, data);
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Only float32, uint8, int8 and int16 are supported "
                         "currently, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

template <KernelType kernel_type>
TfLiteStatus TanhEval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  switch (input->type) {
    case kTfLiteFloat32: {
      if (kernel_type == kReference) {
        reference_ops::Tanh(GetTensorShape(input), GetTensorData<float>(input),
                            GetTensorShape(output),
                            GetTensorData<float>(output));
      } else {
#ifdef TFLITE_KERNEL_USE_XNNPACK
        const size_t channel_dim = 1;
        const size_t batch_size = NumElements(input->dims);
        CpuBackendContext* cpu_backend_context =
            CpuBackendContext::GetFromContext(context);
        pthreadpool_t threadpool =
            cpu_backend_context->get_xnnpack_threadpool();
        xnn_status status = xnn_run_tanh_nc_f32(
            channel_dim, channel_dim, channel_dim, batch_size,
            GetTensorData<float>(input), GetTensorData<float>(output),
            /*flags=*/XNN_FLAG_YIELD_WORKERS, threadpool);
        if (status == xnn_status_success) {
          return kTfLiteOk;
        }
        TFLITE_LOG(TFLITE_LOG_INFO,
                   "Failed to run xnnpack xnn_run_tanh_nc_f32. Error code: %d",
                   status);
#endif
        optimized_ops::Tanh(GetTensorShape(input), GetTensorData<float>(input),
                            GetTensorShape(output),
                            GetTensorData<float>(output));
      }
      return kTfLiteOk;
    } break;
    case kTfLiteInt16: {
      TanhParams params;
      params.input_left_shift = data->input_left_shift;
      if (kernel_type == kReference || (data->input_multiplier > 0)) {
        reference_integer_ops::Tanh(
            data->input_multiplier, data->input_left_shift,
            GetTensorShape(input), GetTensorData<int16_t>(input),
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
        optimized_ops::Tanh16bitPrecision(
            params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<uint8_t>(output));
      } else {
        optimized_integer_ops::LookupTable(
            GetTensorData<uint8_t>(input),
            MatchingFlatSize(GetTensorShape(input), GetTensorShape(output)),
            data->lut_uint8, GetTensorData<uint8_t>(output));
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
        optimized_ops::Tanh16bitPrecision(
            params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(output), GetTensorData<int8_t>(output));
      } else {
        optimized_integer_ops::LookupTable(
            GetTensorData<int8_t>(input),
            MatchingFlatSize(GetTensorShape(input), GetTensorShape(output)),
            data->lut_int8, GetTensorData<int8_t>(output));
      }
      return kTfLiteOk;
    } break;
    default:
      TF_LITE_KERNEL_LOG(context,
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

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  switch (input->type) {
    case kTfLiteFloat32: {
      if (kernel_type == kReference) {
        reference_ops::Logistic(
            GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      } else {
#ifdef TFLITE_KERNEL_USE_XNNPACK
        const size_t channel_dim = 1;
        const size_t batch_size = NumElements(input->dims);
        CpuBackendContext* cpu_backend_context =
            CpuBackendContext::GetFromContext(context);
        pthreadpool_t threadpool =
            cpu_backend_context->get_xnnpack_threadpool();
        xnn_status status = xnn_run_sigmoid_nc_f32(
            channel_dim, channel_dim, channel_dim, batch_size,
            GetTensorData<float>(input), GetTensorData<float>(output),
            /*flags=*/XNN_FLAG_YIELD_WORKERS, threadpool);
        if (status == xnn_status_success) {
          return kTfLiteOk;
        }
        TFLITE_LOG(
            TFLITE_LOG_INFO,
            "Failed to run xnnpack xnn_run_sigmoid_nc_f32. Error code: %d",
            status);
#endif
        optimized_ops::Logistic(
            GetTensorShape(input), GetTensorData<float>(input),
            GetTensorShape(output), GetTensorData<float>(output));
      }
      break;
    }
    case kTfLiteInt16: {
      LogisticParams params;
      if (kernel_type == kReference || (data->input_multiplier > 0)) {
        const int size =
            MatchingFlatSize(GetTensorShape(input), GetTensorShape(output));

        reference_integer_ops::Logistic(
            data->input_multiplier, data->input_left_shift, size,
            GetTensorData<int16_t>(input), GetTensorData<int16_t>(output));
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
        optimized_ops::Logistic16bitPrecision(
            params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(output), GetTensorData<uint8_t>(output));
      } else {
        optimized_integer_ops::LookupTable(
            GetTensorData<uint8_t>(input),
            MatchingFlatSize(GetTensorShape(input), GetTensorShape(output)),
            data->lut_uint8, GetTensorData<uint8_t>(output));
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
        optimized_ops::Logistic16bitPrecision(
            params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(output), GetTensorData<int8_t>(output));
      } else {
        optimized_integer_ops::LookupTable(
            GetTensorData<int8_t>(input),
            MatchingFlatSize(GetTensorShape(input), GetTensorShape(output)),
            data->lut_int8, GetTensorData<int8_t>(output));
      }
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Only float32, uint8, int16 and int8 are supported "
                         "currently, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus SoftmaxFloat(TfLiteContext* context, const TfLiteTensor* input,
                          TfLiteTensor* output, TfLiteSoftmaxParams* params,
                          KernelType kernel_type = kGenericOptimized) {
  SoftmaxParams op_params;
  op_params.beta = params->beta;
  if (kernel_type == kReference) {
    reference_ops::Softmax(op_params, GetTensorShape(input),
                           GetTensorData<float>(input), GetTensorShape(output),
                           GetTensorData<float>(output));
  } else {
    optimized_ops::Softmax(op_params, GetTensorShape(input),
                           GetTensorData<float>(input), GetTensorShape(output),
                           GetTensorData<float>(output),
                           CpuBackendContext::GetFromContext(context));
  }
  return kTfLiteOk;
}

template <typename In, typename Out>
TfLiteStatus SoftmaxQuantized(TfLiteContext* context, const TfLiteTensor* input,
                              TfLiteTensor* output, SoftmaxOpData* data,
                              KernelType kernel_type = kGenericOptimized) {
  if (kernel_type == kReference) {
    reference_ops::Softmax(data->params, GetTensorShape(input),
                           GetTensorData<In>(input), GetTensorShape(output),
                           GetTensorData<Out>(output));
  } else {
    optimized_ops::Softmax(data->params, GetTensorShape(input),
                           GetTensorData<In>(input), GetTensorShape(output),
                           GetTensorData<Out>(output));
  }
  return kTfLiteOk;
}

template <>
TfLiteStatus SoftmaxQuantized<int8_t, int8_t>(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              TfLiteTensor* output,
                                              SoftmaxOpData* data,
                                              KernelType kernel_type) {
  if (kernel_type == kReference) {
    reference_ops::Softmax(data->params, GetTensorShape(input),
                           GetTensorData<int8_t>(input), GetTensorShape(output),
                           GetTensorData<int8_t>(output));
  } else {
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
    optimized_ops::SoftmaxInt8LUT(
        data->params, GetTensorShape(input), GetTensorData<int8_t>(input),
        GetTensorShape(output), GetTensorData<int8_t>(output));
#else
    optimized_ops::Softmax(data->params, GetTensorShape(input),
                           GetTensorData<int8_t>(input), GetTensorShape(output),
                           GetTensorData<int8_t>(output));
#endif
  }
  return kTfLiteOk;
}

template <>
TfLiteStatus SoftmaxQuantized<uint8_t, uint8_t>(TfLiteContext* context,
                                                const TfLiteTensor* input,
                                                TfLiteTensor* output,
                                                SoftmaxOpData* data,
                                                KernelType kernel_type) {
  if (kernel_type == kReference) {
    reference_ops::Softmax(
        data->params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
  } else {
#ifdef TFLITE_SOFTMAX_USE_UINT16_LUT
    optimized_ops::SoftmaxInt8LUT(
        data->params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
#else
    optimized_ops::Softmax(
        data->params, GetTensorShape(input), GetTensorData<uint8_t>(input),
        GetTensorShape(output), GetTensorData<uint8_t>(output));
#endif
  }
  return kTfLiteOk;
}

template <>
TfLiteStatus SoftmaxQuantized<int16, int16>(TfLiteContext* context,
                                            const TfLiteTensor* input,
                                            TfLiteTensor* output,
                                            SoftmaxOpData* data,
                                            KernelType kernel_type) {
  if (NumDimensions(input) >= 1 && NumDimensions(input) <= 4) {
    reference_ops::SoftmaxInt16(
        data->params, GetTensorShape(input), GetTensorData<int16_t>(input),
        GetTensorShape(output), GetTensorData<int16_t>(output));
    return kTfLiteOk;
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "Only 1D, 2D, 3D and 4D tensors supported for int16 "
                       "input with int16 output, got %dD.",
                       NumDimensions(input));
    return kTfLiteError;
  }
}

template <KernelType kernel_type>
TfLiteStatus SoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSoftmaxParams*>(node->builtin_data);
  SoftmaxOpData* data = reinterpret_cast<SoftmaxOpData*>(node->user_data);

  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  switch (input->type) {
    case kTfLiteFloat32: {
      return SoftmaxFloat(context, input, output, params, kernel_type);
    }
    case kTfLiteUInt8: {
      switch (output->type) {
        case kTfLiteUInt8:
          return SoftmaxQuantized<uint8_t, uint8_t>(context, input, output,
                                                    data, kernel_type);
        case kTfLiteInt16:
          return SoftmaxQuantized<uint8_t, int16_t>(context, input, output,
                                                    data, kernel_type);
        default:
          TF_LITE_KERNEL_LOG(context,
                             "Only uint8_t and int16_t outputs are supported "
                             "with uint8_t inputs currently, got %s.",
                             TfLiteTypeGetName(output->type));
          return kTfLiteError;
      }
    }
    case kTfLiteInt8: {
      switch (output->type) {
        case kTfLiteInt8:
          return SoftmaxQuantized<int8_t, int8_t>(context, input, output, data,
                                                  kernel_type);
        case kTfLiteInt16:
          return SoftmaxQuantized<int8_t, int16_t>(context, input, output, data,
                                                   kernel_type);
        default:
          TF_LITE_KERNEL_LOG(context,
                             "Only int8_t and int16_t outputs are supported "
                             "with int8_t inputs currently, got %s.",
                             TfLiteTypeGetName(output->type));
          return kTfLiteError;
      }
    }
    case kTfLiteInt16: {
      return SoftmaxQuantized<int16_t, int16_t>(context, input, output, data,
                                                kernel_type);
    }

    default:
      TF_LITE_KERNEL_LOG(context,
                         "Only float32, uint8_t, Int8_t, Int16_t are supported "
                         "currently, got %s.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

template <KernelType kernel_type>
TfLiteStatus LogSoftmaxEval(TfLiteContext* context, TfLiteNode* node) {
  const LogSoftmaxOpData* data =
      reinterpret_cast<LogSoftmaxOpData*>(node->user_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
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
      const SoftmaxParams& op_params = data->params;
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
      const SoftmaxParams& op_params = data->params;
      if (kernel_type == kGenericOptimized) {
        optimized_ops::LogSoftmax(
            op_params, input->params.scale, GetTensorShape(input),
            GetTensorData<int8_t>(input), GetTensorShape(output),
            GetTensorData<int8_t>(output));
      } else {
        const auto input_shape = GetTensorShape(input);
        const auto output_shape = GetTensorShape(output);
        const int trailing_dim = input_shape.DimensionsCount() - 1;
        const int outer_size =
            MatchingFlatSizeSkipDim(input_shape, trailing_dim, output_shape);
        const int depth =
            MatchingDim(input_shape, trailing_dim, output_shape, trailing_dim);
        reference_integer_ops::LogSoftmax(
            op_params.input_multiplier, op_params.input_left_shift,
            op_params.reverse_scaling_divisor,
            op_params.reverse_scaling_right_shift, op_params.diff_min,
            outer_size, depth, GetTensorData<int8_t>(input),
            GetTensorData<int8_t>(output));
      }
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(
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

template <KernelType kernel_type>
TfLiteStatus PreluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  const TfLiteTensor* alpha;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &alpha));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const PreluOpData* data = reinterpret_cast<PreluOpData*>(node->user_data);
  switch (input->type) {
    case kTfLiteFloat32: {
      if (kernel_type == kGenericOptimized) {
        tflite::ArithmeticParams op_params;
        bool need_broadcast = optimized_ops::ProcessBroadcastShapes(
            GetTensorShape(input), GetTensorShape(alpha), &op_params);
        if (need_broadcast) {
          optimized_ops::BroadcastPReluDispatch(
              op_params, GetTensorShape(input), GetTensorData<float>(input),
              GetTensorShape(alpha), GetTensorData<float>(alpha),
              GetTensorShape(output), GetTensorData<float>(output),
              ApplyPrelu<float>);
        } else {
          const int flat_size =
              MatchingElementsSize(GetTensorShape(input), GetTensorShape(alpha),
                                   GetTensorShape(output));
          optimized_ops::PReluElementWise(
              flat_size, op_params, GetTensorData<float>(alpha),
              GetTensorData<float>(input), GetTensorData<float>(output));
        }
      } else {
        if (data->requires_broadcast) {
          reference_ops::BroadcastBinaryFunction4DSlow<float, float, float>(
              GetTensorShape(input), GetTensorData<float>(input),
              GetTensorShape(alpha), GetTensorData<float>(alpha),
              GetTensorShape(output), GetTensorData<float>(output),
              ApplyPrelu<float>);
        } else {
          reference_ops::BinaryFunction<float, float, float>(
              GetTensorShape(input), GetTensorData<float>(input),
              GetTensorShape(alpha), GetTensorData<float>(alpha),
              GetTensorShape(output), GetTensorData<float>(output),
              ApplyPrelu<float>);
        }
      }
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      PreluParams op_params;
      op_params.input_offset = -input->params.zero_point;
      op_params.alpha_offset = -alpha->params.zero_point;
      op_params.output_offset = output->params.zero_point;
      op_params.output_multiplier_1 = data->output_multiplier_1;
      op_params.output_shift_1 = data->output_shift_1;
      op_params.output_multiplier_2 = data->output_multiplier_2;
      op_params.output_shift_2 = data->output_shift_2;
      if (data->requires_broadcast) {
        reference_ops::BroadcastPrelu4DSlow(
            op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(alpha), GetTensorData<uint8_t>(alpha),
            GetTensorShape(output), GetTensorData<uint8_t>(output));
      } else {
        reference_ops::Prelu(
            op_params, GetTensorShape(input), GetTensorData<uint8_t>(input),
            GetTensorShape(alpha), GetTensorData<uint8_t>(alpha),
            GetTensorShape(output), GetTensorData<uint8_t>(output));
      }
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      PreluParams op_params;
      op_params.input_offset = -input->params.zero_point;
      op_params.alpha_offset = -alpha->params.zero_point;
      op_params.output_offset = output->params.zero_point;
      op_params.output_multiplier_1 = data->output_multiplier_1;
      op_params.output_shift_1 = data->output_shift_1;
      op_params.output_multiplier_2 = data->output_multiplier_2;
      op_params.output_shift_2 = data->output_shift_2;
      if (data->requires_broadcast) {
        reference_ops::BroadcastPrelu4DSlow(
            op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(alpha), GetTensorData<int8_t>(alpha),
            GetTensorShape(output), GetTensorData<int8_t>(output));
      } else {
        reference_ops::Prelu(
            op_params, GetTensorShape(input), GetTensorData<int8_t>(input),
            GetTensorShape(alpha), GetTensorData<int8_t>(alpha),
            GetTensorShape(output), GetTensorData<int8_t>(output));
      }
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Only float32 and uint8 and int8 are supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

template <KernelType kernel_type, typename T>
void QuantizeLeakyRelu(const TfLiteTensor* input, TfLiteTensor* output,
                       const LeakyReluOpData* data) {
  LeakyReluParams op_params;

  op_params.input_offset = input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.output_multiplier_alpha = data->output_multiplier_alpha;
  op_params.output_shift_alpha = data->output_shift_alpha;
  op_params.output_multiplier_identity = data->output_multiplier_identity;
  op_params.output_shift_identity = data->output_shift_identity;
  if (kernel_type != KernelType::kReference && input->type == kTfLiteInt16) {
    optimized_integer_ops::QuantizeLeakyRelu(
        op_params, GetTensorShape(input), GetTensorData<int16>(input),
        GetTensorShape(output), GetTensorData<int16>(output));
  } else {
    reference_ops::QuantizeLeakyRelu(
        op_params, GetTensorShape(input), GetTensorData<T>(input),
        GetTensorShape(output), GetTensorData<T>(output));
  }
}

template <KernelType kernel_type>
TfLiteStatus LeakyReluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const auto* params =
      reinterpret_cast<TfLiteLeakyReluParams*>(node->builtin_data);
  const LeakyReluOpData* data =
      reinterpret_cast<LeakyReluOpData*>(node->user_data);

  LeakyReluParams op_params;
  switch (input->type) {
    case kTfLiteFloat32: {
#ifdef TFLITE_KERNEL_USE_XNNPACK
      const size_t channel_dim = 1;
      const size_t batch_size = NumElements(input->dims);
      CpuBackendContext* cpu_backend_context =
          CpuBackendContext::GetFromContext(context);
      pthreadpool_t threadpool = cpu_backend_context->get_xnnpack_threadpool();
      xnn_status status = xnn_run_leaky_relu_nc_f32(
          channel_dim, channel_dim, channel_dim, batch_size,
          GetTensorData<float>(input), GetTensorData<float>(output),
          params->alpha, /*flags=*/XNN_FLAG_YIELD_WORKERS, threadpool);
      if (status == xnn_status_success) {
        return kTfLiteOk;
      }
      TFLITE_LOG(
          TFLITE_LOG_INFO,
          "Failed to run xnnpack xnn_run_leaky_relu_nc_f32. Error code: %d",
          status);
#endif
      op_params.alpha = params->alpha;
      optimized_ops::LeakyRelu(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteUInt8: {
      QuantizeLeakyRelu<kernel_type, uint8_t>(input, output, data);
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      QuantizeLeakyRelu<kernel_type, int8_t>(input, output, data);
      return kTfLiteOk;
    }
    case kTfLiteInt16: {
      QuantizeLeakyRelu<kernel_type, int16_t>(input, output, data);
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Only float32, int8, int16 and uint8 is supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TfLiteStatus EluPrepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // Use LUT to handle quantized elu path.
  if (input->type == kTfLiteInt8) {
    LUTPopulate<int8_t>(
        input->params.scale, input->params.zero_point, output->params.scale,
        output->params.zero_point,
        [](float value) { return value < 0.0f ? std::expm1(value) : value; },
        data->lut_int8);
  }
  return GenericPrepare(context, node);
}

TfLiteStatus EluEval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  switch (input->type) {
    case kTfLiteFloat32: {
#ifdef TFLITE_KERNEL_USE_XNNPACK
      const size_t channel_dim = 1;
      const size_t batch_size = NumElements(input->dims);
      CpuBackendContext* cpu_backend_context =
          CpuBackendContext::GetFromContext(context);
      pthreadpool_t threadpool = cpu_backend_context->get_xnnpack_threadpool();
      xnn_status status = xnn_run_elu_nc_f32(
          channel_dim, channel_dim, channel_dim, batch_size,
          GetTensorData<float>(input), GetTensorData<float>(output),
          /*alpha=*/1.0f, /*flags=*/XNN_FLAG_YIELD_WORKERS, threadpool);
      if (status == xnn_status_success) {
        return kTfLiteOk;
      }
      TFLITE_LOG(TFLITE_LOG_INFO,
                 "Failed to run xnnpack xnn_run_elu_nc_f32. Error code: %d",
                 status);
#endif
      optimized_ops::Elu(GetTensorShape(input), GetTensorData<float>(input),
                         GetTensorShape(output), GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kTfLiteInt8: {
      OpData* data = reinterpret_cast<OpData*>(node->user_data);
      optimized_integer_ops::LookupTable(
          GetTensorData<int8_t>(input),
          MatchingFlatSize(GetTensorShape(input), GetTensorShape(output)),
          data->lut_int8, GetTensorData<int8_t>(output));
      return kTfLiteOk;
    }
    default:
      TF_LITE_KERNEL_LOG(
          context, "Only float32 and int8 is supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
}

TfLiteStatus GeluPrepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  auto* params = reinterpret_cast<TfLiteGeluParams*>(node->builtin_data);

  // PopulateLookupTable takes a function pointer for TFLM compatibility
  // reasons, we thus can't directly pass a std::function<float(float)> or
  // lambda with capture. Add an if/else condition to avoid capture of
  // params->approximate.
  if (input->type == kTfLiteInt8) {
    LUTPopulate<int8_t>(input->params.scale, input->params.zero_point,
                        output->params.scale, output->params.zero_point,
                        params->approximate
                            ? reference_ops::GeluTransformApproximate
                            : reference_ops::GeluTransform,
                        data->lut_int8);
  } else if (input->type == kTfLiteUInt8) {
    LUTPopulate<uint8_t>(input->params.scale, input->params.zero_point,
                         output->params.scale, output->params.zero_point,
                         params->approximate
                             ? reference_ops::GeluTransformApproximate
                             : reference_ops::GeluTransform,
                         data->lut_uint8);
  }
  return GenericPrepare(context, node);
}

TfLiteStatus GeluEval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  auto* params = reinterpret_cast<TfLiteGeluParams*>(node->builtin_data);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));

  switch (input->type) {
    case kTfLiteFloat32:
      reference_ops::Gelu(GetTensorShape(input), GetTensorData<float>(input),
                          params->approximate, GetTensorShape(output),
                          GetTensorData<float>(output));
      return kTfLiteOk;
    case kTfLiteUInt8:
      optimized_integer_ops::LookupTable(
          GetTensorData<uint8_t>(input),
          MatchingFlatSize(GetTensorShape(input), GetTensorShape(output)),
          data->lut_uint8, GetTensorData<uint8_t>(output));
      return kTfLiteOk;
    case kTfLiteInt8:
      optimized_integer_ops::LookupTable(
          GetTensorData<int8_t>(input),
          MatchingFlatSize(GetTensorShape(input), GetTensorShape(output)),
          data->lut_int8, GetTensorData<int8_t>(output));
      return kTfLiteOk;
    default:
      TF_LITE_KERNEL_LOG(
          context, "Only float32, int8 and uint8 supported currently, got %s.",
          TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace activations

TfLiteRegistration* Register_ELU() {
  static TfLiteRegistration r = {activations::Init, activations::Free,
                                 activations::EluPrepare, activations::EluEval};
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

TfLiteRegistration* Register_RELU_0_TO_1() {
  static TfLiteRegistration r = {activations::ReluInit, activations::ReluFree,
                                 activations::ReluPrepare,
                                 activations::Relu0to1Eval};
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

TfLiteRegistration* Register_SOFTMAX_REF() {
  static TfLiteRegistration r = {
      activations::SoftmaxInit,
      activations::SoftmaxFree,
      activations::SoftmaxPrepare<activations::kReference>,
      activations::SoftmaxEval<activations::kReference>,
      /*profiling_string=*/nullptr,
      /*builtin_code=*/0,
      /*custom_name=*/nullptr,
      /*version=*/0,
      /*registration_external=*/nullptr,
      /*async_kernel=*/nullptr,
      kTfLiteInplaceOpInput0Shared};
  return &r;
}

TfLiteRegistration* Register_SOFTMAX() {
  static TfLiteRegistration r = {
      activations::SoftmaxInit,
      activations::SoftmaxFree,
      activations::SoftmaxPrepare<activations::kGenericOptimized>,
      activations::SoftmaxEval<activations::kGenericOptimized>,
      /*profiling_string=*/nullptr,
      /*builtin_code=*/0,
      /*custom_name=*/nullptr,
      /*version=*/0,
      /*registration_external=*/nullptr,
      /*async_kernel=*/nullptr,
      kTfLiteInplaceOpInput0Shared};
  return &r;
}

TfLiteRegistration* Register_LOG_SOFTMAX_REF() {
  static TfLiteRegistration r = {
      activations::LogSoftmaxInit, activations::LogSoftmaxFree,
      activations::LogSoftmaxPrepare<activations::kReference>,
      activations::LogSoftmaxEval<activations::kReference>};
  return &r;
}

TfLiteRegistration* Register_LOG_SOFTMAX() {
  static TfLiteRegistration r = {
      activations::LogSoftmaxInit, activations::LogSoftmaxFree,
      activations::LogSoftmaxPrepare<activations::kGenericOptimized>,
      activations::LogSoftmaxEval<activations::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_PRELU_REF() {
  static TfLiteRegistration r = {
      activations::PreluInit, activations::PreluFree, activations::PreluPrepare,
      activations::PreluEval<activations::kReference>};
  return &r;
}

TfLiteRegistration* Register_PRELU() {
  static TfLiteRegistration r = {
      activations::PreluInit, activations::PreluFree, activations::PreluPrepare,
      activations::PreluEval<activations::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_LEAKY_RELU_REF() {
  static TfLiteRegistration r = {
      activations::LeakyReluInit, activations::LeakyReluFree,
      activations::LeakyReluPrepare,
      activations::LeakyReluEval<activations::kReference>};
  return &r;
}

TfLiteRegistration* Register_LEAKY_RELU() {
  static TfLiteRegistration r = {
      activations::LeakyReluInit, activations::LeakyReluFree,
      activations::LeakyReluPrepare,
      activations::LeakyReluEval<activations::kGenericOptimized>};
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

TfLiteRegistration* Register_GELU() {
  static TfLiteRegistration r = {activations::Init, activations::Free,
                                 activations::GeluPrepare,
                                 activations::GeluEval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
