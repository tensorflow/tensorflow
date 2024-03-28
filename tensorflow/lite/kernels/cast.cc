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
#include <algorithm>
#include <cmath>
#include <complex>
#include <cstddef>
#include <cstdint>
#include <limits>

#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/core/subgraph.h"
#include "tensorflow/lite/interpreter_options.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace cast {

namespace {

constexpr int kInputTensor = 0;
constexpr int kOutputTensor = 0;

void copyCast(const float* in, int32_t* out, int num_elements) {
  float min_int_float =
      std::nextafterf((float)std::numeric_limits<int32_t>::min(), 0);
  float max_int_float =
      std::nextafterf((float)std::numeric_limits<int32_t>::max(), 0);
  std::transform(in, in + num_elements, out, [=](float a) {
    return static_cast<int32_t>(
        std::max(std::min(a, max_int_float), min_int_float));
  });
}

void copyCast(const float* in, int16_t* out, int num_elements) {
  float min_int_float =
      std::nextafterf((float)std::numeric_limits<int16_t>::min(), 0);
  float max_int_float =
      std::nextafterf((float)std::numeric_limits<int16_t>::max(), 0);
  std::transform(in, in + num_elements, out, [=](float a) {
    return static_cast<int16_t>(
        std::max(std::min(a, max_int_float), min_int_float));
  });
}

template <typename FromT, typename ToT>
void copyCast(const FromT* in, ToT* out, int num_elements) {
  std::transform(in, in + num_elements, out,
                 [](FromT a) { return static_cast<ToT>(a); });
}

template <typename ToT>
void copyCast(const std::complex<float>* in, ToT* out, int num_elements) {
  std::transform(in, in + num_elements, out, [](std::complex<float> a) {
    return static_cast<ToT>(std::real(a));
  });
}

template <>
void copyCast(const std::complex<float>* in, std::complex<float>* out,
              int num_elements) {
  std::transform(in, in + num_elements, out,
                 [](std::complex<float> a) { return a; });
}

template <typename ToT>
void copyCast(const Eigen::half* in, ToT* out, int num_elements) {
  std::transform(in, in + num_elements, out, [](Eigen::half a) {
    return static_cast<ToT>(Eigen::half_impl::half_to_float(a));
  });
}

template <>
void copyCast(const Eigen::half* in, std::complex<float>* out,
              int num_elements) {
  std::transform(in, in + num_elements, out, [](Eigen::half a) {
    return std::complex<float>(Eigen::half_impl::half_to_float(a));
  });
}

template <typename FromT>
void copyCastToFloat16(const FromT* in, Eigen::half* out, int num_elements) {
  std::transform(in, in + num_elements, out, [](FromT a) {
    return Eigen::half_impl::float_to_half_rtne(static_cast<float>(a));
  });
}

template <>
void copyCastToFloat16(const std::complex<float>* in, Eigen::half* out,
                       int num_elements) {
  std::transform(in, in + num_elements, out, [](std::complex<float> a) {
    return Eigen::half_impl::float_to_half_rtne(std::real(a));
  });
}

template <>
void copyCastToFloat16(const Eigen::half* in, Eigen::half* out,
                       int num_elements) {
  std::transform(in, in + num_elements, out, [](Eigen::half a) { return a; });
}

TfLiteStatus castInt4ToFloat(TfLiteContext* context, const TfLiteTensor* in,
                             TfLiteTensor* out, int num_elements) {
  const int8_t* in_data = (const int8_t*)in->data.data;
  float* out_data = (float*)out->data.data;
  int i = 0;
#ifdef __ARM_NEON
  for (; i + 16 <= num_elements / 2; i += 16) {
    const int8x16_t v0_32 = vld1q_s8(&in_data[i]);
    const int8x16_t v0_32_low = vshrq_n_s8(vshlq_n_s8(v0_32, 4), 4);
    const int8x16_t v0_32_high = vshrq_n_s8(v0_32, 4);
    const int8x16x2_t vzipped = vzipq_s8(v0_32_low, v0_32_high);

    const int16x8_t v0_8 = vmovl_s8(vget_low_s8(vzipped.val[0]));
    const int16x8_t v8_15 = vmovl_s8(vget_high_s8(vzipped.val[0]));
    const int16x8_t v16_23 = vmovl_s8(vget_low_s8(vzipped.val[1]));
    const int16x8_t v24_31 = vmovl_s8(vget_high_s8(vzipped.val[1]));

    const int32x4_t v0_3 = vmovl_s16(vget_low_s16(v0_8));
    const int32x4_t v4_7 = vmovl_s16(vget_high_s16(v0_8));
    const int32x4_t v8_11 = vmovl_s16(vget_low_s16(v8_15));
    const int32x4_t v12_15 = vmovl_s16(vget_high_s16(v8_15));
    const int32x4_t v16_19 = vmovl_s16(vget_low_s16(v16_23));
    const int32x4_t v20_23 = vmovl_s16(vget_high_s16(v16_23));
    const int32x4_t v24_27 = vmovl_s16(vget_low_s16(v24_31));
    const int32x4_t v28_31 = vmovl_s16(vget_high_s16(v24_31));

    const float32x4_t v0_3_f = vcvtq_f32_s32(v0_3);
    const float32x4_t v4_7_f = vcvtq_f32_s32(v4_7);
    const float32x4_t v8_11_f = vcvtq_f32_s32(v8_11);
    const float32x4_t v12_15_f = vcvtq_f32_s32(v12_15);
    const float32x4_t v16_19_f = vcvtq_f32_s32(v16_19);
    const float32x4_t v20_23_f = vcvtq_f32_s32(v20_23);
    const float32x4_t v24_27_f = vcvtq_f32_s32(v24_27);
    const float32x4_t v28_31_f = vcvtq_f32_s32(v28_31);

    vst1q_f32(&out_data[i * 2], v0_3_f);
    vst1q_f32(&out_data[i * 2 + 4], v4_7_f);
    vst1q_f32(&out_data[i * 2 + 8], v8_11_f);
    vst1q_f32(&out_data[i * 2 + 12], v12_15_f);
    vst1q_f32(&out_data[i * 2 + 16], v16_19_f);
    vst1q_f32(&out_data[i * 2 + 20], v20_23_f);
    vst1q_f32(&out_data[i * 2 + 24], v24_27_f);
    vst1q_f32(&out_data[i * 2 + 28], v28_31_f);
  }
#endif

  for (; i < (num_elements + 1) / 2; ++i) {
    int8_t byte = in_data[i];
    // Shift left first so that sign is properly extended when shifted right
    int32_t lower = static_cast<int8_t>(byte << 4) >> 4;
    int32_t higher = byte >> 4;
    out_data[2 * i] = (float)lower;
    out_data[2 * i + 1] = (float)higher;
  }
  return kTfLiteOk;
}

template <typename FromT>
TfLiteStatus copyToTensor(TfLiteContext* context, const FromT* in,
                          TfLiteTensor* out, int num_elements) {
  switch (out->type) {
    case kTfLiteInt64:
      copyCast(in, out->data.i64, num_elements);
      break;
    case kTfLiteInt32:
      copyCast(in, out->data.i32, num_elements);
      break;
    case kTfLiteUInt32:
      copyCast(in, out->data.u32, num_elements);
      break;
    case kTfLiteInt16:
      copyCast(in, out->data.i16, num_elements);
      break;
    case kTfLiteUInt16:
      copyCast(in, out->data.ui16, num_elements);
      break;
    case kTfLiteUInt8:
      copyCast(in, out->data.uint8, num_elements);
      break;
    case kTfLiteInt8:
      copyCast(in, out->data.int8, num_elements);
      break;
    case kTfLiteFloat16:
      copyCastToFloat16(in, reinterpret_cast<Eigen::half*>(out->data.f16),
                        num_elements);
      break;
    case kTfLiteFloat32:
      copyCast(in, GetTensorData<float>(out), num_elements);
      break;
    case kTfLiteFloat64:
      copyCast(in, out->data.f64, num_elements);
      break;
    case kTfLiteBool:
      copyCast(in, out->data.b, num_elements);
      break;
    case kTfLiteComplex64:
      copyCast(in, reinterpret_cast<std::complex<float>*>(out->data.c64),
               num_elements);
      break;
    default:
      // Unsupported type.
      TF_LITE_UNSUPPORTED_TYPE(context, out->type, "Cast");
  }
  return kTfLiteOk;
}

TfLiteStatus EvalImpl(TfLiteContext* context, const TfLiteTensor* input,
                      TfLiteTensor* output, const int num_elements) {
  switch (input->type) {
    case kTfLiteInt64:
      return copyToTensor(context, input->data.i64, output, num_elements);
    case kTfLiteInt32:
      return copyToTensor(context, input->data.i32, output, num_elements);
    case kTfLiteUInt32:
      return copyToTensor(context, input->data.u32, output, num_elements);
    case kTfLiteUInt16:
      return copyToTensor(context, input->data.ui16, output, num_elements);
    case kTfLiteInt16:
      return copyToTensor(context, input->data.i16, output, num_elements);
    case kTfLiteUInt8:
      return copyToTensor(context, input->data.uint8, output, num_elements);
    case kTfLiteInt8:
      return copyToTensor(context, input->data.int8, output, num_elements);
    case kTfLiteFloat16:
      return copyToTensor(context,
                          reinterpret_cast<Eigen::half*>(input->data.f16),
                          output, num_elements);
    case kTfLiteFloat32:
      return copyToTensor(context, GetTensorData<float>(input), output,
                          num_elements);
    case kTfLiteFloat64:
      return copyToTensor(context, input->data.f64, output, num_elements);
    case kTfLiteBool:
      return copyToTensor(context, input->data.b, output, num_elements);
    case kTfLiteComplex64:
      return copyToTensor(
          context, reinterpret_cast<std::complex<float>*>(input->data.c64),
          output, num_elements);
    case kTfLiteInt4:
      if (output->type != kTfLiteFloat32) {
        TF_LITE_UNSUPPORTED_TYPE(context, output->type, "Cast");
      }
      return castInt4ToFloat(context, input, output, num_elements);
    default:
      // Unsupported type.
      TF_LITE_UNSUPPORTED_TYPE(context, input->type, "Cast");
  }
  return kTfLiteError;
}

struct OpData {
  bool cached_output = false;
};

void* Init(TfLiteContext* context, const char* /*buffer*/, size_t /*length*/) {
  return new OpData();
}

void Free(TfLiteContext* context, void* op_data) {
  delete reinterpret_cast<OpData*>(op_data);
}

bool OutputCachingEnabled(const TfLiteContext* context) {
  if (context && context->impl_) {
    const InterpreterOptions* options =
        reinterpret_cast<Subgraph*>(context->impl_)->GetOptions();
    if (options) {
      return options->GetCacheConstantCastOp();
    }
  }
  return false;
}

bool ShouldCacheOutput(const TfLiteContext* context,
                       const TfLiteTensor* input) {
  return OutputCachingEnabled(context) && IsConstantTensor(input);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // TODO(ahentz): these two checks would make the new implementation
  // incompatible with some existing models, where params is not specified. It
  // is OK not to have them because toco would have set input and output types
  // to match the parameters.
  // auto* params = reinterpret_cast<TfLiteCastParams*>(node->builtin_data);
  // TF_LITE_ENSURE_EQ(context, input->type, params->in_data_type);
  // TF_LITE_ENSURE_EQ(context, output->type, params->out_data_type);

  if (ShouldCacheOutput(context, input)) {
    output->allocation_type = kTfLiteArenaRwPersistent;
  }

  TF_LITE_ENSURE_OK(
      context,
      context->ResizeTensor(context, output, TfLiteIntArrayCopy(input->dims)));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, kInputTensor, &input));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  const int num_elements = NumElements(input);
  TF_LITE_ENSURE_EQ(context, num_elements, NumElements(output));

  OpData& op_data = *reinterpret_cast<OpData*>(node->user_data);
  if (ShouldCacheOutput(context, input)) {
    if (op_data.cached_output) {
      return kTfLiteOk;
    }
    op_data.cached_output = true;
  }
  return EvalImpl(context, input, output, num_elements);
}

}  // namespace
}  // namespace cast

TfLiteRegistration* Register_CAST() {
  static TfLiteRegistration r = {cast::Init, cast::Free, cast::Prepare,
                                 cast::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
