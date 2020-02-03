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

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace quantize {

// This file has two implementation of Quantize.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct OpData {
  int32_t output_multiplier;
  int output_shift;
};

namespace {
template <KernelType kernel_type, typename output_type>
static inline void AffineQuantize(const tflite::QuantizationParams& op_params,
                                  const RuntimeShape& input_shape,
                                  const float* input_data,
                                  const RuntimeShape& output_shape,
                                  output_type* output_data) {
  if (kernel_type == kReference) {
    reference_ops::AffineQuantize(op_params, input_shape, input_data,
                                  output_shape, output_data);
  } else {
    optimized_ops::AffineQuantize(op_params, input_shape, input_data,
                                  output_shape, output_data);
  }
}

template <KernelType kernel_type, typename input_type, typename output_type>
static inline void Requantize(const input_type* input_data, int32_t size,
                              int32_t effective_scale_multiplier,
                              int32_t effective_scale_shift,
                              int32_t input_zeropoint, int32_t output_zeropoint,
                              output_type* output_data) {
  if (kernel_type == kReference) {
    reference_ops::Requantize(input_data, size, effective_scale_multiplier,
                              effective_scale_shift, input_zeropoint,
                              output_zeropoint, output_data);
  } else {
    optimized_ops::Requantize(input_data, size, effective_scale_multiplier,
                              effective_scale_shift, input_zeropoint,
                              output_zeropoint, output_data);
  }
}

void ReportError(TfLiteContext* context, TfLiteType input_type,
                 TfLiteType output_type) {
  context->ReportError(
      context, "Input type %s with Output type %s is not currently supported.",
      TfLiteTypeGetName(input_type), TfLiteTypeGetName(output_type));
}
}  // namespace

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = static_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  // TODO(b/128934713): Add support for fixed-point per-channel quantization.
  // Currently this only support affine per-layer quantization.
  TF_LITE_ENSURE_EQ(context, output->quantization.type,
                    kTfLiteAffineQuantization);
  const auto* affine_quantization =
      static_cast<TfLiteAffineQuantization*>(output->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  TF_LITE_ENSURE(context, affine_quantization->scale->size == 1);

  if (input->type == kTfLiteFloat32) {
    // Quantize use case.
    TF_LITE_ENSURE(context, output->type == kTfLiteUInt8 ||
                                output->type == kTfLiteInt8 ||
                                output->type == kTfLiteInt16);
  } else {
    // Requantize use case.
    if (input->type == kTfLiteInt16) {
      TF_LITE_ENSURE(context, output->type == kTfLiteInt8);
    } else {
      TF_LITE_ENSURE(context,
                     input->type == kTfLiteInt8 || input->type == kTfLiteUInt8);
      TF_LITE_ENSURE(
          context, output->type == kTfLiteUInt8 || output->type == kTfLiteInt8);
    }
    const double effective_output_scale =
        static_cast<double>(input->params.scale) /
        static_cast<double>(output->params.scale);
    QuantizeMultiplier(effective_output_scale, &data->output_multiplier,
                       &data->output_shift);
  }

  return context->ResizeTensor(context, output,
                               TfLiteIntArrayCopy(input->dims));
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = static_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, 0);
  TfLiteTensor* output = GetOutput(context, node, 0);

  const RuntimeShape input_shape = GetTensorShape(input);
  const RuntimeShape output_shape = GetTensorShape(output);

  switch (input->type) {
    case kTfLiteFloat32: {
      // Float to int8, uint8, int16.
      tflite::QuantizationParams op_params;
      op_params.zero_point = output->params.zero_point;
      op_params.scale = output->params.scale;
      const float* input_data = GetTensorData<float>(input);
      switch (output->type) {
        case kTfLiteInt8:
          AffineQuantize<kernel_type>(op_params, input_shape, input_data,
                                      output_shape,
                                      GetTensorData<int8_t>(output));
          return kTfLiteOk;
        case kTfLiteUInt8:
          AffineQuantize<kernel_type>(op_params, input_shape, input_data,
                                      output_shape,
                                      GetTensorData<uint8_t>(output));
          return kTfLiteOk;
        case kTfLiteInt16:
          AffineQuantize<kernel_type>(op_params, input_shape, input_data,
                                      output_shape,
                                      GetTensorData<int16_t>(output));
          return kTfLiteOk;
        default:
          ReportError(context, input->type, output->type);
          return kTfLiteError;
      }
    }
    case kTfLiteInt16: {
      // int16 to int8.
      switch (output->type) {
        case kTfLiteInt8:
          Requantize<kernel_type>(GetTensorData<int16_t>(input),
                                  MatchingFlatSize(input_shape, output_shape),
                                  data->output_multiplier, data->output_shift,
                                  input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<int8_t>(output));
          return kTfLiteOk;
        default:
          ReportError(context, input->type, output->type);
          return kTfLiteError;
      }
    }
    case kTfLiteInt8: {
      // int8 to int8, uint8.
      const int32_t size = MatchingFlatSize(input_shape, output_shape);
      const int8_t* input_data = GetTensorData<int8_t>(input);
      switch (output->type) {
        case kTfLiteInt8:
          Requantize<kernel_type>(input_data, size, data->output_multiplier,
                                  data->output_shift, input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<int8_t>(output));
          return kTfLiteOk;
        case kTfLiteUInt8:
          Requantize<kernel_type>(input_data, size, data->output_multiplier,
                                  data->output_shift, input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<uint8_t>(output));
          return kTfLiteOk;
        default:
          ReportError(context, input->type, output->type);
          return kTfLiteError;
      }
    }
    case kTfLiteUInt8: {
      // uint8 to int8, uint8.
      const int32_t size = MatchingFlatSize(input_shape, output_shape);
      const uint8_t* input_data = GetTensorData<uint8_t>(input);
      switch (output->type) {
        case kTfLiteInt8:
          Requantize<kernel_type>(input_data, size, data->output_multiplier,
                                  data->output_shift, input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<int8_t>(output));
          return kTfLiteOk;
        case kTfLiteUInt8:
          Requantize<kernel_type>(input_data, size, data->output_multiplier,
                                  data->output_shift, input->params.zero_point,
                                  output->params.zero_point,
                                  GetTensorData<uint8_t>(output));
          return kTfLiteOk;
        default:
          ReportError(context, input->type, output->type);
          return kTfLiteError;
      }
    }
    default:
      ReportError(context, input->type, output->type);
      return kTfLiteError;
  }
}

}  // namespace quantize

// This Op (QUANTIZE) quantizes the input and produces quantized output.
// The input can be either float or quantized. If the input is float,
// AffineQuantize takes scale and zero point and quantize the float value to
// quantized output, in int8 or uint8 format. If the input is quantized value,
// the op requantize the input (of a certain type, with a given scale and zero
// point) to the output of the same or different type with a same or different
// scale and zero point.
TfLiteRegistration* Register_QUANTIZE_OPT() {
  static TfLiteRegistration r = {quantize::Init, quantize::Free,
                                 quantize::Prepare,
                                 quantize::Eval<quantize::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_QUANTIZE_REF() {
  static TfLiteRegistration r = {quantize::Init, quantize::Free,
                                 quantize::Prepare,
                                 quantize::Eval<quantize::kReference>};
  return &r;
}

TfLiteRegistration* Register_QUANTIZE() { return Register_QUANTIZE_OPT(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
