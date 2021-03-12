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
#include "tensorflow/lite/kernels/kernel_util.h"

#include <stdint.h>
#include <stdlib.h>

#include <algorithm>
#include <complex>
#include <limits>
#include <memory>
#ifndef TF_LITE_STATIC_MEMORY
#include <string>
#endif  // TF_LITE_STATIC_MEMORY

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"

namespace tflite {

namespace {

// Assumes tensor_index is a valid index (in bounds)
inline TfLiteTensor* GetTensorAtIndex(const TfLiteContext* context,
                                      int tensor_index) {
  if (context->tensors != nullptr) {
    return &context->tensors[tensor_index];
  } else {
    return context->GetTensor(context, tensor_index);
  }
}

// Validate in a single place to reduce binary size
inline TfLiteStatus ValidateTensorIndexingSafe(const TfLiteContext* context,
                                               int index, int max_size,
                                               const int* tensor_indices,
                                               int* tensor_index) {
  if (index < 0 || index >= max_size) {
    TF_LITE_KERNEL_LOG(const_cast<TfLiteContext*>(context),
                       "Invalid tensor index %d (not in [0, %d))\n", index,
                       max_size);
    return kTfLiteError;
  }
  if (tensor_indices[index] == kTfLiteOptionalTensor) {
    TF_LITE_KERNEL_LOG(const_cast<TfLiteContext*>(context),
                       "Tensor at index %d was optional but was expected\n",
                       index);
    return kTfLiteError;
  }

  *tensor_index = tensor_indices[index];
  return kTfLiteOk;
}

// Same as above but returns -1 for invalid inputs instead of status + logging
// error.
inline int ValidateTensorIndexing(const TfLiteContext* context, int index,
                                  int max_size, const int* tensor_indices) {
  if (index >= 0 && index < max_size) {
    const int tensor_index = tensor_indices[index];
    if (tensor_index != kTfLiteOptionalTensor) {
      return tensor_index;
    }
  }
  return -1;
}

inline TfLiteTensor* GetMutableInput(const TfLiteContext* context,
                                     const TfLiteNode* node, int index) {
  const int tensor_index = ValidateTensorIndexing(
      context, index, node->inputs->size, node->inputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return GetTensorAtIndex(context, tensor_index);
}

inline TfLiteStatus GetMutableInputSafe(const TfLiteContext* context,
                                        const TfLiteNode* node, int index,
                                        const TfLiteTensor** tensor) {
  int tensor_index;
  TF_LITE_ENSURE_OK(
      context, ValidateTensorIndexingSafe(context, index, node->inputs->size,
                                          node->inputs->data, &tensor_index));
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}

}  // anonymous namespace.

const TfLiteTensor* GetInput(const TfLiteContext* context,
                             const TfLiteNode* node, int index) {
  return GetMutableInput(context, node, index);
}

TfLiteStatus GetInputSafe(const TfLiteContext* context, const TfLiteNode* node,
                          int index, const TfLiteTensor** tensor) {
  return GetMutableInputSafe(context, node, index, tensor);
}

TfLiteTensor* GetVariableInput(TfLiteContext* context, const TfLiteNode* node,
                               int index) {
  TfLiteTensor* tensor = GetMutableInput(context, node, index);
  return tensor->is_variable ? tensor : nullptr;
}

TfLiteTensor* GetOutput(TfLiteContext* context, const TfLiteNode* node,
                        int index) {
  const int tensor_index = ValidateTensorIndexing(
      context, index, node->outputs->size, node->outputs->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return GetTensorAtIndex(context, tensor_index);
}

TfLiteStatus GetOutputSafe(const TfLiteContext* context, const TfLiteNode* node,
                           int index, TfLiteTensor** tensor) {
  int tensor_index;
  TF_LITE_ENSURE_OK(
      context, ValidateTensorIndexingSafe(context, index, node->outputs->size,
                                          node->outputs->data, &tensor_index));
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}

const TfLiteTensor* GetOptionalInputTensor(const TfLiteContext* context,
                                           const TfLiteNode* node, int index) {
  return GetInput(context, node, index);
}

#ifndef TF_LITE_STATIC_MEMORY
TfLiteTensor* GetTemporary(TfLiteContext* context, const TfLiteNode* node,
                           int index) {
  const int tensor_index = ValidateTensorIndexing(
      context, index, node->temporaries->size, node->temporaries->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return GetTensorAtIndex(context, tensor_index);
}

TfLiteStatus GetTemporarySafe(const TfLiteContext* context,
                              const TfLiteNode* node, int index,
                              TfLiteTensor** tensor) {
  int tensor_index;
  TF_LITE_ENSURE_OK(context, ValidateTensorIndexingSafe(
                                 context, index, node->temporaries->size,
                                 node->temporaries->data, &tensor_index));
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}

const TfLiteTensor* GetIntermediates(TfLiteContext* context,
                                     const TfLiteNode* node, int index) {
  const int tensor_index = ValidateTensorIndexing(
      context, index, node->intermediates->size, node->intermediates->data);
  if (tensor_index < 0) {
    return nullptr;
  }
  return GetTensorAtIndex(context, tensor_index);
}

TfLiteStatus GetIntermediatesSafe(const TfLiteContext* context,
                                  const TfLiteNode* node, int index,
                                  TfLiteTensor** tensor) {
  int tensor_index;
  TF_LITE_ENSURE_OK(context, ValidateTensorIndexingSafe(
                                 context, index, node->intermediates->size,
                                 node->intermediates->data, &tensor_index));
  *tensor = GetTensorAtIndex(context, tensor_index);
  return kTfLiteOk;
}
#endif  // TF_LITE_STATIC_MEMORY

// Per-axis
TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int* per_channel_shift) {
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  return PopulateConvolutionQuantizationParams(
      context, input, filter, bias, output, activation, multiplier, shift,
      output_activation_min, output_activation_max, per_channel_multiplier,
      per_channel_shift, affine_quantization->scale->size);
}

// Per-axis & per-tensor
TfLiteStatus PopulateConvolutionQuantizationParams(
    TfLiteContext* context, const TfLiteTensor* input,
    const TfLiteTensor* filter, const TfLiteTensor* bias, TfLiteTensor* output,
    const TfLiteFusedActivation& activation, int32_t* multiplier, int* shift,
    int32_t* output_activation_min, int32_t* output_activation_max,
    int32_t* per_channel_multiplier, int* per_channel_shift, int num_channels) {
  TF_LITE_ENSURE_EQ(context, input->quantization.type,
                    kTfLiteAffineQuantization);
  TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                    kTfLiteAffineQuantization);
  // TODO(jianlijianli): Enable bias type check and bias scale == input scale
  // * filter scale for each channel in affine quantization once bias
  // quantization is properly populated.
  // TF_LITE_ENSURE_EQ(context, bias->quantization.type,
  // kTfLiteAffineQuantization);

  // Check data type.
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(filter->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  const bool is_per_channel = affine_quantization->scale->size > 1;
  if (is_per_channel) {
    //  Currently only Int8/Int16 is supported for per channel quantization.
    TF_LITE_ENSURE(context,
                   input->type == kTfLiteInt8 || input->type == kTfLiteInt16);
    TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size, num_channels);
    TF_LITE_ENSURE_EQ(
        context, num_channels,
        filter->dims->data[affine_quantization->quantized_dimension]);
  }

  // Populate multiplier and shift using affine quantization.
  const float input_scale = input->params.scale;
  const float output_scale = output->params.scale;
  const float* filter_scales = affine_quantization->scale->data;
  for (int i = 0; i < num_channels; ++i) {
    // If per-tensor quantization parameter is specified, broadcast it along the
    // quantization dimension (channels_out).
    const float scale = is_per_channel ? filter_scales[i] : filter_scales[0];
    const double filter_scale = static_cast<double>(scale);
    const double effective_output_scale = static_cast<double>(input_scale) *
                                          filter_scale /
                                          static_cast<double>(output_scale);
    int32_t significand;
    int channel_shift;
    QuantizeMultiplier(effective_output_scale, &significand, &channel_shift);
    per_channel_multiplier[i] = significand;
    per_channel_shift[i] = channel_shift;
  }

  // Populate scalar quantization parameters.
  // This check on legacy quantization parameters is kept only for backward
  // compatibility.
  if (input->type == kTfLiteUInt8) {
    // Check bias scale == input scale * filter scale.
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    int exponent;

    // Populate quantization parameters with multiplier and shift.
    QuantizeMultiplier(real_multiplier, multiplier, &exponent);
    *shift = -exponent;
  }
  if (input->type == kTfLiteInt8 || input->type == kTfLiteUInt8 ||
      input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, activation, output, output_activation_min,
        output_activation_max));
  }
  return kTfLiteOk;
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              const TfLiteTensor* bias,
                                              TfLiteTensor* output,
                                              double* multiplier) {
  const double input_product_scale = static_cast<double>(input->params.scale) *
                                     static_cast<double>(filter->params.scale);
  // The following conditions must be guaranteed by the training pipeline.
  if (bias) {
    const double bias_scale = static_cast<double>(bias->params.scale);
    // Here we're making sure the input_product_scale & bias_scale are about the
    // same. Since we have:
    // (output - output_zp) * output_scale =
    // input_product_scale * input_product + bias * bias_scale ---- (0)
    //
    // (0) equals:
    // (input_product + bias) * input_product_scale ----- (1)
    //           +
    // bias * (bias_scale - input_product_scale)   ------ (2)
    //
    // For the real kernel computation, we're doing (1), so we really need to
    // make sure (2) has minimum impact on the output, so:
    // bias * (bias_scale - input_product_scale) / output_scale should be
    // a small number for an integer.
    // Since normally bias should be within a small range.
    // We should expect (bias_scale - input_product_scale) / output_scale to
    // be a small number like 0.02.
    const double scale_diff = std::abs(input_product_scale - bias_scale);
    const double output_scale = static_cast<double>(output->params.scale);

    TF_LITE_ENSURE(context, scale_diff / output_scale <= 0.02);
  }
  return GetQuantizedConvolutionMultipler(context, input, filter, output,
                                          multiplier);
}

TfLiteStatus GetQuantizedConvolutionMultipler(TfLiteContext* context,
                                              const TfLiteTensor* input,
                                              const TfLiteTensor* filter,
                                              TfLiteTensor* output,
                                              double* multiplier) {
  const double input_product_scale =
      static_cast<double>(input->params.scale * filter->params.scale);
  TF_LITE_ENSURE(context, input_product_scale >= 0);
  *multiplier = input_product_scale / static_cast<double>(output->params.scale);

  return kTfLiteOk;
}

namespace {
void CalculateActivationRangeQuantizedImpl(TfLiteFusedActivation activation,
                                           int32_t qmin, int32_t qmax,
                                           TfLiteTensor* output,
                                           int32_t* act_min, int32_t* act_max) {
  const auto scale = output->params.scale;
  const auto zero_point = output->params.zero_point;

  auto quantize = [scale, zero_point](float f) {
    return zero_point + static_cast<int32_t>(TfLiteRound(f / scale));
  };

  if (activation == kTfLiteActRelu) {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = qmax;
  } else if (activation == kTfLiteActRelu6) {
    *act_min = std::max(qmin, quantize(0.0));
    *act_max = std::min(qmax, quantize(6.0));
  } else if (activation == kTfLiteActReluN1To1) {
    *act_min = std::max(qmin, quantize(-1.0));
    *act_max = std::min(qmax, quantize(1.0));
  } else {
    *act_min = qmin;
    *act_max = qmax;
  }
}
}  // namespace

TfLiteStatus CalculateActivationRangeQuantized(TfLiteContext* context,
                                               TfLiteFusedActivation activation,
                                               TfLiteTensor* output,
                                               int32_t* act_min,
                                               int32_t* act_max) {
  int32_t qmin = 0;
  int32_t qmax = 0;
  if (output->type == kTfLiteUInt8) {
    qmin = std::numeric_limits<uint8_t>::min();
    qmax = std::numeric_limits<uint8_t>::max();
  } else if (output->type == kTfLiteInt8) {
    qmin = std::numeric_limits<int8_t>::min();
    qmax = std::numeric_limits<int8_t>::max();
  } else if (output->type == kTfLiteInt16) {
    qmin = std::numeric_limits<int16_t>::min();
    qmax = std::numeric_limits<int16_t>::max();
  } else {
    TF_LITE_ENSURE(context, false);
  }

  CalculateActivationRangeQuantizedImpl(activation, qmin, qmax, output, act_min,
                                        act_max);
  return kTfLiteOk;
}

bool HaveSameShapes(const TfLiteTensor* input1, const TfLiteTensor* input2) {
  return TfLiteIntArrayEqual(input1->dims, input2->dims);
}

#ifndef TF_LITE_STATIC_MEMORY

// TODO(b/172067338): Having this function be part of TF_LITE_STATIC_MEMORY
// build results in a 6KB size increase, even though the function is unsused for
// that build. What appears to be happening is that while the linker drops the
// unsused function, the string library that gets pulled in is not dropped,
// resulting in the increased binary size.
std::string GetShapeDebugString(const TfLiteIntArray* shape) {
  std::string str;
  for (int d = 0; d < shape->size; ++d) {
    if (str.empty())
      str = "[" + std::to_string(shape->data[d]);
    else
      str += ", " + std::to_string(shape->data[d]);
  }
  str += "]";
  return str;
}

TfLiteStatus CalculateShapeForBroadcast(TfLiteContext* context,
                                        const TfLiteTensor* input1,
                                        const TfLiteTensor* input2,
                                        TfLiteIntArray** output_shape) {
  int dims1 = NumDimensions(input1);
  int dims2 = NumDimensions(input2);
  int out_dims = std::max(dims1, dims2);
  if (NumElements(input1) == 0) {
    *output_shape = TfLiteIntArrayCopy(input1->dims);
    return kTfLiteOk;
  }
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> shape(
      TfLiteIntArrayCreate(out_dims), TfLiteIntArrayFree);
  for (int i = 0; i < out_dims; ++i) {
    int d1 = i >= dims1 ? 1 : SizeOfDimension(input1, dims1 - i - 1);
    int d2 = i >= dims2 ? 1 : SizeOfDimension(input2, dims2 - i - 1);
    if (!(d1 == d2 || d1 == 1 || d2 == 1)) {
      context->ReportError(context,
                           "Given shapes, %s and %s, are not broadcastable.",
                           GetShapeDebugString(input1->dims).c_str(),
                           GetShapeDebugString(input2->dims).c_str());
      return kTfLiteError;
    }
    shape->data[out_dims - i - 1] = std::max(d1, d2);
  }
  *output_shape = shape.release();
  return kTfLiteOk;
}

TfLiteStatus CalculateShapeForBroadcast(TfLiteContext* context,
                                        const TfLiteTensor* input1,
                                        const TfLiteTensor* input2,
                                        const TfLiteTensor* input3,
                                        TfLiteIntArray** output_shape) {
  int dims1 = NumDimensions(input1);
  int dims2 = NumDimensions(input2);
  int dims3 = NumDimensions(input3);
  int out_dims = std::max(std::max(dims1, dims2), dims3);
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> shape(
      TfLiteIntArrayCreate(out_dims), TfLiteIntArrayFree);
  for (int i = 0; i < out_dims; ++i) {
    int d1 = i >= dims1 ? 1 : SizeOfDimension(input1, dims1 - i - 1);
    int d2 = i >= dims2 ? 1 : SizeOfDimension(input2, dims2 - i - 1);
    int d3 = i >= dims3 ? 1 : SizeOfDimension(input3, dims3 - i - 1);
    int max_value = std::max(std::max(d1, d2), d3);
    if (!(d1 == 1 || d1 == max_value) || !(d2 == 1 || d2 == max_value) ||
        !(d3 == 1 || d3 == max_value)) {
      context->ReportError(
          context, "Given shapes, %s, %s and %s, are not broadcastable.",
          GetShapeDebugString(input1->dims).c_str(),
          GetShapeDebugString(input2->dims).c_str(),
          GetShapeDebugString(input3->dims).c_str());
      return kTfLiteError;
    }
    shape->data[out_dims - i - 1] = max_value;
  }
  *output_shape = shape.release();
  return kTfLiteOk;
}
#endif  // TF_LITE_STATIC_MEMORY

// Size of string is not constant, return 0 in such case.
int TfLiteTypeGetSize(TfLiteType type) {
  switch (type) {
    case kTfLiteUInt8:
      TF_LITE_ASSERT_EQ(sizeof(uint8_t), 1);
      return 1;
    case kTfLiteInt8:
      TF_LITE_ASSERT_EQ(sizeof(int8_t), 1);
      return 1;
    case kTfLiteBool:
      return sizeof(bool);
    case kTfLiteInt16:
      TF_LITE_ASSERT_EQ(sizeof(int16_t), 2);
      return 2;
    case kTfLiteFloat16:
      TF_LITE_ASSERT_EQ(sizeof(int16_t), 2);
      return 2;
    case kTfLiteFloat32:
      TF_LITE_ASSERT_EQ(sizeof(float), 4);
      return 4;
    case kTfLiteInt32:
      TF_LITE_ASSERT_EQ(sizeof(int32_t), 4);
      return 4;
    case kTfLiteUInt32:
      TF_LITE_ASSERT_EQ(sizeof(uint32_t), 4);
      return 4;
    case kTfLiteInt64:
      TF_LITE_ASSERT_EQ(sizeof(int64_t), 8);
      return 8;
    case kTfLiteUInt64:
      TF_LITE_ASSERT_EQ(sizeof(uint64_t), 8);
      return 8;
    case kTfLiteFloat64:
      TF_LITE_ASSERT_EQ(sizeof(double), 8);
      return 8;
    case kTfLiteComplex64:
      TF_LITE_ASSERT_EQ(sizeof(std::complex<float>), 8);
      return 8;
    case kTfLiteComplex128:
      TF_LITE_ASSERT_EQ(sizeof(std::complex<double>), 16);
      return 16;
    default:
      return 0;
  }
}

}  // namespace tflite
