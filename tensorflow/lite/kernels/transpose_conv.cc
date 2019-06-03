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
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/cpu_backend_support.h"
#include "tensorflow/lite/kernels/eigen_support.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace transpose_conv {

// This file has 2 implementation of TransposeConv.
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
};

constexpr int kOutputShapeTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kDataInputTensor = 2;
constexpr int kOutputTensor = 0;

const int kTensorNotAllocated = -1;

struct OpData {
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers.
  int col2im_id = kTensorNotAllocated;
  int transposed_weights_id = kTensorNotAllocated;
  int scratch_tensor_id = kTensorNotAllocated;

  // col2im is the temporary tensor allocated and used in optimized path for
  // storing col2im data:gemm result for input_matrix x filter_matrix.
  int32_t col2im_index;

  // TfLiteConverter will transpose weights from HWOI to OHWI order.
  // In optimized path, we will transpose them back to HWOI, this temporary
  // tensor is allocated for storing transposed weights.
  int32_t transposed_weights_index;

  // Scratch tensor is used in the quantized path for storing accumulation
  // results.
  int32_t scratch_tensor_index;

  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  bool has_col2im = false;
  bool weights_are_transposed = false;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  eigen_support::IncrementUsageCounter(context);
  cpu_backend_support::IncrementUsageCounter(context);
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  eigen_support::DecrementUsageCounter(context);
  cpu_backend_support::DecrementUsageCounter(context);
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus ResizeTensor(TfLiteContext* context,
                          const TfLiteTensor* shape_tensor,
                          TfLiteTensor* tensor_to_resize) {
  // Currently only support int32 for output shape.
  if (shape_tensor->type != kTfLiteInt32) {
    context->ReportError(context, "Output shape is %d, not int32.",
                         shape_tensor->type);
    return kTfLiteError;
  }

  TfLiteIntArray* shape = TfLiteIntArrayCreate(NumElements(shape_tensor));
  for (int i = 0; i < shape->size; ++i) {
    shape->data[i] = GetTensorData<int32_t>(shape_tensor)[i];
  }

  return context->ResizeTensor(context, tensor_to_resize, shape);
}

// Allocate temporary tensors if necessary.
template <KernelType kernel_type>
static TfLiteStatus AllocateTemporaryTensorsIfRequired(TfLiteContext* context,
                                                       TfLiteType input_type,
                                                       TfLiteType weights_type,
                                                       TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  int temporaries_count = 0;

  // Allocate col2im tensor. Currently it's only used for optimized kernels.
  if (kernel_type == kGenericOptimized) {
    if (data->col2im_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->col2im_id);
    }
    data->col2im_index = temporaries_count;
    data->has_col2im = true;
    ++temporaries_count;
  }

  // Allocate transposed_weights tensor. Currently it's only used for optimized
  // float kernels.
  if (kernel_type == kGenericOptimized && input_type == kTfLiteFloat32) {
    if (data->transposed_weights_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->transposed_weights_id);
    }
    data->transposed_weights_index = temporaries_count;
    data->weights_are_transposed = true;
    ++temporaries_count;
  }

  // Allocate scratch buffer tensor for UInt8 inputs.
  if (input_type == kTfLiteUInt8) {
    if (data->scratch_tensor_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->scratch_tensor_id);
    }
    data->scratch_tensor_index = temporaries_count;
    ++temporaries_count;
  }

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(temporaries_count);

  return kTfLiteOk;
}

TfLiteStatus ResizeCol2ImTensor(TfLiteContext* context,
                                const TfLiteTensor* output_shape,
                                const TfLiteTensor* weights,
                                const TfLiteTensor* input,
                                TfLiteTensor* col2im) {
  if (output_shape->type != kTfLiteInt32) {
    context->ReportError(context, "col2im shape is %d, not int32.",
                         output_shape->type);
    return kTfLiteError;
  }
  TF_LITE_ENSURE_EQ(context, NumElements(output_shape), 4);
  TfLiteIntArray* col2im_shape_array = TfLiteIntArrayCreate(2);
  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& weights_shape = GetTensorShape(weights);
  col2im_shape_array->data[0] = input_shape.Dims(1) * input_shape.Dims(2);
  col2im_shape_array->data[1] =
      weights_shape.Dims(0) * weights_shape.Dims(1) * weights_shape.Dims(2);

  col2im->type = input->type;
  col2im->allocation_type = kTfLiteDynamic;
  return context->ResizeTensor(context, col2im, col2im_shape_array);
}

TfLiteStatus ResizeAndTransposeWeights(TfLiteContext* context,
                                       const TfLiteTensor* weights,
                                       TfLiteTensor* transposed_weights) {
  TfLiteIntArray* transposed_weights_shape_array = TfLiteIntArrayCreate(4);
  const RuntimeShape& input_shape = GetTensorShape(weights);
  transposed_weights_shape_array->data[0] = input_shape.Dims(1);
  transposed_weights_shape_array->data[1] = input_shape.Dims(2);
  transposed_weights_shape_array->data[2] = input_shape.Dims(0);
  transposed_weights_shape_array->data[3] = input_shape.Dims(3);

  transposed_weights->type = weights->type;
  transposed_weights->allocation_type = kTfLiteDynamic;
  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, transposed_weights,
                                              transposed_weights_shape_array));

  // Transpose the weights from from OHWI order to HWOI order.
  TransposeParams transpose_params;
  transpose_params.perm_count = 4;
  transpose_params.perm[0] = 1;
  transpose_params.perm[1] = 2;
  transpose_params.perm[2] = 0;
  transpose_params.perm[3] = 3;

  optimized_ops::Transpose(transpose_params, input_shape,
                           GetTensorData<float>(weights),
                           GetTensorShape(transposed_weights),
                           GetTensorData<float>(transposed_weights));

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // Sanity checks on op
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Retrieve tensors
  const TfLiteTensor* output_shape =
      GetInput(context, node, kOutputShapeTensor);
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Tensor sanity checks
  TF_LITE_ENSURE_EQ(context, NumDimensions(output_shape), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights), 4);
  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, weights->type, input->type);
  TF_LITE_ENSURE_EQ(context, output->type, input->type);
  // Ensure that weights and inputs have the same channel dimension.
  // Note: TOCO will reorder weights in the following format: OHWI.
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input, 3),
                    SizeOfDimension(weights, 3));

  // Allocate col2Im, transposed_weights & scratch Tensor.
  TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequired<kernel_type>(
      context, input->type, weights->type, node));

  OpData* user_data = reinterpret_cast<OpData*>(node->user_data);
  TfLiteTensor* col2im = nullptr;
  if (data->has_col2im) {
    node->temporaries->data[data->col2im_index] = data->col2im_id;
    col2im = GetTemporary(context, node, user_data->col2im_index);
  }

  if (!IsConstantTensor(output_shape)) {
    // Defer resizing until Eval().
    SetTensorToDynamic(output);
    if (data->has_col2im) {
      SetTensorToDynamic(col2im);
    }
  } else {
    TF_LITE_ENSURE_STATUS(ResizeTensor(context, output_shape, output));
    if (data->has_col2im) {
      TF_LITE_ENSURE_STATUS(
          ResizeCol2ImTensor(context, output_shape, weights, input, col2im));
    }
  }

  if (data->weights_are_transposed) {
    node->temporaries->data[data->transposed_weights_index] =
        data->transposed_weights_id;
    TfLiteTensor* transposed_weights =
        GetTemporary(context, node, user_data->transposed_weights_index);
    if (!IsConstantTensor(weights)) {
      SetTensorToDynamic(transposed_weights);
    } else {
      ResizeAndTransposeWeights(context, weights, transposed_weights);
    }
  }

  if (input->type == kTfLiteUInt8) {
    node->temporaries->data[data->scratch_tensor_index] =
        data->scratch_tensor_id;
    TfLiteTensor* scratch_buffer =
        GetTemporary(context, node, data->scratch_tensor_index);
    scratch_buffer->type = kTfLiteInt32;
    scratch_buffer->allocation_type = kTfLiteDynamic;
    if (!IsConstantTensor(output_shape)) {
      SetTensorToDynamic(scratch_buffer);
    } else {
      TF_LITE_ENSURE_STATUS(
          ResizeTensor(context, output_shape, scratch_buffer));
    }

    // Calcuate output multiplier for quantization.
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, weights, output, &real_multiplier));
    int exponent;
    // Populate quantization parameteters with multiplier and shift.
    QuantizeMultiplier(real_multiplier, &data->output_multiplier, &exponent);
    data->output_shift = -exponent;
    // Populate max and min activation range.
    CalculateActivationRangeUint8(kTfLiteActNone, output,
                                  &data->output_activation_min,
                                  &data->output_activation_max);
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
void EvalFloat(TfLiteContext* context, const TfLiteTransposeConvParams* params,
               const OpData* data, const TfLiteTensor* input,
               const TfLiteTensor* weights,
               const TfLiteTensor* transposed_weights, TfLiteTensor* col2im,
               TfLiteTensor* output) {
  tflite::ConvParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  switch (kernel_type) {
    case kReference: {
      reference_ops::TransposeConv(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(weights), GetTensorData<float>(weights),
          GetTensorShape(output), GetTensorData<float>(output),
          GetTensorShape(col2im), GetTensorData<float>(col2im));
      break;
    }
    case kGenericOptimized: {
      optimized_ops::TransposeConvV2(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(transposed_weights),
          GetTensorData<float>(transposed_weights), GetTensorShape(output),
          GetTensorData<float>(output), GetTensorShape(col2im),
          GetTensorData<float>(col2im),
          cpu_backend_support::GetFromContext(context));
      break;
    }
  }
}

void EvalQuantized(const TfLiteTransposeConvParams* params, OpData* data,
                   const TfLiteTensor* input, const TfLiteTensor* weights,
                   TfLiteTensor* col2im, TfLiteTensor* output,
                   TfLiteTensor* scratch_buffer) {
  int32_t input_offset = -input->params.zero_point;
  int32_t filter_offset = -weights->params.zero_point;
  int32_t output_offset = output->params.zero_point;

  tflite::ConvParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.input_offset = input_offset;
  op_params.output_offset = output_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  // TODO(haoliang): Add optimized implementation later.
  reference_ops::TransposeConv(
      op_params, GetTensorShape(input), GetTensorData<uint8>(input),
      GetTensorShape(weights), GetTensorData<uint8>(weights),
      GetTensorShape(output), GetTensorData<uint8>(output),
      GetTensorShape(col2im), GetTensorData<uint8>(col2im),
      GetTensorData<int32_t>(scratch_buffer));
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // Retrieve tensors (All should be allocated by now)
  const TfLiteTensor* output_shape =
      GetInput(context, node, kOutputShapeTensor);
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* input = GetInput(context, node, kDataInputTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  TfLiteTensor* col2im = data->has_col2im
                             ? GetTemporary(context, node, data->col2im_index)
                             : nullptr;
  TfLiteTensor* transposed_weights =
      data->weights_are_transposed
          ? GetTemporary(context, node, data->transposed_weights_index)
          : nullptr;
  const auto* params =
      reinterpret_cast<TfLiteTransposeConvParams*>(node->builtin_data);

  // Resize any deferred dynamic tensors
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeTensor(context, output_shape, output));
  }
  if (data->has_col2im && IsDynamicTensor(col2im)) {
    TF_LITE_ENSURE_OK(context, ResizeCol2ImTensor(context, output_shape,
                                                  weights, input, col2im));
  }

  // Get height and width of the output image.
  const int width = SizeOfDimension(output, 2);
  const int height = SizeOfDimension(output, 1);
  const int filter_width = SizeOfDimension(weights, 2);
  const int filter_height = SizeOfDimension(weights, 1);

  int unused_output_height, unused_output_width;
  data->padding = ComputePaddingHeightWidth(
      params->stride_height, params->stride_width, 1, 1, height, width,
      filter_height, filter_width, params->padding, &unused_output_height,
      &unused_output_width);

  // Currently support float32 and uint8.
  switch (input->type) {
    case kTfLiteFloat32: {
      // Only for GenericOptimized path, we use transposed weights.
      if (data->weights_are_transposed) {
        if (!IsConstantTensor(weights)) {
          ResizeAndTransposeWeights(context, weights, transposed_weights);
        }
      }
      EvalFloat<kernel_type>(context, params, data, input, weights,
                             transposed_weights, col2im, output);
      break;
    }
    case kTfLiteUInt8: {
      // TODO(haoliang): support optimized implementation for quantized
      // TransposeConv.
      TfLiteTensor* scratch_buffer =
          GetTemporary(context, node, data->scratch_tensor_index);
      if (IsDynamicTensor(scratch_buffer)) {
        TF_LITE_ENSURE_OK(context,
                          ResizeTensor(context, output_shape, scratch_buffer));
      }
      EvalQuantized(params, data, input, weights, col2im, output,
                    scratch_buffer);
      break;
    }
    default:
      context->ReportError(context, "Type '%s' is not currently supported.",
                           TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace transpose_conv

TfLiteRegistration* Register_TRANSPOSECONV_REF() {
  static TfLiteRegistration r = {
      transpose_conv::Init, transpose_conv::Free,
      transpose_conv::Prepare<transpose_conv::kReference>,
      transpose_conv::Eval<transpose_conv::kReference>};
  return &r;
}

TfLiteRegistration* Register_TRANSPOSECONV_GENERIC_OPT() {
  static TfLiteRegistration r = {
      transpose_conv::Init, transpose_conv::Free,
      transpose_conv::Prepare<transpose_conv::kGenericOptimized>,
      transpose_conv::Eval<transpose_conv::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_TRANSPOSE_CONV() {
  return Register_TRANSPOSECONV_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
