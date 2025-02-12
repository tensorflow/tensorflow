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

#include <stddef.h>
#include <stdint.h>

#include <vector>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
// NOLINTNEXTLINE - This header file shouldn't go to the top.
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/transpose_conv.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
// NOLINTNEXTLINE - This header file shouldn't go to the top.
#include "tensorflow/lite/kernels/internal/portable_tensor_utils.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/transpose_conv.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/reference/transpose_conv.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
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
constexpr int kBiasTensor = 3;
constexpr int kOutputTensor = 0;

const int kTensorNotAllocated = -1;

struct OpData {
  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers.
  int col2im_id = kTensorNotAllocated;
  int transposed_weights_id = kTensorNotAllocated;
  int scratch_tensor_id = kTensorNotAllocated;
  int input_quantized_id = kTensorNotAllocated;
  int scaling_factors_id = kTensorNotAllocated;
  int input_offset_id = kTensorNotAllocated;

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

  // Indexes are used for hybrid (dynamic range quantization) path.
  int32_t input_quantized_index;
  int32_t scaling_factors_index;
  int32_t input_offset_index;

  TfLitePaddingValues padding;
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;

  // Per channel output multiplier and shift.
  std::vector<int32_t> per_channel_output_multiplier;
  std::vector<int32_t> per_channel_output_shift;

  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;

  bool has_col2im = false;
  bool weights_are_transposed = false;

  TfLiteType quantized_bias_type = kTfLiteNoType;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus ResizeTensor(TfLiteContext* context,
                          const TfLiteTensor* shape_tensor,
                          TfLiteTensor* tensor_to_resize) {
  // Currently only support int32 for output shape.
  if (shape_tensor->type != kTfLiteInt32) {
    TF_LITE_KERNEL_LOG(context, "Output shape is %s, not int32.",
                       TfLiteTypeGetName(shape_tensor->type));
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
  if (kernel_type == kGenericOptimized) {
    if (data->transposed_weights_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->transposed_weights_id);
    }
    data->transposed_weights_index = temporaries_count;
    data->weights_are_transposed = true;
    ++temporaries_count;
  }

  // Allocate scratch buffer tensor
  if (input_type == kTfLiteUInt8 || input_type == kTfLiteInt8 ||
      input_type == kTfLiteInt16) {
    if (data->scratch_tensor_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->scratch_tensor_id);
    }
    data->scratch_tensor_index = temporaries_count;
    ++temporaries_count;
  }

  if (input_type == kTfLiteFloat32 && weights_type == kTfLiteInt8) {
    // Allocate tensor to store the on-the-fly quantized inputs.
    data->input_quantized_index = temporaries_count;
    if (data->input_quantized_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->input_quantized_id));
    }
    ++temporaries_count;

    // Allocate tensor to store the quantization params computed during
    // on-the-fly input quantization.
    data->scaling_factors_index = temporaries_count;
    if (data->scaling_factors_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->scaling_factors_id));
    }
    ++temporaries_count;

    data->input_offset_index = temporaries_count;
    if (data->input_offset_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &data->input_offset_id));
    }
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
    TF_LITE_KERNEL_LOG(context, "col2im shape is %s, not int32.",
                       TfLiteTypeGetName(output_shape->type));
    return kTfLiteError;
  }
  TF_LITE_ENSURE_EQ(context, NumElements(output_shape), 4);
  TfLiteIntArray* col2im_shape_array = TfLiteIntArrayCreate(2);
  const RuntimeShape& input_shape = GetTensorShape(input);
  const RuntimeShape& weights_shape = GetTensorShape(weights);
  col2im_shape_array->data[0] = input_shape.Dims(1) * input_shape.Dims(2);
  col2im_shape_array->data[1] =
      weights_shape.Dims(0) * weights_shape.Dims(1) * weights_shape.Dims(2);

  col2im->type = input->type == kTfLiteFloat32 ? kTfLiteFloat32 : kTfLiteInt32;
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

  // Transpose the weights from OHWI order to HWOI order.
  TransposeParams transpose_params;
  transpose_params.perm_count = 4;
  transpose_params.perm[0] = 1;
  transpose_params.perm[1] = 2;
  transpose_params.perm[2] = 0;
  transpose_params.perm[3] = 3;

  if (weights->type == kTfLiteFloat32) {
    optimized_ops::Transpose(transpose_params, input_shape,
                             GetTensorData<float>(weights),
                             GetTensorShape(transposed_weights),
                             GetTensorData<float>(transposed_weights));
  } else if (weights->type == kTfLiteUInt8) {
    optimized_ops::Transpose(transpose_params, input_shape,
                             GetTensorData<uint8>(weights),
                             GetTensorShape(transposed_weights),
                             GetTensorData<uint8>(transposed_weights));
  } else if (weights->type == kTfLiteInt8) {
    // int16 transpose_conv also with int8 weights
    optimized_ops::Transpose(transpose_params, input_shape,
                             GetTensorData<int8>(weights),
                             GetTensorShape(transposed_weights),
                             GetTensorData<int8>(transposed_weights));
  } else {
    TF_LITE_KERNEL_LOG(
        context,
        "Only float32, uint8, int8, int16 is supported currently, got %s.",
        TfLiteTypeGetName(weights->type));
    return kTfLiteError;
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  auto* params =
      reinterpret_cast<TfLiteTransposeConvParams*>(node->builtin_data);

  bool has_bias = NumInputs(node) == 4;

  // Sanity checks on op
  TF_LITE_ENSURE(context, has_bias || NumInputs(node) == 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  // Retrieve tensors
  const TfLiteTensor* output_shape;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kOutputShapeTensor, &output_shape));
  const TfLiteTensor* weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kWeightsTensor, &weights));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kDataInputTensor, &input));
  const TfLiteTensor* bias = nullptr;

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Tensor sanity checks
  TF_LITE_ENSURE_EQ(context, NumDimensions(output_shape), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 4);
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights), 4);
  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteUInt8 ||
                     input->type == kTfLiteInt8 || input->type == kTfLiteInt16);

  if (has_bias) {
    bias = GetOptionalInputTensor(context, node, kBiasTensor);
    if (bias) {
      if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8) {
        TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt32);
        if (input->type == kTfLiteInt8) {
          TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
        }
      } else if (input->type == kTfLiteInt16) {
        TF_LITE_ENSURE(context, (bias->type == kTfLiteInt64) ||
                                    (bias->type == kTfLiteInt32));
        TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
      } else {
        TF_LITE_ENSURE_TYPES_EQ(context, bias->type, input->type);
      }
      TF_LITE_ENSURE_EQ(context, NumElements(bias),
                        SizeOfDimension(weights, 0));
    }
  }

  if (input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, weights->type, kTfLiteInt8);
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);

    // Check quantized_bias_type is either kTfLiteInt64 or kTfLiteInt32.
    if (params->quantized_bias_type != kTfLiteFloat32) {
      TF_LITE_ENSURE(context, params->quantized_bias_type == kTfLiteInt32 ||
                                  params->quantized_bias_type == kTfLiteInt64);
      TF_LITE_ENSURE(context, (bias == nullptr) ||
                                  bias->type == params->quantized_bias_type);
      data->quantized_bias_type = params->quantized_bias_type;
    }
  }
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input->type);
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
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, user_data->col2im_index, &col2im));
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
    TfLiteTensor* transposed_weights;
    TF_LITE_ENSURE_OK(
        context,
        GetTemporarySafe(context, node, user_data->transposed_weights_index,
                         &transposed_weights));
    if (!IsConstantTensor(weights)) {
      SetTensorToDynamic(transposed_weights);
    } else {
      ResizeAndTransposeWeights(context, weights, transposed_weights);
    }
  }

  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8 ||
      input->type == kTfLiteInt16) {
    node->temporaries->data[data->scratch_tensor_index] =
        data->scratch_tensor_id;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, data->scratch_tensor_index,
                                  &scratch_buffer));

    if (data->quantized_bias_type != kTfLiteNoType) {
      scratch_buffer->type = data->quantized_bias_type;
    } else if (input->type == kTfLiteInt16) {
      scratch_buffer->type = kTfLiteInt64;
    } else {
      scratch_buffer->type = kTfLiteInt32;
    }

    scratch_buffer->allocation_type = kTfLiteDynamic;
    if (!IsConstantTensor(output_shape)) {
      SetTensorToDynamic(scratch_buffer);
    } else {
      TF_LITE_ENSURE_STATUS(
          ResizeTensor(context, output_shape, scratch_buffer));
    }

    TF_LITE_ENSURE_EQ(context, weights->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            weights->quantization.params);
    const int channels_out = weights->dims->data[0];
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context, (affine_quantization->scale->size == 1 ||
                             affine_quantization->scale->size == channels_out));

    data->per_channel_output_multiplier.resize(channels_out);
    data->per_channel_output_shift.resize(channels_out);
    auto* params = reinterpret_cast<TfLiteConvParams*>(node->builtin_data);
    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, weights, bias, output, params->activation,
        &data->output_multiplier, &data->output_shift,
        &data->output_activation_min, &data->output_activation_max,
        data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), channels_out));
  }

  if (input->type == kTfLiteFloat32 && weights->type == kTfLiteInt8) {
    node->temporaries->data[data->input_quantized_index] =
        data->input_quantized_id;
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, data->input_quantized_index,
                                  &input_quantized));
    input_quantized->type = kTfLiteInt8;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }

    node->temporaries->data[data->scaling_factors_index] =
        data->scaling_factors_id;
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, data->scaling_factors_index,
                                  &scaling_factors));
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    // Only one scale factor per batch is typically necessary. See optimized
    // implementation for why we need to allocate for the height of the inputs
    // flattened to 2D.
    const int channels_in = weights->dims->data[3];
    TF_LITE_ENSURE(context, channels_in != 0);
    const int height = NumElements(input) / channels_in;
    int scaling_dims[1] = {height};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = height;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }

    auto* affine_quantization = reinterpret_cast<TfLiteAffineQuantization*>(
        weights->quantization.params);
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);

    const int channels_out =
        weights->dims->data[affine_quantization->quantized_dimension];
    if (affine_quantization->scale->size != channels_out) {
      TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size, 1);
      TfLiteFloatArrayFree(affine_quantization->scale);
      affine_quantization->scale = TfLiteFloatArrayCreate(channels_out);
      for (int i = 0; i < channels_out; ++i) {
        affine_quantization->scale->data[i] = weights->params.scale;
      }
    } else {
      TF_LITE_ENSURE_EQ(context, affine_quantization->scale->size,
                        channels_out);
    }

    node->temporaries->data[data->input_offset_index] = data->input_offset_id;
    TfLiteTensor* input_offsets;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, data->input_offset_index,
                                       &input_offsets));
    input_offsets->type = kTfLiteInt32;
    input_offsets->allocation_type = kTfLiteArenaRw;
    // See above comment for the need to allocate for height of inputs.
    TF_LITE_ENSURE(context, channels_in != 0);
    const int input_offset_dims[1] = {height};
    if (!TfLiteIntArrayEqualsArray(input_offsets->dims, 1, input_offset_dims)) {
      TfLiteIntArray* input_offsets_size = TfLiteIntArrayCreate(1);
      input_offsets_size->data[0] = input_offset_dims[0];
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_offsets,
                                                       input_offsets_size));
    }
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
void EvalFloat(TfLiteContext* context, const TfLiteTransposeConvParams* params,
               const OpData* data, const TfLiteTensor* input,
               const TfLiteTensor* weights, const TfLiteTensor* bias,
               const TfLiteTensor* transposed_weights, TfLiteTensor* col2im,
               TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  tflite::ConvParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  switch (kernel_type) {
    case kReference: {
      reference_ops::TransposeConv(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(weights), GetTensorData<float>(weights),
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output),
          GetTensorShape(col2im), GetTensorData<float>(col2im));
      break;
    }
    case kGenericOptimized: {
      optimized_ops::TransposeConvV2(
          op_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(transposed_weights),
          GetTensorData<float>(transposed_weights), GetTensorShape(bias),
          GetTensorData<float>(bias), GetTensorShape(output),
          GetTensorData<float>(output), GetTensorShape(col2im),
          GetTensorData<float>(col2im),
          CpuBackendContext::GetFromContext(context));
      break;
    }
  }
}

template <KernelType kernel_type>
void EvalQuantized(TfLiteContext* context,
                   const TfLiteTransposeConvParams* params, OpData* data,
                   const TfLiteTensor* input, const TfLiteTensor* weights,
                   const TfLiteTensor* transposed_weights,
                   const TfLiteTensor* bias, TfLiteTensor* col2im,
                   TfLiteTensor* output, TfLiteTensor* scratch_buffer) {
  int32_t input_offset = -input->params.zero_point;
  int32_t filter_offset = -weights->params.zero_point;
  int32_t output_offset = output->params.zero_point;

  tflite::ConvParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.input_offset = input_offset;
  op_params.output_offset = output_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = -data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  switch (kernel_type) {
    case kReference: {
      reference_ops::TransposeConv(
          op_params, GetTensorShape(input), GetTensorData<uint8>(input),
          GetTensorShape(weights), GetTensorData<uint8>(weights),
          GetTensorShape(bias), GetTensorData<int32_t>(bias),
          GetTensorShape(output), GetTensorData<uint8>(output),
          GetTensorShape(col2im), GetTensorData<uint8>(col2im),
          GetTensorData<int32_t>(scratch_buffer));
      break;
    }
    case kGenericOptimized: {
      optimized_ops::TransposeConvV2(
          op_params, GetTensorShape(input), GetTensorData<uint8>(input),
          GetTensorShape(transposed_weights),
          GetTensorData<uint8>(transposed_weights), GetTensorShape(bias),
          GetTensorData<int32>(bias), GetTensorShape(output),
          GetTensorData<uint8>(output), GetTensorShape(col2im),
          GetTensorData<int32>(col2im), GetTensorData<int32>(scratch_buffer),
          CpuBackendContext::GetFromContext(context));
      break;
    }
  }
}

template <KernelType kernel_type>
void EvalQuantizedPerChannel(
    TfLiteContext* context, const TfLiteTransposeConvParams* params,
    OpData* data, const TfLiteTensor* input, const TfLiteTensor* weights,
    const TfLiteTensor* transposed_weights, const TfLiteTensor* bias,
    TfLiteTensor* col2im, TfLiteTensor* output, TfLiteTensor* scratch_buffer) {
  tflite::ConvParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  // Need to flip the sign of input offset to add it directly to the quantized
  // buffer.
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  switch (kernel_type) {
    case kReference: {
      reference_integer_ops::TransposeConv(
          op_params, data->per_channel_output_multiplier.data(),
          data->per_channel_output_shift.data(), GetTensorShape(input),
          GetTensorData<int8>(input), GetTensorShape(weights),
          GetTensorData<int8>(weights), GetTensorShape(bias),
          GetTensorData<int32>(bias), GetTensorShape(output),
          GetTensorData<int8>(output), GetTensorShape(col2im),
          GetTensorData<int8>(col2im), GetTensorData<int32_t>(scratch_buffer));
      break;
    }
    case kGenericOptimized: {
      optimized_integer_ops::TransposeConvV2(
          op_params, data->per_channel_output_multiplier.data(),
          data->per_channel_output_shift.data(), GetTensorShape(input),
          GetTensorData<int8>(input), GetTensorShape(transposed_weights),
          GetTensorData<int8>(transposed_weights), GetTensorShape(bias),
          GetTensorData<int32>(bias), GetTensorShape(output),
          GetTensorData<int8>(output), GetTensorShape(col2im),
          GetTensorData<int32>(col2im), GetTensorData<int32>(scratch_buffer),
          CpuBackendContext::GetFromContext(context));
      break;
    }
  }
}

template <KernelType kernel_type>
void EvalQuantizedPerChannel16x8(
    TfLiteContext* context, const TfLiteTransposeConvParams* params,
    OpData* data, const TfLiteTensor* input, const TfLiteTensor* weights,
    const TfLiteTensor* transposed_weights, const TfLiteTensor* bias,
    TfLiteTensor* col2im, TfLiteTensor* output, TfLiteTensor* scratch_buffer) {
  tflite::ConvParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  // Need to flip the sign of input offset to add it directly to the quantized
  // buffer.
  op_params.input_offset = -input->params.zero_point;
  op_params.output_offset = output->params.zero_point;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  // To prevent 32bit accum overflow for 16x8 quantization, it enables the
  // optimized path only when zero_point is 0.
  bool has_non_zero_point = input->params.zero_point ||
                            weights->params.zero_point ||
                            output->params.zero_point;

  if (data->quantized_bias_type == kTfLiteInt32) {
    if (kernel_type == kReference || has_non_zero_point) {
      reference_integer_ops::TransposeConv(
          op_params, data->per_channel_output_multiplier.data(),
          data->per_channel_output_shift.data(), GetTensorShape(input),
          GetTensorData<int16>(input), GetTensorShape(weights),
          GetTensorData<int8>(weights), GetTensorShape(bias),
          GetTensorData<int32_t>(bias), GetTensorShape(output),
          GetTensorData<int16>(output), GetTensorShape(col2im),
          GetTensorData<int8>(col2im), GetTensorData<int32_t>(scratch_buffer));
    } else {
      optimized_integer_ops::TransposeConvV2(
          op_params, data->per_channel_output_multiplier.data(),
          data->per_channel_output_shift.data(), GetTensorShape(input),
          GetTensorData<int16>(input), GetTensorShape(transposed_weights),
          GetTensorData<int8>(transposed_weights), GetTensorShape(bias),
          GetTensorData<int32_t>(bias), GetTensorShape(output),
          GetTensorData<int16>(output), GetTensorShape(col2im),
          GetTensorData<int32>(col2im), GetTensorData<int32>(scratch_buffer),
          CpuBackendContext::GetFromContext(context));
    }
  } else {
    TFLITE_DCHECK(!has_non_zero_point);
    // Fallback to reference kernel when bias_type is int64 as
    // there is no optimized kernel for int64 bias yet.
    reference_integer_ops::TransposeConv(
        op_params, data->per_channel_output_multiplier.data(),
        data->per_channel_output_shift.data(), GetTensorShape(input),
        GetTensorData<int16>(input), GetTensorShape(weights),
        GetTensorData<int8>(weights), GetTensorShape(bias),
        GetTensorData<int64_t>(bias), GetTensorShape(output),
        GetTensorData<int16>(output), GetTensorShape(col2im),
        GetTensorData<int8>(col2im), GetTensorData<int64_t>(scratch_buffer));
  }
}

TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                        const TfLiteTransposeConvParams* params, OpData* data,
                        const TfLiteTensor* input, const TfLiteTensor* weights,
                        const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  const int batch_size = SizeOfDimension(input, 0);
  TF_LITE_ENSURE(context, batch_size != 0);
  const int input_size = NumElements(input) / batch_size;
  TfLiteTensor* quantized_input_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_quantized_index,
                                     &quantized_input_tensor));
  int8_t* quantized_input_ptr_batch =
      GetTensorData<int8_t>(quantized_input_tensor);
  TfLiteTensor* scaling_factors_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->scaling_factors_index,
                                     &scaling_factors_tensor));
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors_tensor);
  TfLiteTensor* input_offset_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, data->input_offset_index,
                                     &input_offset_tensor));
  int32_t* input_offset_ptr = GetTensorData<int32_t>(input_offset_tensor);

  for (int b = 0; b < batch_size; ++b) {
    const int offset = b * input_size;
    tensor_utils::AsymmetricQuantizeFloats(
        GetTensorData<float>(input) + offset, input_size,
        quantized_input_ptr_batch + offset, &scaling_factors_ptr[b],
        &input_offset_ptr[b]);
  }

  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(weights->quantization.params);

  tflite::ConvParams op_params;
  op_params.padding_type = PaddingType::kSame;
  op_params.padding_values.width = data->padding.width;
  op_params.padding_values.height = data->padding.height;
  op_params.padding_values.width_offset = data->padding.width_offset;
  op_params.padding_values.height_offset = data->padding.height_offset;
  op_params.stride_width = params->stride_width;
  op_params.stride_height = params->stride_height;
  op_params.float_activation_min = output_activation_min;
  op_params.float_activation_max = output_activation_max;

  reference_ops::HybridTransposeConv(
      op_params, scaling_factors_ptr, GetTensorShape(input),
      quantized_input_ptr_batch, GetTensorShape(weights),
      GetTensorData<int8_t>(weights), GetTensorShape(bias),
      GetTensorData<float>(bias), GetTensorShape(output),
      GetTensorData<float>(output), affine_quantization->scale->data,
      input_offset_ptr);

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  // Retrieve tensors (All should be allocated by now)
  const TfLiteTensor* output_shape;
  TF_LITE_ENSURE_OK(
      context, GetInputSafe(context, node, kOutputShapeTensor, &output_shape));
  const TfLiteTensor* weights;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kWeightsTensor, &weights));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kDataInputTensor, &input));
  const TfLiteTensor* bias =
      (NumInputs(node) == 4)
          ? GetOptionalInputTensor(context, node, kBiasTensor)
          : nullptr;
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
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

  // Prevent divisions by 0
  TF_LITE_ENSURE(context, params->stride_height > 0);
  TF_LITE_ENSURE(context, params->stride_width > 0);

  // Resize any deferred dynamic tensors
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeTensor(context, output_shape, output));
  }
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input, 0),
                    SizeOfDimension(output, 0));
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

  // Currently support float32, uint8, int8, int16.
  switch (input->type) {
    case kTfLiteFloat32: {
      if (weights->type == kTfLiteInt8) {
        TF_LITE_ENSURE_OK(context, EvalHybrid(context, node, params, data,
                                              input, weights, bias, output));
      } else {
        // Only for GenericOptimized path, we use transposed weights.
        if (data->weights_are_transposed) {
          if (!IsConstantTensor(weights)) {
            ResizeAndTransposeWeights(context, weights, transposed_weights);
          }
        }
        EvalFloat<kernel_type>(context, params, data, input, weights, bias,
                               transposed_weights, col2im, output);
      }
      break;
    }
    case kTfLiteUInt8: {
      TfLiteTensor* scratch_buffer;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, data->scratch_tensor_index,
                                    &scratch_buffer));
      if (IsDynamicTensor(scratch_buffer)) {
        TF_LITE_ENSURE_OK(context,
                          ResizeTensor(context, output_shape, scratch_buffer));
      }
      if (data->weights_are_transposed) {
        if (!IsConstantTensor(weights)) {
          ResizeAndTransposeWeights(context, weights, transposed_weights);
        }
      }
      EvalQuantized<kernel_type>(context, params, data, input, weights,
                                 transposed_weights, bias, col2im, output,
                                 scratch_buffer);
      break;
    }
    case kTfLiteInt8: {
      TfLiteTensor* scratch_buffer;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, data->scratch_tensor_index,
                                    &scratch_buffer));
      if (IsDynamicTensor(scratch_buffer)) {
        TF_LITE_ENSURE_OK(context,
                          ResizeTensor(context, output_shape, scratch_buffer));
      }
      if (data->weights_are_transposed && !IsConstantTensor(weights)) {
        ResizeAndTransposeWeights(context, weights, transposed_weights);
      }
      EvalQuantizedPerChannel<kernel_type>(context, params, data, input,
                                           weights, transposed_weights, bias,
                                           col2im, output, scratch_buffer);
      break;
    }
    case kTfLiteInt16: {
      TfLiteTensor* scratch_buffer;
      TF_LITE_ENSURE_OK(
          context, GetTemporarySafe(context, node, data->scratch_tensor_index,
                                    &scratch_buffer));
      if (IsDynamicTensor(scratch_buffer)) {
        TF_LITE_ENSURE_OK(context,
                          ResizeTensor(context, output_shape, scratch_buffer));
      }
      if (data->weights_are_transposed && !IsConstantTensor(weights)) {
        ResizeAndTransposeWeights(context, weights, transposed_weights);
      }
      EvalQuantizedPerChannel16x8<kernel_type>(
          context, params, data, input, weights, transposed_weights, bias,
          col2im, output, scratch_buffer);
      break;
    }
    default:
      TF_LITE_KERNEL_LOG(context, "Type '%s' is not currently supported.",
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
