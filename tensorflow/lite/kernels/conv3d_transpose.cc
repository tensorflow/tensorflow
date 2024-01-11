/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/internal/reference/conv3d_transpose.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv3d_transpose.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace conv3d_transpose {

enum KernelType {
  kReference,
  kGenericOptimized,
};

const int kTensorNotAllocated = -1;

struct OpData {
  Padding3DValues padding;

  // IDs are the arbitrary identifiers used by TF Lite to identify and access
  // memory buffers.
  int col2im_id = kTensorNotAllocated;
  int scratch_tensor_id = kTensorNotAllocated;
  // The index of col2im tensor in the temporaries list.
  int col2im_index;

  bool need_col2im = false;

  // Scratch tensor is used in the quantized path for storing accumulation
  // results.
  int32_t scratch_tensor_index;

  // Per channel output multiplier and shift.
  std::vector<int32_t> per_channel_output_multiplier;
  std::vector<int> per_channel_output_shift;

  int32_t output_activation_min;
  int32_t output_activation_max;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* opdata = new OpData;
  return opdata;
}

void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
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

static TfLiteStatus AllocateTemporaryTensorsIfRequired(TfLiteContext* context,
                                                       TfLiteType input_type,
                                                       TfLiteNode* node,
                                                       KernelType kernel_type) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  int temporaries_count = 0;

  // Allocate col2im tensor for the optimized kernel.
  if (kernel_type == kGenericOptimized && input_type != kTfLiteInt8 &&
      input_type != kTfLiteInt16) {
    if (data->col2im_id == kTensorNotAllocated) {
      context->AddTensors(context, 1, &data->col2im_id);
    }
    data->col2im_index = temporaries_count++;
    data->need_col2im = true;
  }

  // Allocate scratch buffer tensor
  if (input_type == kTfLiteInt8 || input_type == kTfLiteInt16) {
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

TfLiteStatus ResizeOutputAndTemporaryTensors(
    TfLiteContext* context, OpData* opdata, TfLiteConv3DTransposeParams* params,
    const TfLiteTensor* shape_tensor, const TfLiteTensor* filter,
    const TfLiteTensor* input, TfLiteTensor* col2im, TfLiteTensor* output) {
  auto shape_data = GetTensorData<int32_t>(shape_tensor);
  // Output and input tensor must have the same batch size.
  TF_LITE_ENSURE_EQ(context, shape_data[0], SizeOfDimension(input, 0));
  // The number of channels of output must be divisible by that of filter.
  TF_LITE_ENSURE_EQ(context, shape_data[4] % SizeOfDimension(filter, 3), 0);

  // Compute padding.
  const RuntimeShape& filter_shape = GetTensorShape(filter);
  const int depth = shape_data[1];
  const int height = shape_data[2];
  const int width = shape_data[3];
  const int filter_depth = filter_shape.Dims(0);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  int unused_out_width, unused_out_height, unused_out_depth;
  opdata->padding = ComputePadding3DValues(
      params->stride_height, params->stride_width, params->stride_depth,
      params->dilation_height_factor, params->dilation_width_factor,
      params->dilation_depth_factor, height, width, depth, filter_height,
      filter_width, filter_depth, params->padding, &unused_out_height,
      &unused_out_width, &unused_out_depth);
  // Computed shape must match the shape of the input tensor.
  TF_LITE_ENSURE_EQ(context, unused_out_depth, SizeOfDimension(input, 1));
  TF_LITE_ENSURE_EQ(context, unused_out_height, SizeOfDimension(input, 2));
  TF_LITE_ENSURE_EQ(context, unused_out_width, SizeOfDimension(input, 3));

  TfLiteIntArray* output_shape =
      TfLiteIntArrayCreate(NumElements(shape_tensor));
  for (int i = 0; i < output_shape->size; ++i) {
    output_shape->data[i] = GetTensorData<int32_t>(shape_tensor)[i];
  }

  TF_LITE_ENSURE_STATUS(context->ResizeTensor(context, output, output_shape));

  // Resize col2im tensor.
  if (opdata->need_col2im) {
    TfLiteIntArray* col2im_shape_array = TfLiteIntArrayCreate(2);
    const RuntimeShape& input_shape = GetTensorShape(input);
    col2im_shape_array->data[0] =
        input_shape.Dims(1) * input_shape.Dims(2) * input_shape.Dims(3);
    col2im_shape_array->data[1] =
        filter_depth * filter_height * filter_width * filter_shape.Dims(3);

    col2im->type = kTfLiteFloat32;
    col2im->allocation_type = kTfLiteDynamic;
    return context->ResizeTensor(context, col2im, col2im_shape_array);
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(KernelType kernel_type, TfLiteContext* context,
                     TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteConv3DTransposeParams*>(node->builtin_data);
  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);
  // Check number of inputs/outputs.
  TF_LITE_ENSURE(context, node->inputs->size == 3 || node->inputs->size == 4);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* output_shape;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &output_shape));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &input));

  // Check dimensionality of inputs/outputs.
  TF_LITE_ENSURE_EQ(context, output_shape->dims->size, 1);
  TF_LITE_ENSURE_EQ(context, NumElements(output_shape), 5);
  TF_LITE_ENSURE_EQ(context, input->dims->size, 5);
  TF_LITE_ENSURE_EQ(context, filter->dims->size, 5);

  // Input and filter must have the same number of channels.
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(input, 4),
                    SizeOfDimension(filter, 4));

  // Check types.
  TfLiteType input_type = input->type;
  TF_LITE_ENSURE(context, input_type == kTfLiteFloat32 ||
                              input_type == kTfLiteInt8 ||
                              input_type == kTfLiteInt16);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input_type);
  TF_LITE_ENSURE_TYPES_EQ(context, output_shape->type, kTfLiteInt32);
  if (input_type == kTfLiteInt8 || input_type == kTfLiteInt16) {
    TF_LITE_ENSURE_TYPES_EQ(context, filter->type, kTfLiteInt8);
  } else {
    TF_LITE_ENSURE_TYPES_EQ(context, filter->type, kTfLiteFloat32);
  }

  if (input_type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  TF_LITE_ENSURE_EQ(context, filter->params.zero_point, 0);

  // Check bias.
  const TfLiteTensor* bias = GetInput(context, node, 3);
  if (bias) {
    if (input_type == kTfLiteInt8) {
      TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt32);
      TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    } else if (input_type == kTfLiteInt16) {
      TF_LITE_ENSURE_TYPES_EQ(context, bias->type, kTfLiteInt64);
      TF_LITE_ENSURE_EQ(context, bias->params.zero_point, 0);
    } else {
      TF_LITE_ENSURE_TYPES_EQ(context, bias->type, input_type);
    }
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 3));
  }

  // GenericOptimized kernel currently doesn't support dilation.
  if (params->dilation_depth_factor > 1 || params->dilation_height_factor > 1 ||
      params->dilation_width_factor > 1) {
    kernel_type = kReference;
  }

  // Allocate temporary tensors.
  TF_LITE_ENSURE_STATUS(AllocateTemporaryTensorsIfRequired(context, input->type,
                                                           node, kernel_type));

  if (input_type != kTfLiteFloat32) {
    node->temporaries->data[opdata->scratch_tensor_index] =
        opdata->scratch_tensor_id;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, opdata->scratch_tensor_index,
                                  &scratch_buffer));
    if (input->type == kTfLiteInt16) {
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

    TF_LITE_ENSURE_EQ(context, filter->quantization.type,
                      kTfLiteAffineQuantization);
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    int channels_out = filter->dims->data[3];
    TF_LITE_ENSURE(context, affine_quantization);
    TF_LITE_ENSURE(context, affine_quantization->scale);
    TF_LITE_ENSURE(context,
                   (affine_quantization->scale->size == 1) ||
                       (affine_quantization->scale->size == channels_out));
    for (int i = 0; i < affine_quantization->zero_point->size; ++i) {
      TF_LITE_ENSURE_EQ(context, affine_quantization->zero_point->data[i], 0);
    }

    opdata->per_channel_output_multiplier.resize(channels_out);
    opdata->per_channel_output_shift.resize(channels_out);
    TF_LITE_ENSURE_STATUS(tflite::PopulateConvolutionQuantizationParams(
        context, input, filter, bias, output, params->activation, nullptr,
        nullptr, &opdata->output_activation_min, &opdata->output_activation_max,
        opdata->per_channel_output_multiplier.data(),
        opdata->per_channel_output_shift.data(), channels_out));
  }

  // Check temporary tensors.
  TfLiteTensor* col2im = nullptr;
  if (opdata->need_col2im) {
    node->temporaries->data[opdata->col2im_index] = opdata->col2im_id;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node,
                                                opdata->col2im_index, &col2im));
  }

  // Resize the output tensor.
  if (!IsConstantOrPersistentTensor(output_shape)) {
    SetTensorToDynamic(output);
    if (opdata->need_col2im) {
      SetTensorToDynamic(col2im);
    }
  } else {
    TF_LITE_ENSURE_STATUS(ResizeOutputAndTemporaryTensors(
        context, opdata, params, output_shape, filter, input, col2im, output));
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return Prepare(kernel_type, context, node);
}

void EvalFloat(KernelType kernel_type, TfLiteContext* context, TfLiteNode* node,
               TfLiteConv3DTransposeParams* params, OpData* opdata,
               const TfLiteTensor* input, const TfLiteTensor* filter,
               const TfLiteTensor* bias, TfLiteTensor* col2im,
               TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  Conv3DTransposeParams runtime_params;
  runtime_params.padding_values = opdata->padding;
  runtime_params.stride_depth = params->stride_depth;
  runtime_params.stride_height = params->stride_height;
  runtime_params.stride_width = params->stride_width;
  runtime_params.dilation_depth = params->dilation_depth_factor;
  runtime_params.dilation_height = params->dilation_height_factor;
  runtime_params.dilation_width = params->dilation_width_factor;
  runtime_params.float_activation_min = output_activation_min;
  runtime_params.float_activation_max = output_activation_max;

  switch (kernel_type) {
    case kReference: {
      reference_ops::Conv3DTranspose(
          runtime_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), GetTensorData<float>(filter),
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output));
      break;
    }
    case kGenericOptimized: {
      optimized_ops::Conv3DTranspose(
          runtime_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), GetTensorData<float>(filter),
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output),
          GetTensorShape(col2im), GetTensorData<float>(col2im),
          CpuBackendContext::GetFromContext(context));
    } break;
  }
}

template <typename InputType, typename BiasType>
void EvalQuantizedPerChannel(KernelType kernel_type, TfLiteContext* context,
                             TfLiteNode* node, TfLiteConv3DParams* params,
                             OpData* opdata, const TfLiteTensor* input,
                             const TfLiteTensor* filter,
                             const TfLiteTensor* bias, TfLiteTensor* output,
                             TfLiteTensor* scratch_buffer) {
  Conv3DTransposeParams runtime_params;
  runtime_params.input_offset = -input->params.zero_point;
  runtime_params.output_offset = output->params.zero_point;
  runtime_params.padding_values = opdata->padding;
  runtime_params.stride_depth = params->stride_depth;
  runtime_params.stride_height = params->stride_height;
  runtime_params.stride_width = params->stride_width;
  runtime_params.dilation_depth = params->dilation_depth_factor;
  runtime_params.dilation_height = params->dilation_height_factor;
  runtime_params.dilation_width = params->dilation_width_factor;
  switch (kernel_type) {
    case kGenericOptimized:
    case kReference: {
      reference_integer_ops::Conv3DTransposePerChannel<InputType, BiasType>(
          runtime_params, opdata->per_channel_output_multiplier.data(),
          opdata->per_channel_output_shift.data(), GetTensorShape(input),
          GetTensorData<InputType>(input), GetTensorShape(filter),
          GetTensorData<int8_t>(filter), GetTensorShape(bias),
          GetTensorData<BiasType>(bias), GetTensorShape(output),
          GetTensorData<InputType>(output),
          GetTensorData<BiasType>(scratch_buffer));
      break;
    }
  }
}

TfLiteStatus Eval(KernelType kernel_type, TfLiteContext* context,
                  TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteConv3DTransposeParams*>(node->builtin_data);
  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* output_shape;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &output_shape));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 2, &input));
  const TfLiteTensor* bias = GetInput(context, node, 3);
  TfLiteTensor* col2im = opdata->need_col2im
                             ? GetTemporary(context, node, opdata->col2im_index)
                             : nullptr;

  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputAndTemporaryTensors(
                                   context, opdata, params, output_shape,
                                   filter, input, col2im, output));
  }

  // GenericOptimized kernel currently doesn't support dilation.
  if (params->dilation_depth_factor > 1 || params->dilation_height_factor > 1 ||
      params->dilation_width_factor > 1) {
    kernel_type = kReference;
  }

  TfLiteTensor* scratch_buffer = nullptr;
  if (input->type == kTfLiteInt8 || input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, opdata->scratch_tensor_index,
                                  &scratch_buffer));
    if (IsDynamicTensor(scratch_buffer)) {
      TF_LITE_ENSURE_OK(context,
                        ResizeTensor(context, output_shape, scratch_buffer));
    }
  }

  switch (input->type) {
    case kTfLiteFloat32:
      EvalFloat(kernel_type, context, node, params, opdata, input, filter, bias,
                col2im, output);
      break;
    case kTfLiteInt8:
      EvalQuantizedPerChannel<int8_t, int32_t>(kernel_type, context, node,
                                               params, opdata, input, filter,
                                               bias, output, scratch_buffer);
      break;
    case kTfLiteInt16:
      EvalQuantizedPerChannel<int16_t, int64_t>(kernel_type, context, node,
                                                params, opdata, input, filter,
                                                bias, output, scratch_buffer);
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s currently not supported.",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  return Eval(kernel_type, context, node);
}

}  // namespace conv3d_transpose

TfLiteRegistration* Register_CONV_3D_TRANSPOSE_REF() {
  static TfLiteRegistration r = {
      conv3d_transpose::Init, conv3d_transpose::Free,
      conv3d_transpose::Prepare<conv3d_transpose::kReference>,
      conv3d_transpose::Eval<conv3d_transpose::kReference>};
  return &r;
}

TfLiteRegistration* Register_CONV_3D_TRANSPOSE_GENERIC_OPT() {
  static TfLiteRegistration r = {
      conv3d_transpose::Init, conv3d_transpose::Free,
      conv3d_transpose::Prepare<conv3d_transpose::kGenericOptimized>,
      conv3d_transpose::Eval<conv3d_transpose::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_CONV_3D_TRANSPOSE() {
  return Register_CONV_3D_TRANSPOSE_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
