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

#include "tensorflow/lite/kernels/internal/reference/conv3d.h"

#include <cstddef>
#include <cstdint>
#include <vector>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/padding.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace conv3d {

enum KernelType {
  kReference,
  kGenericOptimized,
};

// Struct to carry data from Prepare to Eval.
const int kTensorNotAllocated = -1;
static constexpr size_t kMaxIm2colBufferSizeMobile = 1024 * 1024 * 1024;  // 1GB

struct OpData {
  Padding3DValues padding;
  int im2col_tensor_id = kTensorNotAllocated;

  bool need_im2col = false;

  // Disable im2col if the temporary im2col tensor requires too much memory
  // (i.e. >= kMaxIm2colBufferSizeMobile).
  bool im2col_oversized = false;

  int32_t im2col_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* opdata = new OpData;
  return opdata;
}

void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus AllocateTemporaryTensorsIfRequired(
    KernelType kernel_type, TfLiteContext* context, TfLiteNode* node,
    OpData* opdata, TfLiteConv3DParams* params, const TfLiteTensor* filter,
    size_t im2col_bytes) {
  int temporaries_count = 0;
  const bool need_dilated_im2col = params->dilation_width_factor != 1 ||
                                   params->dilation_height_factor != 1 ||
                                   params->dilation_depth_factor != 1;
  const bool need_non_dilated_im2col =
      params->stride_depth != 1 || params->stride_width != 1 ||
      params->stride_height != 1 || filter->dims->data[2] != 1 ||
      filter->dims->data[1] != 1 || filter->dims->data[0] != 1;

  opdata->need_im2col = (kernel_type == kGenericOptimized) &&
                        (need_dilated_im2col || need_non_dilated_im2col);

  // On mobile platforms, the generic optimized kernel will not be used if the
  // temporary im2col tensor requires too much memory.
  if (IsMobilePlatform() && opdata->need_im2col &&
      im2col_bytes >= kMaxIm2colBufferSizeMobile) {
    opdata->need_im2col = false;
    opdata->im2col_oversized = true;
  }

  if (opdata->need_im2col) {
    if (opdata->im2col_tensor_id == kTensorNotAllocated) {
      TF_LITE_ENSURE_OK(
          context, context->AddTensors(context, 1, &opdata->im2col_tensor_id));
    }
    opdata->im2col_index = temporaries_count++;
  }

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(temporaries_count);
  return kTfLiteOk;
}

TfLiteStatus Prepare(KernelType kernel_type, TfLiteContext* context,
                     TfLiteNode* node) {
  auto* params = static_cast<TfLiteConv3DParams*>(node->builtin_data);
  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);

  // Check number of inputs/outputs.
  TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));

  // Check dimensionality of input, filter.
  TF_LITE_ENSURE_EQ(context, input->dims->size, 5);
  TF_LITE_ENSURE_EQ(context, filter->dims->size, 5);

  // Check input channels matching filter.
  TF_LITE_ENSURE_EQ(context, input->dims->data[4], filter->dims->data[3]);

  // Check types.
  TfLiteType input_type = input->type;
  TF_LITE_ENSURE_TYPES_EQ(context, input_type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, filter->type, kTfLiteFloat32);
  TF_LITE_ENSURE_TYPES_EQ(context, output->type, input_type);

  // Check bias.
  const TfLiteTensor* bias = GetInput(context, node, 2);
  if (bias) {
    TF_LITE_ENSURE_TYPES_EQ(context, bias->type, input_type);
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 4));
  }

  // Filter has shape of [filter_depth, filter_height, filter_width,
  // in_channels, out_channels].
  int batches = input->dims->data[0];
  int channels_out = filter->dims->data[4];
  int depth = input->dims->data[1];
  int height = input->dims->data[2];
  int width = input->dims->data[3];
  int filter_depth = filter->dims->data[0];
  int filter_height = filter->dims->data[1];
  int filter_width = filter->dims->data[2];
  int input_channel = filter->dims->data[3];

  // Matching GetWindowedOutputSize in TensorFlow.
  int out_width, out_height, out_depth;
  opdata->padding = ComputePadding3DValues(
      params->stride_height, params->stride_width, params->stride_depth,
      params->dilation_height_factor, params->dilation_width_factor,
      params->dilation_depth_factor, height, width, depth, filter_height,
      filter_width, filter_depth, params->padding, &out_height, &out_width,
      &out_depth);

  TfLiteIntArray* output_size = TfLiteIntArrayCreate(5);
  output_size->data[0] = batches;
  output_size->data[1] = out_depth;
  output_size->data[2] = out_height;
  output_size->data[3] = out_width;
  output_size->data[4] = channels_out;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size));

  // Allocate temporary tensors.
  size_t input_type_size;
  TF_LITE_ENSURE_STATUS(GetSizeOfType(context, input->type, &input_type_size));
  const size_t im2col_bytes = batches * out_depth * out_height * out_width *
                              input_channel * filter_depth * filter_height *
                              filter_width * input_type_size;
  TF_LITE_ENSURE_OK(context, AllocateTemporaryTensorsIfRequired(
                                 kernel_type, context, node, opdata, params,
                                 filter, im2col_bytes));

  if (opdata->need_im2col) {
    TfLiteIntArray* im2col_size = TfLiteIntArrayCreate(5);
    im2col_size->data[0] = output_size->data[0];
    im2col_size->data[1] = output_size->data[1];
    im2col_size->data[2] = output_size->data[2];
    im2col_size->data[3] = output_size->data[3];
    im2col_size->data[4] =
        input_channel * filter_depth * filter_height * filter_width;

    TfLiteTensor* im2col;
    node->temporaries->data[opdata->im2col_index] = opdata->im2col_tensor_id;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node,
                                                opdata->im2col_index, &im2col));
    im2col->type = input->type;
    im2col->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      context->ResizeTensor(context, im2col, im2col_size));
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return Prepare(kernel_type, context, node);
}

TfLiteStatus EvalFloat(KernelType kernel_type, TfLiteContext* context,
                       TfLiteNode* node, TfLiteConv3DParams* params,
                       OpData* opdata, const TfLiteTensor* input,
                       const TfLiteTensor* filter, const TfLiteTensor* bias,
                       TfLiteTensor* im2col, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);

  Conv3DParams runtime_params;
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
      reference_ops::Conv3D(runtime_params, GetTensorShape(input),
                            GetTensorData<float>(input), GetTensorShape(filter),
                            GetTensorData<float>(filter), GetTensorShape(bias),
                            GetTensorData<float>(bias), GetTensorShape(output),
                            GetTensorData<float>(output));
      return kTfLiteOk;
    }
    case kGenericOptimized: {
      return optimized_ops::Conv3D(
          runtime_params, GetTensorShape(input), GetTensorData<float>(input),
          GetTensorShape(filter), GetTensorData<float>(filter),
          GetTensorShape(bias), GetTensorData<float>(bias),
          GetTensorShape(output), GetTensorData<float>(output),
          GetTensorShape(im2col), GetTensorData<float>(im2col),
          CpuBackendContext::GetFromContext(context));
    }
  }
}

TfLiteStatus Eval(KernelType kernel_type, TfLiteContext* context,
                  TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteConv3DParams*>(node->builtin_data);
  OpData* opdata = reinterpret_cast<OpData*>(node->user_data);

  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &output));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  const TfLiteTensor* filter;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 1, &filter));
  const TfLiteTensor* bias = GetInput(context, node, 2);

  TfLiteTensor* im2col = opdata->need_im2col
                             ? &context->tensors[opdata->im2col_tensor_id]
                             : nullptr;
  // Fallback to reference execution path when im2col is needed but disabled.
  if (opdata->im2col_oversized) {
    kernel_type = kReference;
  }

  switch (input->type) {
    case kTfLiteFloat32:
      return EvalFloat(kernel_type, context, node, params, opdata, input,
                       filter, bias, im2col, output);
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

}  // namespace conv3d

TfLiteRegistration* Register_CONV_3D_REF() {
  static TfLiteRegistration r = {conv3d::Init, conv3d::Free,
                                 conv3d::Prepare<conv3d::kReference>,
                                 conv3d::Eval<conv3d::kReference>};
  return &r;
}

TfLiteRegistration* Register_CONV_3D_GENERIC_OPT() {
  static TfLiteRegistration r = {conv3d::Init, conv3d::Free,
                                 conv3d::Prepare<conv3d::kGenericOptimized>,
                                 conv3d::Eval<conv3d::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_CONV_3D() {
  return Register_CONV_3D_GENERIC_OPT();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
