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

#include "tensorflow/lite/kernels/internal/reference/strided_slice.h"

#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace strided_slice {

enum KernelType {
  kReference,
  // TODO(soroosh): add kGenericOptimized
};

constexpr int kInputTensor = 0;
constexpr int kBeginTensor = 1;
constexpr int kEndTensor = 2;
constexpr int kStridesTensor = 3;
constexpr int kOutputTensor = 0;

struct StridedSliceContext {
  StridedSliceContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteStridedSliceParams*>(node->builtin_data);
    input = GetInput(context, node, kInputTensor);
    begin = GetInput(context, node, kBeginTensor);
    end = GetInput(context, node, kEndTensor);
    strides = GetInput(context, node, kStridesTensor);
    output = GetOutput(context, node, kOutputTensor);
    dims = NumDimensions(input);
  }
  const TfLiteStridedSliceParams* params;
  const TfLiteTensor* input;
  const TfLiteTensor* begin;
  const TfLiteTensor* end;
  const TfLiteTensor* strides;
  TfLiteTensor* output;
  int dims;
};

StridedSliceParams BuildStridedSliceParams(StridedSliceContext* op_context) {
  StridedSliceParams op_params;
  op_params.start_indices_count = op_context->dims;
  op_params.stop_indices_count = op_context->dims;
  op_params.strides_count = op_context->dims;

  for (int i = 0; i < op_context->dims; ++i) {
    op_params.start_indices[i] = GetTensorData<int32_t>(op_context->begin)[i];
    op_params.stop_indices[i] = GetTensorData<int32_t>(op_context->end)[i];
    op_params.strides[i] = GetTensorData<int32_t>(op_context->strides)[i];
  }

  op_params.begin_mask = op_context->params->begin_mask;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = op_context->params->end_mask;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = op_context->params->shrink_axis_mask;
  return op_params;
}

// Processes the indexing tensors (begin, end and strides) to resize the
// output tensor. This function is callable from both Prepare() and Eval() as
// long as the caller ensures the indexing tensors are present.
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                StridedSliceContext* op_context) {
  std::vector<int> output_shape_vector;
  StridedSliceParams op_params = BuildStridedSliceParams(op_context);
  RuntimeShape input_shape = GetTensorShape(op_context->input);

  for (int idx = op_context->dims - 1; idx >= 0; --idx) {
    int32_t stride = GetTensorData<int32_t>(op_context->strides)[idx];
    TF_LITE_ENSURE_MSG(context, stride != 0, "stride value has to be non-zero");

    int32_t begin =
        ::tflite::strided_slice::StartForAxis(op_params, input_shape, idx);
    int32_t end = ::tflite::strided_slice::StopForAxis(op_params, input_shape,
                                                       idx, begin);

    // When shrinking an axis, the end position does not matter (and can be
    // incorrect when negative indexing is used, see Issue #19260). Always use
    // begin + 1 to generate a length 1 slice, since begin has
    // already been adjusted for negative indices by GetBeginValueAtIndex.
    const bool shrink_axis = op_context->params->shrink_axis_mask & (1 << idx);
    if (shrink_axis) {
      end = begin + 1;
    }

    // This is valid for both positive and negative strides
    int32_t dim_shape = std::ceil((end - begin) / static_cast<float>(stride));
    dim_shape = dim_shape < 0 ? 0 : dim_shape;
    if (!shrink_axis) {
      output_shape_vector.push_back(dim_shape);
    }
  }

  TfLiteIntArray* output_shape =
      TfLiteIntArrayCreate(output_shape_vector.size());

  std::reverse_copy(output_shape_vector.begin(), output_shape_vector.end(),
                    output_shape->data);

  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, op_context->output, output_shape));

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  StridedSliceContext op_context(context, node);

  // Ensure validity of input tensor and its dimension
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.begin), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.end), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.strides), 1);
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);
  // Only INT32 begin/end/strides are supported
  // TODO(soroosh) add support for INT64
  TF_LITE_ENSURE_EQ(context, op_context.begin->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, op_context.end->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, op_context.strides->type, kTfLiteInt32);
  TF_LITE_ENSURE_MSG(context, op_context.dims <= 4,
                     "StridedSlice op only supports 1D-4D input arrays.");

  // TODO(soroosh): add the following missing functionalities
  TF_LITE_ENSURE_MSG(context, op_context.params->ellipsis_mask == 0,
                     "ellipsis_mask is not implemented yet.");
  TF_LITE_ENSURE_MSG(context, op_context.params->new_axis_mask == 0,
                     "new_axis_mask is not implemented yet.");

  // Postpone allocation of output if any of the indexing tensors is not
  // constant
  if (!(IsConstantTensor(op_context.begin) &&
        IsConstantTensor(op_context.end) &&
        IsConstantTensor(op_context.strides))) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }
  return ResizeOutputTensor(context, &op_context);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  StridedSliceContext op_context(context, node);

  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }
  StridedSliceParams op_params = BuildStridedSliceParams(&op_context);

#define TF_LITE_STRIDED_SLICE(kernel_type, data_type)                    \
  kernel_type::StridedSlice(op_params, GetTensorShape(op_context.input), \
                            GetTensorData<data_type>(op_context.input),  \
                            GetTensorShape(op_context.output),           \
                            GetTensorData<data_type>(op_context.output))

  switch (op_context.input->type) {
    case kTfLiteFloat32:
      if (kernel_type == kReference) {
        TF_LITE_STRIDED_SLICE(reference_ops, float);
      }
      break;
    case kTfLiteInt32:
      if (kernel_type == kReference) {
        TF_LITE_STRIDED_SLICE(reference_ops, int32_t);
      }
      break;
    case kTfLiteInt64:
      if (kernel_type == kReference) {
        TF_LITE_STRIDED_SLICE(reference_ops, int64_t);
      }
      break;
    case kTfLiteUInt8:
      if (kernel_type == kReference) {
        TF_LITE_STRIDED_SLICE(reference_ops, uint8_t);
      }
      break;
    case kTfLiteInt8:
      if (kernel_type == kReference) {
        TF_LITE_STRIDED_SLICE(reference_ops, int8_t);
      }
      break;
    case kTfLiteInt16:
      if (kernel_type == kReference) {
        TF_LITE_STRIDED_SLICE(reference_ops, int16_t);
      }
      break;
    case kTfLiteBool:
      if (kernel_type == kReference) {
        TF_LITE_STRIDED_SLICE(reference_ops, bool);
      }
      break;
    default:
      context->ReportError(context,
                           "Type %d is currently not supported "
                           "by StridedSlice.",
                           op_context.input->type);
      return kTfLiteError;
  }
#undef TF_LITE_STRIDED_SLICE
  return kTfLiteOk;
}

}  // namespace strided_slice

TfLiteRegistration* Register_STRIDED_SLICE_REF() {
  static TfLiteRegistration r = {
      nullptr, nullptr, strided_slice::Prepare,
      strided_slice::Eval<strided_slice::kReference>};
  return &r;
}

// TODO(soroosh): add optimized
TfLiteRegistration* Register_STRIDED_SLICE() {
  return Register_STRIDED_SLICE_REF();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
