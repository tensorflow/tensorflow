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

#include <math.h>
#include <stdint.h>

#include <algorithm>
#include <cmath>
#include <vector>

#include "Eigen/Core"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/strided_slice_logic.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace strided_slice {

enum KernelType {
  kReference,
  kGenericOptimized,
};

constexpr int kInputTensor = 0;
constexpr int kBeginTensor = 1;
constexpr int kEndTensor = 2;
constexpr int kStridesTensor = 3;
constexpr int kOutputTensor = 0;

struct OpData {
  // Indicates that 'Eval' is a noop as the output as written during 'Prepare'.
  bool noop;
};

struct StridedSliceContext {
  StridedSliceContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteStridedSliceParams*>(node->builtin_data);
    input = GetInput(context, node, kInputTensor);
    begin = GetInput(context, node, kBeginTensor);
    end = GetInput(context, node, kEndTensor);
    strides = GetInput(context, node, kStridesTensor);
    output = GetOutput(context, node, kOutputTensor);
    input_dims = NumDimensions(input);
  }
  const TfLiteStridedSliceParams* params;
  const TfLiteTensor* input;
  const TfLiteTensor* begin;
  const TfLiteTensor* end;
  const TfLiteTensor* strides;
  TfLiteTensor* output;

  // Equivalent input shape after adding axis according to new_axis_mask.
  RuntimeShape effective_input_shape;
  int input_dims;
};

StridedSliceParams BuildStridedSliceParams(StridedSliceContext* op_context,
                                           bool start_and_end_indices) {
  StridedSliceParams op_params{};

  // The ellipsis_mask and new_axis_mask in op_params are not used. Those masks
  // are processed here to update begin_mask, end_mask and the index range.
  op_params.begin_mask = 0;
  op_params.ellipsis_mask = 0;
  op_params.end_mask = 0;
  op_params.new_axis_mask = 0;
  op_params.shrink_axis_mask = 0;
  op_params.offset = op_context->params->offset;

  // Count indexes where the new_axis_mask is set but the ellipsis_mask is not.
  const int begin_count = GetTensorShape(op_context->begin).Dims(0);
  int num_add_axis = 0;
  for (int i = 0; i < begin_count; ++i) {
    if (!((1 << i) & op_context->params->ellipsis_mask) &&
        ((1 << i) & op_context->params->new_axis_mask)) {
      num_add_axis++;
    }
  }

  // Calculate the dims of input after adding new axises.
  const int effective_dims = op_context->input_dims + num_add_axis;

  // If begin, end and strides are not fully provided, it means Ellipsis should
  // be expanded to multiple dimensions (Ex: for spec [Ellipsis, 2] on a 3D
  // input, the Ellipsis should be applied for the first 2 dimensions). Besides,
  // If the new_axis_mask and the ellipsis_mask are set at the same index, the
  // new_axis_mask will have no effect.
  int effective_ellipsis_mask = 0, effective_new_axis_mask = 0;
  int ellipsis_start_idx = effective_dims, expanded_ellipsis = 0;
  for (int i = 0; i < effective_dims;) {
    if ((1 << i) & op_context->params->ellipsis_mask) {
      ellipsis_start_idx = i;
      int ellipsis_end_idx = std::max(
          i + 1,
          std::min(i + 1 + num_add_axis + op_context->input_dims - begin_count,
                   effective_dims));
      expanded_ellipsis = ellipsis_end_idx - ellipsis_start_idx - 1;

      // Set bit for effective_ellipsis_mask.
      for (; i < ellipsis_end_idx; ++i) {
        effective_ellipsis_mask |= (1 << i);
      }
      continue;
    }

    if ((1 << (i - expanded_ellipsis)) & op_context->params->new_axis_mask) {
      effective_new_axis_mask |= (1 << i);
    }
    ++i;
  }

  // Calculate effective_input_shape and its corresponding begin, end, strides.
  const int32_t* begin_data = GetTensorData<int32_t>(op_context->begin);
  const int32_t* end_data = GetTensorData<int32_t>(op_context->end);
  const int32_t* strides_data = GetTensorData<int32_t>(op_context->strides);
  const RuntimeShape input_shape = GetTensorShape(op_context->input);
  int added_ellipsis = 0, added_axises = 0;
  op_context->effective_input_shape.Resize(effective_dims);

  for (int i = 0; i < effective_dims; ++i) {
    if ((1 << i) & effective_ellipsis_mask) {
      // If ellipsis_mask, set the begin_mask and end_mask at that index.
      added_ellipsis = std::max(0, i - ellipsis_start_idx);
      op_params.begin_mask |= (1 << i);
      op_params.end_mask |= (1 << i);
      op_params.strides[i] = 1;
      op_context->effective_input_shape.SetDim(
          i, input_shape.Dims(i - added_axises));
    } else if ((1 << i) & effective_new_axis_mask) {
      // If new_axis_mask is set, it is equivalent to adding a new dim of 1 to
      // input tensor. Store added shape to effective_input_shape.
      op_params.start_indices[i] = 0;
      op_params.stop_indices[i] = 1;
      op_params.strides[i] = 1;
      op_context->effective_input_shape.SetDim(i, 1);
      added_axises++;
    } else if (i >= begin_count + expanded_ellipsis) {
      op_params.start_indices[i] = 0;
      op_params.stop_indices[i] = 0;
      op_params.strides[i] = 1;
      op_params.begin_mask |= (1 << i);
      op_params.end_mask |= (1 << i);
      op_context->effective_input_shape.SetDim(
          i, input_shape.Dims(i - added_axises));
    } else {
      const int orig_idx = i - added_ellipsis;
      if (start_and_end_indices) {
        op_params.start_indices[i] = begin_data[orig_idx];
        op_params.stop_indices[i] = end_data[orig_idx];
      }
      op_params.strides[i] = strides_data[orig_idx];
      if (op_context->params->begin_mask & (1 << orig_idx)) {
        op_params.begin_mask |= (1 << i);
      }
      if (op_context->params->end_mask & (1 << orig_idx)) {
        op_params.end_mask |= (1 << i);
      }
      if (op_context->params->shrink_axis_mask & (1 << orig_idx)) {
        op_params.shrink_axis_mask |= (1 << i);
      }
      op_context->effective_input_shape.SetDim(
          i, input_shape.Dims(i - added_axises));
    }
  }
  op_params.start_indices_count = effective_dims;
  op_params.stop_indices_count = effective_dims;
  op_params.strides_count = effective_dims;

  return op_params;
}

// Processes the indexing tensors (begin, end and strides) to resize the
// output tensor. This function is callable from both Prepare() and Eval() as
// long as the caller ensures the indexing tensors are present.
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                StridedSliceContext* op_context) {
  std::vector<int> output_shape_vector;
  StridedSliceParams op_params =
      BuildStridedSliceParams(op_context, !op_context->params->offset);
  const RuntimeShape effective_input_shape = op_context->effective_input_shape;
  TF_LITE_ENSURE_MSG(
      context, effective_input_shape.DimensionsCount() <= 5,
      "StridedSlice op only supports up to 5D output including added axis.");

  const int32_t* end_data = GetTensorData<int32_t>(op_context->end);
  for (int idx = effective_input_shape.DimensionsCount() - 1; idx >= 0; --idx) {
    int32_t stride = op_params.strides[idx];
    TF_LITE_ENSURE_MSG(context, stride != 0, "stride value has to be non-zero");

    int32_t dim_shape = 0;
    const bool shrink_axis = op_params.shrink_axis_mask & (1 << idx);
    if (shrink_axis) continue;
    if (op_params.offset) {
      dim_shape = end_data[idx];
    } else {
      int32_t begin = ::tflite::strided_slice::StridedSliceStartForAxis(
          op_params, effective_input_shape, idx);
      int32_t end = ::tflite::strided_slice::StridedSliceEndForAxis(
          op_params, effective_input_shape, idx, begin);

      // This is valid for both positive and negative strides
      dim_shape = end - begin;
    }
    dim_shape = std::ceil((dim_shape) / static_cast<float>(stride));
    dim_shape = dim_shape < 0 ? 0 : dim_shape;
    output_shape_vector.push_back(dim_shape);
  }

  TfLiteIntArray* output_shape =
      TfLiteIntArrayCreate(output_shape_vector.size());

  std::reverse_copy(output_shape_vector.begin(), output_shape_vector.end(),
                    output_shape->data);

  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, op_context->output, output_shape));

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node) {
  StridedSliceContext op_context(context, node);

  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  }
  StridedSliceParams op_params = BuildStridedSliceParams(&op_context, true);

  switch (op_context.input->type) {
    case kTfLiteFloat32:
      reference_ops::StridedSlice<float>(
          op_params, op_context.effective_input_shape, op_context.input,
          GetTensorShape(op_context.output), op_context.output);
      break;
    case kTfLiteFloat16:
      reference_ops::StridedSlice<Eigen::half>(
          op_params, op_context.effective_input_shape, op_context.input,
          GetTensorShape(op_context.output), op_context.output);
      break;
    case kTfLiteBFloat16:
      reference_ops::StridedSlice<Eigen::bfloat16>(
          op_params, op_context.effective_input_shape, op_context.input,
          GetTensorShape(op_context.output), op_context.output);
      break;
    case kTfLiteInt32:
      reference_ops::StridedSlice<int32_t>(
          op_params, op_context.effective_input_shape, op_context.input,
          GetTensorShape(op_context.output), op_context.output);
      break;
    case kTfLiteInt64:
      reference_ops::StridedSlice<int64_t>(
          op_params, op_context.effective_input_shape, op_context.input,
          GetTensorShape(op_context.output), op_context.output);
      break;
    case kTfLiteUInt8:
      reference_ops::StridedSlice<uint8_t>(
          op_params, op_context.effective_input_shape, op_context.input,
          GetTensorShape(op_context.output), op_context.output);
      break;
    case kTfLiteUInt32:
      reference_ops::StridedSlice<uint32_t>(
          op_params, op_context.effective_input_shape, op_context.input,
          GetTensorShape(op_context.output), op_context.output);
      break;
    case kTfLiteInt8:
      reference_ops::StridedSlice<int8_t>(
          op_params, op_context.effective_input_shape, op_context.input,
          GetTensorShape(op_context.output), op_context.output);
      break;
    case kTfLiteInt16:
      reference_ops::StridedSlice<int16_t>(
          op_params, op_context.effective_input_shape, op_context.input,
          GetTensorShape(op_context.output), op_context.output);
      break;
    case kTfLiteBool:
      reference_ops::StridedSlice<bool>(
          op_params, op_context.effective_input_shape, op_context.input,
          GetTensorShape(op_context.output), op_context.output);
      break;
    case kTfLiteString:
      reference_ops::StridedSlice<string>(
          op_params, op_context.effective_input_shape, op_context.input,
          GetTensorShape(op_context.output), op_context.output);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Type %s is currently not supported "
                         "by StridedSlice.",
                         TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }
#undef TF_LITE_STRIDED_SLICE
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  op_data->noop = false;
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 4);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  StridedSliceContext op_context(context, node);

  // Ensure validity of input tensor and its dimension
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.begin), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.end), 1);
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.strides), 1);
  TF_LITE_ENSURE_EQ(context, NumElements(op_context.begin),
                    NumElements(op_context.end));
  TF_LITE_ENSURE_EQ(context, op_context.input->type, op_context.output->type);

  // Only INT32 begin/end/strides are supported
  // TODO(b/253465311): add support for INT64
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.begin->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.end->type, kTfLiteInt32);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.strides->type, kTfLiteInt32);
  TF_LITE_ENSURE_MSG(context, op_context.input_dims <= 5,
                     "StridedSlice op only supports 1D-5D input arrays.");

  // Postpone allocation of output if any of the indexing tensors is not
  // constant
  bool offset = op_context.params->offset;
  bool output_shape_known = IsConstantOrPersistentTensor(op_context.strides);
  output_shape_known &=
      offset || (IsConstantOrPersistentTensor(op_context.begin) &&
                 IsConstantOrPersistentTensor(op_context.end));
  if (!output_shape_known) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }
  if (IsConstantOrPersistentTensor(op_context.input) &&
      IsConstantOrPersistentTensor(op_context.begin) &&
      IsConstantOrPersistentTensor(op_context.end)) {
    SetTensorToPersistentRo(op_context.output);
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
    op_data->noop = true;
    return EvalImpl<kGenericOptimized>(context, node);
  }
  return ResizeOutputTensor(context, &op_context);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  StridedSliceContext op_context(context, node);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  if (op_data->noop) {
    return kTfLiteOk;
  }
  return EvalImpl<kernel_type>(context, node);
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return new OpData;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}
}  // namespace strided_slice

TfLiteRegistration* Register_STRIDED_SLICE_REF() {
  static TfLiteRegistration r = {
      strided_slice::Init, strided_slice::Free, strided_slice::Prepare,
      strided_slice::Eval<strided_slice::kReference>};
  return &r;
}

TfLiteRegistration* Register_STRIDED_SLICE() {
  static TfLiteRegistration r = {
      strided_slice::Init, strided_slice::Free, strided_slice::Prepare,
      strided_slice::Eval<strided_slice::kGenericOptimized>};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
