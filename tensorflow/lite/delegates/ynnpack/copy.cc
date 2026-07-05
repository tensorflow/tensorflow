/* Copyright 2026 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/delegates/ynnpack/copy.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include "ynnpack/include/ynnpack.h"  // from @XNNPACK
#include "tensorflow/lite/builtin_ops.h"
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/ynnpack/utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ynnpack {

namespace {

int Clamp(int v, int min, int max) {
  return v < min ? min : (v > max ? max : v);
}

int StridedSliceStartForAxis(const TfLiteStridedSliceParams* params,
                             const TfLiteIntArray* input_dims, int axis,
                             int32_t start, int32_t stride) {
  const int32_t axis_size = input_dims->data[axis];
  const int32_t begin_mask = (params->begin_mask & (1 << axis));
  if (start < 0) {
    start += axis_size;
  }
  if (stride > 0) {
    start = Clamp(start, 0, axis_size);
  } else {
    start = Clamp(start, -1, axis_size - 1);
  }
  if (begin_mask) {
    if (stride > 0) {
      start = 0;
    } else {
      start = axis_size - 1;
    }
  }
  return start;
}

int StridedSliceEndForAxis(const TfLiteStridedSliceParams* params,
                           const TfLiteIntArray* input_dims, int axis,
                           int start, int32_t end, int32_t stride) {
  const auto shrink_axis_mask = params->shrink_axis_mask;
  const bool shrink_axis = shrink_axis_mask & (1 << axis);
  const int axis_size = input_dims->data[axis];
  if (shrink_axis) {
    if (start >= axis_size) {
      return start;
    } else {
      return start + 1;
    }
  }
  if (params->offset) {
    end += start;
  }
  const int32_t end_mask = (params->end_mask & (1 << axis));
  if (end < 0) {
    end += axis_size;
  }
  if (stride > 0) {
    end = Clamp(end, 0, axis_size);
  } else {
    end = Clamp(end, -1, axis_size - 1);
  }
  if (end_mask) {
    if (stride > 0) {
      end = axis_size;
    } else {
      end = -1;
    }
  }
  return end;
}

TfLiteStatus GetReshapeTargetShape(TfLiteContext* context,
                                   const TfLiteNode* tflite_node,
                                   int32_t* target_shape,
                                   int* target_shape_size) {
  if (tflite_node->inputs->size == 2) {
    const TfLiteTensor& shape_tensor =
        context->tensors[tflite_node->inputs->data[1]];
    if (shape_tensor.allocation_type == kTfLiteMmapRo) {
      int num_elements = tflite::NumElements(&shape_tensor);
      TF_LITE_ENSURE_MSG(context, num_elements <= YNN_MAX_TENSOR_RANK,
                         "Reshape shape elements %d exceeds max %d",
                         num_elements, YNN_MAX_TENSOR_RANK);
      *target_shape_size = num_elements;
      const int32_t* shape_data =
          reinterpret_cast<const int32_t*>(shape_tensor.data.raw);
      std::copy_n(shape_data, num_elements, target_shape);
      return kTfLiteOk;
    }
  }

  const auto* params =
      static_cast<const TfLiteReshapeParams*>(tflite_node->builtin_data);
  if (params && params->num_dimensions > 0) {
    TF_LITE_ENSURE_MSG(context, params->num_dimensions <= YNN_MAX_TENSOR_RANK,
                       "Reshape shape dimensions %d exceeds max %d",
                       params->num_dimensions, YNN_MAX_TENSOR_RANK);
    *target_shape_size = params->num_dimensions;
    std::copy_n(params->shape, params->num_dimensions, target_shape);
    return kTfLiteOk;
  }

  TF_LITE_ENSURE_MSG(context, false, "Failed to get target shape for Reshape");
}

}  // namespace

TfLiteStatus IsTransposeSupported(const TfLiteRegistration* registration,
                                  const TfLiteNode* node,
                                  TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& perm = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type input_ynn_type = GetYnnType(input.type);
  ynn_type output_ynn_type = GetYnnType(output.type);
  TF_LITE_ENSURE(context, input_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, input.type, output.type);

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  TF_LITE_ENSURE(context, QuantizationParamsEqual(input, output));

  TF_LITE_ENSURE(context, perm.allocation_type == kTfLiteMmapRo);
  TF_LITE_ENSURE_EQ(context, perm.type, kTfLiteInt32);

  TF_LITE_ENSURE_EQ(context, perm.dims->size, 1);
  TF_LITE_ENSURE_EQ(context, perm.dims->data[0], input.dims->size);
  TF_LITE_ENSURE_MSG(context, input.dims->size <= YNN_MAX_TENSOR_RANK,
                     "Input rank exceeds max rank");

  const int32_t* perm_data = reinterpret_cast<const int32_t*>(perm.data.raw);
  TF_LITE_ENSURE(context, perm_data != nullptr);
  bool seen[YNN_MAX_TENSOR_RANK] = {false};
  for (int i = 0; i < input.dims->size; ++i) {
    int32_t axis = perm_data[i];
    TF_LITE_ENSURE_MSG(context, axis >= 0 && axis < input.dims->size,
                       "Invalid perm axis");
    TF_LITE_ENSURE_MSG(context, !seen[axis], "Duplicate perm axis");
    seen[axis] = true;
  }

  return kTfLiteOk;
}

TfLiteStatus IsSliceSupported(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context) {
  const bool is_strided =
      (registration->builtin_code == kTfLiteBuiltinStridedSlice);
  if (is_strided) {
    TF_LITE_ENSURE_EQ(context, node->inputs->size, 4);
  } else {
    TF_LITE_ENSURE_EQ(context, node->inputs->size, 3);
  }
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& begin = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& end_or_size = context->tensors[node->inputs->data[2]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type input_ynn_type = GetYnnType(input.type);
  ynn_type output_ynn_type = GetYnnType(output.type);
  TF_LITE_ENSURE(context, input_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, input.type, output.type);

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  TF_LITE_ENSURE(context, QuantizationParamsEqual(input, output));

  TF_LITE_ENSURE(context, begin.allocation_type == kTfLiteMmapRo);
  TF_LITE_ENSURE(context, end_or_size.allocation_type == kTfLiteMmapRo);

  TF_LITE_ENSURE_EQ(context, begin.type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, end_or_size.type, kTfLiteInt32);

  TF_LITE_ENSURE_EQ(context, begin.dims->size, 1);
  TF_LITE_ENSURE_EQ(context, end_or_size.dims->size, 1);
  TF_LITE_ENSURE_EQ(context, begin.dims->data[0], input.dims->size);
  TF_LITE_ENSURE_EQ(context, end_or_size.dims->data[0], input.dims->size);

  const int32_t* begin_data = reinterpret_cast<const int32_t*>(begin.data.raw);
  const int32_t* end_or_size_data =
      reinterpret_cast<const int32_t*>(end_or_size.data.raw);

  if (is_strided) {
    const TfLiteTensor& strides = context->tensors[node->inputs->data[3]];
    TF_LITE_ENSURE(context, strides.allocation_type == kTfLiteMmapRo);
    TF_LITE_ENSURE_EQ(context, strides.type, kTfLiteInt32);
    TF_LITE_ENSURE_EQ(context, strides.dims->size, 1);
    TF_LITE_ENSURE_EQ(context, strides.dims->data[0], input.dims->size);

    const int32_t* strides_data =
        reinterpret_cast<const int32_t*>(strides.data.raw);
    for (int i = 0; i < strides.dims->data[0]; ++i) {
      TF_LITE_ENSURE_MSG(context, strides_data[i] > 0,
                         "Negative or zero strides are not supported");
    }

    const auto* params =
        static_cast<const TfLiteStridedSliceParams*>(node->builtin_data);
    TF_LITE_ENSURE(context, params != nullptr);

    TF_LITE_ENSURE_MSG(context, params->ellipsis_mask == 0,
                       "ellipsis_mask is not supported");
    TF_LITE_ENSURE_MSG(context, params->new_axis_mask == 0,
                       "new_axis_mask is not supported");
    TF_LITE_ENSURE_MSG(context, params->shrink_axis_mask == 0,
                       "shrink_axis_mask is not supported");

    for (int i = 0; i < input.dims->size; ++i) {
      int start = StridedSliceStartForAxis(params, input.dims, i, begin_data[i],
                                           strides_data[i]);
      int stop = StridedSliceEndForAxis(params, input.dims, i, start,
                                        end_or_size_data[i], strides_data[i]);
      TF_LITE_ENSURE_MSG(context, stop > start,
                         "0-sized outputs not supported");
    }
  } else {
    for (int i = 0; i < input.dims->size; ++i) {
      int32_t size_val = end_or_size_data[i];
      if (size_val < 0) {
        TF_LITE_ENSURE_EQ(context, size_val, -1);
        size_val = input.dims->data[i] - begin_data[i];
      }
      TF_LITE_ENSURE_MSG(context, size_val > 0,
                         "0-sized outputs not supported");
    }
  }

  return kTfLiteOk;
}

TfLiteStatus IsExpandDimsSupported(const TfLiteRegistration* registration,
                                   const TfLiteNode* node,
                                   TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& axis = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type input_ynn_type = GetYnnType(input.type);
  ynn_type output_ynn_type = GetYnnType(output.type);
  TF_LITE_ENSURE(context, input_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, input.type, output.type);

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  TF_LITE_ENSURE(context, QuantizationParamsEqual(input, output));

  TF_LITE_ENSURE(context, axis.allocation_type == kTfLiteMmapRo);
  TF_LITE_ENSURE_EQ(context, axis.type, kTfLiteInt32);
  TF_LITE_ENSURE_MSG(context, axis.dims->size <= 1,
                     "Axis tensor must be 0D or 1D");
  if (axis.dims->size == 1) {
    TF_LITE_ENSURE_EQ(context, axis.dims->data[0], 1);
  }

  int32_t axis_val = *reinterpret_cast<const int32_t*>(axis.data.raw);
  if (axis_val < 0) {
    axis_val += input.dims->size + 1;
  }
  TF_LITE_ENSURE_MSG(context, axis_val >= 0 && axis_val <= input.dims->size,
                     "Invalid axis value");

  TF_LITE_ENSURE_MSG(context, input.dims->size + 1 <= YNN_MAX_TENSOR_RANK,
                     "Target rank exceeds max rank");

  return kTfLiteOk;
}

TfLiteStatus IsConcatenationSupported(const TfLiteRegistration* registration,
                                      const TfLiteNode* node,
                                      TfLiteContext* context) {
  TF_LITE_ENSURE(context, node->inputs->size >= 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];
  ynn_type output_ynn_type = GetYnnType(output.type);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  const auto* params =
      static_cast<const TfLiteConcatenationParams*>(node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);

  for (int i = 0; i < node->inputs->size; ++i) {
    int input_index = node->inputs->data[i];
    if (input_index == kTfLiteOptionalTensor) continue;
    const TfLiteTensor& input = context->tensors[input_index];
    ynn_type input_ynn_type = GetYnnType(input.type);
    TF_LITE_ENSURE(context, input_ynn_type != ynn_type_invalid);
    TF_LITE_ENSURE_EQ(context, input.type, output.type);
    TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  }

  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  TF_LITE_ENSURE(context,
                 IsActivationSupported(params->activation, output.type));

  return kTfLiteOk;
}

TfLiteStatus IsReshapeSupported(const TfLiteRegistration* registration,
                                const TfLiteNode* node,
                                TfLiteContext* context) {
  TF_LITE_ENSURE(context, node->inputs->size == 1 || node->inputs->size == 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  // Reject 0-size tensors as YNNPACK/Slinky may not support them.
  TF_LITE_ENSURE_MSG(context, tflite::NumElements(&input) > 0,
                     "0-size tensors are not supported");

  ynn_type input_ynn_type = GetYnnType(input.type);
  ynn_type output_ynn_type = GetYnnType(output.type);
  TF_LITE_ENSURE(context, input_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, input.type, output.type);

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  TF_LITE_ENSURE(context, QuantizationParamsEqual(input, output));

  if (node->inputs->size == 2) {
    const TfLiteTensor& shape = context->tensors[node->inputs->data[1]];
    TF_LITE_ENSURE(context, shape.allocation_type == kTfLiteMmapRo);
    TF_LITE_ENSURE_EQ(context, shape.type, kTfLiteInt32);
  }

  int32_t target_shape[YNN_MAX_TENSOR_RANK];
  int target_shape_size = 0;
  TF_LITE_ENSURE_STATUS(
      GetReshapeTargetShape(context, node, target_shape, &target_shape_size));

  // This is a weird special case that apparently old models use, indicating
  // scalar input and scalar output. Let's not handle it.
  if (target_shape_size == 1 && target_shape[0] == 0) {
    return kTfLiteError;
  }

  return kTfLiteOk;
}

TfLiteStatus IsPadSupported(const TfLiteRegistration* registration,
                            const TfLiteNode* node, TfLiteContext* context) {
  if (registration->builtin_code == kTfLiteBuiltinPad) {
    TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  } else if (registration->builtin_code == kTfLiteBuiltinPadv2) {
    TF_LITE_ENSURE_EQ(context, node->inputs->size, 3);
  } else {
    TF_LITE_ENSURE_MSG(context, false, "Unsupported pad op");
  }
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& paddings = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type input_ynn_type = GetYnnType(input.type);
  ynn_type output_ynn_type = GetYnnType(output.type);
  TF_LITE_ENSURE(context, input_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, input.type, output.type);

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  TF_LITE_ENSURE(context, QuantizationParamsEqual(input, output));

  TF_LITE_ENSURE(context, paddings.allocation_type == kTfLiteMmapRo);
  TF_LITE_ENSURE(
      context, paddings.type == kTfLiteInt32 || paddings.type == kTfLiteInt64);
  TF_LITE_ENSURE(context, paddings.data.raw != nullptr);

  TF_LITE_ENSURE_EQ(context, paddings.dims->size, 2);
  TF_LITE_ENSURE_EQ(context, paddings.dims->data[0], input.dims->size);
  TF_LITE_ENSURE_EQ(context, paddings.dims->data[1], 2);

  if (registration->builtin_code == kTfLiteBuiltinPadv2) {
    const TfLiteTensor& constant_value =
        context->tensors[node->inputs->data[2]];
    TF_LITE_ENSURE_EQ(context, constant_value.type, input.type);
    TF_LITE_ENSURE(context, constant_value.allocation_type == kTfLiteMmapRo);
    TF_LITE_ENSURE(context, constant_value.data.raw != nullptr);
    TF_LITE_ENSURE_EQ(context, tflite::NumElements(&constant_value), 1);
  }

  // Check that paddings are non-negative, and reject padding on extent 1
  // dimensions. This is a workaround for an issue in YNNPACK: it treats
  // statically extent 1 dimensions as broadcast dimensions, which is not what
  // we want if we're trying to pad them.
  if (paddings.type == kTfLiteInt32) {
    const int32_t* paddings_data =
        reinterpret_cast<const int32_t*>(paddings.data.raw);
    for (int i = 0; i < input.dims->size; ++i) {
      int32_t pre_pad = paddings_data[i * 2];
      int32_t post_pad = paddings_data[i * 2 + 1];
      TF_LITE_ENSURE_MSG(context, pre_pad >= 0 && post_pad >= 0,
                         "Negative paddings are not supported");
      if (input.dims->data[i] == 1) {
        TF_LITE_ENSURE_MSG(context, pre_pad == 0 && post_pad == 0,
                           "Padding on extent 1 dimension is not supported");
      }
    }
  } else if (paddings.type == kTfLiteInt64) {
    const int64_t* paddings_data =
        reinterpret_cast<const int64_t*>(paddings.data.raw);
    for (int i = 0; i < input.dims->size; ++i) {
      int64_t pre_pad = paddings_data[i * 2];
      int64_t post_pad = paddings_data[i * 2 + 1];
      TF_LITE_ENSURE_MSG(context, pre_pad >= 0 && post_pad >= 0,
                         "Negative paddings are not supported");
      if (input.dims->data[i] == 1) {
        TF_LITE_ENSURE_MSG(context, pre_pad == 0 && post_pad == 0,
                           "Padding on extent 1 dimension is not supported");
      }
    }
  }

  return kTfLiteOk;
}

TfLiteStatus DefineTransposeNode(TfLiteContext* context,
                                 ynn_subgraph_t subgraph,
                                 TensorToValueIdMap& tensor_to_value_id,
                                 const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 2);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);
  int input_tensor_index = node.inputs[0];
  int perm_tensor_index = node.inputs[1];
  int output_tensor_index = node.outputs[0];

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  const TfLiteTensor& perm_tensor = context->tensors[perm_tensor_index];
  const int32_t* perm_data =
      reinterpret_cast<const int32_t*>(perm_tensor.data.raw);
  int rank = perm_tensor.dims->data[0];

  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
      subgraph, rank, perm_data, input_val_id, &output_val_id, 0));

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

TfLiteStatus DefineSliceNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                             TensorToValueIdMap& tensor_to_value_id,
                             const NodeInfo& node) {
  const bool is_strided = (node.builtin_code == kTfLiteBuiltinStridedSlice);
  if (is_strided) {
    TF_LITE_ENSURE_EQ(context, node.inputs.size(), 4);
  } else {
    TF_LITE_ENSURE_EQ(context, node.inputs.size(), 3);
  }
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int input_tensor_index = node.inputs[0];
  int begin_tensor_index = node.inputs[1];
  int end_or_size_tensor_index = node.inputs[2];
  int output_tensor_index = node.outputs[0];

  const TfLiteTensor& input_tensor = context->tensors[input_tensor_index];
  const TfLiteTensor& begin_tensor = context->tensors[begin_tensor_index];
  const TfLiteTensor& end_or_size_tensor =
      context->tensors[end_or_size_tensor_index];
  TfLiteTensor& output_tensor = context->tensors[output_tensor_index];

  const int32_t* begin_data =
      reinterpret_cast<const int32_t*>(begin_tensor.data.raw);
  const int32_t* end_or_size_data =
      reinterpret_cast<const int32_t*>(end_or_size_tensor.data.raw);

  int rank = input_tensor.dims->size;
  TF_LITE_ENSURE_MSG(context, rank <= YNN_MAX_TENSOR_RANK,
                     "Rank %d exceeds max %d", rank, YNN_MAX_TENSOR_RANK);

  // Validate size values for Slice before allocating output_shape
  if (!is_strided) {
    for (int i = 0; i < rank; ++i) {
      TF_LITE_ENSURE_MSG(context, end_or_size_data[i] >= -1,
                         "Invalid size value %d for Slice at index %d",
                         end_or_size_data[i], i);
    }
  }

  int32_t axes[YNN_MAX_TENSOR_RANK];
  int64_t begins[YNN_MAX_TENSOR_RANK];
  int64_t ends[YNN_MAX_TENSOR_RANK];
  int64_t strides[YNN_MAX_TENSOR_RANK];

  const int32_t* strides_data = nullptr;
  const TfLiteStridedSliceParams* params = nullptr;
  if (is_strided) {
    int strides_tensor_index = node.inputs[3];
    const TfLiteTensor& strides_tensor = context->tensors[strides_tensor_index];
    strides_data = reinterpret_cast<const int32_t*>(strides_tensor.data.raw);

    TfLiteNode* tflite_node;
    TfLiteRegistration* registration;
    TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
        context, node.node_index, &tflite_node, &registration));
    params =
        static_cast<const TfLiteStridedSliceParams*>(tflite_node->builtin_data);
  }

  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(rank);
  for (int i = 0; i < rank; ++i) {
    axes[i] = i;
    if (is_strided) {
      int32_t start = StridedSliceStartForAxis(params, input_tensor.dims, i,
                                               begin_data[i], strides_data[i]);
      int32_t stop =
          StridedSliceEndForAxis(params, input_tensor.dims, i, start,
                                 end_or_size_data[i], strides_data[i]);
      begins[i] = start;
      ends[i] = stop;
      strides[i] = strides_data[i];

      int32_t stride = strides_data[i];
      int32_t size = 0;
      if (stop > start) {
        size = (stop - start + stride - 1) / stride;
      }
      output_shape->data[i] = size;
    } else {
      begins[i] = begin_data[i];
      int32_t size_val = end_or_size_data[i];
      if (size_val == -1) {
        size_val = input_tensor.dims->data[i] - begin_data[i];
      }
      ends[i] = begin_data[i] + size_val;
      output_shape->data[i] = size_val;
    }
  }

  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, &output_tensor, output_shape));

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_slice(
      subgraph, rank, axes, begins, ends, is_strided ? strides : nullptr,
      input_val_id, &output_val_id, 0));

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

TfLiteStatus DefineConcatenationNode(TfLiteContext* context,
                                     ynn_subgraph_t subgraph,
                                     TensorToValueIdMap& tensor_to_value_id,
                                     const NodeInfo& node) {
  TF_LITE_ENSURE(context, !node.inputs.empty());
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);
  int output_tensor_index = node.outputs[0];
  const TfLiteTensor& output_tensor = context->tensors[output_tensor_index];
  bool is_output_quantized = IsQuantized(output_tensor);
  ynn_type internal_type = GetYnnType(output_tensor.type);

  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));
  const auto* params =
      static_cast<const TfLiteConcatenationParams*>(tflite_node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);

  std::vector<uint32_t> input_val_ids;
  input_val_ids.reserve(node.inputs.size());

  for (int input_tensor_index : node.inputs) {
    if (input_tensor_index == kTfLiteOptionalTensor) continue;
    uint32_t val_id = GetOrCreateValueId(context, subgraph, tensor_to_value_id,
                                         input_tensor_index);
    TF_LITE_ENSURE(context, val_id != YNN_INVALID_VALUE_ID);

    const TfLiteTensor& input_tensor = context->tensors[input_tensor_index];

    if (is_output_quantized &&
        (!IsQuantized(input_tensor) ||
         !QuantizationParamsEqual(input_tensor, output_tensor))) {
      // Mismatched quantization.
      // Dequantize to float.
      uint32_t float_val_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(
          DequantizeIfNeeded(context, subgraph, tensor_to_value_id,
                             input_tensor_index, val_id, &float_val_id));

      // Requantize to output parameters.
      uint32_t scale_id = YNN_INVALID_VALUE_ID;
      uint32_t zp_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_STATUS(DefineQuantizationParams(
          context, subgraph, output_tensor, &scale_id, &zp_id));
      ynn_type ynn_type = GetYnnType(output_tensor.type);
      uint32_t requant_val_id = YNN_INVALID_VALUE_ID;
      TF_LITE_ENSURE_YNN_STATUS(ynn_define_quantize(subgraph, float_val_id,
                                                    ynn_type, zp_id, scale_id,
                                                    &requant_val_id, 0));

      input_val_ids.push_back(requant_val_id);
    } else {
      // Identical quantization or both float.
      input_val_ids.push_back(val_id);
    }
  }

  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  TfLiteFusedActivation activation = params->activation;

  // We can write directly to output_val_id if there is no activation.
  uint32_t concat_output_val_id =
      activation == kTfLiteActNone ? output_val_id : YNN_INVALID_VALUE_ID;

  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_concatenate(subgraph, params->axis, input_val_ids.size(),
                             input_val_ids.data(), &concat_output_val_id, 0));

  if (activation != kTfLiteActNone) {
    TF_LITE_ENSURE_STATUS(ApplyActivation(context, subgraph, activation,
                                          concat_output_val_id, output_val_id,
                                          output_tensor_index, internal_type));
  }

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

TfLiteStatus DefineReshapeNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                               TensorToValueIdMap& tensor_to_value_id,
                               const NodeInfo& node) {
  TF_LITE_ENSURE(context, node.inputs.size() == 1 || node.inputs.size() == 2);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);
  int input_tensor_index = node.inputs[0];
  int output_tensor_index = node.outputs[0];

  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));

  int32_t target_shape[YNN_MAX_TENSOR_RANK];
  int target_shape_size = 0;
  TF_LITE_ENSURE_STATUS(GetReshapeTargetShape(
      context, tflite_node, target_shape, &target_shape_size));

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  size_t ynn_dims[YNN_MAX_TENSOR_RANK];
  for (int i = 0; i < target_shape_size; ++i) {
    if (target_shape[i] == -1) {
      ynn_dims[i] = 0;
    } else {
      ynn_dims[i] = target_shape[i];
    }
  }

  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_reshape(
      subgraph, target_shape_size, ynn_dims, input_val_id, &output_val_id, 0));

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

TfLiteStatus DefineExpandDimsNode(TfLiteContext* context,
                                  ynn_subgraph_t subgraph,
                                  TensorToValueIdMap& tensor_to_value_id,
                                  const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 2);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int input_tensor_index = node.inputs[0];
  int axis_tensor_index = node.inputs[1];
  int output_tensor_index = node.outputs[0];

  const TfLiteTensor& input_tensor = context->tensors[input_tensor_index];
  const TfLiteTensor& axis_tensor = context->tensors[axis_tensor_index];
  TfLiteTensor& output_tensor = context->tensors[output_tensor_index];

  TF_LITE_ENSURE(context, axis_tensor.dims->size <= 1);
  if (axis_tensor.dims->size == 1) {
    TF_LITE_ENSURE_EQ(context, axis_tensor.dims->data[0], 1);
  }

  int32_t axis_val = *reinterpret_cast<const int32_t*>(axis_tensor.data.raw);
  int input_rank = input_tensor.dims->size;
  if (axis_val < 0) {
    axis_val += input_rank + 1;
  }
  TF_LITE_ENSURE_MSG(context, axis_val >= 0 && axis_val <= input_rank,
                     "Invalid axis value %d for input rank %d", axis_val,
                     input_rank);

  int target_rank = input_rank + 1;
  TF_LITE_ENSURE_MSG(context, target_rank <= YNN_MAX_TENSOR_RANK,
                     "Target rank %d exceeds max %d", target_rank,
                     YNN_MAX_TENSOR_RANK);

  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(target_rank);
  int out_idx = 0;
  for (int i = 0; i < input_rank; ++i) {
    if (i == axis_val) {
      output_shape->data[out_idx++] = 1;
    }
    output_shape->data[out_idx++] = input_tensor.dims->data[i];
  }
  if (axis_val == input_rank) {
    output_shape->data[out_idx++] = 1;
  }

  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, &output_tensor, output_shape));

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
      subgraph, 1, &axis_val, input_val_id, &output_val_id, 0));

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

TfLiteStatus DefinePadNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                           TensorToValueIdMap& tensor_to_value_id,
                           const NodeInfo& node) {
  if (node.builtin_code == kTfLiteBuiltinPad) {
    TF_LITE_ENSURE_EQ(context, node.inputs.size(), 2);
  } else if (node.builtin_code == kTfLiteBuiltinPadv2) {
    TF_LITE_ENSURE_EQ(context, node.inputs.size(), 3);
  } else {
    TF_LITE_ENSURE_MSG(context, false, "Unsupported pad op");
  }
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int input_tensor_index = node.inputs[0];
  int paddings_tensor_index = node.inputs[1];
  int output_tensor_index = node.outputs[0];

  const TfLiteTensor& input_tensor = context->tensors[input_tensor_index];
  const TfLiteTensor& paddings_tensor = context->tensors[paddings_tensor_index];

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  ynn_type internal_type = GetYnnType(input_tensor.type);

  uint32_t padding_val_id = YNN_INVALID_VALUE_ID;
  double pad_value = 0.0;

  if (node.builtin_code == kTfLiteBuiltinPadv2) {
    int constant_value_tensor_index = node.inputs[2];
    const TfLiteTensor& constant_value_tensor =
        context->tensors[constant_value_tensor_index];
    TF_LITE_ENSURE_STATUS(GetTfLiteTensorValueAsDouble(
        context, constant_value_tensor, 0, &pad_value));
    if (IsQuantized(constant_value_tensor)) {
      float scale = constant_value_tensor.params.scale;
      int32_t zp = constant_value_tensor.params.zero_point;
      pad_value = scale * (pad_value - zp);
    }
  }

  if (IsQuantized(input_tensor)) {
    float scale = input_tensor.params.scale;
    int32_t zp = input_tensor.params.zero_point;
    double quantized_pad_val = std::round(pad_value / scale) + zp;
    if (input_tensor.type == kTfLiteInt8) {
      quantized_pad_val = std::max(-128.0, std::min(127.0, quantized_pad_val));
    } else if (input_tensor.type == kTfLiteUInt8) {
      quantized_pad_val = std::max(0.0, std::min(255.0, quantized_pad_val));
    }
    pad_value = quantized_pad_val;
  }

  TF_LITE_ENSURE_STATUS(DefineScalarConstant(context, subgraph, internal_type,
                                             pad_value, &padding_val_id));

  int rank = input_tensor.dims->size;
  TF_LITE_ENSURE_MSG(context, rank <= YNN_MAX_TENSOR_RANK,
                     "Rank %d exceeds max %d", rank, YNN_MAX_TENSOR_RANK);
  int32_t axes[YNN_MAX_TENSOR_RANK];
  int64_t pre_paddings[YNN_MAX_TENSOR_RANK];
  int64_t post_paddings[YNN_MAX_TENSOR_RANK];

  if (paddings_tensor.type == kTfLiteInt32) {
    const int32_t* paddings_data =
        reinterpret_cast<const int32_t*>(paddings_tensor.data.raw);
    for (int i = 0; i < rank; ++i) {
      axes[i] = i;
      pre_paddings[i] = paddings_data[i * 2];
      post_paddings[i] = paddings_data[i * 2 + 1];
    }
  } else if (paddings_tensor.type == kTfLiteInt64) {
    const int64_t* paddings_data =
        reinterpret_cast<const int64_t*>(paddings_tensor.data.raw);
    for (int i = 0; i < rank; ++i) {
      axes[i] = i;
      pre_paddings[i] = paddings_data[i * 2];
      post_paddings[i] = paddings_data[i * 2 + 1];
    }
  } else {
    TF_LITE_ENSURE_MSG(context, false, "Unsupported paddings type %d",
                       paddings_tensor.type);
  }

  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_static_pad(subgraph, rank, axes, pre_paddings, post_paddings,
                            input_val_id, padding_val_id, &output_val_id, 0));

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

TfLiteStatus IsSplitSupported(const TfLiteRegistration* registration,
                              const TfLiteNode* node, TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  TF_LITE_ENSURE(context, node->outputs->size >= 1);

  const TfLiteTensor& split_dim = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& value = context->tensors[node->inputs->data[1]];

  TF_LITE_ENSURE_EQ(context, split_dim.type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, split_dim.allocation_type, kTfLiteMmapRo);

  ynn_type value_ynn_type = GetYnnType(value.type);
  TF_LITE_ENSURE(context, value_ynn_type != ynn_type_invalid);

  for (int i = 0; i < node->outputs->size; ++i) {
    const TfLiteTensor& output = context->tensors[node->outputs->data[i]];
    TF_LITE_ENSURE_EQ(context, value.type, output.type);
    TF_LITE_ENSURE(context, IsSupportedQuantization(output));
    TF_LITE_ENSURE(context, QuantizationParamsEqual(value, output));
  }

  TF_LITE_ENSURE(context, IsSupportedQuantization(value));
  TF_LITE_ENSURE_MSG(context, value.dims->size <= YNN_MAX_TENSOR_RANK,
                     "Input rank exceeds max rank");

  return kTfLiteOk;
}

TfLiteStatus DefineSplitNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                             TensorToValueIdMap& tensor_to_value_id,
                             const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 2);
  TF_LITE_ENSURE(context, !node.outputs.empty());

  int split_dim_tensor_index = node.inputs[0];
  int value_tensor_index = node.inputs[1];

  const TfLiteTensor& split_dim_tensor =
      context->tensors[split_dim_tensor_index];
  const TfLiteTensor& value_tensor = context->tensors[value_tensor_index];

  int32_t axis = *reinterpret_cast<const int32_t*>(split_dim_tensor.data.raw);
  if (axis < 0) {
    axis += value_tensor.dims->size;
  }
  TF_LITE_ENSURE_MSG(context, axis >= 0 && axis < value_tensor.dims->size,
                     "Invalid axis %d for rank %d", axis,
                     value_tensor.dims->size);

  uint32_t value_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, value_tensor_index);
  TF_LITE_ENSURE(context, value_val_id != YNN_INVALID_VALUE_ID);

  int num_outputs = node.outputs.size();
  std::vector<uint32_t> output_val_ids(num_outputs);
  for (int i = 0; i < num_outputs; ++i) {
    output_val_ids[i] = GetOrCreateValueId(context, subgraph,
                                           tensor_to_value_id, node.outputs[i]);
    TF_LITE_ENSURE(context, output_val_ids[i] != YNN_INVALID_VALUE_ID);
  }

  TF_LITE_ENSURE_YNN_STATUS(ynn_define_even_split(
      subgraph, axis, value_val_id, num_outputs, output_val_ids.data(), 0));

  for (int i = 0; i < num_outputs; ++i) {
    tensor_to_value_id[node.outputs[i]] = output_val_ids[i];
  }

  return kTfLiteOk;
}

TfLiteStatus IsSpaceToDepthSupported(const TfLiteRegistration* registration,
                                     const TfLiteNode* node,
                                     TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type input_ynn_type = GetYnnType(input.type);
  ynn_type output_ynn_type = GetYnnType(output.type);
  TF_LITE_ENSURE(context, input_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, input.type, output.type);

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  TF_LITE_ENSURE(context, QuantizationParamsEqual(input, output));

  TF_LITE_ENSURE_EQ(context, input.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, output.dims->size, 4);

  return kTfLiteOk;
}

TfLiteStatus IsDepthToSpaceSupported(const TfLiteRegistration* registration,
                                     const TfLiteNode* node,
                                     TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type input_ynn_type = GetYnnType(input.type);
  ynn_type output_ynn_type = GetYnnType(output.type);
  TF_LITE_ENSURE(context, input_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, input.type, output.type);

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  TF_LITE_ENSURE(context, QuantizationParamsEqual(input, output));

  TF_LITE_ENSURE_EQ(context, input.dims->size, 4);
  TF_LITE_ENSURE_EQ(context, output.dims->size, 4);

  return kTfLiteOk;
}

TfLiteStatus DefineSpaceToDepthNode(TfLiteContext* context,
                                    ynn_subgraph_t subgraph,
                                    TensorToValueIdMap& tensor_to_value_id,
                                    const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 1);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int input_tensor_index = node.inputs[0];
  int output_tensor_index = node.outputs[0];

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));
  const auto* params =
      static_cast<const TfLiteSpaceToDepthParams*>(tflite_node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);
  const size_t block_size = params->block_size;

  // Stencil copy to split [n, y_dy, x_dx, c] -> [n, y, x, dy, dx, c]
  uint32_t tiled_id = YNN_INVALID_VALUE_ID;
  const int32_t stencil_axes[] = {1, 2};
  const int32_t new_axes[] = {3, 4};
  const size_t strides[] = {block_size, block_size};
  const size_t extents[] = {block_size, block_size};
  const size_t dilations[] = {1, 1};

  TF_LITE_ENSURE_YNN_STATUS(ynn_define_stencil_copy(
      subgraph, 2, stencil_axes, new_axes, extents, strides, dilations,
      input_val_id, /*padding_id=*/YNN_INVALID_VALUE_ID, &tiled_id,
      /*flags=*/0));

  // Fuse [n, y, x, dy, dx, c] -> [n, y, x, dy_dx_c]
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_fuse_dim(subgraph, 3, 3, tiled_id,
                                                &output_val_id, /*flags=*/0));

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

TfLiteStatus DefineDepthToSpaceNode(TfLiteContext* context,
                                    ynn_subgraph_t subgraph,
                                    TensorToValueIdMap& tensor_to_value_id,
                                    const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 1);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int input_tensor_index = node.inputs[0];
  int output_tensor_index = node.outputs[0];

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));
  const auto* params =
      static_cast<const TfLiteDepthToSpaceParams*>(tflite_node->builtin_data);
  TF_LITE_ENSURE(context, params != nullptr);
  const size_t block_size = params->block_size;

  // Split [n, y, x, dy_dx_c] -> [n, y, x, dy, dx, c]
  uint32_t transposed_id = YNN_INVALID_VALUE_ID;
  const size_t splits[] = {block_size, block_size, 0};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_split_dim(
      subgraph, 3, 3, splits, input_val_id, &transposed_id, /*flags=*/0));

  // Transpose [n, y, x, dy, dx, c] -> [n, y, dy, x, dx, c]
  uint32_t tiled_id = YNN_INVALID_VALUE_ID;
  const int32_t to_space_axes[] = {0, 1, 3, 2, 4, 5};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
      subgraph, 6, to_space_axes, transposed_id, &tiled_id, /*flags=*/0));

  // Fuse [n, y, dy, x, dx, c] -> [n, y_dy, x_dx, c]
  const int32_t fuse_axes[] = {1, 3};
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_fuse_dims(
      subgraph, 2, fuse_axes, tiled_id, &output_val_id, /*flags=*/0));

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

TfLiteStatus IsGatherSupported(const TfLiteRegistration* registration,
                               const TfLiteNode* node, TfLiteContext* context) {
  TF_LITE_ENSURE(context, node->inputs->size == 2 || node->inputs->size == 3);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& input = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& indices = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type input_ynn_type = GetYnnType(input.type);
  ynn_type indices_ynn_type = GetYnnType(indices.type);
  ynn_type output_ynn_type = GetYnnType(output.type);

  TF_LITE_ENSURE(context, input_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, indices_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, input.type, output.type);
  TF_LITE_ENSURE(context, indices.type == kTfLiteInt32 ||
                              indices.type == kTfLiteInt8 ||
                              indices.type == kTfLiteUInt8);

  // Axis must be constant.
  int32_t axis = 0;
  if (node->inputs->size == 3) {
    const TfLiteTensor& axis_tensor = context->tensors[node->inputs->data[2]];
    TF_LITE_ENSURE_EQ(context, axis_tensor.type, kTfLiteInt32);
    TF_LITE_ENSURE_EQ(context, tflite::NumElements(&axis_tensor), 1);
    TF_LITE_ENSURE_MSG(context, axis_tensor.allocation_type == kTfLiteMmapRo,
                       "Gather axis must be constant");
    axis = axis_tensor.data.i32[0];
  } else {
    const auto* params =
        reinterpret_cast<const TfLiteGatherParams*>(node->builtin_data);
    TF_LITE_ENSURE(context, params != nullptr);
    axis = params->axis;
  }

  if (axis < 0) {
    axis += input.dims->size;
  }
  TF_LITE_ENSURE(context, 0 <= axis && axis < input.dims->size);

  TF_LITE_ENSURE(context, IsSupportedQuantization(input));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  return kTfLiteOk;
}

TfLiteStatus DefineGatherNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                              TensorToValueIdMap& tensor_to_value_id,
                              const NodeInfo& node) {
  TF_LITE_ENSURE(context, node.inputs.size() == 2 || node.inputs.size() == 3);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  TfLiteNode* tflite_node;
  TfLiteRegistration* reg;
  TF_LITE_ENSURE_STATUS(context->GetNodeAndRegistration(
      context, node.node_index, &tflite_node, &reg));

  int input_tensor_index = node.inputs[0];
  int indices_tensor_index = node.inputs[1];
  int output_tensor_index = node.outputs[0];

  const TfLiteTensor& input_tensor = context->tensors[input_tensor_index];
  const TfLiteTensor& indices_tensor = context->tensors[indices_tensor_index];

  uint32_t input_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, input_tensor_index);
  uint32_t indices_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, indices_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, input_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, indices_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  // Get axis.
  int32_t axis = 0;
  if (node.inputs.size() == 3) {
    int axis_tensor_index = node.inputs[2];
    const TfLiteTensor& axis_tensor = context->tensors[axis_tensor_index];
    axis = axis_tensor.data.i32[0];
  } else {
    const auto* params =
        reinterpret_cast<const TfLiteGatherParams*>(tflite_node->builtin_data);
    axis = params->axis;
  }

  if (axis < 0) {
    axis += input_tensor.dims->size;
  }

  int batch_dims = 0;
  if (tflite_node->builtin_data != nullptr) {
    const auto* params =
        reinterpret_cast<const TfLiteGatherParams*>(tflite_node->builtin_data);
    batch_dims = params->batch_dims;
    if (batch_dims < 0) {
      batch_dims += indices_tensor.dims->size;
    }
  }

  int b_g = axis - batch_dims;
  int s = input_tensor.dims->size - 1 - axis;
  int k_g = indices_tensor.dims->size - batch_dims;
  bool gather_is_scalar = (k_g == 0);

  // 1. Expand input if k_g > 1
  uint32_t final_input_val_id = input_val_id;
  if (k_g > 1) {
    int32_t new_input_axes[YNN_MAX_TENSOR_RANK];
    size_t new_input_axes_size = 0;
    for (int i = 0; i < k_g - 1; ++i) {
      TF_LITE_ENSURE(context, new_input_axes_size < YNN_MAX_TENSOR_RANK);
      new_input_axes[new_input_axes_size++] = axis + 1 + i;
    }
    uint32_t expanded_input_val_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, new_input_axes_size, new_input_axes, input_val_id,
        &expanded_input_val_id, 0));
    final_input_val_id = expanded_input_val_id;
  }

  // 2. Expand indices
  uint32_t final_indices_val_id = indices_val_id;

  if (b_g > 0 || s > 0 || gather_is_scalar) {
    int32_t new_axes[YNN_MAX_TENSOR_RANK];
    size_t new_axes_size = 0;

    if (!gather_is_scalar) {
      // Expand indices to [Batch..., 1_{b_g}, I..., 1_s]
      for (int i = 0; i < b_g; ++i) {
        TF_LITE_ENSURE(context, new_axes_size < YNN_MAX_TENSOR_RANK);
        new_axes[new_axes_size++] = batch_dims + i;
      }
      for (int i = 0; i < s; ++i) {
        TF_LITE_ENSURE(context, new_axes_size < YNN_MAX_TENSOR_RANK);
        new_axes[new_axes_size++] = batch_dims + b_g + k_g + i;
      }
    } else {
      // Expand indices to [Batch..., 1_{b_g}, 1_J, 1_s]
      new_axes_size = b_g + 1 + s;
      TF_LITE_ENSURE(context, new_axes_size <= YNN_MAX_TENSOR_RANK);
      std::iota(new_axes, new_axes + new_axes_size, batch_dims);
    }

    uint32_t expanded_indices_val_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, new_axes_size, new_axes, indices_val_id,
        &expanded_indices_val_id, 0));
    final_indices_val_id = expanded_indices_val_id;
  }

  // 3. Gather
  uint32_t gather_output_val_id =
      gather_is_scalar ? YNN_INVALID_VALUE_ID : output_val_id;

  size_t gather_output_rank =
      gather_is_scalar ? input_tensor.dims->size
                       : (input_tensor.dims->size + indices_tensor.dims->size -
                          batch_dims - 1);
  TF_LITE_ENSURE_YNN_STATUS(ynn_define_gather(
      subgraph, 1, &axis, gather_output_rank, final_input_val_id,
      final_indices_val_id, &gather_output_val_id, 0));

  // 4. Slice gather axis if gather_is_scalar
  if (gather_is_scalar) {
    int32_t slice_axes[] = {axis};
    int64_t slice_begins[] = {0};
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_slice(
        subgraph, 1, slice_axes, slice_begins, nullptr, nullptr,
        gather_output_val_id, &output_val_id, YNN_NODE_FLAG_SLICE_DIMS));
  }

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

TfLiteStatus IsGatherNdSupported(const TfLiteRegistration* registration,
                                 const TfLiteNode* node,
                                 TfLiteContext* context) {
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 2);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor& params = context->tensors[node->inputs->data[0]];
  const TfLiteTensor& indices = context->tensors[node->inputs->data[1]];
  const TfLiteTensor& output = context->tensors[node->outputs->data[0]];

  ynn_type params_ynn_type = GetYnnType(params.type);
  ynn_type indices_ynn_type = GetYnnType(indices.type);
  ynn_type output_ynn_type = GetYnnType(output.type);

  TF_LITE_ENSURE(context, params_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, indices_ynn_type != ynn_type_invalid);
  TF_LITE_ENSURE(context, output_ynn_type != ynn_type_invalid);

  TF_LITE_ENSURE_EQ(context, params.type, output.type);
  TF_LITE_ENSURE(context, indices.type == kTfLiteInt32 ||
                              indices.type == kTfLiteInt8 ||
                              indices.type == kTfLiteUInt8);

  // Indices must have at least rank 1.
  TF_LITE_ENSURE(context, indices.dims->size >= 1);

  // We need to know the last dimension of indices at compile time.
  int W = indices.dims->data[indices.dims->size - 1];
  TF_LITE_ENSURE(context, W > 0);
  TF_LITE_ENSURE(context, W <= params.dims->size);

  TF_LITE_ENSURE(context, IsSupportedQuantization(params));
  TF_LITE_ENSURE(context, IsSupportedQuantization(output));

  return kTfLiteOk;
}

TfLiteStatus DefineGatherNdNode(TfLiteContext* context, ynn_subgraph_t subgraph,
                                TensorToValueIdMap& tensor_to_value_id,
                                const NodeInfo& node) {
  TF_LITE_ENSURE_EQ(context, node.inputs.size(), 2);
  TF_LITE_ENSURE_EQ(context, node.outputs.size(), 1);

  int params_tensor_index = node.inputs[0];
  int indices_tensor_index = node.inputs[1];
  int output_tensor_index = node.outputs[0];

  const TfLiteTensor& indices_tensor = context->tensors[indices_tensor_index];

  uint32_t params_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, params_tensor_index);
  uint32_t indices_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, indices_tensor_index);
  uint32_t output_val_id = GetOrCreateValueId(
      context, subgraph, tensor_to_value_id, output_tensor_index);

  TF_LITE_ENSURE(context, params_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, indices_val_id != YNN_INVALID_VALUE_ID);
  TF_LITE_ENSURE(context, output_val_id != YNN_INVALID_VALUE_ID);

  int num_axes = indices_tensor.dims->data[indices_tensor.dims->size - 1];

  // The gathered axes are 0, 1, ..., W - 1.
  int32_t axes[YNN_MAX_TENSOR_RANK];
  TF_LITE_ENSURE(context, num_axes <= YNN_MAX_TENSOR_RANK);
  std::iota(axes, axes + num_axes, 0);

  const TfLiteTensor& params_tensor = context->tensors[params_tensor_index];

  int batch_rank = indices_tensor.dims->size - 1;
  int part_rank = params_tensor.dims->size - num_axes;
  int output_rank = batch_rank + part_rank;

  uint32_t final_indices_val_id = indices_val_id;

  int R = indices_tensor.dims->size;
  if (num_axes == 1) {
    // Slice indices to remove the last dimension of size 1.
    int32_t slice_axes[] = {R - 1};
    int64_t slice_begins[] = {0};
    uint32_t sliced_indices_val_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_slice(
        subgraph, 1, slice_axes, slice_begins, nullptr, nullptr, indices_val_id,
        &sliced_indices_val_id, YNN_NODE_FLAG_SLICE_DIMS));
    final_indices_val_id = sliced_indices_val_id;
  } else {
    // Transpose indices to move the last dimension (W) to the front.
    int32_t perm[YNN_MAX_TENSOR_RANK];
    TF_LITE_ENSURE(context, R <= YNN_MAX_TENSOR_RANK);
    std::iota(perm, perm + R, 0);
    std::rotate(perm, perm + R - 1, perm + R);

    uint32_t transposed_indices_val_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_transpose(
        subgraph, R, perm, indices_val_id, &transposed_indices_val_id, 0));
    final_indices_val_id = transposed_indices_val_id;
  }

  // Now expand the tensor by appending dummy dimensions at the end if needed.
  int num_dummy_dims = output_rank - batch_rank;
  if (num_dummy_dims > 0) {
    int start_axis = (num_axes > 1) ? (batch_rank + 1) : batch_rank;
    int32_t expand_axes[YNN_MAX_TENSOR_RANK];
    TF_LITE_ENSURE(context, num_dummy_dims <= YNN_MAX_TENSOR_RANK);
    std::iota(expand_axes, expand_axes + num_dummy_dims, start_axis);

    uint32_t expanded_indices_val_id = YNN_INVALID_VALUE_ID;
    TF_LITE_ENSURE_YNN_STATUS(ynn_define_static_expand_dims(
        subgraph, num_dummy_dims, expand_axes, final_indices_val_id,
        &expanded_indices_val_id, 0));
    final_indices_val_id = expanded_indices_val_id;
  }

  TF_LITE_ENSURE_YNN_STATUS(
      ynn_define_gather(subgraph, num_axes, axes, output_rank, params_val_id,
                        final_indices_val_id, &output_val_id, 0));

  tensor_to_value_id[output_tensor_index] = output_val_id;
  return kTfLiteOk;
}

}  // namespace ynnpack
}  // namespace tflite
