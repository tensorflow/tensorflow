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
#include "tensorflow/lite/kernels/internal/reference/pad.h"

#include <stdint.h>

#include <cstddef>
#include <limits>
#include <memory>
#include <type_traits>

#include "absl/types/span.h"
#include "Eigen/Core"  // from @eigen_archive
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace pad {

// This file has two implementations of Pad.
enum KernelType {
  kReference,
  kGenericOptimized,
};

/// Validates that the node has the Pad/PADV2 arity and index arrays.
TfLiteStatus ValidateNodeArity(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE(context, node != nullptr);
  TF_LITE_ENSURE(context, node->inputs != nullptr);
  TF_LITE_ENSURE(context, node->outputs != nullptr);
  TF_LITE_ENSURE(context, node->inputs->size >= 0);
  TF_LITE_ENSURE(context, node->outputs->size >= 0);
  const int num_inputs = NumInputs(node);
  TF_LITE_ENSURE(context, num_inputs == 2 || num_inputs == 3);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  return kTfLiteOk;
}

/// Validates that a tensor has usable dimension metadata.
TfLiteStatus ValidateTensorShape(TfLiteContext* context,
                                 const TfLiteTensor* tensor) {
  TF_LITE_ENSURE(context, tensor != nullptr);
  TF_LITE_ENSURE(context, tensor->dims != nullptr);
  TF_LITE_ENSURE(context, tensor->dims->size >= 0);
  return kTfLiteOk;
}

struct PadContext {
  PadContext(TfLiteContext* context, TfLiteNode* node) {
    status = ValidateNodeArity(context, node);
    if (status != kTfLiteOk) {
      return;
    }
    status = GetInputSafe(context, node, 0, &input);
    if (status != kTfLiteOk) {
      return;
    }
    status = GetInputSafe(context, node, 1, &paddings);
    if (status != kTfLiteOk) {
      return;
    }
    if (NumInputs(node) == 3 &&
        node->inputs->data[2] != kTfLiteOptionalTensor) {
      status = GetInputSafe(context, node, 2, &constant_values);
      if (status != kTfLiteOk) {
        return;
      }
    }
    status = GetOutputSafe(context, node, 0, &output);
    if (status != kTfLiteOk) {
      return;
    }
    status = ValidateTensorShape(context, input);
    if (status != kTfLiteOk) {
      return;
    }
    status = ValidateTensorShape(context, paddings);
    if (status != kTfLiteOk) {
      return;
    }
    if (constant_values != nullptr) {
      status = ValidateTensorShape(context, constant_values);
      if (status != kTfLiteOk) {
        return;
      }
    }
    resizing_category = ResizingCategory::kGenericResize;
    dims = NumDimensions(input);
    switch (paddings->type) {
      case kTfLiteInt64: {
        status = SetResizingCategory<int64_t>(context);
        break;
      }
      case kTfLiteInt32:
        status = SetResizingCategory<int32_t>(context);
        break;
      case kTfLiteInt8:
        status = SetResizingCategory<int8_t>(context);
        break;
      case kTfLiteInt16:
        status = SetResizingCategory<int16_t>(context);
        break;
      case kTfLiteBool:
        status = SetResizingCategory<bool>(context);
        break;
      default:
        TF_LITE_KERNEL_LOG(context,
                           "Padding type %s is currently not supported by Pad.",
                           TfLiteTypeGetName(paddings->type));
        status = kTfLiteError;
    }
  }

  /// Updates `resizing_category` based on the constant paddings pattern.
  template <typename padding_integer_type>
  TfLiteStatus SetResizingCategory(TfLiteContext* context) {
    resizing_category = ResizingCategory::kGenericResize;
    int paddings_total = 0;
    TF_LITE_ENSURE_MSG(
        context, CheckedNumElements(paddings, paddings_total) == kTfLiteOk,
        "Pad paddings size overflowed.");
    // Paddings will be a n,2 array, and we need to detect 4D arrays with the
    // pattern { {0,0}, {a, b}, {c, d}, {0,0} }.
    if (!IsConstantTensor(paddings) || paddings_total != 8 ||
        NumDimensions(paddings) != 2 || SizeOfDimension(paddings, 1) != 2) {
      return kTfLiteOk;
    }
    const padding_integer_type* paddings_data =
        GetTensorData<padding_integer_type>(paddings);
    TF_LITE_ENSURE(context, paddings_data != nullptr);
    const absl::Span<const padding_integer_type> paddings_values(
        paddings_data, paddings_total);
    if ((paddings_values[0] == 0 && paddings_values[1] == 0) &&
        (paddings_values[6] == 0 && paddings_values[7] == 0)) {
      resizing_category = ResizingCategory::kImageStyle;
    }
    return kTfLiteOk;
  }

  const TfLiteTensor* constant_values = nullptr;
  const TfLiteTensor* input = nullptr;
  const TfLiteTensor* paddings = nullptr;
  TfLiteTensor* output = nullptr;
  int dims = 0;
  ResizingCategory resizing_category = ResizingCategory::kGenericResize;
  TfLiteStatus status = kTfLiteOk;
};

/// Validates that the paddings tensor is a 2-column matrix.
TfLiteStatus ValidatePaddingsMatrixShape(TfLiteContext* context,
                                         const TfLiteTensor* paddings) {
  TF_LITE_ENSURE_OK(context, ValidateTensorShape(context, paddings));
  TF_LITE_ENSURE_EQ(context, NumDimensions(paddings), 2);
  TF_LITE_ENSURE(context, SizeOfDimension(paddings, 0) >= 0);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(paddings, 1), 2);
  return kTfLiteOk;
}

/// Validates that the paddings tensor has shape `[input_rank, 2]`.
TfLiteStatus ValidatePaddingsShape(TfLiteContext* context,
                                   const PadContext& op_context) {
  TF_LITE_ENSURE_OK(context, ValidateTensorShape(context, op_context.input));
  TF_LITE_ENSURE_EQ(context, op_context.dims, NumDimensions(op_context.input));
  TF_LITE_ENSURE_OK(context,
                    ValidatePaddingsMatrixShape(context, op_context.paddings));
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(op_context.paddings, 0),
                    op_context.dims);
  return kTfLiteOk;
}

/// Validates that PADV2's constant value tensor is a scalar when present.
TfLiteStatus ValidateConstantValuesShape(TfLiteContext* context,
                                         const PadContext& op_context) {
  if (op_context.constant_values == nullptr) {
    return kTfLiteOk;
  }
  TF_LITE_ENSURE_OK(context,
                    ValidateTensorShape(context, op_context.constant_values));
  int constant_values_count = 0;
  TF_LITE_ENSURE_MSG(context,
                     CheckedNumElements(op_context.constant_values,
                                        constant_values_count) == kTfLiteOk,
                     "Pad constant_values size overflowed.");
  TF_LITE_ENSURE_EQ(context, constant_values_count, 1);
  return kTfLiteOk;
}

/// Validates that a tensor with elements has a data buffer.
TfLiteStatus ValidateTensorData(TfLiteContext* context,
                                const TfLiteTensor* tensor) {
  TF_LITE_ENSURE_OK(context, ValidateTensorShape(context, tensor));
  size_t count = 0;
  TF_LITE_ENSURE_MSG(context, CheckedNumElements(tensor, count) == kTfLiteOk,
                     "Pad tensor size overflowed.");
  TF_LITE_ENSURE(context, count == 0 || tensor->data.raw != nullptr);
  return kTfLiteOk;
}

/// Converts a padding value to the int type used by PadParams.
template <typename PaddingIntegerType>
TfLiteStatus GetPaddingValueAsInt(TfLiteContext* context,
                                  PaddingIntegerType padding, int* value) {
  TF_LITE_ENSURE(context, value != nullptr);
  if constexpr (std::is_same_v<PaddingIntegerType, int64_t>) {
    const int64_t int32_min =
        static_cast<int64_t>(std::numeric_limits<int32_t>::min());
    const int64_t int32_max =
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    TF_LITE_ENSURE_MSG(context, padding >= int32_min && padding <= int32_max,
                       "INT64 padding overflow. Only support value between "
                       "INT32_MIN and INT32_MAX.");
  }
  CheckedInt<int> checked_padding(padding);
  TF_LITE_ENSURE_MSG(context, checked_padding.Status() == kTfLiteOk,
                     "Pad value overflowed.");
  *value = checked_padding.Value();
  return kTfLiteOk;
}

/// Computes one output dimension from an input dimension and padding pair.
TfLiteStatus GetPaddedOutputDimension(TfLiteContext* context, int input_dim,
                                      int left_padding, int right_padding,
                                      int* output_dim) {
  TF_LITE_ENSURE(context, output_dim != nullptr);
  TF_LITE_ENSURE_MSG(context, input_dim >= 0,
                     "Pad input dimensions must be non-negative.");
  const CheckedInt<int> checked_output_dim =
      CheckedInt<int>(input_dim) + left_padding + right_padding;
  TF_LITE_ENSURE_MSG(context, checked_output_dim.Status() == kTfLiteOk,
                     "Pad output dimension overflowed.");
  TF_LITE_ENSURE_MSG(context, checked_output_dim >= 0,
                     "Pad output dimension has to be greater than equal to 0.");
  *output_dim = checked_output_dim.Value();
  return kTfLiteOk;
}

/// Resizes output array based on the input size and padding params.
TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const PadContext& op_context,
                                const tflite::PadParams& op_params) {
  TF_LITE_ENSURE_OK(context, ValidateTensorShape(context, op_context.input));
  TF_LITE_ENSURE(context, op_context.output != nullptr);
  TfLiteIntArray* input_size = op_context.input->dims;
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> output_size(
      TfLiteIntArrayCopy(input_size), TfLiteIntArrayFree);
  TF_LITE_ENSURE(context, output_size != nullptr);
  const RuntimeShape input_shape = GetTensorShape(op_context.input);
  for (int idx = 0; idx < op_context.dims; ++idx) {
    TF_LITE_ENSURE_OK(context,
                      GetPaddedOutputDimension(context, input_shape.Dims(idx),
                                               op_params.left_padding[idx],
                                               op_params.right_padding[idx],
                                               &output_size->data[idx]));
  }
  size_t output_num_elements = 0;
  TF_LITE_ENSURE_MSG(
      context,
      CheckedNumElements(output_size.get(), output_num_elements) == kTfLiteOk,
      "Pad output size overflowed.");
  if (op_context.resizing_category == ResizingCategory::kImageStyle) {
    int output_num_elements_int = 0;
    TF_LITE_ENSURE_MSG(context,
                       CheckedNumElements(output_size.get(),
                                          output_num_elements_int) == kTfLiteOk,
                       "Pad image-style output size overflowed.");
  }
  return context->ResizeTensor(context, op_context.output,
                               output_size.release());
}

/// Validates static output dimensions against the input shape and paddings.
TfLiteStatus ValidateOutputShape(TfLiteContext* context,
                                 const PadContext& op_context,
                                 const tflite::PadParams& op_params) {
  TF_LITE_ENSURE_OK(context, ValidateTensorShape(context, op_context.input));
  TF_LITE_ENSURE_OK(context, ValidateTensorShape(context, op_context.output));
  const RuntimeShape input_shape = GetTensorShape(op_context.input);
  const RuntimeShape output_shape = GetTensorShape(op_context.output);
  TF_LITE_ENSURE_EQ(context, output_shape.DimensionsCount(), op_context.dims);
  size_t output_num_elements = 0;
  TF_LITE_ENSURE_MSG(context, output_shape.CheckedFlatSize(output_num_elements),
                     "Pad output size overflowed.");
  for (int idx = 0; idx < op_context.dims; ++idx) {
    int expected_output_dim = 0;
    TF_LITE_ENSURE_OK(context,
                      GetPaddedOutputDimension(context, input_shape.Dims(idx),
                                               op_params.left_padding[idx],
                                               op_params.right_padding[idx],
                                               &expected_output_dim));
    TF_LITE_ENSURE_EQ(context, output_shape.Dims(idx), expected_output_dim);
  }
  return kTfLiteOk;
}

/// Extracts validated pad params for a specific paddings tensor type.
template <typename PaddingIntegerType>
TfLiteStatus GetPadParams(TfLiteContext* context, const PadContext& op_context,
                          tflite::PadParams* op_params) {
  TF_LITE_ENSURE(context, op_params != nullptr);
  TF_LITE_ENSURE(context, op_context.paddings != nullptr);
  *op_params = tflite::PadParams();
  TF_LITE_ENSURE(
      context, op_context.dims <= reference_ops::PadKernelMaxDimensionCount());
  if (!(op_context.paddings->type == kTfLiteInt64 &&
        std::is_same_v<PaddingIntegerType, int64_t>) &&
      !(op_context.paddings->type == kTfLiteInt32 &&
        std::is_same_v<PaddingIntegerType, int32_t>) &&
      !(op_context.paddings->type == kTfLiteInt8 &&
        std::is_same_v<PaddingIntegerType, int8_t>) &&
      !(op_context.paddings->type == kTfLiteInt16 &&
        std::is_same_v<PaddingIntegerType, int16_t>) &&
      !(op_context.paddings->type == kTfLiteBool &&
        std::is_same_v<PaddingIntegerType, bool>)) {
    TF_LITE_KERNEL_LOG(context, "Padding type %s doesn't match typename.",
                       TfLiteTypeGetName(op_context.paddings->type));
    return kTfLiteError;
  }
  TF_LITE_ENSURE_OK(context, ValidatePaddingsShape(context, op_context));
  int paddings_total = 0;
  TF_LITE_ENSURE_MSG(
      context,
      CheckedNumElements(op_context.paddings, paddings_total) == kTfLiteOk,
      "Pad paddings size overflowed.");
  const PaddingIntegerType* paddings_data =
      GetTensorData<PaddingIntegerType>(op_context.paddings);
  TF_LITE_ENSURE(context, paddings_data != nullptr || paddings_total == 0);
  const absl::Span<const PaddingIntegerType> paddings_values(paddings_data,
                                                             paddings_total);
  if constexpr (std::is_same_v<PaddingIntegerType, int64_t>) {
    const int64_t int32_min =
        static_cast<int64_t>(std::numeric_limits<int32_t>::min());
    const int64_t int32_max =
        static_cast<int64_t>(std::numeric_limits<int32_t>::max());
    for (const int64_t padding : paddings_values) {
      TF_LITE_ENSURE_MSG(context, padding >= int32_min && padding <= int32_max,
                         "INT64 padding overflow. Only support value between "
                         "INT32_MIN and INT32_MAX.");
    }
  }
  op_params->left_padding_count = op_context.dims;
  op_params->right_padding_count = op_context.dims;
  for (int idx = op_context.dims - 1; idx >= 0; --idx) {
    const int padding_index = idx * 2;
    int left_padding = 0;
    int right_padding = 0;
    TF_LITE_ENSURE_OK(
        context, GetPaddingValueAsInt(context, paddings_values[padding_index],
                                      &left_padding));
    TF_LITE_ENSURE_OK(context, GetPaddingValueAsInt(
                                   context, paddings_values[padding_index + 1],
                                   &right_padding));
    TF_LITE_ENSURE_MSG(context, left_padding >= 0 && right_padding >= 0,
                       "Pad value has to be greater than equal to 0.");
    op_params->left_padding[idx] = left_padding;
    op_params->right_padding[idx] = right_padding;
  }
  return kTfLiteOk;
}

/// Extracts validated pad params from the paddings tensor.
TfLiteStatus GetPadParams(TfLiteContext* context, const PadContext& op_context,
                          tflite::PadParams* op_params) {
  TF_LITE_ENSURE(context, op_params != nullptr);
  TF_LITE_ENSURE(context, op_context.paddings != nullptr);
  switch (op_context.paddings->type) {
    case kTfLiteInt64: {
      return GetPadParams<int64_t>(context, op_context, op_params);
    }
    case kTfLiteInt32: {
      return GetPadParams<int32_t>(context, op_context, op_params);
    }
    case kTfLiteInt8: {
      return GetPadParams<int8_t>(context, op_context, op_params);
    }
    case kTfLiteInt16: {
      return GetPadParams<int16_t>(context, op_context, op_params);
    }
    case kTfLiteBool: {
      return GetPadParams<bool>(context, op_context, op_params);
    }
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Padding type %s is currently not supported by Pad.",
                         TfLiteTypeGetName(op_context.paddings->type));
  }
  return kTfLiteError;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  PadContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, op_context.status);
  TF_LITE_ENSURE(context, op_context.input != nullptr);
  TF_LITE_ENSURE(context, op_context.paddings != nullptr);
  TF_LITE_ENSURE(context, op_context.output != nullptr);
  TF_LITE_ENSURE_OK(context,
                    ValidatePaddingsMatrixShape(context, op_context.paddings));
  const bool paddings_are_constant =
      IsConstantOrPersistentTensor(op_context.paddings);
  if (op_context.dims != 0 && !paddings_are_constant) {
    TF_LITE_ENSURE_OK(context, ValidatePaddingsShape(context, op_context));
  }
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                          op_context.output->type);
  if (op_context.constant_values != nullptr) {
    TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                            op_context.constant_values->type);
  }
  TF_LITE_ENSURE_OK(context, ValidateConstantValuesShape(context, op_context));

  // Ensure we do not exceed maximum dimension count.
  TF_LITE_ENSURE(
      context, op_context.dims <= reference_ops::PadKernelMaxDimensionCount());

  // Exit early if paddings is a non-const tensor or the given input is an
  // unranked input. Set output tensor to dynamic so output size can be
  // determined in Eval.
  if (NumDimensions(op_context.input) == 0 || !paddings_are_constant) {
    SetTensorToDynamic(op_context.output);
    return kTfLiteOk;
  }
  tflite::PadParams op_params;
  TF_LITE_ENSURE_OK(context, GetPadParams(context, op_context, &op_params));
  return ResizeOutputTensor(context, op_context, op_params);
}

template <typename integer_type>
TfLiteStatus EvalInt(TfLiteContext* context, const PadContext& op_context,
                     const tflite::PadParams& op_params) {
  integer_type pad_value;
  if (op_context.constant_values == nullptr) {
    // Quantized Pad requires that 0 is represented in the quantized
    // range.
    TF_LITE_ENSURE(context, op_context.output->params.zero_point >=
                                std::numeric_limits<integer_type>::min());
    TF_LITE_ENSURE(context, op_context.output->params.zero_point <=
                                std::numeric_limits<integer_type>::max());
    pad_value = static_cast<integer_type>(op_context.output->params.zero_point);
  } else {
    // Quantized Pad requires that 'constant_values' is represented in the
    // same quantized range as the input and output tensors.
    TF_LITE_ENSURE_EQ(context, op_context.output->params.zero_point,
                      op_context.constant_values->params.zero_point);
    TF_LITE_ENSURE_EQ(context, op_context.output->params.scale,
                      op_context.constant_values->params.scale);
    pad_value = *GetTensorData<integer_type>(op_context.constant_values);
  }
  const integer_type pad_value_copy = pad_value;
  if (op_context.resizing_category == ResizingCategory::kImageStyle) {
    optimized_ops::PadImageStyle(
        op_params, GetTensorShape(op_context.input),
        GetTensorData<integer_type>(op_context.input), &pad_value_copy,
        GetTensorShape(op_context.output),
        GetTensorData<integer_type>(op_context.output));
  } else {
    optimized_ops::Pad(op_params, GetTensorShape(op_context.input),
                       GetTensorData<integer_type>(op_context.input),
                       &pad_value_copy, GetTensorShape(op_context.output),
                       GetTensorData<integer_type>(op_context.output));
  }

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  PadContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, op_context.status);
  TF_LITE_ENSURE(context, op_context.input != nullptr);
  TF_LITE_ENSURE(context, op_context.paddings != nullptr);
  TF_LITE_ENSURE(context, op_context.output != nullptr);
  TF_LITE_ENSURE_OK(context, ValidateConstantValuesShape(context, op_context));
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                          op_context.output->type);
  if (op_context.constant_values != nullptr) {
    TF_LITE_ENSURE_TYPES_EQ(context, op_context.input->type,
                            op_context.constant_values->type);
  }
  TF_LITE_ENSURE(
      context, op_context.dims <= reference_ops::PadKernelMaxDimensionCount());

  tflite::PadParams op_params;
  TF_LITE_ENSURE_OK(context, GetPadParams(context, op_context, &op_params));

  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeOutputTensor(context, op_context, op_params));
  } else {
    TF_LITE_ENSURE_OK(context,
                      ValidateOutputShape(context, op_context, op_params));
  }

  TF_LITE_ENSURE_OK(context, ValidateTensorData(context, op_context.input));
  TF_LITE_ENSURE_OK(context, ValidateTensorData(context, op_context.output));
  if (op_context.constant_values != nullptr) {
    TF_LITE_ENSURE_OK(context,
                      ValidateTensorData(context, op_context.constant_values));
  }

#define TF_LITE_PAD(type, op_name, scalar, pad_value)                     \
  const scalar pad_value_copy = pad_value;                                \
                                                                          \
  type::op_name(op_params, GetTensorShape(op_context.input),              \
                GetTensorData<scalar>(op_context.input), &pad_value_copy, \
                GetTensorShape(op_context.output),                        \
                GetTensorData<scalar>(op_context.output))
  switch (op_context.input->type) {
    case kTfLiteFloat32: {
      float pad_value = op_context.constant_values == nullptr
                            ? 0.f
                            : *GetTensorData<float>(op_context.constant_values);
      if (kernel_type == kReference) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(reference_ops, PadImageStyle, float, pad_value);
        } else {
          TF_LITE_PAD(reference_ops, Pad, float, pad_value);
        }
      } else if (kernel_type == kGenericOptimized) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(optimized_ops, PadImageStyle, float, pad_value);
        } else {
          TF_LITE_PAD(optimized_ops, Pad, float, pad_value);
        }
      }
    } break;
    case kTfLiteFloat16: {
      Eigen::half pad_value =
          op_context.constant_values == nullptr
              ? static_cast<Eigen::half>(0.f)
              : *GetTensorData<Eigen::half>(op_context.constant_values);
      if (kernel_type == kReference) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(reference_ops, PadImageStyle, Eigen::half, pad_value);
        } else {
          TF_LITE_PAD(reference_ops, Pad, Eigen::half, pad_value);
        }
      } else if (kernel_type == kGenericOptimized) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(optimized_ops, PadImageStyle, Eigen::half, pad_value);
        } else {
          TF_LITE_PAD(optimized_ops, Pad, Eigen::half, pad_value);
        }
      }
    } break;
    case kTfLiteBFloat16: {
      Eigen::bfloat16 pad_value =
          op_context.constant_values == nullptr
              ? static_cast<Eigen::bfloat16>(0.f)
              : *GetTensorData<Eigen::bfloat16>(op_context.constant_values);
      if (kernel_type == kReference) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(reference_ops, PadImageStyle, Eigen::bfloat16, pad_value);
        } else {
          TF_LITE_PAD(reference_ops, Pad, Eigen::bfloat16, pad_value);
        }
      } else if (kernel_type == kGenericOptimized) {
        if (op_context.resizing_category == ResizingCategory::kImageStyle) {
          TF_LITE_PAD(optimized_ops, PadImageStyle, Eigen::bfloat16, pad_value);
        } else {
          TF_LITE_PAD(optimized_ops, Pad, Eigen::bfloat16, pad_value);
        }
      }
    } break;
    case kTfLiteUInt8: {
      TF_LITE_ENSURE_OK(context,
                        EvalInt<uint8_t>(context, op_context, op_params));
    } break;
    case kTfLiteInt8: {
      if (op_context.input->quantization.type != kTfLiteNoQuantization) {
        TF_LITE_ENSURE_OK(context,
                          EvalInt<int8_t>(context, op_context, op_params));
      } else {
        int8_t pad_value =
            op_context.constant_values == nullptr
                ? 0
                : *GetTensorData<int8_t>(op_context.constant_values);
        if (kernel_type == kReference) {
          TF_LITE_PAD(reference_ops, Pad, int8_t, pad_value);
        } else if (kernel_type == kGenericOptimized) {
          TF_LITE_PAD(optimized_ops, Pad, int8_t, pad_value);
        }
      }
    } break;
    case kTfLiteInt16: {
      if (op_context.input->quantization.type != kTfLiteNoQuantization) {
        TF_LITE_ENSURE_OK(context,
                          EvalInt<int16_t>(context, op_context, op_params));
      } else {
        int16_t pad_value =
            op_context.constant_values == nullptr
                ? 0
                : *GetTensorData<int16_t>(op_context.constant_values);
        if (kernel_type == kReference) {
          TF_LITE_PAD(reference_ops, Pad, int16_t, pad_value);
        } else if (kernel_type == kGenericOptimized) {
          TF_LITE_PAD(optimized_ops, Pad, int16_t, pad_value);
        }
      }
    } break;
    case kTfLiteInt32: {
      int32_t pad_value =
          op_context.constant_values == nullptr
              ? 0
              : *GetTensorData<int32_t>(op_context.constant_values);
      if (kernel_type == kReference) {
        TF_LITE_PAD(reference_ops, Pad, int32_t, pad_value);
      } else if (kernel_type == kGenericOptimized) {
        TF_LITE_PAD(optimized_ops, Pad, int32_t, pad_value);
      }
    } break;
    case kTfLiteInt64: {
      int64_t pad_value =
          op_context.constant_values == nullptr
              ? 0L
              : *GetTensorData<int64_t>(op_context.constant_values);
      if (kernel_type == kReference) {
        TF_LITE_PAD(reference_ops, Pad, int64_t, pad_value);
      } else if (kernel_type == kGenericOptimized) {
        TF_LITE_PAD(optimized_ops, Pad, int64_t, pad_value);
      }
    } break;
    case kTfLiteBool: {
      bool pad_value = op_context.constant_values == nullptr
                           ? false
                           : *GetTensorData<bool>(op_context.constant_values);
      if (kernel_type == kReference) {
        TF_LITE_PAD(reference_ops, Pad, bool, pad_value);
      } else if (kernel_type == kGenericOptimized) {
        TF_LITE_PAD(optimized_ops, Pad, bool, pad_value);
      }
    } break;
    default:
      TF_LITE_KERNEL_LOG(context, "Type %s is currently not supported by Pad.",
                         TfLiteTypeGetName(op_context.input->type));
      return kTfLiteError;
  }
#undef TF_LITE_PAD
  return kTfLiteOk;
}

}  // namespace pad

TfLiteRegistration* Register_PAD_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, pad::Prepare,
                                 pad::Eval<pad::kReference>};
  return &r;
}

TfLiteRegistration* Register_PAD_GENERIC_OPT() {
  static TfLiteRegistration r = {nullptr, nullptr, pad::Prepare,
                                 pad::Eval<pad::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_PAD() { return Register_PAD_GENERIC_OPT(); }

// Also register Pad as PadV2.
TfLiteRegistration* Register_PADV2_REF() {
  static TfLiteRegistration r = {nullptr, nullptr, pad::Prepare,
                                 pad::Eval<pad::kReference>};
  return &r;
}

TfLiteRegistration* Register_PADV2_GENERIC_OPT() {
  static TfLiteRegistration r = {nullptr, nullptr, pad::Prepare,
                                 pad::Eval<pad::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_PADV2() { return Register_PADV2_GENERIC_OPT(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
