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
#include "tensorflow/lite/kernels/internal/reference/reduce.h"

#include <stddef.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <tuple>

#include "absl/algorithm/container.h"
#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/inlined_vector.h"
#include "absl/types/span.h"
#include "ruy/profiler/instrumentation.h"  // from @ruy
#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/integer_ops/mean.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/optimized/reduce.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reduce {

constexpr size_t kMaxConstantOutputTensorSize = 8;
/// Stores reducer axes inline for common rank <= 4 tensors.
using ResolvedAxisVector = absl::InlinedVector<int, 4>;
using TfLiteIntArrayUniquePtr =
    std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)>;

// This file has reference implementation of reduce_* operators.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct OpData {
  int32_t multiplier = 0;
  int shift = 0;
  // The index of the temporary tensor where the quantized inputs are cached.
  int scratch_tensor_index = -1;
  // Indicates that 'Eval' is a noop as the output as written during 'Prepare'.
  bool noop = false;
};

/// Validates that the node has the reducer op arity and parameter storage.
TfLiteStatus ValidateNode(TfLiteContext* absl_nonnull context,
                          TfLiteNode* node) {
  TF_LITE_ENSURE(context, node != nullptr);
  TF_LITE_ENSURE(context, node->inputs != nullptr);
  TF_LITE_ENSURE(context, node->outputs != nullptr);
  TF_LITE_ENSURE(context, node->inputs->size >= 0);
  TF_LITE_ENSURE(context, node->outputs->size >= 0);
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  TF_LITE_ENSURE(context, node->builtin_data != nullptr);
  TF_LITE_ENSURE(context, node->user_data != nullptr);
  return kTfLiteOk;
}

/// Validates that a tensor has usable dimension metadata.
TfLiteStatus ValidateTensorShape(TfLiteContext* absl_nonnull context,
                                 const TfLiteTensor* absl_nonnull tensor) {
  TF_LITE_ENSURE(context, tensor->dims != nullptr);
  TF_LITE_ENSURE(context, tensor->dims->size >= 0);
  return kTfLiteOk;
}

/// Validates that a tensor with elements has a data buffer.
TfLiteStatus ValidateTensorData(TfLiteContext* absl_nonnull context,
                                const TfLiteTensor* absl_nonnull tensor) {
  size_t count = 0;
  TF_LITE_ENSURE_MSG(context, CheckedNumElements(tensor, count) == kTfLiteOk,
                     "Reduce tensor shape is invalid or size overflowed.");
  TF_LITE_ENSURE(context, count == 0 || tensor->data.raw != nullptr);
  return kTfLiteOk;
}

/// Validates tensor data for reducer APIs that use int-sized offsets.
TfLiteStatus ValidateTensorDataForIntIndexedKernel(
    TfLiteContext* absl_nonnull context,
    const TfLiteTensor* absl_nonnull tensor) {
  int count = 0;
  TF_LITE_ENSURE_MSG(context, CheckedNumElements(tensor, count) == kTfLiteOk,
                     "Reduce tensor shape is invalid or size overflowed.");
  TF_LITE_ENSURE(context, count == 0 || tensor->data.raw != nullptr);
  return kTfLiteOk;
}

struct OpContext {
  OpContext(TfLiteContext* absl_nonnull context, TfLiteNode* node) {
    status = ValidateNode(context, node);
    if (status != kTfLiteOk) {
      return;
    }
    params = reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);
    status = GetInputSafe(context, node, 0, &input);
    if (status != kTfLiteOk) {
      return;
    }
    status = GetInputSafe(context, node, 1, &axis);
    if (status != kTfLiteOk) {
      return;
    }
    status = GetOutputSafe(context, node, 0, &output);
    if (status != kTfLiteOk) {
      return;
    }
    status = ValidateTensorShape(context, input);
    if (status != kTfLiteOk) {
      return;
    }
    status = ValidateTensorShape(context, axis);
  }
  TfLiteReducerParams* params = nullptr;
  const TfLiteTensor* input = nullptr;
  const TfLiteTensor* axis = nullptr;
  TfLiteTensor* output = nullptr;
  TfLiteStatus status = kTfLiteOk;
};

/// Resolves and validates the axis tensor for an input rank.
TfLiteStatus ResolveAxisForShape(
    TfLiteContext* absl_nonnull context,
    const TfLiteTensor* absl_nonnull axis_tensor, int input_num_dims,
    ResolvedAxisVector* absl_nonnull resolved_axis) {
  TF_LITE_ENSURE_MSG(context, input_num_dims >= 0,
                     "Reduce input rank must be non-negative.");
  size_t num_axis = 0;
  TF_LITE_ENSURE_MSG(context,
                     CheckedNumElements(axis_tensor, num_axis) == kTfLiteOk,
                     "Reduce tensor shape is invalid or size overflowed.");
  const int* axis_data = GetTensorData<int>(axis_tensor);
  TF_LITE_ENSURE(context, axis_data != nullptr || num_axis == 0);
  const absl::Span<const int> axis_values =
      absl::MakeConstSpan(axis_data, num_axis);
  resolved_axis->clear();
  if (input_num_dims == 0) {
    return kTfLiteOk;
  }
  for (int current : axis_values) {
    if (current < 0) {
      current += input_num_dims;
    }
    TF_LITE_ENSURE_MSG(context, current >= 0 && current < input_num_dims,
                       "Invalid axis index.");
    if (!absl::c_contains(*resolved_axis, current)) {
      resolved_axis->push_back(current);
    }
  }
  return kTfLiteOk;
}

/// Computes a checked flat size as an int for legacy reducer kernel APIs.
TfLiteStatus GetFlatSizeAsInt(TfLiteContext* absl_nonnull context,
                              const RuntimeShape& shape,
                              int* absl_nonnull flat_size) {
  size_t checked_flat_size = 0;
  TF_LITE_ENSURE_MSG(context, shape.CheckedFlatSize(checked_flat_size),
                     "Reduce tensor size overflowed.");
  CheckedInt<int> narrowed_flat_size(checked_flat_size);
  TF_LITE_ENSURE_MSG(context, narrowed_flat_size.Status() == kTfLiteOk,
                     "Reduce tensor size overflowed.");
  *flat_size = narrowed_flat_size.Value();
  return kTfLiteOk;
}

/// Builds parameters for the specialized 4D mean kernels.
TfLiteStatus BuildMeanParamsFor4D(TfLiteContext* absl_nonnull context,
                                  const OpContext& op_context,
                                  tflite::MeanParams* absl_nonnull op_params) {
  TF_LITE_ENSURE_EQ(context, NumDimensions(op_context.input), 4);
  ResolvedAxisVector resolved_axis;
  TF_LITE_ENSURE_OK(
      context,
      ResolveAxisForShape(context, op_context.axis,
                          NumDimensions(op_context.input), &resolved_axis));
  TF_LITE_ENSURE(context, resolved_axis.size() <= 4);
  op_params->axis_count = static_cast<int8_t>(resolved_axis.size());
  int i = 0;
  for (; i < op_params->axis_count; ++i) {
    op_params->axis[i] = static_cast<int16_t>(resolved_axis[i]);
  }
  for (; i < 4; ++i) {
    op_params->axis[i] = 1;
  }
  return kTfLiteOk;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // Creates three temp tensors to store index and axis for internal
  // implementation only.
  return new OpData();
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

template <KernelType kernel_type>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node);

// Resizes the temp tensor that stores resolved axis.
TfLiteStatus ResizeTempAxis(TfLiteContext* absl_nonnull context,
                            OpContext* absl_nonnull op_context,
                            TfLiteTensor* absl_nonnull resolved_axis) {
  TfLiteIntArrayUniquePtr axis_size(TfLiteIntArrayCreate(1),
                                    TfLiteIntArrayFree);
  TF_LITE_ENSURE(context, axis_size != nullptr);
  TF_LITE_ENSURE_MSG(
      context,
      CheckedNumElements(op_context->axis, axis_size->data[0]) == kTfLiteOk,
      "Reduce tensor shape is invalid or size overflowed.");
  return context->ResizeTensor(context, resolved_axis, axis_size.release());
}

// Resizes the temp tensor that stores temp sum of reduced elements.
TfLiteStatus ResizeTempAccum(TfLiteContext* absl_nonnull context,
                             OpContext* absl_nonnull op_context,
                             TfLiteTensor* absl_nonnull temp_accum) {
  TfLiteIntArrayUniquePtr size(TfLiteIntArrayCreate(1), TfLiteIntArrayFree);
  TF_LITE_ENSURE(context, size != nullptr);
  TF_LITE_ENSURE_MSG(
      context,
      CheckedNumElements(op_context->output, size->data[0]) == kTfLiteOk,
      "Reduce tensor shape is invalid or size overflowed.");
  return context->ResizeTensor(context, temp_accum, size.release());
}

// Returns the output shape.
TfLiteStatus GetOutputShape(TfLiteContext* absl_nonnull context,
                            OpContext* absl_nonnull op_context,
                            TfLiteIntArrayUniquePtr& output_shape) {
  TF_LITE_ENSURE_OK(context, ValidateTensorShape(context, op_context->input));
  const TfLiteIntArray* input_dims = op_context->input->dims;
  int input_num_dims = NumDimensions(op_context->input);
  ResolvedAxisVector resolved_axis;
  TF_LITE_ENSURE_OK(
      context, ResolveAxisForShape(context, op_context->axis, input_num_dims,
                                   &resolved_axis));
  if (input_num_dims == 0) {
    output_shape.reset(TfLiteIntArrayCreate(0));
    TF_LITE_ENSURE(context, output_shape != nullptr);
    return kTfLiteOk;
  }
  if (op_context->params->keep_dims) {
    output_shape.reset(TfLiteIntArrayCreate(input_num_dims));
    TF_LITE_ENSURE(context, output_shape != nullptr);
    for (int idx = 0; idx < input_num_dims; ++idx) {
      if (absl::c_contains(resolved_axis, idx)) {
        output_shape->data[idx] = 1;
      } else {
        output_shape->data[idx] = input_dims->data[idx];
      }
    }
    return kTfLiteOk;
  } else {
    // Determines output dimensions.
    output_shape.reset(TfLiteIntArrayCreate(
        input_num_dims - static_cast<int>(resolved_axis.size())));
    TF_LITE_ENSURE(context, output_shape != nullptr);
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx) {
      if (absl::c_contains(resolved_axis, idx)) {
        ++num_skip_axis;
      } else {
        output_shape->data[idx - num_skip_axis] = input_dims->data[idx];
      }
    }
    return kTfLiteOk;
  }
}

// Resizes output array based on the input size and resolved axis.
TfLiteStatus ResizeOutputTensor(TfLiteContext* absl_nonnull context,
                                OpContext* absl_nonnull op_context) {
  TfLiteIntArrayUniquePtr output_dims(nullptr, TfLiteIntArrayFree);
  TF_LITE_ENSURE_OK(context, GetOutputShape(context, op_context, output_dims));
  TF_LITE_ENSURE(context, output_dims != nullptr);
  return context->ResizeTensor(context, op_context->output,
                               output_dims.release());
}

// Resizes the temp tensor that stores normalized dims.
TfLiteStatus ResizeTempDims(TfLiteContext* absl_nonnull context,
                            OpContext* absl_nonnull op_context,
                            TfLiteTensor* absl_nonnull normalized_dims) {
  TF_LITE_ENSURE_OK(context, ValidateTensorShape(context, op_context->input));
  TfLiteIntArrayUniquePtr dims_size(TfLiteIntArrayCreate(1),
                                    TfLiteIntArrayFree);
  TF_LITE_ENSURE(context, dims_size != nullptr);
  dims_size->data[0] = (op_context->input->dims->size);
  return context->ResizeTensor(context, normalized_dims, dims_size.release());
}

// Initializes temp tensors to store index and resolved axis.
TfLiteStatus InitializeTemporaries(TfLiteContext* absl_nonnull context,
                                   TfLiteNode* absl_nonnull node,
                                   OpContext* absl_nonnull op_context) {
  // Creates a temp index to iterate through input data.
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, op_data != nullptr);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = nullptr;
  TfLiteIntArrayUniquePtr temporaries(TfLiteIntArrayCreate(4),
                                      TfLiteIntArrayFree);
  TF_LITE_ENSURE(context, temporaries != nullptr);
  node->temporaries = temporaries.release();
  node->temporaries->data[0] = op_data->scratch_tensor_index;
  TfLiteTensor* scratch_tensor;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/0, &scratch_tensor));
  scratch_tensor->type = kTfLiteInt32;
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TfLiteIntArrayUniquePtr index_size(TfLiteIntArrayCreate(1),
                                     TfLiteIntArrayFree);
  TF_LITE_ENSURE(context, index_size != nullptr);
  index_size->data[0] = NumDimensions(op_context->input);
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_tensor,
                                                   index_size.release()));

  // Creates a temp tensor to store resolved axis given input data.
  node->temporaries->data[1] = op_data->scratch_tensor_index + 1;
  TfLiteTensor* resolved_axis;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
  resolved_axis->type = kTfLiteInt32;
  // Creates a temporary accumulation tensor to store temp sums when calculating
  // mean or temp prod when calculating reduce prod.
  node->temporaries->data[2] = op_data->scratch_tensor_index + 2;
  TfLiteTensor* temp_accum;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/2, &temp_accum));
  switch (op_context->input->type) {
    case kTfLiteFloat32:
      temp_accum->type = kTfLiteFloat32;
      break;
    case kTfLiteInt32:
      temp_accum->type = kTfLiteInt64;
      break;
    case kTfLiteInt64:
      temp_accum->type = kTfLiteInt64;
      break;
    case kTfLiteUInt8:
    case kTfLiteInt8:
    case kTfLiteInt16:
      temp_accum->type = kTfLiteInt32;
      break;
    case kTfLiteBool:
      temp_accum->type = kTfLiteBool;
      break;
    default:
      return kTfLiteError;
  }
  // Creates a temp tensor to store normalized shape given input data.
  node->temporaries->data[3] = op_data->scratch_tensor_index + 3;
  TfLiteTensor* normalized_dims;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/3, &normalized_dims));
  normalized_dims->type = kTfLiteInt32;
  return kTfLiteOk;
}

TfLiteStatus PrepareSimple(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, op_context.status);
  TF_LITE_ENSURE(context, op_context.axis != nullptr);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, op_data != nullptr);
  if (op_data->scratch_tensor_index == -1) {
    TF_LITE_ENSURE_STATUS(
        context->AddTensors(context, 4, &op_data->scratch_tensor_index));
  }
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.axis->type, kTfLiteInt32);
  TF_LITE_ENSURE_OK(context, InitializeTemporaries(context, node, &op_context));
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  data->noop = IsConstantOrPersistentTensor(op_context.input) &&
               IsConstantOrPersistentTensor(op_context.axis);
  if (data->noop) {
    // Constant reductions should only be used for small outputs, typically
    // coming from Shape tensors. Constant reductions on larger tensors could
    // increase memory usage due to the output not being stored in the Arena.
    TfLiteIntArrayUniquePtr output_shape(nullptr, TfLiteIntArrayFree);
    TF_LITE_ENSURE_OK(context,
                      GetOutputShape(context, &op_context, output_shape));
    size_t output_num_elements = 0;
    TF_LITE_ENSURE_MSG(context,
                       CheckedNumElements(output_shape.get(),
                                          output_num_elements) == kTfLiteOk,
                       "Reduce output shape is invalid or size overflowed.");
    data->noop &= output_num_elements <= kMaxConstantOutputTensorSize;
  }

  if (op_context.input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, op_context.input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, op_context.output->params.zero_point, 0);
  }

  TfLiteTensor* resolved_axis;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
  TfLiteTensor* normalized_dims;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/3, &normalized_dims));

  if (!IsConstantOrPersistentTensor(op_context.input)) {
    SetTensorToDynamic(normalized_dims);
  } else {
    TfLiteTensorDataFree(normalized_dims);
    normalized_dims->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context,
                      ResizeTempDims(context, &op_context, normalized_dims));
  }
  // Leaves work to Eval if axis is not constant; else resizes output.
  if (!IsConstantOrPersistentTensor(op_context.axis)) {
    SetTensorToDynamic(op_context.output);
    SetTensorToDynamic(resolved_axis);
    return kTfLiteOk;
  }
  TfLiteTensorDataFree(resolved_axis);
  resolved_axis->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context,
                    ResizeTempAxis(context, &op_context, resolved_axis));
  TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
  return kTfLiteOk;
}

TfLiteStatus PrepareAllOrAny(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, ValidateNode(context, node));
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TF_LITE_ENSURE_OK(context, ValidateTensorShape(context, input));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteBool);
  return PrepareSimple(context, node);
}

TfLiteStatus PrepareMeanOrSum(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node));
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, data != nullptr);

  // reduce_mean requires a buffer to store intermediate sum result.
  OpContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, op_context.status);
  if (op_context.input->type == kTfLiteInt8 ||
      op_context.input->type == kTfLiteUInt8 ||
      op_context.input->type == kTfLiteInt16) {
    const double real_multiplier =
        static_cast<double>(op_context.input->params.scale) /
        static_cast<double>(op_context.output->params.scale);
    int exponent;
    QuantizeMultiplier(real_multiplier, &data->multiplier, &exponent);
    data->shift = exponent;
  }

  if (op_context.input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, op_context.input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, op_context.output->params.zero_point, 0);
  }

  TfLiteTensor* temp_sum;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/2, &temp_sum));
  if (!IsConstantOrPersistentTensor(op_context.axis)) {
    SetTensorToDynamic(temp_sum);
    return kTfLiteOk;
  }
  temp_sum->allocation_type = kTfLiteArenaRw;
  return ResizeTempAccum(context, &op_context, temp_sum);
}

double GetQuantProdScaling(double input_scale, double output_scale,
                           int reduced_axis_size) {
  // The scaling after taking the product of all the quantized values should
  // be (input_scale**reduced_axis_size)/output_scale but to avoid overflowing
  // the accumulator we instead scale each multiplication by
  // input_scale/nth_root(output_scale, reduced_axis_size).
  return input_scale / std::pow(output_scale, 1.0 / reduced_axis_size);
}

TfLiteStatus PrepareProd(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node));

  OpContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, op_context.status);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, data != nullptr);

  TfLiteTensor* temp_prod;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/2, &temp_prod));

  if (op_context.input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, op_context.input->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, op_context.output->params.zero_point, 0);
  }

  if (!IsConstantOrPersistentTensor(op_context.axis)) {
    SetTensorToDynamic(temp_prod);
    return kTfLiteOk;
  }

  int input_size = 0;
  TF_LITE_ENSURE_OK(
      context,
      GetFlatSizeAsInt(context, GetTensorShape(op_context.input), &input_size));
  int output_size = 0;
  TF_LITE_ENSURE_OK(context,
                    GetFlatSizeAsInt(context, GetTensorShape(op_context.output),
                                     &output_size));
  // We support both quantized and non-quantized int8/int16 inputs
  if (op_context.input->quantization.type != kTfLiteNoQuantization) {
    if (op_context.input->quantization.type != kTfLiteAffineQuantization) {
      TF_LITE_KERNEL_LOG(context, "Unsupported quantization type: %d",
                         op_context.input->quantization.type);
      return kTfLiteError;
    }
    if (op_context.input->type != kTfLiteInt8 &&
        op_context.input->type != kTfLiteInt16) {
      TF_LITE_KERNEL_LOG(context, "Unsupported quantized data type: %d",
                         op_context.input->type);
      return kTfLiteError;
    }
    if (input_size != 0 && output_size != 0) {
      const int reduced_axis_size = input_size / output_size;
      const double scaling = GetQuantProdScaling(
          static_cast<double>(op_context.input->params.scale),
          static_cast<double>(op_context.output->params.scale),
          reduced_axis_size);
      QuantizeMultiplier(scaling, &data->multiplier, &data->shift);
    }
  }

  if (data->noop) {
    SetTensorToDynamic(temp_prod);
    SetTensorToPersistentRo(op_context.output);
    TF_LITE_ENSURE_OK(context,
                      ResizeTempAccum(context, &op_context, temp_prod));
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));

    TfLiteTensor* resolved_axis;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
    SetTensorToDynamic(resolved_axis);
    TF_LITE_ENSURE_OK(context,
                      ResizeTempAxis(context, &op_context, resolved_axis));
    TfLiteTensor* normalized_dims;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/3,
                                                &normalized_dims));
    SetTensorToDynamic(normalized_dims);
    TF_LITE_ENSURE_OK(context,
                      ResizeTempDims(context, &op_context, normalized_dims));
    return EvalImpl<kGenericOptimized>(context, node);
  } else {
    temp_prod->allocation_type = kTfLiteArenaRw;
    return ResizeTempAccum(context, &op_context, temp_prod);
  }
}

template <typename T, typename U, KernelType kernel_type>
TfLiteStatus Mean(TfLiteContext* absl_nonnull context,
                  const OpContext* absl_nonnull op_context, int* temp_index,
                  int* resolved_axis, U* temp_sum) {
  int num_axis = 0;
  TF_LITE_ENSURE_MSG(
      context, CheckedNumElements(op_context->axis, num_axis) == kTfLiteOk,
      "Reduce tensor shape is invalid or size overflowed.");
  if (kernel_type == kGenericOptimized) {
    TF_LITE_ENSURE_OK(context, ValidateTensorDataForIntIndexedKernel(
                                   context, op_context->input));
    TF_LITE_ENSURE_OK(context, ValidateTensorDataForIntIndexedKernel(
                                   context, op_context->output));
  }
  auto args = std::tuple(
      GetTensorData<T>(op_context->input), &op_context->input->dims->data[0],
      op_context->input->dims->size, GetTensorData<T>(op_context->output),
      &op_context->output->dims->data[0], op_context->output->dims->size,
      GetTensorData<int>(op_context->axis), num_axis,
      op_context->params->keep_dims, temp_index, resolved_axis, temp_sum);
  if (kernel_type == kReference) {
    TF_LITE_ENSURE(context, std::apply(reference_ops::Mean<T, U>, args));
  } else {
    TF_LITE_ENSURE(context, std::apply(optimized_ops::Mean<T, U>, args));
  }
  return kTfLiteOk;
}

template <typename T, KernelType kernel_type>
TfLiteStatus QuantizedMeanOrSum(TfLiteContext* absl_nonnull context,
                                const OpContext& op_context,
                                const OpData* absl_nonnull op_data,
                                TfLiteTensor* absl_nonnull temp_index,
                                TfLiteTensor* absl_nonnull resolved_axis,
                                TfLiteTensor* absl_nonnull temp_sum,
                                bool compute_sum) {
  int num_axis = 0;
  TF_LITE_ENSURE_MSG(context,
                     CheckedNumElements(op_context.axis, num_axis) == kTfLiteOk,
                     "Reduce tensor shape is invalid or size overflowed.");
  if (kernel_type == kGenericOptimized) {
    TF_LITE_ENSURE_OK(context, ValidateTensorDataForIntIndexedKernel(
                                   context, op_context.input));
    TF_LITE_ENSURE_OK(context, ValidateTensorDataForIntIndexedKernel(
                                   context, op_context.output));
    TF_LITE_ENSURE(
        context,
        optimized_ops::QuantizedMeanOrSum(
            GetTensorData<T>(op_context.input),
            op_context.input->params.zero_point, op_context.input->params.scale,
            op_context.input->dims->data, op_context.input->dims->size,
            GetTensorData<T>(op_context.output),
            op_context.output->params.zero_point,
            op_context.output->params.scale, op_context.output->dims->data,
            op_context.output->dims->size, GetTensorData<int>(op_context.axis),
            num_axis, op_context.params->keep_dims,
            GetTensorData<int>(temp_index), GetTensorData<int>(resolved_axis),
            GetTensorData<int32_t>(temp_sum), compute_sum));
  } else {
    TF_LITE_ENSURE(
        context,
        reference_ops::QuantizedMeanOrSum(
            GetTensorData<uint8_t>(op_context.input),
            op_context.input->params.zero_point, op_context.input->dims->data,
            op_context.input->dims->size,
            GetTensorData<uint8_t>(op_context.output), op_data->multiplier,
            op_data->shift, op_context.output->params.zero_point,
            op_context.output->dims->data, op_context.output->dims->size,
            GetTensorData<int>(op_context.axis), num_axis,
            op_context.params->keep_dims, GetTensorData<int>(temp_index),
            GetTensorData<int>(resolved_axis), GetTensorData<int32_t>(temp_sum),
            compute_sum));
  }
  return kTfLiteOk;
}

template <typename integer_type>
TfLiteStatus EvalQuantizedMean(TfLiteContext* absl_nonnull context,
                               const OpContext& op_context, int num_axis,
                               OpData* absl_nonnull data,
                               TfLiteTensor* absl_nonnull temp_index,
                               TfLiteTensor* absl_nonnull resolved_axis,
                               TfLiteTensor* absl_nonnull temp_sum) {
  const TfLiteTensor* input = op_context.input;
  TfLiteTensor* output = op_context.output;

  TF_LITE_ENSURE(
      context,
      reference_ops::QuantizedMeanOrSum(
          GetTensorData<integer_type>(input), input->params.zero_point,
          input->dims->data, input->dims->size,
          GetTensorData<integer_type>(output), data->multiplier, data->shift,
          output->params.zero_point, output->dims->data, output->dims->size,
          GetTensorData<int>(op_context.axis), num_axis,
          op_context.params->keep_dims, GetTensorData<int>(temp_index),
          GetTensorData<int>(resolved_axis), GetTensorData<int32_t>(temp_sum),
          /*compute_sum=*/false));

  return kTfLiteOk;
}

template <typename T>
TfLiteStatus InitializeMeanOutputTyped(TfLiteContext* absl_nonnull context,
                                       TfLiteTensor* absl_nonnull output) {
  TF_LITE_ENSURE_OK(context, ValidateTensorShape(context, output));
  RuntimeShape output_shape = GetTensorShape(output);
  size_t flat_size = 0;
  TF_LITE_ENSURE_MSG(context, output_shape.CheckedFlatSize(flat_size),
                     "Reduce output size overflowed.");
  T* output_data = GetTensorData<T>(output);
  TF_LITE_ENSURE(context, output_data != nullptr || flat_size == 0);
  T nan_value = std::numeric_limits<T>::quiet_NaN();
  for (size_t idx = 0; idx < flat_size; ++idx) {
    *output_data++ = nan_value;
  }
  return kTfLiteOk;
}

/// Initializes mean output for empty inputs.
TfLiteStatus InitializeMeanOutput(TfLiteContext* absl_nonnull context,
                                  TfLiteTensor* absl_nonnull output) {
  switch (output->type) {
    case kTfLiteFloat32:
      return InitializeMeanOutputTyped<float>(context, output);
    case kTfLiteInt32:
      return InitializeMeanOutputTyped<int>(context, output);
    case kTfLiteInt64:
      return InitializeMeanOutputTyped<int64_t>(context, output);
    case kTfLiteUInt8:
      return InitializeMeanOutputTyped<uint8_t>(context, output);
    case kTfLiteInt8:
      return InitializeMeanOutputTyped<int8_t>(context, output);
    case kTfLiteInt16:
      return InitializeMeanOutputTyped<int16_t>(context, output);
    default:
      return kTfLiteError;
  }
}

template <KernelType kernel_type>
TfLiteStatus EvalMean(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, op_context.status);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, data != nullptr);

  int num_axis = 0;
  TF_LITE_ENSURE_MSG(context,
                     CheckedNumElements(op_context.axis, num_axis) == kTfLiteOk,
                     "Reduce tensor shape is invalid or size overflowed.");
  TfLiteTensor* temp_index;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/0, &temp_index));
  TfLiteTensor* resolved_axis;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
  TfLiteTensor* temp_sum;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/2, &temp_sum));
  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context.output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeTempAxis(context, &op_context, resolved_axis));
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
    TF_LITE_ENSURE_OK(context, ResizeTempAccum(context, &op_context, temp_sum));
  }
  TfLiteTensor* normalized_dims;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/3, &normalized_dims));
  if (IsDynamicTensor(normalized_dims)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeTempDims(context, &op_context, normalized_dims));
  }

  // Return early when input is empty.
  const TfLiteTensor* input = op_context.input;
  TF_LITE_ENSURE_OK(context, ValidateTensorData(context, input));
  TF_LITE_ENSURE_OK(context, ValidateTensorData(context, op_context.axis));
  TF_LITE_ENSURE_OK(context, ValidateTensorData(context, op_context.output));
  RuntimeShape input_shape = GetTensorShape(input);
  size_t input_flat_size = 0;
  TF_LITE_ENSURE_MSG(context, input_shape.CheckedFlatSize(input_flat_size),
                     "Reduce input size overflowed.");
  if (input_flat_size == 0) {
    TF_LITE_ENSURE_OK(context,
                      InitializeMeanOutput(context, op_context.output));
    return kTfLiteOk;
  }

  if (kernel_type == kGenericOptimized) {
    // Use optimized ops if available.
    switch (input->type) {
      case kTfLiteInt8: {
        if (op_context.params->keep_dims && NumDimensions(input) == 4) {
          tflite::MeanParams op_params;
          TF_LITE_ENSURE_OK(
              context, BuildMeanParamsFor4D(context, op_context, &op_params));
          if (op_params.axis_count == 2 &&
              ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1))) {
            TF_LITE_ENSURE_OK(
                context, ValidateTensorDataForIntIndexedKernel(context, input));
            TF_LITE_ENSURE_OK(context, ValidateTensorDataForIntIndexedKernel(
                                           context, op_context.output));
            optimized_integer_ops::Mean(
                op_params, input_shape, GetTensorData<int8_t>(input),
                input->params.zero_point, input->params.scale,
                GetTensorShape(op_context.output),
                GetTensorData<int8_t>(op_context.output),
                op_context.output->params.zero_point,
                op_context.output->params.scale,
                CpuBackendContext::GetFromContext(context));
            return kTfLiteOk;
          }
        }
      } break;
      case kTfLiteUInt8: {
        if (op_context.params->keep_dims && NumDimensions(input) == 4) {
          tflite::MeanParams op_params;
          TF_LITE_ENSURE_OK(
              context, BuildMeanParamsFor4D(context, op_context, &op_params));
          if (op_params.axis_count == 2 &&
              ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
               (op_params.axis[0] == 2 && op_params.axis[1] == 1))) {
            TF_LITE_ENSURE_OK(
                context, ValidateTensorDataForIntIndexedKernel(context, input));
            TF_LITE_ENSURE_OK(context, ValidateTensorDataForIntIndexedKernel(
                                           context, op_context.output));
            optimized_ops::Mean(op_params, input_shape,
                                GetTensorData<uint8_t>(input),
                                input->params.zero_point, input->params.scale,
                                GetTensorShape(op_context.output),
                                GetTensorData<uint8_t>(op_context.output),
                                op_context.output->params.zero_point,
                                op_context.output->params.scale,
                                CpuBackendContext::GetFromContext(context));
            return kTfLiteOk;
          }
        }
      } break;
      default:
        break;
    }
  }

  switch (op_context.input->type) {
    case kTfLiteFloat32: {
      TfLiteStatus mean_status = Mean<float, float, kernel_type>(
          context, &op_context, GetTensorData<int>(temp_index),
          GetTensorData<int>(resolved_axis), GetTensorData<float>(temp_sum));
      TF_LITE_ENSURE_OK(context, mean_status);
    } break;
    case kTfLiteInt32: {
      TfLiteStatus mean_status = Mean<int, int64_t, kernel_type>(
          context, &op_context, GetTensorData<int>(temp_index),
          GetTensorData<int>(resolved_axis), GetTensorData<int64_t>(temp_sum));
      TF_LITE_ENSURE_OK(context, mean_status);
    } break;
    case kTfLiteInt64: {
      TfLiteStatus mean_status = Mean<int64_t, int64_t, kernel_type>(
          context, &op_context, GetTensorData<int>(temp_index),
          GetTensorData<int>(resolved_axis), GetTensorData<int64_t>(temp_sum));
      TF_LITE_ENSURE_OK(context, mean_status);
    } break;
    case kTfLiteInt8: {
      TF_LITE_ENSURE_OK(context, EvalQuantizedMean<int8_t>(
                                     context, op_context, num_axis, data,
                                     temp_index, resolved_axis, temp_sum));
    } break;
    case kTfLiteInt16: {
      TF_LITE_ENSURE_OK(context, EvalQuantizedMean<int16_t>(
                                     context, op_context, num_axis, data,
                                     temp_index, resolved_axis, temp_sum));
    } break;
    case kTfLiteUInt8: {
      TF_LITE_ENSURE_OK(context, EvalQuantizedMean<uint8_t>(
                                     context, op_context, num_axis, data,
                                     temp_index, resolved_axis, temp_sum));
    } break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <typename T>
struct EvalData {
  std::function<T(T, T)> reduce_func;
  const T* input_data;
  T output;
};

/// Returns true when the resolved axes cover every input dimension.
bool IsReduceAllDims(int num_resolved_axis, int num_dims) {
  return num_resolved_axis == num_dims;
}

// Worker for reducing single interval. Interval is identified by index
// from [start, end).
template <typename T>
struct ReduceWorkerTask : cpu_backend_threadpool::Task {
  ReduceWorkerTask(EvalData<T>* absl_nonnull eval_data, int start, int end)
      : eval_data(eval_data), start(start), end(end) {}
  void Run() override {
    auto* input_data = eval_data->input_data;
    T& output = eval_data->output;
    auto& reducer = eval_data->reduce_func;
    for (int i = start; i < end; ++i) {
      output = reducer(output, input_data[i]);
    }
  }

 private:
  EvalData<T>* eval_data;
  int start;
  int end;
};

/// Applies a reducer over all input elements into a single output element.
template <typename T>
TfLiteStatus ReduceAllDims(const T* input_data,
                           absl::Span<const int> input_dims,
                           T* absl_nonnull output_data, T init_value,
                           T reducer(const T current, const T in),
                           TfLiteContext* absl_nonnull context) {
  EvalData<T> eval_data;
  eval_data.reduce_func = reducer;
  eval_data.input_data = input_data;
  eval_data.output = init_value;

  int num_elems = 0;
  TF_LITE_ENSURE_MSG(context,
                     CheckedNumElements(input_dims, num_elems) == kTfLiteOk,
                     "Reduce input size overflowed.");
  TF_LITE_ENSURE(context, input_data != nullptr || num_elems == 0);

  // Fetch backend context and number of threads.
  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  int thread_count = std::max(1, cpu_backend_context->max_num_threads());
  const int kMinElementsPerThread = 1024;
  if (num_elems / thread_count < kMinElementsPerThread) thread_count = 1;

  if (thread_count == 1) {
    output_data[0] = num_elems > 0 ? input_data[0] : init_value;
    for (int i = 1; i < num_elems; ++i) {
      output_data[0] = reducer(output_data[0], input_data[i]);
    }
    return kTfLiteOk;
  }
  absl::InlinedVector<ReduceWorkerTask<T>, 4> tasks;
  absl::InlinedVector<EvalData<T>, 4> data;
  tasks.reserve(thread_count);
  data.reserve(thread_count);
  int start = 0;
  for (int i = 0; i < thread_count; ++i) {
    data.push_back(eval_data);
    int end = start + (num_elems - start) / (thread_count - i);
    tasks.emplace_back(&data.back(), start, end);
    start = end;
  }
  // Run all tasks on the thread pool.
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
  // Reduce all data from different workers.
  output_data[0] = data[0].output;
  for (size_t i = 1; i < data.size(); ++i) {
    output_data[0] = reducer(output_data[0], data[i].output);
  }
  return kTfLiteOk;
}

// The underlying logic for Reduce Sum/Prod/Max/Min/Any
template <typename T, KernelType kernel_type>
TfLiteStatus EvalType(TfLiteContext* absl_nonnull context,
                      TfLiteNode* absl_nonnull node,
                      OpContext* absl_nonnull op_context,
                      ReduceType reduce_type) {
  TF_LITE_ENSURE_OK(context, op_context->status);
  int num_axis = 0;
  TF_LITE_ENSURE_MSG(
      context, CheckedNumElements(op_context->axis, num_axis) == kTfLiteOk,
      "Reduce tensor shape is invalid or size overflowed.");
  TfLiteTensor* temp_index;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/0, &temp_index));
  TfLiteTensor* resolved_axis;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(op_context->output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeTempAxis(context, op_context, resolved_axis));
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, op_context));
  }

  const TfLiteTensor* input = op_context->input;
  TF_LITE_ENSURE_OK(context, ValidateTensorData(context, input));
  TF_LITE_ENSURE_OK(context, ValidateTensorData(context, op_context->axis));
  TF_LITE_ENSURE_OK(context, ValidateTensorData(context, op_context->output));
  if (kernel_type == kGenericOptimized) {
    TF_LITE_ENSURE_OK(context,
                      ValidateTensorDataForIntIndexedKernel(context, input));
    TF_LITE_ENSURE_OK(context, ValidateTensorDataForIntIndexedKernel(
                                   context, op_context->output));
  }
  if (input->type == kTfLiteUInt8 || input->type == kTfLiteInt8 ||
      input->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, input->params.scale,
                      op_context->output->params.scale);
    TF_LITE_ENSURE_EQ(context, input->params.zero_point,
                      op_context->output->params.zero_point);
  }
  if (kernel_type == kReference) {
    T init_value = 0;
    T (*reducer)(const T current, const T in);
    switch (reduce_type) {
      case kSum:
        reducer = [](const T current, const T in) -> T { return in + current; };
        init_value = T(0);
        break;
      case kProd:
        init_value = static_cast<T>(1);
        reducer = [](const T current, const T in) -> T { return in * current; };
        break;
      case kMax:
        init_value = std::numeric_limits<T>::lowest();
        reducer = [](const T current, const T in) -> T {
          return (in > current) ? in : current;
        };
        break;
      case kMin:
        init_value = std::numeric_limits<T>::max();
        reducer = [](const T current, const T in) -> T {
          return (in < current) ? in : current;
        };
        break;
      case kAny:
        init_value = false;
        reducer = [](const T current, const T in) -> T {
          return in || current;
        };
        break;
      case kAll:
        init_value = true;
        reducer = [](const T current, const T in) -> T {
          return in && current;
        };
        break;
      default:
        TF_LITE_KERNEL_LOG(context, "Unsupported ReduceType: %d", reduce_type);
        return kTfLiteError;
    }

    ResolvedAxisVector resolved_axis_values;
    TF_LITE_ENSURE_OK(
        context, ResolveAxisForShape(context, op_context->axis,
                                     input->dims->size, &resolved_axis_values));
    const int num_resolved_axis = static_cast<int>(resolved_axis_values.size());
    int* resolved_axis_data = GetTensorData<int>(resolved_axis);
    TF_LITE_ENSURE(
        context, resolved_axis_data != nullptr || resolved_axis_values.empty());
    absl::c_copy(resolved_axis_values, resolved_axis_data);

    if (IsReduceAllDims(num_resolved_axis, input->dims->size)) {
      TF_LITE_ENSURE_OK(
          context, ReduceAllDims(GetTensorData<T>(input),
                                 absl::Span<const int>(
                                     input->dims->data,
                                     static_cast<size_t>(input->dims->size)),
                                 GetTensorData<T>(op_context->output),
                                 init_value, reducer, context));
      return kTfLiteOk;
    }
    TF_LITE_ENSURE(
        context,
        reference_ops::ReduceGeneric<T>(
            GetTensorData<T>(input), input->dims->data, input->dims->size,
            GetTensorData<T>(op_context->output),
            op_context->output->dims->data, op_context->output->dims->size,
            GetTensorData<int>(op_context->axis), num_axis,
            op_context->params->keep_dims, GetTensorData<int>(temp_index),
            GetTensorData<int>(resolved_axis), init_value, reducer));
    return kTfLiteOk;
  } else {
    TfLiteTensor* normalized_dims;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/3,
                                                &normalized_dims));
    if (IsDynamicTensor(normalized_dims)) {
      TF_LITE_ENSURE_OK(context,
                        ResizeTempDims(context, op_context, normalized_dims));
    }
    TF_LITE_ENSURE(
        context,
        optimized_ops::ReduceGeneric<T>(
            GetTensorData<T>(input), input->dims->data, input->dims->size,
            GetTensorData<T>(op_context->output),
            op_context->output->dims->data, op_context->output->dims->size,
            GetTensorData<int>(op_context->axis), num_axis,
            GetTensorData<int>(resolved_axis),
            GetTensorData<int>(normalized_dims), reduce_type));
    return kTfLiteOk;
  }
}

// The entry point that handles input types and then calls template functions to
// handle ReduceType.
template <KernelType kernel_type, ReduceType reduce_type>
TfLiteStatus EvalGeneric(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, op_context.status);
  switch (op_context.input->type) {
    case kTfLiteFloat32:
      return EvalType<float, kernel_type>(context, node, &op_context,
                                          reduce_type);
      break;
    case kTfLiteInt32:
      return EvalType<int, kernel_type>(context, node, &op_context,
                                        reduce_type);
      break;
    case kTfLiteInt64:
      return EvalType<int64_t, kernel_type>(context, node, &op_context,
                                            reduce_type);
      break;
    case kTfLiteUInt8:
      return EvalType<uint8_t, kernel_type>(context, node, &op_context,
                                            reduce_type);
      break;
    case kTfLiteInt8:
      return EvalType<int8_t, kernel_type>(context, node, &op_context,
                                           reduce_type);
      break;
    case kTfLiteInt16:
      return EvalType<int16_t, kernel_type>(context, node, &op_context,
                                            reduce_type);
      break;
    case kTfLiteBool:
      return EvalType<bool, kernel_type>(context, node, &op_context,
                                         reduce_type);
      break;
    default:
      return kTfLiteError;
  }
}

template <KernelType kernel_type>
TfLiteStatus EvalSum(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, op_context.status);
  ruy::profiler::ScopeLabel label("Sum");
  const auto& input = op_context.input;
  const bool quantized = input->type == kTfLiteUInt8 ||
                         input->type == kTfLiteInt8 ||
                         input->type == kTfLiteInt16;
  if (quantized) {
    const OpData* op_data = reinterpret_cast<const OpData*>(node->user_data);
    TF_LITE_ENSURE(context, op_data != nullptr);
    TfLiteTensor* temp_index;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/0, &temp_index));
    TfLiteTensor* resolved_axis;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
    TfLiteTensor* temp_sum;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, /*index=*/2, &temp_sum));
    // Resize the output tensor if the output tensor is dynamic.
    if (IsDynamicTensor(op_context.output)) {
      TF_LITE_ENSURE_OK(context,
                        ResizeTempAxis(context, &op_context, resolved_axis));
      TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, &op_context));
      TF_LITE_ENSURE_OK(context,
                        ResizeTempAccum(context, &op_context, temp_sum));
    }
    TF_LITE_ENSURE_OK(context, ValidateTensorData(context, op_context.input));
    TF_LITE_ENSURE_OK(context, ValidateTensorData(context, op_context.axis));
    TF_LITE_ENSURE_OK(context, ValidateTensorData(context, op_context.output));

    if (input->type == kTfLiteUInt8) {
      return QuantizedMeanOrSum<uint8_t, kernel_type>(
          context, op_context, op_data, temp_index, resolved_axis, temp_sum,
          /*compute_sum=*/true);
    }
    if (input->type == kTfLiteInt8) {
      return QuantizedMeanOrSum<int8_t, kernel_type>(
          context, op_context, op_data, temp_index, resolved_axis, temp_sum,
          /*compute_sum=*/true);
    }
    if (input->type == kTfLiteInt16) {
      return QuantizedMeanOrSum<int16_t, kernel_type>(
          context, op_context, op_data, temp_index, resolved_axis, temp_sum,
          /*compute_sum=*/true);
    }
  } else {
    return EvalGeneric<kernel_type, kSum>(context, node);
  }

  return kTfLiteOk;
}

template <KernelType kernel_type, typename T>
TfLiteStatus EvalQuantizedProd(TfLiteContext* absl_nonnull context,
                               TfLiteNode* absl_nonnull node,
                               OpContext* absl_nonnull op_context) {
  TF_LITE_ENSURE_OK(context, op_context->status);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);
  TF_LITE_ENSURE(context, data != nullptr);

  int num_axis = 0;
  TF_LITE_ENSURE_MSG(
      context, CheckedNumElements(op_context->axis, num_axis) == kTfLiteOk,
      "Reduce tensor shape is invalid or size overflowed.");
  TfLiteTensor* temp_index;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/0, &temp_index));
  TfLiteTensor* resolved_axis;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/1, &resolved_axis));
  TfLiteTensor* temp_prod;
  TF_LITE_ENSURE_OK(context,
                    GetTemporarySafe(context, node, /*index=*/2, &temp_prod));
  TfLiteTensor* normalized_dims;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/3, &normalized_dims));
  const TfLiteTensor* input = op_context->input;
  TfLiteTensor* output = op_context->output;
  TF_LITE_ENSURE_OK(context, ValidateTensorData(context, input));
  TF_LITE_ENSURE_OK(context, ValidateTensorData(context, op_context->axis));
  ResolvedAxisVector resolved_axis_values;
  TF_LITE_ENSURE_OK(
      context, ResolveAxisForShape(context, op_context->axis, input->dims->size,
                                   &resolved_axis_values));

  const absl::Span<const int> input_dims = absl::MakeConstSpan(
      input->dims->data, static_cast<size_t>(input->dims->size));
  const bool input_has_zero_dim = absl::c_contains(input_dims, 0);

  if (IsDynamicTensor(normalized_dims)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeTempDims(context, op_context, normalized_dims));
  }
  // Resize the output tensor if the output tensor is dynamic.
  if (IsDynamicTensor(output)) {
    TF_LITE_ENSURE_OK(context,
                      ResizeTempAxis(context, op_context, resolved_axis));
    TF_LITE_ENSURE_OK(context, ResizeOutputTensor(context, op_context));
    TF_LITE_ENSURE_OK(context, ResizeTempAccum(context, op_context, temp_prod));

    if (!input_has_zero_dim) {
      int input_size = 0;
      TF_LITE_ENSURE_OK(
          context,
          GetFlatSizeAsInt(context, GetTensorShape(input), &input_size));
      int output_size = 0;
      TF_LITE_ENSURE_OK(
          context,
          GetFlatSizeAsInt(context, GetTensorShape(output), &output_size));
      TF_LITE_ENSURE(context, input_size != 0);
      TF_LITE_ENSURE(context, output_size != 0);

      const int reduced_axis_size = input_size / output_size;
      const double scaling = GetQuantProdScaling(
          static_cast<double>(input->params.scale),
          static_cast<double>(output->params.scale), reduced_axis_size);
      QuantizeMultiplier(scaling, &data->multiplier, &data->shift);
    }
  }
  TF_LITE_ENSURE_OK(context, ValidateTensorData(context, output));
  // Return early when input shape has zero dim after validating the dynamic
  // axis and sizing any dynamic output tensors.
  if (input_has_zero_dim) {
    return kTfLiteOk;
  }

  if (kernel_type == kReference) {
    TF_LITE_ENSURE(
        context,
        reference_ops::QuantizedReduceProd<T>(
            GetTensorData<T>(input), input->params.zero_point,
            GetTensorShape(input), GetTensorData<T>(output),
            output->params.zero_point, GetTensorShape(output),
            GetTensorData<int>(op_context->axis), num_axis,
            op_context->params->keep_dims, GetTensorData<int>(temp_index),
            GetTensorData<int>(resolved_axis), GetTensorData<int32>(temp_prod),
            data->multiplier, data->shift));
    return kTfLiteOk;
  } else {
    TF_LITE_ENSURE(
        context,
        optimized_ops::QuantizedReduceProd<T>(
            GetTensorData<T>(input), input->params.zero_point,
            GetTensorShape(input), GetTensorData<T>(output),
            output->params.zero_point, GetTensorShape(output),
            GetTensorData<int>(op_context->axis), num_axis,
            GetTensorData<int>(resolved_axis),
            GetTensorData<int>(normalized_dims),
            GetTensorData<int32>(temp_prod), data->multiplier, data->shift));
    return kTfLiteOk;
  }
}

template <KernelType kernel_type>
TfLiteStatus EvalProd(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, ValidateNode(context, node));
  const OpData* data = reinterpret_cast<const OpData*>(node->user_data);
  TF_LITE_ENSURE(context, data != nullptr);
  if (data->noop) {
    return kTfLiteOk;
  }
  return EvalImpl<kernel_type>(context, node);
}

template <KernelType kernel_type>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, op_context.status);
  // As we need to support both quantized and non-quantized int8/int16 inputs,
  // we separate the evaluation between EvalQuantizedProd for quantized
  // int8/int16 inputs and EvalGeneric for non-quantized int8/int16 (and
  // other non-quantized types).
  if (op_context.input->quantization.type != kTfLiteNoQuantization) {
    if (op_context.input->type == kTfLiteInt8) {
      return EvalQuantizedProd<kernel_type, int8_t>(context, node, &op_context);
    } else if (op_context.input->type == kTfLiteInt16) {
      return EvalQuantizedProd<kernel_type, int16_t>(context, node,
                                                     &op_context);
    } else {
      TF_LITE_KERNEL_LOG(context, "Unsupported quantized data type: %d",
                         op_context.input->type);
      return kTfLiteError;
    }
  } else {
    return EvalGeneric<kernel_type, kProd>(context, node);
  }
}

}  // namespace reduce

using ops::builtin::reduce::ReduceType;

TfLiteRegistration* Register_MEAN_OPT() {
  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::PrepareMeanOrSum,
                                 reduce::EvalMean<reduce::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_MEAN_REF() {
  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::PrepareMeanOrSum,
                                 reduce::EvalMean<reduce::kReference>};
  return &r;
}

TfLiteRegistration* Register_SUM_REF() {
  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::PrepareMeanOrSum,
                                 reduce::EvalSum<reduce::kReference>};
  return &r;
}

TfLiteRegistration* Register_SUM_OPT() {
  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::PrepareMeanOrSum,
                                 reduce::EvalSum<reduce::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_PROD_REF() {
  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::PrepareProd,
                                 reduce::EvalProd<reduce::kReference>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_PROD_OPT() {
  static TfLiteRegistration r = {reduce::Init, reduce::Free,
                                 reduce::PrepareProd,
                                 reduce::EvalProd<reduce::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_MAX_REF() {
  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareSimple,
      reduce::EvalGeneric<reduce::kReference, ReduceType::kMax>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_MAX_OPT() {
  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareSimple,
      reduce::EvalGeneric<reduce::kGenericOptimized, ReduceType::kMax>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_MIN_REF() {
  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareSimple,
      reduce::EvalGeneric<reduce::kReference, ReduceType::kMin>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_MIN_OPT() {
  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareSimple,
      reduce::EvalGeneric<reduce::kGenericOptimized, ReduceType::kMin>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_ANY_REF() {
  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareAllOrAny,
      reduce::EvalGeneric<reduce::kReference, ReduceType::kAny>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_ANY_OPT() {
  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareAllOrAny,
      reduce::EvalGeneric<reduce::kGenericOptimized, ReduceType::kAny>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_ALL_REF() {
  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareAllOrAny,
      reduce::EvalGeneric<reduce::kReference, ReduceType::kAll>};
  return &r;
}

TfLiteRegistration* Register_REDUCE_ALL_OPT() {
  static TfLiteRegistration r = {
      reduce::Init, reduce::Free, reduce::PrepareAllOrAny,
      reduce::EvalGeneric<reduce::kGenericOptimized, ReduceType::kAll>};
  return &r;
}

TfLiteRegistration* Register_MEAN() { return Register_MEAN_OPT(); }

TfLiteRegistration* Register_SUM() { return Register_SUM_OPT(); }
TfLiteRegistration* Register_REDUCE_PROD() {
  return Register_REDUCE_PROD_OPT();
}
TfLiteRegistration* Register_REDUCE_MAX() { return Register_REDUCE_MAX_OPT(); }
TfLiteRegistration* Register_REDUCE_MIN() { return Register_REDUCE_MIN_OPT(); }
TfLiteRegistration* Register_REDUCE_ANY() { return Register_REDUCE_ANY_OPT(); }
TfLiteRegistration* Register_REDUCE_ALL() { return Register_REDUCE_ALL_OPT(); }

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
