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

#include <cstdint>
#include <limits>

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
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace reduce {

const int kMaxConstantOutputTensorSize = 8;
// This file has reference implementation of reduce_* operators.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct OpData {
  int32_t multiplier;
  int shift;
  // The index of the temporary tensor where the quantized inputs are cached.
  int scratch_tensor_index;
  // Indicates that 'Eval' is a noop as the output as written during 'Prepare'.
  bool noop;
};

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteReducerParams*>(node->builtin_data);
    input = GetInput(context, node, 0);
    axis = GetInput(context, node, 1);
    output = GetOutput(context, node, 0);
  }
  TfLiteReducerParams* params;
  const TfLiteTensor* input;
  const TfLiteTensor* axis;
  TfLiteTensor* output;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // Creates three temp tensors to store index and axis for internal
  // implementation only.
  auto* op_data = new OpData();
  context->AddTensors(context, 4, &op_data->scratch_tensor_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

template <KernelType kernel_type>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node);

// Resizes the temp tensor that stores resolved axis.
TfLiteStatus ResizeTempAxis(TfLiteContext* context, OpContext* op_context,
                            TfLiteTensor* resolved_axis) {
  TfLiteIntArray* axis_size = TfLiteIntArrayCreate(1);
  axis_size->data[0] = static_cast<int>(NumElements(op_context->axis));
  return context->ResizeTensor(context, resolved_axis, axis_size);
}

// Resizes the temp tensor that stores temp sum of reduced elements.
TfLiteStatus ResizeTempAccum(TfLiteContext* context, OpContext* op_context,
                             TfLiteTensor* temp_accum) {
  TfLiteIntArray* size = TfLiteIntArrayCreate(1);
  size->data[0] = static_cast<int>(NumElements(op_context->output));
  return context->ResizeTensor(context, temp_accum, size);
}

// Returns the output shape.
TfLiteStatus GetOutputShape(TfLiteContext* context, OpContext* op_context,
                            TfLiteIntArray** output_shape) {
  size_t num_axis = NumElements(op_context->axis);
  const TfLiteIntArray* input_dims = op_context->input->dims;
  int input_num_dims = NumDimensions(op_context->input);
  if (input_num_dims == 0) {
    *output_shape = TfLiteIntArrayCreate(0);
    return kTfLiteOk;
  }
  const int* axis = GetTensorData<int>(op_context->axis);
  if (op_context->params->keep_dims) {
    TfLiteIntArray* output_dims = TfLiteIntArrayCreate(input_num_dims);
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx) {
          is_axis = true;
          break;
        }
      }
      if (is_axis) {
        output_dims->data[idx] = 1;
      } else {
        output_dims->data[idx] = input_dims->data[idx];
      }
    }
    *output_shape = output_dims;
    return kTfLiteOk;
  } else {
    // Calculates size of reducing axis.
    int num_reduce_axis = num_axis;
    for (int i = 0; i < num_axis; ++i) {
      int current = axis[i];
      if (current < 0) {
        current += input_num_dims;
      }
      TF_LITE_ENSURE(context, current >= 0 && current < input_num_dims);
      for (int j = 0; j < i; ++j) {
        int previous = axis[j];
        if (previous < 0) {
          previous += input_num_dims;
        }
        if (current == previous) {
          --num_reduce_axis;
          break;
        }
      }
    }
    // Determines output dimensions.
    TfLiteIntArray* output_dims =
        TfLiteIntArrayCreate(input_num_dims - num_reduce_axis);
    int num_skip_axis = 0;
    for (int idx = 0; idx < input_num_dims; ++idx) {
      bool is_axis = false;
      for (int axis_idx = 0; axis_idx < num_axis; ++axis_idx) {
        if (axis[axis_idx] == idx || axis[axis_idx] + input_num_dims == idx) {
          ++num_skip_axis;
          is_axis = true;
          break;
        }
      }
      if (!is_axis) {
        output_dims->data[idx - num_skip_axis] = input_dims->data[idx];
      }
    }
    *output_shape = output_dims;
    return kTfLiteOk;
  }
}

// Resizes output array based on the input size and resolved axis.
TfLiteStatus ResizeOutputTensor(TfLiteContext* context, OpContext* op_context) {
  TfLiteIntArray* output_dims;
  TF_LITE_ENSURE_OK(context, GetOutputShape(context, op_context, &output_dims));
  return context->ResizeTensor(context, op_context->output, output_dims);
}

// Resizes the temp tensor that stores normalized dims.
TfLiteStatus ResizeTempDims(TfLiteContext* context, OpContext* op_context,
                            TfLiteTensor* normalized_dims) {
  TfLiteIntArray* dims_size = TfLiteIntArrayCreate(1);
  dims_size->data[0] = (op_context->input->dims->size);
  return context->ResizeTensor(context, normalized_dims, dims_size);
}

// Initializes temp tensors to store index and resolved axis.
TfLiteStatus InitializeTemporaries(TfLiteContext* context, TfLiteNode* node,
                                   OpContext* op_context) {
  // Creates a temp index to iterate through input data.
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(4);
  node->temporaries->data[0] = op_data->scratch_tensor_index;
  TfLiteTensor* scratch_tensor;
  TF_LITE_ENSURE_OK(
      context, GetTemporarySafe(context, node, /*index=*/0, &scratch_tensor));
  scratch_tensor->type = kTfLiteInt32;
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TfLiteIntArray* index_size = TfLiteIntArrayCreate(1);
  index_size->data[0] = NumDimensions(op_context->input);
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, scratch_tensor, index_size));

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
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpContext op_context(context, node);
  TF_LITE_ENSURE_TYPES_EQ(context, op_context.axis->type, kTfLiteInt32);
  TF_LITE_ENSURE_OK(context, InitializeTemporaries(context, node, &op_context));
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  data->noop = IsConstantOrPersistentTensor(op_context.input) &&
               IsConstantOrPersistentTensor(op_context.axis);
  if (data->noop) {
    // Constant reductions should only be used for small outputs, typically
    // coming from Shape tensors. Constant reductions on larger tensors could
    // increase memory usage due to the output not being stored in the Arena.
    TfLiteIntArray* output_shape;
    TF_LITE_ENSURE_OK(context,
                      GetOutputShape(context, &op_context, &output_shape));
    int output_num_elements = 1;
    for (int i = 0; i < output_shape->size; ++i) {
      output_num_elements *= output_shape->data[i];
    }
    data->noop &= output_num_elements <= kMaxConstantOutputTensorSize;
    TfLiteIntArrayFree(output_shape);
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
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TF_LITE_ENSURE_TYPES_EQ(context, input->type, kTfLiteBool);
  return PrepareSimple(context, node);
}

TfLiteStatus PrepareMeanOrSum(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_OK(context, PrepareSimple(context, node));
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // reduce_mean requires a buffer to store intermediate sum result.
  OpContext op_context(context, node);
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
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

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

  const int input_size = GetTensorShape(op_context.input).FlatSize();
  const int output_size = GetTensorShape(op_context.output).FlatSize();
  // We support both quantized and non-quantized int8/int16 inputs
  if (op_context.input->quantization.type != kTfLiteNoQuantization &&
      (op_context.input->type == kTfLiteInt8 ||
       op_context.input->type == kTfLiteInt16) &&
      input_size != 0 && output_size != 0) {
    const int reduced_axis_size = input_size / output_size;
    const double scaling = GetQuantProdScaling(
        static_cast<double>(op_context.input->params.scale),
        static_cast<double>(op_context.output->params.scale),
        reduced_axis_size);
    QuantizeMultiplier(scaling, &data->multiplier, &data->shift);
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

void ResolveAxis(const int* axis_data, int axis_count,
                 tflite::MeanParams* op_params) {
  int i = 0;
  for (; i < axis_count; ++i) {
    op_params->axis[i] = static_cast<int16>(axis_data[i]);
  }
  for (; i < 4; ++i) {
    op_params->axis[i] = 1;
  }
}

template <typename T, typename U, KernelType kernel_type>
TfLiteStatus Mean(TfLiteContext* context, const OpContext* op_context,
                  int* temp_index, int* resolved_axis, U* temp_sum) {
  int num_axis = static_cast<int>(NumElements(op_context->axis));
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
TfLiteStatus QuantizedMeanOrSum(TfLiteContext* context,
                                const OpContext& op_context,
                                const OpData* op_data, TfLiteTensor* temp_index,
                                TfLiteTensor* resolved_axis,
                                TfLiteTensor* temp_sum, bool compute_sum) {
  int num_axis = static_cast<int>(NumElements(op_context.axis));
  if (kernel_type == kGenericOptimized) {
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
TfLiteStatus EvalQuantizedMean(TfLiteContext* context,
                               const OpContext& op_context, int num_axis,
                               OpData* data, TfLiteTensor* temp_index,
                               TfLiteTensor* resolved_axis,
                               TfLiteTensor* temp_sum) {
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
void InitializeMeanOutputTyped(TfLiteTensor* output) {
  RuntimeShape output_shape = GetTensorShape(output);
  const size_t flat_size = output_shape.FlatSize();
  T* output_data = GetTensorData<T>(output);
  T nan_value = std::numeric_limits<T>::quiet_NaN();
  for (int idx = 0; idx < flat_size; ++idx) {
    *output_data++ = nan_value;
  }
}

TfLiteStatus InitializeMeanOutput(TfLiteTensor* output) {
  switch (output->type) {
    case kTfLiteFloat32:
      InitializeMeanOutputTyped<float>(output);
      break;
    case kTfLiteInt32:
      InitializeMeanOutputTyped<int>(output);
      break;
    case kTfLiteInt64:
      InitializeMeanOutputTyped<int64_t>(output);
      break;
    case kTfLiteUInt8:
      InitializeMeanOutputTyped<uint8_t>(output);
      break;
    case kTfLiteInt8:
      InitializeMeanOutputTyped<int8_t>(output);
      break;
    case kTfLiteInt16:
      InitializeMeanOutputTyped<int16_t>(output);
      break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalMean(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  int num_axis = static_cast<int>(NumElements(op_context.axis));
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
  RuntimeShape input_shape = GetTensorShape(input);
  if (input_shape.FlatSize() == 0) {
    TF_LITE_ENSURE_OK(context, InitializeMeanOutput(op_context.output));
    return kTfLiteOk;
  }

  if (kernel_type == kGenericOptimized) {
    // Use optimized ops if available.
    switch (input->type) {
      case kTfLiteInt8: {
        tflite::MeanParams op_params;
        op_params.axis_count = num_axis;
        ResolveAxis(GetTensorData<int>(op_context.axis), num_axis, &op_params);
        if (op_context.params->keep_dims && NumDimensions(input) == 4 &&
            op_params.axis_count == 2 &&
            ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
             (op_params.axis[0] == 2 && op_params.axis[1] == 1))) {
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
      } break;
      case kTfLiteUInt8: {
        tflite::MeanParams op_params;
        op_params.axis_count = num_axis;
        ResolveAxis(GetTensorData<int>(op_context.axis), num_axis, &op_params);
        if (op_context.params->keep_dims && NumDimensions(input) == 4 &&
            op_params.axis_count == 2 &&
            ((op_params.axis[0] == 1 && op_params.axis[1] == 2) ||
             (op_params.axis[0] == 2 && op_params.axis[1] == 1))) {
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
      } break;
      default:
        break;
    }
  }

  switch (op_context.input->type) {
    case kTfLiteFloat32:
      Mean<float, float, kernel_type>(
          context, &op_context, GetTensorData<int>(temp_index),
          GetTensorData<int>(resolved_axis), GetTensorData<float>(temp_sum));
      break;
    case kTfLiteInt32:
      Mean<int, int64_t, kernel_type>(
          context, &op_context, GetTensorData<int>(temp_index),
          GetTensorData<int>(resolved_axis), GetTensorData<int64_t>(temp_sum));
      break;
    case kTfLiteInt64:
      Mean<int64_t, int64_t, kernel_type>(
          context, &op_context, GetTensorData<int>(temp_index),
          GetTensorData<int>(resolved_axis), GetTensorData<int64_t>(temp_sum));
      break;
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

// Returns true if 'axis' holds all dims [0 ... N-1] where N is num_dims.
bool IsReduceAllDims(const TfLiteTensor* axis, int num_axis, int num_dims) {
  int dims_mask = 0;
  for (int i = 0; i < num_axis; ++i) {
    dims_mask |= 1 << (axis->data.i32[i]);
  }
  return num_dims == 0 ? dims_mask == 0 : (dims_mask == (1 << num_dims) - 1);
}

// Worker for reducing single interval. Interval is identified by index
// from [start, end).
template <typename T>
struct ReduceWorkerTask : cpu_backend_threadpool::Task {
  ReduceWorkerTask(EvalData<T>* eval_data, int start, int end)
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

// Apply reduce operation using the 'reducer' function on all of 'input_data'.
// and reduce all to single element.
template <typename T>
void ReduceAllDims(const T* input_data, const int* input_dims,
                   const int input_num_dims, T* output_data, T init_value,
                   T reducer(const T current, const T in),
                   TfLiteContext* context) {
  EvalData<T> eval_data;
  eval_data.reduce_func = reducer;
  eval_data.input_data = input_data;
  eval_data.output = init_value;

  int num_elems = NumElements(input_dims, input_num_dims);

  // Fetch backend context and number of threads.
  CpuBackendContext* cpu_backend_context =
      CpuBackendContext::GetFromContext(context);
  int thread_count = cpu_backend_context->max_num_threads();
  const int kMinElementsPerThread = 1024;
  if (num_elems / thread_count < kMinElementsPerThread) thread_count = 1;

  if (thread_count == 1) {
    output_data[0] = num_elems > 0 ? input_data[0] : init_value;
    for (int i = 1; i < num_elems; ++i) {
      output_data[0] = reducer(output_data[0], input_data[i]);
    }
    return;
  }
  std::vector<ReduceWorkerTask<T>> tasks;
  std::vector<EvalData<T>> data;
  tasks.reserve(thread_count);
  data.reserve(thread_count);
  int start = 0;
  for (int i = 0; i < thread_count; ++i) {
    data.push_back(eval_data);
    int end = start + (num_elems - start) / (thread_count - i);
    tasks.emplace_back(ReduceWorkerTask<T>(&data.back(), start, end));
    start = end;
  }
  // Run all tasks on the thread pool.
  cpu_backend_threadpool::Execute(tasks.size(), tasks.data(),
                                  cpu_backend_context);
  // Reduce all data from different workers.
  output_data[0] = data[0].output;
  for (int i = 1; i < data.size(); ++i) {
    output_data[0] = reducer(output_data[0], data[i].output);
  }
}

// The underlying logic for Reduce Sum/Prod/Max/Min/Any
template <typename T, KernelType kernel_type>
TfLiteStatus EvalType(TfLiteContext* context, TfLiteNode* node,
                      OpContext* op_context, ReduceType reduce_type) {
  int64_t num_axis = NumElements(op_context->axis);
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

    int num_resolved_axis = 0;
    TF_LITE_ENSURE_MSG(
        context,
        tflite::reference_ops::ResolveAxis(
            input->dims->size, GetTensorData<int>(op_context->axis), num_axis,
            GetTensorData<int>(resolved_axis), &num_resolved_axis),
        "Invalid axis index.");

    if (IsReduceAllDims(resolved_axis, num_resolved_axis, input->dims->size)) {
      ReduceAllDims(GetTensorData<T>(input), input->dims->data,
                    input->dims->size, GetTensorData<T>(op_context->output),
                    init_value, reducer, context);
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
  ruy::profiler::ScopeLabel label("Sum");
  const auto& input = op_context.input;
  const bool eight_bit_quantized =
      input->type == kTfLiteUInt8 || input->type == kTfLiteInt8;
  if (eight_bit_quantized) {
    const OpData* op_data = reinterpret_cast<const OpData*>(node->user_data);
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
  } else {
    return EvalGeneric<kernel_type, kSum>(context, node);
  }

  return kTfLiteOk;
}

template <KernelType kernel_type, typename T>
TfLiteStatus EvalQuantizedProd(TfLiteContext* context, TfLiteNode* node,
                               OpContext* op_context) {
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const int64_t num_axis = NumElements(op_context->axis);
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

  // Return early when input shape has zero dim.
  for (int i = 0; i < input->dims->size; ++i) {
    if (input->dims->data[i] == 0) return kTfLiteOk;
  }

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

    const int input_size = GetTensorShape(input).FlatSize();
    const int output_size = GetTensorShape(output).FlatSize();
    TF_LITE_ENSURE(context, input_size != 0);
    TF_LITE_ENSURE(context, output_size != 0);

    const int reduced_axis_size = input_size / output_size;
    const double scaling = GetQuantProdScaling(
        static_cast<double>(input->params.scale),
        static_cast<double>(output->params.scale), reduced_axis_size);
    QuantizeMultiplier(scaling, &data->multiplier, &data->shift);
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
  if (reinterpret_cast<const OpData*>(node->user_data)->noop) {
    return kTfLiteOk;
  }
  return EvalImpl<kernel_type>(context, node);
}

template <KernelType kernel_type>
TfLiteStatus EvalImpl(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
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
