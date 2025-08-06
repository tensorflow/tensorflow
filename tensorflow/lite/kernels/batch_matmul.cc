/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/kernels/internal/reference/batch_matmul.h"

#include <stddef.h>
#include <string.h>

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/core/c/builtin_op_data.h"
#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/kernels/cpu_backend_context.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/kernels/internal/optimized/batch_matmul.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace batch_matmul {

static const int kInputLHSTensor = 0;
static const int kInputRHSTensor = 1;
static const int kOutputTensor = 0;

static const int kNumTempTensorsForAdjoints = 2;
static const int kNumTempTensorsForHybrid = 5;

// This file has two implementations of Transpose.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct OpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;
  // The range of the fused activation layer. For example for kNone and
  // uint8_t these would be 0 and 255.
  int32_t output_activation_min;
  int32_t output_activation_max;
  // The index of the temporary tensors where we store transposed LHS/RHS.
  int scratch_tensor_index;
  bool rhs_transposed;
  bool compute_row_sums = false;
};

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteBatchMatMulParams*>(node->builtin_data);
    lhs = GetInput(context, node, kInputLHSTensor);
    rhs = GetInput(context, node, kInputRHSTensor);
    output = GetOutput(context, node, 0);
  }
  TfLiteBatchMatMulParams* params;
  const TfLiteTensor* lhs;
  const TfLiteTensor* rhs;
  TfLiteTensor* output;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  // If the RHS is constant, we only transpose once.
  op_data->rhs_transposed = false;
  // Creates the temp tensors to store the transposed LHS and/or RHS, and
  // extra buffers for the quantized case.
  op_data->scratch_tensor_index = -1;
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete static_cast<OpData*>(buffer);
}

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const RuntimeShape& extended_lhs_shape,
                                const RuntimeShape& extended_rhs_shape,
                                bool adj_x, bool adj_y, int output_rank,
                                TfLiteTensor* output) {
  TfLiteIntArray* output_shape = TfLiteIntArrayCreate(output_rank);
  // Fill in any broadcast dimensions.
  for (int i = 0; i < output_rank - 2; ++i) {
    const int lhs_dim = extended_lhs_shape.Dims(i);
    const int rhs_dim = extended_rhs_shape.Dims(i);
    int broadcast_dim = lhs_dim;
    if ((lhs_dim != rhs_dim) && (lhs_dim == 1)) {
      broadcast_dim = rhs_dim;
    }
    output_shape->data[i] = broadcast_dim;
  }
  // Fill in the matmul dimensions.
  int lhs_rows_index = adj_x ? output_rank - 1 : output_rank - 2;
  int rhs_cols_index = adj_y ? output_rank - 2 : output_rank - 1;

  output_shape->data[output_rank - 2] = extended_lhs_shape.Dims(lhs_rows_index);
  output_shape->data[output_rank - 1] = extended_rhs_shape.Dims(rhs_cols_index);
  TfLiteStatus stat = context->ResizeTensor(context, output, output_shape);
  return stat;
}

// Initializes temp tensors to store transposed operands.
TfLiteStatus InitializeTemporaries(TfLiteContext* context, TfLiteNode* node,
                                   OpContext* op_context) {
  // Create temporary tensors to hold transposed LHS/RHS.
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* lhs = op_context->lhs;
  const TfLiteTensor* rhs = op_context->rhs;
  TfLiteIntArrayFree(node->temporaries);
  // For "hybrid" quantization, we impose the constraint that the LHS
  // is float (typically an activation from a prior layer) and the RHS
  // is quantized int8.
  bool is_hybrid =
      (op_context->lhs->type == kTfLiteFloat32 && rhs->type == kTfLiteInt8);
  if (is_hybrid) {
    node->temporaries = TfLiteIntArrayCreate(kNumTempTensorsForAdjoints +
                                             kNumTempTensorsForHybrid);
  } else {
    node->temporaries = TfLiteIntArrayCreate(kNumTempTensorsForAdjoints);
  }

  const int lhs_rank = NumDimensions(lhs);
  const int rhs_rank = NumDimensions(rhs);
  const int batch_size = op_context->params->adj_x
                             ? lhs->dims->data[lhs_rank - 1]
                             : lhs->dims->data[lhs_rank - 2];
  const int num_units = op_context->params->adj_y
                            ? rhs->dims->data[rhs_rank - 2]
                            : rhs->dims->data[rhs_rank - 1];

  // Temp tensor for Transposed LHS;
  {
    node->temporaries->data[0] = op_data->scratch_tensor_index;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/0, &scratch_buffer));
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(lhs_rank);
    for (int i = 0; i < lhs_rank - 2; ++i) {
      scratch_buffer_size->data[i] = lhs->dims->data[i];
    }
    // Swap last two dimensions.
    scratch_buffer_size->data[lhs_rank - 2] = lhs->dims->data[lhs_rank - 1];
    scratch_buffer_size->data[lhs_rank - 1] = lhs->dims->data[lhs_rank - 2];

    scratch_buffer->type = op_context->lhs->type;
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // We need a temp buffer for the RHS if we need to transpose the RHS. We
  // transpose by default, so that the two inputs (LHS and RHS) are in a proper
  // layout for our fast matrix multiplication routines. If the transpose flag
  // is set by the caller, the data is already in the desired layout.
  {
    node->temporaries->data[1] = op_data->scratch_tensor_index + 1;
    TfLiteTensor* scratch_buffer;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/1, &scratch_buffer));
    scratch_buffer->name = "BatchMatMul_scratch_buffer";
    const TfLiteTensor* rhs = op_context->rhs;
    int rhs_rank = NumDimensions(rhs);
    TfLiteIntArray* scratch_buffer_size = TfLiteIntArrayCreate(rhs_rank);
    for (int i = 0; i < rhs_rank - 2; ++i) {
      scratch_buffer_size->data[i] = rhs->dims->data[i];
    }
    // Swap last two dimensions.
    scratch_buffer_size->data[rhs_rank - 2] = rhs->dims->data[rhs_rank - 1];
    scratch_buffer_size->data[rhs_rank - 1] = rhs->dims->data[rhs_rank - 2];

    if (IsConstantTensor(op_context->rhs)) {
      scratch_buffer->allocation_type = kTfLiteArenaRwPersistent;
    } else {
      scratch_buffer->allocation_type = kTfLiteArenaRw;
    }
    scratch_buffer->type = op_context->rhs->type;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }

  // If we have to perform on-the-fly quantization (with quantized weights and
  // float inputs) first we need to quantize the inputs. Allocate temporary
  // buffer to store the intermediate quantized values, the batch scaling
  // factors, the accumulator buffer (optimized version), the input offsets,
  // and the sums of the rows for each weights matrix.
  // RHS = weights, LHS = inputs
  if (is_hybrid) {
    // Calculate the total number of LHS batches.
    int num_batches = 1;
    for (int i = 0; i < lhs_rank - 2; ++i) {
      num_batches *= lhs->dims->data[i];
    }
    int num_weights_matrices = 1;
    for (int i = 0; i < rhs_rank - 2; ++i) {
      num_weights_matrices *= rhs->dims->data[i];
    }
    op_data->compute_row_sums = true;
    node->temporaries->data[2] = op_data->scratch_tensor_index + 2;
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/2,
                                                &input_quantized));
    input_quantized->type = op_context->rhs->type;
    input_quantized->allocation_type = kTfLiteArenaRw;

    TfLiteIntArray* input_quantized_size =
        TfLiteIntArrayCopy(op_context->lhs->dims);
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                     input_quantized_size));

    node->temporaries->data[3] = op_data->scratch_tensor_index + 3;
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/3,
                                                &scaling_factors));
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    // Total size of scaling factors is batch size * number of total batches
    int scaling_dims[1] = {num_batches * batch_size};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = scaling_dims[0];
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }

    node->temporaries->data[4] = op_data->scratch_tensor_index + 4;
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/4, &accum_scratch));
    accum_scratch->type = kTfLiteInt32;
    accum_scratch->allocation_type = kTfLiteArenaRw;
    int accum_scratch_dims[2] = {num_units, batch_size};
    if (!TfLiteIntArrayEqualsArray(accum_scratch->dims, 2,
                                   accum_scratch_dims)) {
      TfLiteIntArray* accum_size = TfLiteIntArrayCreate(2);
      accum_size->data[0] = num_units;
      accum_size->data[1] = batch_size;
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, accum_scratch, accum_size));
    }

    node->temporaries->data[5] = op_data->scratch_tensor_index + 5;
    TfLiteTensor* input_offsets;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/5, &input_offsets));
    input_offsets->type = kTfLiteInt32;
    input_offsets->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqualsArray(input_offsets->dims, 1, scaling_dims)) {
      TfLiteIntArray* input_offsets_size = TfLiteIntArrayCreate(1);
      input_offsets_size->data[0] = num_batches * batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_offsets,
                                                       input_offsets_size));
    }
    node->temporaries->data[6] = op_data->scratch_tensor_index + 6;
    TfLiteTensor* row_sums;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, /*index=*/6, &row_sums));
    row_sums->type = kTfLiteInt32;
    row_sums->allocation_type = kTfLiteArenaRwPersistent;
    int row_sums_dims[1] = {num_weights_matrices * num_units};
    if (!TfLiteIntArrayEqualsArray(row_sums->dims, 1, row_sums_dims)) {
      TfLiteIntArray* row_sums_size = TfLiteIntArrayCreate(1);
      row_sums_size->data[0] = row_sums_dims[0];
      TF_LITE_ENSURE_OK(
          context, context->ResizeTensor(context, row_sums, row_sums_size));
    }
  }

  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  if (op_data->scratch_tensor_index == -1) {
    context->AddTensors(context,
                        kNumTempTensorsForAdjoints + kNumTempTensorsForHybrid,
                        &op_data->scratch_tensor_index);
  }
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, InitializeTemporaries(context, node, &op_context));

  bool adj_x = op_context.params->adj_x;
  bool adj_y = op_context.params->adj_y;

  const TfLiteTensor* lhs_data;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputLHSTensor, &lhs_data));
  const TfLiteTensor* rhs_data;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputRHSTensor, &rhs_data));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if ((lhs_data->type == kTfLiteInt8 || lhs_data->type == kTfLiteInt16) &&
      output->type != kTfLiteInt32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, lhs_data, rhs_data, output, &real_multiplier));
    int exponent;
    QuantizeMultiplier(real_multiplier, &op_data->output_multiplier, &exponent);
    op_data->output_shift = exponent;
    // BatchMatMul has no fused activation functions. Therefore, set
    // output activation min and max to min and max of int8_t or int16_t
    // type.
    if (lhs_data->type == kTfLiteInt8) {
      op_data->output_activation_min = std::numeric_limits<int8_t>::min();
      op_data->output_activation_max = std::numeric_limits<int8_t>::max();
    } else {
      op_data->output_activation_min = std::numeric_limits<int16_t>::min();
      op_data->output_activation_max = std::numeric_limits<int16_t>::max();
    }
  }

  if (lhs_data->type == kTfLiteInt16) {
    TF_LITE_ENSURE_EQ(context, lhs_data->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, rhs_data->params.zero_point, 0);
    TF_LITE_ENSURE_EQ(context, output->params.zero_point, 0);
  }

  TF_LITE_ENSURE(context, lhs_data->type == kTfLiteFloat32 ||
                              lhs_data->type == kTfLiteInt8 ||
                              lhs_data->type == kTfLiteInt16);
  TF_LITE_ENSURE(context, rhs_data->type == kTfLiteFloat32 ||
                              rhs_data->type == kTfLiteInt8 ||
                              rhs_data->type == kTfLiteInt16);
  // Either we have a hybrid quantization with a float32 and an int8 input,
  // otherwise both inputs should be of the same type.
  TF_LITE_ENSURE(context, (lhs_data->type == kTfLiteFloat32 &&
                           rhs_data->type == kTfLiteInt8) ||
                              lhs_data->type == rhs_data->type);
  // Support dimensions between 2 and 5, inclusive.
  TF_LITE_ENSURE(context, NumDimensions(lhs_data) >= 2);
  TF_LITE_ENSURE(context, NumDimensions(lhs_data) <= 5);
  TF_LITE_ENSURE(context, NumDimensions(rhs_data) >= 2);
  TF_LITE_ENSURE(context, NumDimensions(rhs_data) <= 5);

  const int lhs_rank = NumDimensions(lhs_data);
  const int rhs_rank = NumDimensions(rhs_data);
  const int output_rank = std::max(lhs_rank, rhs_rank);
  const RuntimeShape extended_lhs_shape =
      RuntimeShape::ExtendedShape(output_rank, GetTensorShape(lhs_data));
  const RuntimeShape extended_rhs_shape =
      RuntimeShape::ExtendedShape(output_rank, GetTensorShape(rhs_data));

  // Ensure any batch dimensions obey broacasting rules.
  for (int i = 0; i < output_rank - 2; ++i) {
    const int lhs_dim = extended_lhs_shape.Dims(i);
    const int rhs_dim = extended_rhs_shape.Dims(i);
    if (lhs_dim != rhs_dim) {
      if (lhs_dim != 1) {
        TF_LITE_ENSURE_EQ(context, rhs_dim, 1);
      }
    }
  }
  // Ensure other dimensions work for matrix multiplication.
  int accum_dim_lhs = adj_x ? extended_lhs_shape.Dims(output_rank - 2)
                            : extended_lhs_shape.Dims(output_rank - 1);
  int accum_dim_rhs = adj_y ? extended_rhs_shape.Dims(output_rank - 1)
                            : extended_rhs_shape.Dims(output_rank - 2);

  TF_LITE_ENSURE_EQ(context, accum_dim_lhs, accum_dim_rhs);
  TfLiteStatus status =
      ResizeOutputTensor(context, extended_lhs_shape, extended_rhs_shape, adj_x,
                         adj_y, output_rank, output);
  return status;
}

template <typename scalar>
void TransposeRowsColumnsImpl(const TfLiteTensor* tensor_in,
                              const scalar* input, TfLiteTensor* tensor_out,
                              scalar* output) {
  RuntimeShape transposed_shape(GetTensorShape(tensor_in));
  RuntimeShape shape(GetTensorShape(tensor_in));
  TransposeParams params;
  int rank = NumDimensions(tensor_in);
  params.perm_count = rank;
  for (int i = 0; i < rank - 2; ++i) {
    params.perm[i] = i;
  }
  // Transpose the last two dimensions.
  params.perm[rank - 2] = rank - 1;
  params.perm[rank - 1] = rank - 2;
  transposed_shape.SetDim(rank - 1, shape.Dims(rank - 2));
  transposed_shape.SetDim(rank - 2, shape.Dims(rank - 1));
  optimized_ops::Transpose(params, shape, input, transposed_shape, output);
}

TfLiteStatus TransposeRowsColumns(TfLiteContext* context,
                                  const TfLiteTensor* tensor_in,
                                  TfLiteTensor* tensor_out) {
  if (tensor_in->type == kTfLiteFloat32) {
    TransposeRowsColumnsImpl<float>(tensor_in, GetTensorData<float>(tensor_in),
                                    tensor_out,
                                    GetTensorData<float>(tensor_out));
    return kTfLiteOk;
  } else if (tensor_in->type == kTfLiteInt8) {
    TransposeRowsColumnsImpl<int8_t>(
        tensor_in, GetTensorData<int8_t>(tensor_in), tensor_out,
        GetTensorData<int8_t>(tensor_out));
    return kTfLiteOk;
  } else if (tensor_in->type == kTfLiteInt16) {
    TransposeRowsColumnsImpl<int16_t>(
        tensor_in, GetTensorData<int16_t>(tensor_in), tensor_out,
        GetTensorData<int16_t>(tensor_out));
    return kTfLiteOk;
  } else {
    TF_LITE_KERNEL_LOG(
        context, "Can only transpose tensors with float, int8 or int16 type.");
    return kTfLiteError;
  }
}

RuntimeShape SwapRowColumnDims(const RuntimeShape& shape) {
  RuntimeShape swapped_shape(shape);
  const int32_t dims = shape.DimensionsCount();
  swapped_shape.SetDim(dims - 2, shape.Dims(dims - 1));
  swapped_shape.SetDim(dims - 1, shape.Dims(dims - 2));
  return swapped_shape;
}

TfLiteStatus VerifyPerChannelQuantization(TfLiteContext* context,
                                          const TfLiteTensor* tensor) {
  TF_LITE_ENSURE_EQ(context, tensor->quantization.type,
                    kTfLiteAffineQuantization);
  const auto* affine_quantization =
      reinterpret_cast<TfLiteAffineQuantization*>(tensor->quantization.params);
  TF_LITE_ENSURE(context, affine_quantization);
  TF_LITE_ENSURE(context, affine_quantization->scale);
  return affine_quantization->scale->size > 1 ? kTfLiteOk : kTfLiteError;
}

TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node, OpData* data,
                        const RuntimeShape& input_shape,
                        const TfLiteTensor* input,
                        const RuntimeShape& filter_shape,
                        const TfLiteTensor* filter,
                        TfLiteTensor* input_quantized,
                        TfLiteTensor* scaling_factors,
                        TfLiteTensor* accum_scratch, TfLiteTensor* row_sums,
                        TfLiteTensor* input_offsets, TfLiteTensor* output) {
  const auto* params =
      reinterpret_cast<TfLiteBatchMatMulParams*>(node->builtin_data);
  const int32_t num_input_dims = input_shape.DimensionsCount();

  // Input row/cols have been swapped at this point, so dims are
  // {input_size, num_batches}
  const int input_size = input_shape.Dims(num_input_dims - 2);
  const int batch_size = input_shape.Dims(num_input_dims - 1);

  int num_batches_to_quantize = batch_size;
  for (int i = 0; i < input_shape.DimensionsCount() - 2; ++i) {
    num_batches_to_quantize *= input_shape.Dims(i);
  }
  // Quantize input from float to uint8 + quantization params (scaling factor).
  const int scaling_factor_size = GetTensorShape(scaling_factors).FlatSize();
  TF_LITE_ENSURE(context, scaling_factor_size >= num_batches_to_quantize);
  float* scaling_factors_ptr = GetTensorData<float>(scaling_factors);
  int32_t* input_offset_ptr = nullptr;
  int32_t* row_sums_ptr = nullptr;
  input_offset_ptr = GetTensorData<int32_t>(input_offsets);
  row_sums_ptr = GetTensorData<int32_t>(row_sums);
  if (!params->asymmetric_quantize_inputs) {
    memset(input_offset_ptr, 0, input_offsets->bytes);
  }
  int8_t* quant_data = GetTensorData<int8_t>(input_quantized);
  const int8_t* filter_data = GetTensorData<int8_t>(filter);
  const float* input_ptr = GetTensorData<float>(input);
  // Quantize each batch independently.
  tensor_utils::BatchQuantizeFloats(input_ptr, num_batches_to_quantize,
                                    input_size, quant_data, scaling_factors_ptr,
                                    input_offset_ptr,
                                    params->asymmetric_quantize_inputs);
  float* per_channel_scale_ptr = nullptr;
  if (VerifyPerChannelQuantization(context, filter) == kTfLiteOk) {
    //  Per channel quantization.
    const auto* affine_quantization =
        reinterpret_cast<TfLiteAffineQuantization*>(
            filter->quantization.params);
    TF_LITE_ENSURE_EQ(
        context, affine_quantization->scale->size,
        filter->dims->data[affine_quantization->quantized_dimension]);
    per_channel_scale_ptr = affine_quantization->scale->data;
  } else {
    // Per tensor quantization.
    for (int b = 0; b < num_batches_to_quantize; ++b) {
      // Incorporate scaling of the filter
      scaling_factors_ptr[b] *= filter->params.scale;
    }
  }

  RuntimeShape output_shape = GetTensorShape(output);
  int output_size = 1;
  for (int i = 0; i < output_shape.DimensionsCount(); ++i) {
    output_size *= output_shape.Dims(i);
  }
  std::fill_n(GetTensorData<float>(output), output_size, 0.0f);
  reference_ops::BatchMatMul(filter_shape, filter_data, input_shape, quant_data,
                             scaling_factors_ptr, input_offset_ptr,
                             row_sums_ptr, GetTensorShape(output),
                             GetTensorData<float>(output),
                             &(data->compute_row_sums), per_channel_scale_ptr);

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalInt8Int8(TfLiteContext* context, const OpData* data,
                          const RuntimeShape& lhs_shape,
                          const TfLiteTensor* lhs,
                          const RuntimeShape& rhs_shape,
                          const TfLiteTensor* rhs,
                          const RuntimeShape& output_shape,
                          TfLiteTensor* output, bool transpose_lhs) {
  // Reuse params struct from FullyConnected Op.
  FullyConnectedParams op_params;
  int32_t input_offset = -lhs->params.zero_point;
  int32_t filter_offset = -rhs->params.zero_point;
  int32_t output_offset = output->params.zero_point;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;
  op_params.lhs_cacheable = IsConstantTensor(lhs);
  op_params.rhs_cacheable = IsConstantTensor(rhs);

  if (kernel_type == kReference) {
    reference_ops::BatchMatMul<int8_t, int32_t>(
        op_params, rhs_shape, GetTensorData<int8_t>(rhs), lhs_shape,
        GetTensorData<int8_t>(lhs), GetTensorShape(output),
        GetTensorData<int8_t>(output));
  } else {
    optimized_ops::BatchMatMul(
        op_params, rhs_shape, GetTensorData<int8_t>(rhs), lhs_shape,
        GetTensorData<int8_t>(lhs), GetTensorShape(output),
        GetTensorData<int8_t>(output),
        CpuBackendContext::GetFromContext(context), transpose_lhs);
  }
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalInt8Int32(TfLiteContext* context, const OpData* data,
                           const RuntimeShape& lhs_shape,
                           const TfLiteTensor* lhs,
                           const RuntimeShape& rhs_shape,
                           const TfLiteTensor* rhs,
                           const RuntimeShape& output_shape,
                           TfLiteTensor* output) {
  // Set BatchMatMul lhs param to rhs(filter) and rhs param to lhs(input). For
  // the reason, see comment of Eval() function.
  reference_ops::BatchMatMul<int8, int8, int32>(
      rhs_shape, GetTensorData<int8>(rhs), lhs_shape, GetTensorData<int8>(lhs),
      GetTensorShape(output), GetTensorData<int32>(output));
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalInt16(TfLiteContext* context, const OpData* data,
                       const RuntimeShape& lhs_shape, const TfLiteTensor* lhs,
                       const RuntimeShape& rhs_shape, const TfLiteTensor* rhs,
                       const RuntimeShape& output_shape, TfLiteTensor* output) {
  // Reuse params struct from FullyConnected Op.
  FullyConnectedParams op_params;
  int32_t input_offset = -lhs->params.zero_point;
  int32_t filter_offset = -rhs->params.zero_point;
  int32_t output_offset = output->params.zero_point;
  op_params.input_offset = input_offset;
  op_params.weights_offset = filter_offset;
  op_params.output_offset = output_offset;
  op_params.output_multiplier = data->output_multiplier;
  op_params.output_shift = data->output_shift;
  op_params.quantized_activation_min = data->output_activation_min;
  op_params.quantized_activation_max = data->output_activation_max;

  // optimized_ops not yet implemented for int16_t, use reference_ops in all
  // cases.
  reference_ops::BatchMatMul<int16_t, int64_t>(
      op_params, rhs_shape, GetTensorData<int16_t>(rhs), lhs_shape,
      GetTensorData<int16_t>(lhs), GetTensorShape(output),
      GetTensorData<int16_t>(output));
  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           OpData* data, const RuntimeShape& lhs_shape,
                           const TfLiteTensor* lhs,
                           const RuntimeShape& rhs_shape,
                           const TfLiteTensor* rhs, TfLiteTensor* output,
                           bool transpose_lhs) {
  if (lhs->type == kTfLiteFloat32 && rhs->type == kTfLiteInt8) {
    TfLiteTensor* input_quantized;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/2,
                                                &input_quantized));
    TfLiteTensor* scaling_factors;
    TF_LITE_ENSURE_OK(context, GetTemporarySafe(context, node, /*index=*/3,
                                                &scaling_factors));
    TfLiteTensor* accum_scratch;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/4, &accum_scratch));
    TfLiteTensor* input_offsets;
    TF_LITE_ENSURE_OK(
        context, GetTemporarySafe(context, node, /*index=*/5, &input_offsets));
    TfLiteTensor* row_sums;
    TF_LITE_ENSURE_OK(context,
                      GetTemporarySafe(context, node, /*index=*/6, &row_sums));
    return EvalHybrid(context, node, data, lhs_shape, lhs, rhs_shape, rhs,
                      input_quantized, scaling_factors, accum_scratch, row_sums,
                      input_offsets, output);
  } else if (lhs->type == kTfLiteInt8 && rhs->type == kTfLiteInt8) {
    if (output->type == kTfLiteInt8) {
      return EvalInt8Int8<kernel_type>(context, data, lhs_shape, lhs, rhs_shape,
                                       rhs, GetTensorShape(output), output,
                                       transpose_lhs);
    } else {
      return EvalInt8Int32<kernel_type>(context, data, lhs_shape, lhs,
                                        rhs_shape, rhs, GetTensorShape(output),
                                        output);
    }
  } else if (lhs->type == kTfLiteInt16 && rhs->type == kTfLiteInt16) {
    return EvalInt16<kernel_type>(context, data, lhs_shape, lhs, rhs_shape, rhs,
                                  GetTensorShape(output), output);
  } else {
    TF_LITE_KERNEL_LOG(
        context,
        "Currently only hybrid, int8 and int16 quantization are supported.\n");
    return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteTensor* GetTempRhs(TfLiteContext* context, TfLiteNode* node,
                         const TfLiteTensor* rhs) {
  TfLiteTensor* transposed_rhs = GetTemporary(context, node, 1);
  if (transposed_rhs == nullptr) {
    return nullptr;
  }

  TfLiteIntArrayFree(transposed_rhs->dims);
  transposed_rhs->dims = TfLiteIntArrayCopy(rhs->dims);
  std::swap(transposed_rhs->dims->data[transposed_rhs->dims->size - 1],
            transposed_rhs->dims->data[transposed_rhs->dims->size - 2]);
  if (rhs->type == kTfLiteInt8 || rhs->type == kTfLiteInt16) {
    // Get the quantization params from the RHS tensor.
    transposed_rhs->params.scale = rhs->params.scale;
    transposed_rhs->params.zero_point = rhs->params.zero_point;
    if (rhs->quantization.type == kTfLiteAffineQuantization) {
      transposed_rhs->quantization.type = rhs->quantization.type;
      if (transposed_rhs->quantization.params) {
        auto* transposed_rhs_affine_quantization =
            reinterpret_cast<TfLiteAffineQuantization*>(
                transposed_rhs->quantization.params);
        TfLiteIntArrayFree(transposed_rhs_affine_quantization->zero_point);
        TfLiteFloatArrayFree(transposed_rhs_affine_quantization->scale);
        free(transposed_rhs->quantization.params);
      }
      transposed_rhs->quantization.params =
          malloc(sizeof(TfLiteAffineQuantization));
      const auto* rhs_affine_quantization =
          reinterpret_cast<TfLiteAffineQuantization*>(rhs->quantization.params);
      auto* transposed_rhs_affine_quantization =
          reinterpret_cast<TfLiteAffineQuantization*>(
              transposed_rhs->quantization.params);
      int quantized_dimension = rhs_affine_quantization->quantized_dimension;
      if (quantized_dimension == rhs->dims->size - 1) {
        quantized_dimension = rhs->dims->size - 2;
      } else if (quantized_dimension == rhs->dims->size - 2) {
        quantized_dimension = rhs->dims->size - 1;
      }
      transposed_rhs_affine_quantization->quantized_dimension =
          quantized_dimension;
      transposed_rhs_affine_quantization->zero_point =
          TfLiteIntArrayCopy(rhs_affine_quantization->zero_point);
      transposed_rhs_affine_quantization->scale =
          TfLiteFloatArrayCopy(rhs_affine_quantization->scale);
    }
  }
  return transposed_rhs;
}

TfLiteTensor* GetTempLhs(TfLiteContext* context, TfLiteNode* node,
                         const TfLiteTensor* lhs) {
  TfLiteTensor* transposed_lhs = GetTemporary(context, node, 0);
  if (transposed_lhs == nullptr) {
    return nullptr;
  }

  if (lhs->type == kTfLiteInt8 || lhs->type == kTfLiteInt16) {
    // Get the quantization params from the LHS tensor.
    transposed_lhs->params.scale = lhs->params.scale;
    transposed_lhs->params.zero_point = lhs->params.zero_point;
  }
  return transposed_lhs;
}

// Perform a batch matrix multiply on
// LHS <..., A, B>  X  RHS<..., B, C>
// where the leading dimensions of LHS and RHS obey broadcasting rules
// (this Op will apply broadcasting rules).
// We assume that LHS and RHS are both row oriented (adjacent values in memory
// are in the same row) and will output in the same memory layout. However,
// our fast GEMM libraries assume RCC layout (LHS row oriented,
// RHS column oriented, output column oriented). Therefore, we perform
// RHS <..., C, B> X LHS <..., B, A>
// where output is a C X A column-oriented, which is equivalent to
// A X C row-oriented.
template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpContext op_context(context, node);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  const TfLiteTensor* lhs;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputLHSTensor, &lhs));
  const TfLiteTensor* rhs;
  TF_LITE_ENSURE_OK(context,
                    GetInputSafe(context, node, kInputRHSTensor, &rhs));
  TfLiteTensor* output;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, kOutputTensor, &output));
  RuntimeShape orig_lhs_shape = GetTensorShape(lhs);
  RuntimeShape orig_rhs_shape = GetTensorShape(rhs);

  bool adj_y = op_context.params->adj_y;
  bool adj_x = op_context.params->adj_x;

  int32_t rhs_dims_count = orig_rhs_shape.DimensionsCount();
  int32_t lhs_dims_count = orig_lhs_shape.DimensionsCount();
  // Compress ops where rhs shape is [..., 1, X, Y] and lhs shape is
  // [..., Q, R, S] which is equivalent to rhs: [..., X, Y] and
  // lhs: [..., Q * R, S].
  if (rhs_dims_count > 2 && lhs_dims_count > 2) {
    int rhs_one = orig_rhs_shape.DimsData()[rhs_dims_count - 3];
    if (rhs_one == 1) {
      int32_t* lhs_dims = orig_lhs_shape.DimsData();
      int32_t* rhs_dims = orig_rhs_shape.DimsData();
      RuntimeShape tmp_l(lhs_dims_count - 1, lhs_dims);
      tmp_l.SetDim(lhs_dims_count - 3,
                   lhs_dims[lhs_dims_count - 3] * lhs_dims[lhs_dims_count - 2]);
      tmp_l.SetDim(lhs_dims_count - 2, lhs_dims[lhs_dims_count - 1]);
      orig_lhs_shape.ReplaceWith(tmp_l.DimensionsCount(), tmp_l.DimsData());
      RuntimeShape tmp_r(rhs_dims_count - 1, orig_rhs_shape.DimsData());
      tmp_r.SetDim(rhs_dims_count - 3, rhs_dims[rhs_dims_count - 2]);
      tmp_r.SetDim(rhs_dims_count - 2, rhs_dims[rhs_dims_count - 1]);
      orig_rhs_shape.ReplaceWith(tmp_r.DimensionsCount(), tmp_r.DimsData());
    }
  }
  rhs_dims_count = orig_rhs_shape.DimensionsCount();
  lhs_dims_count = orig_lhs_shape.DimensionsCount();
  const TfLiteTensor* rhs_tensor = rhs;
  bool implicit_transpose_possible = true;
  if (lhs->type == kTfLiteFloat32 || kernel_type == kReference ||
      rhs->type == kTfLiteInt16 ||
      (rhs->type == kTfLiteInt8 && output->type == kTfLiteInt32)) {
    implicit_transpose_possible = false;
  }
  bool do_implicit_transpose = !adj_y && implicit_transpose_possible;
  if (!adj_y && !implicit_transpose_possible) {
    rhs_tensor = GetTempRhs(context, node, rhs);
  }
  const TfLiteTensor* lhs_tensor = adj_x ? GetTempLhs(context, node, lhs) : lhs;
  if (!adj_y && !implicit_transpose_possible) {
    // TODO(b/154760341) Constant tensors should already be transposed, but
    // we transpose once if necessary for now.
    if (!(IsConstantTensor(rhs) && op_data->rhs_transposed)) {
      TransposeRowsColumns(context, rhs, GetTemporary(context, node, 1));
      op_data->rhs_transposed = true;
    }
  }
  if (adj_x) {
    TransposeRowsColumns(context, lhs, GetTemporary(context, node, 0));
  }
  RuntimeShape rhs_shape = (adj_y && !do_implicit_transpose)
                               ? orig_rhs_shape
                               : SwapRowColumnDims(orig_rhs_shape);
  RuntimeShape lhs_shape =
      adj_x ? orig_lhs_shape : SwapRowColumnDims(orig_lhs_shape);

  switch (rhs->type) {
    case kTfLiteFloat32:
      // Note we pass RHS args first, LHS args second. See note above.
      reference_ops::BatchMatMul(rhs_shape, GetTensorData<float>(rhs_tensor),
                                 lhs_shape, GetTensorData<float>(lhs_tensor),
                                 GetTensorShape(output),
                                 GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
    case kTfLiteInt16:
      EvalQuantized<kernel_type>(context, node, op_data, lhs_shape, lhs_tensor,
                                 rhs_shape, rhs_tensor, output,
                                 do_implicit_transpose);
      break;
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Currently BatchMatMul doesn't support type: %s",
                         TfLiteTypeGetName(lhs->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace batch_matmul

TfLiteRegistration* Register_BATCH_MATMUL_REF() {
  static TfLiteRegistration r = {batch_matmul::Init, batch_matmul::Free,
                                 batch_matmul::Prepare,
                                 batch_matmul::Eval<batch_matmul::kReference>};
  return &r;
}

TfLiteRegistration* Register_BATCH_MATMUL_GENERIC_OPTIMIZED() {
  static TfLiteRegistration r = {
      batch_matmul::Init, batch_matmul::Free, batch_matmul::Prepare,
      batch_matmul::Eval<batch_matmul::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_BATCH_MATMUL() {
  return Register_BATCH_MATMUL_GENERIC_OPTIMIZED();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
