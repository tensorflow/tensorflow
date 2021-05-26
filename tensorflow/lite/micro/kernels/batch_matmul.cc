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

#include "tensorflow/lite/kernels/internal/reference/batch_matmul.h"

#include <algorithm>
#include <cstdint>
#include <limits>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/process_broadcast_shapes.h"
#include "tensorflow/lite/kernels/internal/reference/transpose.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/micro/kernels/kernel_util.h"

namespace tflite {
namespace {

constexpr int kInputLHSTensor = 0;
constexpr int kInputRHSTensor = 1;
constexpr int kOutputTensor = 0;

constexpr int kInvalidScratchBufferIndex = -1;

struct QuantizationOpData {
  // The scaling factor from input to output (aka the 'real multiplier') can
  // be represented as a fixed point multiplier plus a left shift.
  int32_t output_multiplier;
  int output_shift;  // exponent

  // The range of the fused activation layer. For example for kNone and
  // int8_t these would be -128 and 127.
  int32_t output_activation_min;
  int32_t output_activation_max;

  int32_t lhs_zero_point;
  int32_t rhs_zero_point;
  int32_t output_zero_point;
};

struct HybridOpData {
  float filter_scale;  // RHS tensor scale

  // scratch buffer indices
  int input_quantized_index;
  int scaling_factors_index;
  int input_offsets_index;

  // row_sums_buffer may be re-used across eval calls
  int32_t* row_sums_buffer;

  bool compute_row_sums;
};

struct OpData {
  union {
    QuantizationOpData* quantization;
    HybridOpData* hybrid;
  };

  // Transpose tensors and state
  TfLiteEvalTensor* lhs_transposed_tensor;
  TfLiteEvalTensor* rhs_transposed_tensor;
  bool rhs_is_transposed;
  bool lhs_is_constant_tensor;
  bool rhs_is_constant_tensor;
};

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    params = static_cast<TfLiteBatchMatMulParams*>(node->builtin_data);
    opdata = static_cast<OpData*>(node->user_data);
  }

  TfLiteBatchMatMulParams* params;
  OpData* opdata;
};

struct PrepareOpContext : OpContext {
  PrepareOpContext(TfLiteContext* context, TfLiteNode* node)
      : OpContext(context, node) {
    lhs = GetInput(context, node, kInputLHSTensor);
    rhs = GetInput(context, node, kInputRHSTensor);
    output = GetOutput(context, node, kOutputTensor);
  }

  const TfLiteTensor* lhs;
  const TfLiteTensor* rhs;
  TfLiteTensor* output;
};

struct EvalOpContext : OpContext {
  EvalOpContext(TfLiteContext* context, TfLiteNode* node)
      : OpContext(context, node) {
    lhs = tflite::micro::GetEvalInput(context, node, kInputLHSTensor);
    rhs = tflite::micro::GetEvalInput(context, node, kInputRHSTensor);
    output = tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  }

  const TfLiteEvalTensor* lhs;
  const TfLiteEvalTensor* rhs;
  TfLiteEvalTensor* output;
};

TfLiteStatus ResizeOutputTensor(TfLiteContext* context, TfLiteNode* node,
                                const RuntimeShape& extended_lhs_shape,
                                const RuntimeShape& extended_rhs_shape,
                                bool adj_x, bool adj_y, int output_rank,
                                TfLiteTensor* output) {
  auto orig_size = NumElements(output);

  // make sure output tensor dims are not in the FlatBuffer
  TfLiteEvalTensor* output_eval =
      tflite::micro::GetEvalOutput(context, node, kOutputTensor);
  TF_LITE_ENSURE_OK(context, tflite::micro::CreateWritableTensorDimsWithCopy(
                                 context, output, output_eval));

  // Fill in any broadcast dimensions.
  for (int i = 0; i < output_rank - 2; ++i) {
    const int lhs_dim = extended_lhs_shape.Dims(i);
    const int rhs_dim = extended_rhs_shape.Dims(i);
    int broadcast_dim = lhs_dim;
    if ((lhs_dim != rhs_dim) && (lhs_dim == 1)) {
      broadcast_dim = rhs_dim;
    }
    output->dims->data[i] = broadcast_dim;
  }
  // Fill in the matmul dimensions.
  int lhs_rows_index = adj_x ? output_rank - 1 : output_rank - 2;
  int rhs_cols_index = adj_y ? output_rank - 2 : output_rank - 1;

  output->dims->data[output_rank - 2] = extended_lhs_shape.Dims(lhs_rows_index);
  output->dims->data[output_rank - 1] = extended_rhs_shape.Dims(rhs_cols_index);
  output->dims->size = output_rank;

  // Check that output tensor has not been resized
  // since TFLM doesn't support tensor resizing.
  TF_LITE_ENSURE_EQ(context, orig_size, NumElements(output));

  return kTfLiteOk;
}

TfLiteEvalTensor* AllocInitTransposeTensorFromTfLiteTensor(
    TfLiteContext* context, const TfLiteTensor& tensor) {
  TfLiteEvalTensor* eval_tensor = static_cast<TfLiteEvalTensor*>(
      context->AllocatePersistentBuffer(context, sizeof(TfLiteEvalTensor)));

  eval_tensor->type = tensor.type;

  const int tensor_rank = NumDimensions(&tensor);
  auto eval_dims_size = TfLiteIntArrayGetSizeInBytes(tensor_rank);
  eval_tensor->dims = static_cast<TfLiteIntArray*>(
      context->AllocatePersistentBuffer(context, eval_dims_size));
  eval_tensor->dims->size = tensor_rank;
  for (int i = 0; i < tensor_rank - 2; ++i) {
    eval_tensor->dims->data[i] = tensor.dims->data[i];
  }
  // Swap last two dimensions.
  eval_tensor->dims->data[tensor_rank - 2] = tensor.dims->data[tensor_rank - 1];
  eval_tensor->dims->data[tensor_rank - 1] = tensor.dims->data[tensor_rank - 2];

  size_t eval_data_size = static_cast<size_t>(NumElements(&tensor));
  if (tensor.type == kTfLiteFloat32) {
    eval_data_size *= sizeof(float);
  }
  eval_tensor->data.data =
      context->AllocatePersistentBuffer(context, eval_data_size);

  return eval_tensor;
}

// Initializes tensors to store transposed operands.
// Allocate storage for hybrid quantization if needed.
// Allocate normal quantization data if needed.
TfLiteStatus InitializeTemporaries(TfLiteContext* context, TfLiteNode* node,
                                   const PrepareOpContext& op_context) {
  OpData* op_data = op_context.opdata;
  const TfLiteTensor* lhs = op_context.lhs;
  const TfLiteTensor* rhs = op_context.rhs;

  // For "hybrid" quantization, we impose the constraint that the LHS
  // is float (typically an activation from a prior layer) and the RHS
  // is quantized int8.
  bool is_hybrid = (lhs->type == kTfLiteFloat32 && rhs->type == kTfLiteInt8);
  if (is_hybrid) {
    op_data->hybrid = static_cast<decltype(op_data->hybrid)>(
        context->AllocatePersistentBuffer(context, sizeof(*op_data->hybrid)));
    TF_LITE_ENSURE(context, op_data->hybrid != nullptr);
    op_data->hybrid->input_quantized_index = kInvalidScratchBufferIndex;
    op_data->hybrid->scaling_factors_index = kInvalidScratchBufferIndex;
    op_data->hybrid->row_sums_buffer = nullptr;
    op_data->hybrid->input_offsets_index = kInvalidScratchBufferIndex;
  } else if (lhs->type == kTfLiteInt8) {
    op_data->quantization = static_cast<decltype(op_data->quantization)>(
        context->AllocatePersistentBuffer(context,
                                          sizeof(*op_data->quantization)));
    TF_LITE_ENSURE(context, op_data->quantization != nullptr);
  } else {
    op_data->quantization = nullptr;  // also op_data->hybrid
  }

  // tensor for Transposed LHS;
  if (op_context.params->adj_x) {
    op_data->lhs_transposed_tensor =
        AllocInitTransposeTensorFromTfLiteTensor(context, *lhs);
  } else {
    op_data->lhs_transposed_tensor = nullptr;
  }

  // We need a buffer for the RHS if we need to transpose the RHS. We
  // transpose by default, so that the two inputs (LHS and RHS) are in a proper
  // layout for our fast matrix multiplication routines. If the transpose flag
  // is set by the caller, the data is already in the desired layout.
  if (!op_context.params->adj_y) {
    op_data->rhs_transposed_tensor =
        AllocInitTransposeTensorFromTfLiteTensor(context, *rhs);
  } else {
    op_data->rhs_transposed_tensor = nullptr;
  }

  // If we have to perform on-the-fly quantization (with quantized weights and
  // float inputs) first we need to quantize the inputs. Allocate temporary
  // buffer to store the intermediate quantized values, the batch scaling
  // factors, the input offsets, and persistent storage for the sums of the
  // rows for each weights matrix.
  // RHS = weights, LHS = inputs
  if (is_hybrid) {
    const int lhs_rank = NumDimensions(lhs);
    const int rhs_rank = NumDimensions(rhs);
    const int batch_size = op_context.params->adj_x
                               ? lhs->dims->data[lhs_rank - 1]
                               : lhs->dims->data[lhs_rank - 2];
    const int num_units = rhs->dims->data[rhs_rank - 1];

    // Calculate the total number of LHS batches.
    int num_batches = 1;
    for (int i = 0; i < lhs_rank - 2; ++i) {
      num_batches *= lhs->dims->data[i];
    }
    int num_weights_matrices = 1;
    for (int i = 0; i < rhs_rank - 2; ++i) {
      num_weights_matrices *= rhs->dims->data[i];
    }

    const size_t input_quantized_size = static_cast<size_t>(
        NumElements(lhs->dims) * TfLiteTypeGetSize(rhs->type));
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                   context, input_quantized_size,
                                   &op_data->hybrid->input_quantized_index));

    const size_t scaling_factors_size =
        static_cast<size_t>(batch_size * num_batches * sizeof(float));
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                   context, scaling_factors_size,
                                   &op_data->hybrid->scaling_factors_index));

    const size_t input_offsets_size =
        static_cast<size_t>(batch_size * num_batches * sizeof(int32_t));
    TF_LITE_ENSURE_OK(context, context->RequestScratchBufferInArena(
                                   context, input_offsets_size,
                                   &op_data->hybrid->input_offsets_index));

    const size_t row_sums_size =
        static_cast<size_t>(num_weights_matrices * num_units * sizeof(int32_t));
    op_data->hybrid->row_sums_buffer = static_cast<int32_t*>(
        context->AllocatePersistentBuffer(context, row_sums_size));
    TF_LITE_ENSURE(context, op_data->hybrid->row_sums_buffer != nullptr);

    op_data->hybrid->compute_row_sums = true;
    op_data->hybrid->filter_scale = rhs->params.scale;
  }

  return kTfLiteOk;
}

template <typename scalar>
void TransposeRowsColumnsImpl(const TfLiteEvalTensor& tensor_in,
                              const scalar* input, TfLiteEvalTensor* tensor_out,
                              scalar* output) {
  RuntimeShape transposed_shape(tflite::micro::GetTensorShape(&tensor_in));
  RuntimeShape shape(transposed_shape);
  TransposeParams params;
  int rank = shape.DimensionsCount();
  params.perm_count = rank;
  for (int i = 0; i < rank - 2; ++i) {
    params.perm[i] = i;
  }
  // Transpose the last two dimensions.
  params.perm[rank - 2] = rank - 1;
  params.perm[rank - 1] = rank - 2;
  transposed_shape.SetDim(rank - 1, shape.Dims(rank - 2));
  transposed_shape.SetDim(rank - 2, shape.Dims(rank - 1));
  reference_ops::Transpose(params, shape, input, transposed_shape, output);
}

TfLiteStatus TransposeRowsColumns(TfLiteContext* context,
                                  const TfLiteEvalTensor& tensor_in,
                                  TfLiteEvalTensor* tensor_out) {
  if (tensor_in.type == kTfLiteFloat32) {
    TransposeRowsColumnsImpl<float>(
        tensor_in, tflite::micro::GetTensorData<float>(&tensor_in), tensor_out,
        tflite::micro::GetTensorData<float>(tensor_out));
    return kTfLiteOk;
  } else if (tensor_in.type == kTfLiteInt8) {
    TransposeRowsColumnsImpl<int8_t>(
        tensor_in, tflite::micro::GetTensorData<int8_t>(&tensor_in), tensor_out,
        tflite::micro::GetTensorData<int8_t>(tensor_out));
    return kTfLiteOk;
  } else {
    TF_LITE_KERNEL_LOG(context,
                       "BATCH_MATMUL can only transpose tensors with float, "
                       "int8 type.");
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

TfLiteStatus CalculateOpData(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  PrepareOpContext op_context(context, node);
  const TfLiteTensor* lhs_data = op_context.lhs;
  TF_LITE_ENSURE(context, lhs_data != nullptr);
  const TfLiteTensor* rhs_data = op_context.rhs;
  TF_LITE_ENSURE(context, rhs_data != nullptr);
  TfLiteTensor* output = op_context.output;
  TF_LITE_ENSURE(context, output != nullptr);

  TF_LITE_ENSURE(context, lhs_data->type == kTfLiteFloat32 ||
                              lhs_data->type == kTfLiteInt8);
  TF_LITE_ENSURE(context, rhs_data->type == kTfLiteFloat32 ||
                              rhs_data->type == kTfLiteInt8);
  // Either we have a hybrid quantization with a float32 and an int8 input,
  // otherwise both inputs should be of the same type.
  TF_LITE_ENSURE(context, (lhs_data->type == kTfLiteFloat32 &&
                           rhs_data->type == kTfLiteInt8) ||
                              lhs_data->type == rhs_data->type);

  const int lhs_rank = NumDimensions(lhs_data);
  const int rhs_rank = NumDimensions(rhs_data);
  // Support dimensions between 2 and 4, inclusive.
  TF_LITE_ENSURE(context, lhs_rank >= 2);
  TF_LITE_ENSURE(context, lhs_rank <= 4);
  TF_LITE_ENSURE(context, rhs_rank >= 2);
  TF_LITE_ENSURE(context, rhs_rank <= 4);

  TF_LITE_ENSURE_OK(context, InitializeTemporaries(context, node, op_context));

  OpData* op_data = op_context.opdata;
  // If the RHS is constant, we only transpose once.
  op_data->rhs_is_transposed = false;
  op_data->lhs_is_constant_tensor = IsConstantTensor(lhs_data);
  op_data->rhs_is_constant_tensor = IsConstantTensor(rhs_data);

  bool adj_x = op_context.params->adj_x;
  bool adj_y = op_context.params->adj_y;

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  if (lhs_data->type == kTfLiteInt8) {
    TF_LITE_ENSURE(context, op_data->quantization != nullptr);
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, lhs_data, rhs_data, output, &real_multiplier));
    QuantizeMultiplier(real_multiplier,
                       &op_data->quantization->output_multiplier,
                       &op_data->quantization->output_shift);
    // BatchMatMul has no fused activation functions. Therefore, set
    // output activation min and max to min and max of int8_t type.
    op_data->quantization->output_activation_min =
        std::numeric_limits<int8_t>::min();
    op_data->quantization->output_activation_max =
        std::numeric_limits<int8_t>::max();

    // set zero_point for Int8 only
    op_data->quantization->lhs_zero_point = lhs_data->params.zero_point;
    op_data->quantization->rhs_zero_point = rhs_data->params.zero_point;
    op_data->quantization->output_zero_point = output->params.zero_point;
  }

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
      ResizeOutputTensor(context, node, extended_lhs_shape, extended_rhs_shape,
                         adj_x, adj_y, output_rank, output);
  return status;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  TFLITE_DCHECK(context->AllocatePersistentBuffer != nullptr);
  return context->AllocatePersistentBuffer(context, sizeof(OpData));
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  return CalculateOpData(context, node);
}

TfLiteStatus EvalHybrid(TfLiteContext* context, TfLiteNode* node,
                        const OpData& data, const RuntimeShape& input_shape,
                        const TfLiteEvalTensor& input,
                        const RuntimeShape& filter_shape,
                        const TfLiteEvalTensor& filter,
                        TfLiteEvalTensor* output) {
  const auto* params =
      static_cast<TfLiteBatchMatMulParams*>(node->builtin_data);
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
  float* scaling_factors_ptr = static_cast<float*>(
      context->GetScratchBuffer(context, data.hybrid->scaling_factors_index));
  int32_t* input_offset_ptr = static_cast<int32_t*>(
      context->GetScratchBuffer(context, data.hybrid->input_offsets_index));
  int32_t* row_sums_ptr = data.hybrid->row_sums_buffer;
  if (!params->asymmetric_quantize_inputs) {
    std::fill_n(input_offset_ptr, num_batches_to_quantize, 0);
  }

  int8_t* quant_data = static_cast<int8_t*>(
      context->GetScratchBuffer(context, data.hybrid->input_quantized_index));
  const int8_t* filter_data = tflite::micro::GetTensorData<int8_t>(&filter);
  const float* input_ptr = tflite::micro::GetTensorData<float>(&input);
  // Quantize each batch independently.
  tensor_utils::BatchQuantizeFloats(input_ptr, num_batches_to_quantize,
                                    input_size, quant_data, scaling_factors_ptr,
                                    input_offset_ptr,
                                    params->asymmetric_quantize_inputs);
  for (int b = 0; b < num_batches_to_quantize; ++b) {
    // Incorporate scaling of the filter.
    scaling_factors_ptr[b] *= data.hybrid->filter_scale;
  }

  RuntimeShape output_shape = tflite::micro::GetTensorShape(output);
  int output_size = NumElements(output->dims);
  std::fill_n(tflite::micro::GetTensorData<float>(output), output_size, 0.0f);
  reference_ops::BatchMatMul(
      filter_shape, filter_data, input_shape, quant_data, scaling_factors_ptr,
      input_offset_ptr, row_sums_ptr, tflite::micro::GetTensorShape(output),
      tflite::micro::GetTensorData<float>(output),
      &(data.hybrid->compute_row_sums));

  return kTfLiteOk;
}

TfLiteStatus EvalInt8(TfLiteContext* context, const OpData& data,
                      const RuntimeShape& lhs_shape,
                      const TfLiteEvalTensor& lhs,
                      const RuntimeShape& rhs_shape,
                      const TfLiteEvalTensor& rhs,
                      const RuntimeShape& output_shape,
                      TfLiteEvalTensor* output) {
  TF_LITE_ENSURE(context, data.quantization != nullptr);

  // Reuse params struct from FullyConnected Op.
  FullyConnectedParams op_params;
  op_params.input_offset = -data.quantization->lhs_zero_point;
  op_params.weights_offset =
      -data.quantization->rhs_zero_point;  // filter offset
  op_params.output_offset = data.quantization->output_zero_point;
  op_params.output_multiplier = data.quantization->output_multiplier;
  op_params.output_shift = data.quantization->output_shift;
  op_params.quantized_activation_min = data.quantization->output_activation_min;
  op_params.quantized_activation_max = data.quantization->output_activation_max;
  op_params.lhs_cacheable = data.lhs_is_constant_tensor;
  op_params.rhs_cacheable = data.rhs_is_constant_tensor;

  // Note we pass RHS args first, LHS args second. See note for Eval.
  reference_ops::BatchMatMul<int8_t, int32_t>(
      op_params, rhs_shape, tflite::micro::GetTensorData<int8_t>(&rhs),
      lhs_shape, tflite::micro::GetTensorData<int8_t>(&lhs), output_shape,
      tflite::micro::GetTensorData<int8_t>(output));

  return kTfLiteOk;
}

TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           const OpData& data, const RuntimeShape& lhs_shape,
                           const TfLiteEvalTensor& lhs,
                           const RuntimeShape& rhs_shape,
                           const TfLiteEvalTensor& rhs,
                           TfLiteEvalTensor* output) {
  if (lhs.type == kTfLiteFloat32 && rhs.type == kTfLiteInt8) {
    TF_LITE_ENSURE(context, data.hybrid != nullptr);
    TF_LITE_ENSURE(context, data.hybrid->row_sums_buffer != nullptr);
    TF_LITE_ENSURE(context, data.hybrid->input_quantized_index !=
                                kInvalidScratchBufferIndex);
    TF_LITE_ENSURE(context, data.hybrid->scaling_factors_index !=
                                kInvalidScratchBufferIndex);
    TF_LITE_ENSURE(context, data.hybrid->input_offsets_index !=
                                kInvalidScratchBufferIndex);
    return EvalHybrid(context, node, data, lhs_shape, lhs, rhs_shape, rhs,
                      output);
  } else if (lhs.type == kTfLiteInt8 && rhs.type == kTfLiteInt8) {
    return EvalInt8(context, data, lhs_shape, lhs, rhs_shape, rhs,
                    tflite::micro::GetTensorShape(output), output);
  } else {
    TF_LITE_KERNEL_LOG(
        context, "BATCH_MATMUL only supports hybrid, int8 quantization.\n");
  }
  return kTfLiteError;
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
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  EvalOpContext op_context(context, node);
  OpData* op_data = op_context.opdata;
  const TfLiteEvalTensor* lhs = op_context.lhs;
  const TfLiteEvalTensor* rhs = op_context.rhs;
  TfLiteEvalTensor* output = op_context.output;
  RuntimeShape orig_lhs_shape = tflite::micro::GetTensorShape(lhs);
  RuntimeShape orig_rhs_shape = tflite::micro::GetTensorShape(rhs);

  bool adj_y = op_context.params->adj_y;
  bool adj_x = op_context.params->adj_x;

  TfLiteEvalTensor* rhs_tensor = adj_y ? const_cast<TfLiteEvalTensor*>(rhs)
                                       : op_data->rhs_transposed_tensor;
  TfLiteEvalTensor* lhs_tensor = adj_x ? op_data->lhs_transposed_tensor
                                       : const_cast<TfLiteEvalTensor*>(lhs);
  TF_LITE_ENSURE(context, rhs_tensor != nullptr);
  TF_LITE_ENSURE(context, lhs_tensor != nullptr);
  if (!adj_y) {
    // OLD-TODO(b/154760341) Constant tensors should already be transposed, but
    // we transpose once if necessary for now.
    if (!(op_data->rhs_is_constant_tensor && op_data->rhs_is_transposed)) {
      TransposeRowsColumns(context, *rhs, rhs_tensor);
      op_data->rhs_is_transposed = true;
    }
  }
  if (adj_x) {
    TransposeRowsColumns(context, *lhs, lhs_tensor);
  }
  RuntimeShape rhs_shape =
      adj_y ? orig_rhs_shape : SwapRowColumnDims(orig_rhs_shape);
  RuntimeShape lhs_shape =
      adj_x ? orig_lhs_shape : SwapRowColumnDims(orig_lhs_shape);

  switch (rhs->type) {
    case kTfLiteFloat32:
      // Note we pass RHS args first, LHS args second. See note above.
      reference_ops::BatchMatMul(
          rhs_shape, tflite::micro::GetTensorData<float>(rhs_tensor), lhs_shape,
          tflite::micro::GetTensorData<float>(lhs_tensor),
          tflite::micro::GetTensorShape(output),
          tflite::micro::GetTensorData<float>(output));
      break;
    case kTfLiteInt8:
      return EvalQuantized(context, node, *op_data, lhs_shape, *lhs_tensor,
                           rhs_shape, *rhs_tensor, output);
    default:
      TF_LITE_KERNEL_LOG(context,
                         "Currently BATCH_MATMUL doesn't support type: %s",
                         TfLiteTypeGetName(lhs->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace

TfLiteRegistration Register_BATCH_MATMUL() {
  return {/*init=*/Init,
          /*free=*/nullptr,
          /*prepare=*/Prepare,
          /*invoke=*/Eval,
          /*profiling_string=*/nullptr,
          /*builtin_code=*/0,
          /*custom_name=*/nullptr,
          /*version=*/0};
}

}  // namespace tflite
