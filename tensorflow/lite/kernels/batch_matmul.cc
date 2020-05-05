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

#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/optimized/batch_matmul.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace batch_matmul {

static const int kInputLHSTensor = 0;
static const int kInputRHSTensor = 1;
static const int kOutputTensor = 0;

// This file has two implementations of Transpose.
enum KernelType {
  kReference,
  kGenericOptimized,
};

struct OpData {
  // The index of the temporary tensors where we store transposed LHS/RHS.
  int scratch_tensor_index;
  bool rhs_transposed;
};

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    params = reinterpret_cast<TfLiteBatchMatMulParams*>(node->builtin_data);
    lhs = GetInput(context, node, 0);
    rhs = GetInput(context, node, 1);
    output = GetOutput(context, node, 0);
  }
  TfLiteBatchMatMulParams* params;
  const TfLiteTensor* lhs;
  const TfLiteTensor* rhs;
  TfLiteTensor* output;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // Creates two temp tensors to store the transposed LHS and/or RHS if
  // needed.
  auto* op_data = new OpData();
  // If the RHS is constant, we only transpose once.
  op_data->rhs_transposed = false;
  context->AddTensors(context, 2, &op_data->scratch_tensor_index);
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
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(2);
  node->temporaries->data[0] = op_data->scratch_tensor_index;
  node->temporaries->data[1] = op_data->scratch_tensor_index + 1;
  // Temp tensor for Transposed LHS;
  if (op_context->params->adj_x) {
    TfLiteTensor* scratch_buffer = GetTemporary(context, node, /*index=*/0);
    const TfLiteTensor* lhs = op_context->lhs;
    int lhs_rank = NumDimensions(lhs);
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
  const bool rhs_needs_temp = !(op_context->params->adj_y);
  if (rhs_needs_temp) {
    TfLiteTensor* scratch_buffer = GetTemporary(context, node, /*index=*/1);
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
    scratch_buffer->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_buffer,
                                                     scratch_buffer_size));
  }
  return kTfLiteOk;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);

  OpContext op_context(context, node);
  TF_LITE_ENSURE_OK(context, InitializeTemporaries(context, node, &op_context));

  bool adj_x = op_context.params->adj_x;
  bool adj_y = op_context.params->adj_y;

  const TfLiteTensor* lhs_data = GetInput(context, node, kInputLHSTensor);
  const TfLiteTensor* rhs_data = GetInput(context, node, kInputRHSTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, lhs_data->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, rhs_data->type, kTfLiteFloat32);
  // Support dimensions between 2 and 4, inclusive.
  TF_LITE_ENSURE(context, NumDimensions(lhs_data) >= 2);
  TF_LITE_ENSURE(context, NumDimensions(lhs_data) <= 4);
  TF_LITE_ENSURE(context, NumDimensions(rhs_data) >= 2);
  TF_LITE_ENSURE(context, NumDimensions(rhs_data) <= 4);

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
void TransposeRowsColumns(const TfLiteTensor* tensor_in, const scalar* input,
                          TfLiteTensor* tensor_out, scalar* output) {
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

RuntimeShape SwapRowColumnDims(const RuntimeShape& shape) {
  RuntimeShape swapped_shape(shape);
  const int32_t dims = shape.DimensionsCount();
  swapped_shape.SetDim(dims - 2, shape.Dims(dims - 1));
  swapped_shape.SetDim(dims - 1, shape.Dims(dims - 2));
  return swapped_shape;
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
  const TfLiteTensor* lhs = GetInput(context, node, kInputLHSTensor);
  const TfLiteTensor* rhs = GetInput(context, node, kInputRHSTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  RuntimeShape orig_lhs_shape = GetTensorShape(lhs);
  RuntimeShape orig_rhs_shape = GetTensorShape(rhs);

  bool adj_y = op_context.params->adj_y;
  bool adj_x = op_context.params->adj_x;

  const TfLiteTensor* rhs_tensor = adj_y ? rhs : GetTemporary(context, node, 1);
  const TfLiteTensor* lhs_tensor = adj_x ? GetTemporary(context, node, 0) : lhs;
  if (!adj_y) {
    // TODO(b/154760341) Constant tensors should already be transposed, but
    // we transpose once if necessary for now.
    if (!(IsConstantTensor(rhs) && op_data->rhs_transposed)) {
      TransposeRowsColumns<float>(
          rhs, GetTensorData<float>(rhs), GetTemporary(context, node, 1),
          GetTensorData<float>(GetTemporary(context, node, 1)));
      op_data->rhs_transposed = true;
    }
  }
  if (adj_x) {
    TransposeRowsColumns<float>(
        lhs, GetTensorData<float>(lhs), GetTemporary(context, node, 0),
        GetTensorData<float>(GetTemporary(context, node, 0)));
  }
  RuntimeShape rhs_shape =
      adj_y ? orig_rhs_shape : SwapRowColumnDims(orig_rhs_shape);
  RuntimeShape lhs_shape =
      adj_x ? orig_lhs_shape : SwapRowColumnDims(orig_lhs_shape);

  switch (lhs->type) {
    case kTfLiteFloat32:
      // Note we pass RHS args first, LHS args second. See note above.
      if (kernel_type == kGenericOptimized) {
        optimized_ops::BatchMatMul(rhs_shape, GetTensorData<float>(rhs_tensor),
                                   lhs_shape, GetTensorData<float>(lhs_tensor),
                                   GetTensorShape(output),
                                   GetTensorData<float>(output),
                                   CpuBackendContext::GetFromContext(context));
      } else {
        reference_ops::BatchMatMul(rhs_shape, GetTensorData<float>(rhs_tensor),
                                   lhs_shape, GetTensorData<float>(lhs_tensor),
                                   GetTensorShape(output),
                                   GetTensorData<float>(output));
      }
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
