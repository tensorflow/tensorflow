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

TfLiteStatus ResizeOutputTensor(TfLiteContext* context,
                                const RuntimeShape& extended_lhs_shape,
                                const RuntimeShape& extended_rhs_shape,
                                int output_rank, TfLiteTensor* output) {
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
  output_shape->data[output_rank - 2] =
      extended_lhs_shape.Dims(output_rank - 2);
  output_shape->data[output_rank - 1] =
      extended_rhs_shape.Dims(output_rank - 1);
  TfLiteStatus stat = context->ResizeTensor(context, output, output_shape);
  return stat;
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 1);
  const TfLiteTensor* lhs_data = GetInput(context, node, kInputLHSTensor);
  const TfLiteTensor* rhs_data = GetInput(context, node, kInputRHSTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  TF_LITE_ENSURE_EQ(context, lhs_data->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, rhs_data->type, kTfLiteFloat32);
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
  TF_LITE_ENSURE_EQ(context, extended_lhs_shape.Dims(output_rank - 1),
                    extended_rhs_shape.Dims(output_rank - 2));
  return ResizeOutputTensor(context, extended_lhs_shape, extended_rhs_shape,
                            output_rank, output);
}

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* lhs = GetInput(context, node, kInputLHSTensor);
  const TfLiteTensor* rhs = GetInput(context, node, kInputRHSTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  switch (lhs->type) {
    case kTfLiteFloat32:
      if (kernel_type == kGenericOptimized) {
        optimized_ops::BatchMatMul(
            GetTensorShape(lhs), GetTensorData<float>(lhs), GetTensorShape(rhs),
            GetTensorData<float>(rhs), GetTensorShape(output),
            GetTensorData<float>(output),
            CpuBackendContext::GetFromContext(context));
      } else {
        reference_ops::BatchMatMul(
            GetTensorShape(lhs), GetTensorData<float>(lhs), GetTensorShape(rhs),
            GetTensorData<float>(rhs), GetTensorShape(output),
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
  static TfLiteRegistration r = {nullptr, nullptr, batch_matmul::Prepare,
                                 batch_matmul::Eval<batch_matmul::kReference>};
  return &r;
}

TfLiteRegistration* Register_BATCH_MATMUL_GENERIC_OPTIMIZED() {
  static TfLiteRegistration r = {
      nullptr, nullptr, batch_matmul::Prepare,
      batch_matmul::Eval<batch_matmul::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_BATCH_MATMUL() {
  return Register_BATCH_MATMUL_GENERIC_OPTIMIZED();
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
