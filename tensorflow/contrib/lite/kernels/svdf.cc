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
#include <unistd.h>
#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace svdf {

constexpr int kInputTensor = 0;
constexpr int kWeightsFeatureTensor = 1;
constexpr int kWeightsTimeTensor = 2;
constexpr int kBiasTensor = 3;
constexpr int kStateTensor = 0;
constexpr int kOutputTensor = 1;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* scratch_tensor_index = new int;
  context->AddTensors(context, 1, scratch_tensor_index);
  return scratch_tensor_index;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<int*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  int* scratch_tensor_index = reinterpret_cast<int*>(node->user_data);

  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 4);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 2);

  TfLiteTensor* input = &context->tensors[node->inputs->data[kInputTensor]];
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ASSERT_EQ(num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];
  TF_LITE_ASSERT_EQ(input->dims->data[1], weights_feature->dims->data[1]);
  TF_LITE_ASSERT_EQ(weights_time->dims->data[0], num_filters);

  TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  if (bias) {
    TF_LITE_ASSERT_EQ(bias->dims->data[0], num_units);
  }

  TfLiteTensor* state = GetOutput(context, node, kStateTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Resize state.
  // For each batch, the state is a 2-D tensor: memory_size * num_filters
  // The left most column is used to save current cycle activation.
  // The right most column is used to save temporary output which will be
  // reduced to num_units outputs.
  TfLiteIntArray* state_size_array = TfLiteIntArrayCreate(2);
  state_size_array->data[0] = batch_size;
  state_size_array->data[1] = memory_size * num_filters;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, state, state_size_array));

  // Mark state as a persistent tensor.
  state->allocation_type = kTfLiteArenaRwPersistent;

  // Resize output.
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(2);
  output_size_array->data[0] = batch_size;
  output_size_array->data[1] = num_units;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));

  // Resize scratch.
  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(1);
  node->temporaries->data[0] = *scratch_tensor_index;

  TfLiteIntArray* scratch_size_array = TfLiteIntArrayCreate(2);
  scratch_size_array->data[0] = batch_size;
  scratch_size_array->data[1] = num_filters;

  TfLiteTensor* scratch_tensor = GetTemporary(context, node, /*index=*/0);
  scratch_tensor->type = input->type;
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_tensor,
                                                   scratch_size_array));

  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);

  TfLiteTensor* state = GetOutput(context, node, kStateTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TfLiteTensor* scratch = GetTemporary(context, node, /*index=*/0);

  TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);

  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int input_size = input->dims->data[1];
  const int num_filters = weights_feature->dims->data[0];
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];

  // Clear the activation (state left most column).
  // TODO(ghodrat): Add a test which initialize state with invalid values in
  // left most column and make sure it passes.
  for (int b = 0; b < batch_size; b++) {
    float* state_ptr_batch = state->data.f + b * memory_size * num_filters;
    for (int c = 0; c < num_filters; c++) {
      float* state_ptr = state_ptr_batch + c * memory_size;
      state_ptr[memory_size - 1] = 0.0;
    }
  }

  // Compute conv1d(inputs, weights_feature).
  // The state left most column is used to save current cycle activation. This
  // is achieved by starting at state->data.f[memory_size - 1] and having the
  // stride equal to memory_size.
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      weights_feature->data.f, num_filters, input_size, input->data.f,
      batch_size, &state->data.f[memory_size - 1], memory_size);

  // Compute matmul(state, weights_time).
  // The right most column is used to save temporary output (with the size of
  // num_filters). This is achieved by starting at state->data.f and having the
  // stride equal to memory_size.
  for (int b = 0; b < batch_size; b++) {
    float* state_ptr_batch = state->data.f + b * memory_size * num_filters;
    float* scratch_ptr_batch = scratch->data.f + b * num_filters;
    tensor_utils::BatchVectorBatchVectorDotProduct(
        weights_time->data.f, state_ptr_batch, memory_size, num_filters,
        scratch_ptr_batch, /*result_stride=*/1);
  }

  // Initialize output with bias if provided.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(bias->data.f, num_units, batch_size,
                                          output->data.f);
  } else {
    tensor_utils::ZeroVector(output->data.f, batch_size * num_units);
  }

  // Reduction sum
  for (int b = 0; b < batch_size; b++) {
    float* output_ptr_batch = output->data.f + b * num_units;
    float* scratch_ptr_batch = scratch->data.f + b * num_filters;
    tensor_utils::ReductionSumVector(scratch_ptr_batch, output_ptr_batch,
                                     num_units, rank);
  }

  // Apply activation.
  for (int b = 0; b < batch_size; b++) {
    float* output_ptr_batch = output->data.f + b * num_units;
    tensor_utils::ApplyActivationToVector(output_ptr_batch, num_units,
                                          params->activation, output_ptr_batch);
  }

  // Right shift the state.
  for (int b = 0; b < batch_size; b++) {
    float* state_ptr_batch = state->data.f + b * memory_size * num_filters;
    for (int f = 0; f < num_filters; f++) {
      tensor_utils::VectorShiftLeft(state_ptr_batch, memory_size,
                                    /*shift_value=*/0.0);
      state_ptr_batch += memory_size;
    }
  }
  return kTfLiteOk;
}

}  // namespace svdf

TfLiteRegistration* Register_SVDF() {
  static TfLiteRegistration r = {svdf::Init, svdf::Free, svdf::Prepare,
                                 svdf::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
