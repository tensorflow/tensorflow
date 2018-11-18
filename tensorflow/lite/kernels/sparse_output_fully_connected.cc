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
// SparseOutputFullyConnected is a fully connected layer that uses a single
// row in the weights and bias via a lookup.
#include "tensorflow/lite/context.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace sparse_output_fully_connected {

// Input tensors of size {n_batch, n_input}
constexpr int kInputTensor = 0;
// Auxiliary input tensor of size { 1 }
constexpr int kInputLookupTensor = 1;

// Weights tensor of size { n_embeddings , n_input }
constexpr int kWeightsTensor = 2;
// Bias tensor of size { n_embeddings }
constexpr int kBiasTensor = 3;

// Output tensor.
constexpr int kOutputTensor = 0;

// Temporary tensors.
enum TemporaryTensor {
  kInputQuantized = 0,
  kScalingFactors = 1,
  kNumTemporaryTensors = 2
};

// Struct to hold op data.
struct OpData {
  int scratch_tensor_index;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* data = new OpData;
  context->AddTensors(context, /*tensors_to_add=*/kNumTemporaryTensors,
                      &data->scratch_tensor_index);
  return data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  TF_LITE_ENSURE_EQ(context, node->inputs->size, 4);
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 2);
  const int n_batch = SizeOfDimension(input, 0);
  const int n_input = SizeOfDimension(input, 1);

  const TfLiteTensor* lookup = GetInput(context, node, kInputLookupTensor);
  TF_LITE_ENSURE_EQ(context, lookup->type, kTfLiteInt32);
  TF_LITE_ENSURE_EQ(context, NumDimensions(lookup), 1);
  // Only support single lookup.
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(lookup, 0), 1);

  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  TF_LITE_ENSURE_EQ(context, NumDimensions(weights), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(weights, 1), n_input);

  const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
  TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(weights, 0));

  const bool is_hybrid_op =
      (weights->type == kTfLiteUInt8 && input->type == kTfLiteFloat32);

  // Resize output.
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(1);
  output_size_array->data[0] = 1;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));

  if (is_hybrid_op) {
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(kNumTemporaryTensors);

    // Allocate temporary tensors to store quantized values of input.
    node->temporaries->data[kInputQuantized] = op_data->scratch_tensor_index;
    TfLiteTensor* input_quantized =
        GetTemporary(context, node, /*index=*/kInputQuantized);
    input_quantized->type = kTfLiteUInt8;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }

    // Tell interpreter to allocate temporary tensors to store scaling factors.
    node->temporaries->data[kScalingFactors] =
        op_data->scratch_tensor_index + kScalingFactors;
    TfLiteTensor* scaling_factors =
        GetTemporary(context, node, /*index=*/kScalingFactors);
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    int scaling_dims[1] = {n_batch};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = n_batch;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }
  }
  return kTfLiteOk;
}

TfLiteStatus EvalFloat(const TfLiteTensor* input, const TfLiteTensor* lookup,
                       const TfLiteTensor* weights, const TfLiteTensor* bias,
                       TfLiteTensor* output) {
  const int n_batch = SizeOfDimension(input, 0);
  const int n_input = SizeOfDimension(input, 1);

  const float* input_ptr_batch = input->data.f;

  // Initialize pointer to right row according to lookup value.
  int32 lookup_index = lookup->data.i32[0];
  const float* weights_ptr = weights->data.f + lookup_index * n_input;

  // Initialize output to bias.
  if (bias) {
    float* bias_ptr = bias->data.f + lookup_index;
    tensor_utils::VectorBatchVectorAssign(bias_ptr, 1, n_batch, output->data.f);
  } else {
    tensor_utils::ZeroVector(output->data.f, n_batch * 1);
  }

  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      weights_ptr, /*m_rows=*/1, n_input, input_ptr_batch, n_batch,
      output->data.f, /*result_stride=*/1);

  return kTfLiteOk;
}

TfLiteStatus EvalHybrid(const TfLiteTensor* input, const TfLiteTensor* lookup,
                        const TfLiteTensor* weights, const TfLiteTensor* bias,
                        TfLiteTensor* scaling_factors,
                        TfLiteTensor* input_quantized, TfLiteTensor* output) {
  const int n_batch = SizeOfDimension(input, 0);
  const int n_input = SizeOfDimension(input, 1);

  const float* input_ptr_batch = input->data.f;
  // Initialize the pointer to storage for quantized values and
  // scaling factors.
  int8_t* quantized_input_ptr_batch =
      reinterpret_cast<int8_t*>(input_quantized->data.uint8);
  float* scaling_factors_ptr = scaling_factors->data.f;

  // Initialize pointer to right row according to lookup value.
  int32 lookup_index = lookup->data.i32[0];
  int8_t* weights_ptr =
      reinterpret_cast<int8_t*>(weights->data.uint8) + lookup_index * n_input;

  // Initialize output to bias.
  if (bias) {
    float* bias_ptr = bias->data.f + lookup_index;
    tensor_utils::VectorBatchVectorAssign(bias_ptr, 1, n_batch, output->data.f);
  } else {
    tensor_utils::ZeroVector(output->data.f, n_batch * 1);
  }

  if (!tensor_utils::IsZeroVector(input_ptr_batch, n_batch * n_input)) {
    // Quantize input from float to int8.
    float unused_min, unused_max;
    for (int b = 0; b < n_batch; ++b) {
      const int offset = b * n_input;
      tensor_utils::SymmetricQuantizeFloats(
          input_ptr_batch + offset, n_input, quantized_input_ptr_batch + offset,
          &unused_min, &unused_max, &scaling_factors_ptr[b]);
      scaling_factors_ptr[b] *= weights->params.scale;
    }

    tensor_utils::MatrixBatchVectorMultiplyAccumulate(
        weights_ptr, /*m_rows=*/1, n_input, quantized_input_ptr_batch,
        scaling_factors_ptr, n_batch, output->data.f, /*result_stride=*/1);
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* lookup = GetInput(context, node, kInputLookupTensor);
  const TfLiteTensor* weights = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetInput(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (weights->type) {
    case kTfLiteFloat32: {
      return EvalFloat(input, lookup, weights, bias, output);
    }
    case kTfLiteUInt8: {
      TfLiteTensor* input_quantized =
          GetTemporary(context, node, /*index=*/kInputQuantized);
      TfLiteTensor* scaling_factors =
          GetTemporary(context, node, /*index=*/kScalingFactors);
      return EvalHybrid(input, lookup, weights, bias, scaling_factors,
                        input_quantized, output);
    }
    default:
      context->ReportError(context, "Type %d is not currently supported.",
                           weights->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace sparse_output_fully_connected

TfLiteRegistration* Register_SPARSE_OUTPUT_FULLY_CONNECTED() {
  static TfLiteRegistration r = {sparse_output_fully_connected::Init,
                                 sparse_output_fully_connected::Free,
                                 sparse_output_fully_connected::Prepare,
                                 sparse_output_fully_connected::Eval};
  return &r;
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
