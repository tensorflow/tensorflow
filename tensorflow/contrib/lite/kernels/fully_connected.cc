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

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <limits>

#include "tensorflow/contrib/lite/builtin_op_data.h"
#include "tensorflow/contrib/lite/context.h"
#include "tensorflow/contrib/lite/kernels/activation_functor.h"
#include "tensorflow/contrib/lite/kernels/gemm_support.h"
#include "tensorflow/contrib/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/quantization_util.h"
#include "tensorflow/contrib/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor.h"
#include "tensorflow/contrib/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/contrib/lite/kernels/kernel_util.h"
#include "tensorflow/contrib/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace fully_connected {

// This file has four implementations of FullyConnected
enum KernelType {
  kReference,
  kGenericOptimized,  // Neon-free
  kNeonOptimized,
  kPie,  // Used by the PIE team
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
  // The index of the temporary tensor where the quantized inputs are cached.
  int input_quantized_index;
};

constexpr int kInputTensor = 0;
constexpr int kWeightsTensor = 1;
constexpr int kBiasTensor = 2;
constexpr int kOutputTensor = 0;
constexpr int kShuffledInputWorkspaceTensor = 1;
constexpr int kScratchBufferTensor = 1;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  // This is a builtin op, so we don't use the contents in 'buffer', if any.
  // Instead, we allocate a new object to carry information from Prepare() to
  // Eval().
  gemm_support::IncrementUsageCounter(context);
  auto* op_data = new OpData();
  context->AddTensors(context, 1, &op_data->input_quantized_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  gemm_support::DecrementUsageCounter(context);
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 3);
  // Shuffled formats need a workspace to store the shuffled input activations.
  const int expected_outputs_count =
      params->weights_format == kTfLiteFullyConnectedWeightsFormatDefault ? 1
                                                                          : 2;
  TF_LITE_ENSURE_EQ(context, node->outputs->size, expected_outputs_count);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  int input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    input_size *= input->dims->data[i];
  }

  TF_LITE_ENSURE_EQ(context, NumDimensions(filter), 2);
  const int batch_size = input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  TF_LITE_ENSURE_EQ(context, input_size, batch_size * filter->dims->data[1]);
  if (bias) {
    TF_LITE_ENSURE_EQ(context, NumElements(bias), SizeOfDimension(filter, 0));
  }

  // Note that quantized inference requires that all tensors have their
  // parameters set. This is usually done during quantized training.
  TfLiteType data_type = input->type;
  if (data_type != kTfLiteFloat32) {
    double real_multiplier = 0.0;
    TF_LITE_ENSURE_STATUS(GetQuantizedConvolutionMultipler(
        context, input, filter, bias, output, &real_multiplier));
    TF_LITE_ENSURE(context, real_multiplier < 1.0);
    QuantizeMultiplierSmallerThanOneExp(
        real_multiplier, &data->output_multiplier, &data->output_shift);
    data->output_shift *= -1;
    TF_LITE_ENSURE_STATUS(CalculateActivationRangeQuantized(
        context, params->activation, output, &data->output_activation_min,
        &data->output_activation_max));
  }

  // If we have to perform on-the-fly quantization (with quantized weights and
  // float inputs) first we need to quantize the inputs. Allocate a temporary
  // buffer to store the intermediate quantized values.
  if (input->type == kTfLiteFloat32 && filter->type == kTfLiteUInt8) {
    TfLiteIntArrayFree(node->temporaries);
    node->temporaries = TfLiteIntArrayCreate(1);
    node->temporaries->data[0] = data->input_quantized_index;

    TfLiteTensor* input_quantized =
        &context->tensors[node->temporaries->data[0]];
    input_quantized->type = kTfLiteUInt8;
    input_quantized->allocation_type = kTfLiteArenaRw;

    // TODO(raziel): add this logic to ResizeTensor.
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }
  }

  // Resize output.
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(2);
  output_size_array->data[0] = batch_size;
  output_size_array->data[1] = num_units;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));
  return kTfLiteOk;
}

TfLiteStatus EvalPie(TfLiteContext* context, TfLiteNode* node,
                     TfLiteFullyConnectedParams* params, OpData* data,
                     const TfLiteTensor* input, const TfLiteTensor* filter,
                     const TfLiteTensor* bias, TfLiteTensor* output) {
  int total_input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    total_input_size *= input->dims->data[i];
  }

  int input_size = filter->dims->data[1];
  const int batch_size = total_input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  // Output = bias if bias tensor exists.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(bias->data.f, num_units, batch_size,
                                          output->data.f);
  } else {
    tensor_utils::ZeroVector(output->data.f, batch_size * num_units);
  }

  // Compute output += weight * input
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      filter->data.f, num_units, input_size, input->data.f, batch_size,
      output->data.f, /*result_stride=*/1);

  // Apply activation function
  tensor_utils::ApplyActivationToVector(output->data.f, batch_size * num_units,
                                        params->activation, output->data.f);

  return kTfLiteOk;
}

TfLiteStatus EvalPieQuantized(TfLiteContext* context, TfLiteNode* node,
                              TfLiteFullyConnectedParams* params, OpData* data,
                              const TfLiteTensor* input,
                              const TfLiteTensor* filter,
                              const TfLiteTensor* bias,
                              TfLiteTensor* input_quantized,
                              TfLiteTensor* output) {
  // Check the types for this hybrid Op.
  TF_LITE_ENSURE_EQ(context, input->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, filter->type, kTfLiteUInt8);
  TF_LITE_ENSURE_EQ(context, bias->type, kTfLiteFloat32);
  TF_LITE_ENSURE_EQ(context, output->type, kTfLiteFloat32);

  int total_input_size = 1;
  for (int i = 0; i < input->dims->size; i++) {
    total_input_size *= input->dims->data[i];
  }

  const int input_size = filter->dims->data[1];
  const int batch_size = total_input_size / filter->dims->data[1];
  const int num_units = filter->dims->data[0];

  // Output = bias if bias tensor exists.
  if (bias) {
    tensor_utils::VectorBatchVectorAssign(bias->data.f, num_units, batch_size,
                                          output->data.f);
  } else {
    tensor_utils::ZeroVector(output->data.f, batch_size * num_units);
  }

  // Save matrix multiplication computation for all zero input.
  if (tensor_utils::IsZeroVector(input->data.f, total_input_size)) {
    tensor_utils::ApplyActivationToVector(output->data.f,
                                          batch_size * num_units,
                                          params->activation, output->data.f);
    return kTfLiteOk;
  }

  // Quantize input from float to uint8 + quantization params (scaling factor).
  float min, max;
  float* scaling_factors = new float[batch_size];

  // Quantize each batch independently.
  for (int b = 0; b < batch_size; ++b) {
    const int offset = b * input_size;
    tensor_utils::SymmetricQuantizeFloats(
        input->data.f + offset, input_size,
        reinterpret_cast<int8_t*>(input_quantized->data.uint8) + offset, &min,
        &max, &scaling_factors[b]);
    // Incorporate scaling of the filter.
    scaling_factors[b] *= filter->params.scale;
  }

  // Compute output += weight * quantized_input
  tensor_utils::MatrixBatchVectorMultiplyAccumulate(
      reinterpret_cast<int8_t*>(filter->data.uint8), num_units, input_size,
      reinterpret_cast<int8_t*>(input_quantized->data.uint8), scaling_factors,
      batch_size, output->data.f, /*result_stride=*/1);

  // Apply activation function to floats.
  tensor_utils::ApplyActivationToVector(output->data.f, batch_size * num_units,
                                        params->activation, output->data.f);
  delete[] scaling_factors;

  return kTfLiteOk;
}

#define TF_LITE_MACRO_DISPATCH(macro_name, params, target_namespace) \
  if (params->activation == kTfLiteActNone) {                        \
    macro_name(target_namespace, kNone);                             \
  }                                                                  \
  if (params->activation == kTfLiteActRelu) {                        \
    macro_name(target_namespace, kRelu);                             \
  }                                                                  \
  if (params->activation == kTfLiteActRelu6) {                       \
    macro_name(target_namespace, kRelu6);                            \
  }

template <KernelType kernel_type>
TfLiteStatus EvalQuantized(TfLiteContext* context, TfLiteNode* node,
                           TfLiteFullyConnectedParams* params, OpData* data,
                           const TfLiteTensor* input,
                           const TfLiteTensor* filter, const TfLiteTensor* bias,
                           TfLiteTensor* output) {
  gemmlowp::GemmContext* gemm_context = gemm_support::GetFromContext(context);

  int32_t input_offset = -input->params.zero_point;
  int32_t filter_offset = -filter->params.zero_point;
  int32_t output_offset = output->params.zero_point;
#define TF_LITE_FULLY_CONNECTED(type, output_data_type)                     \
  type::FullyConnected(                                                     \
      GetTensorData<uint8_t>(input), GetTensorDims(input), input_offset,    \
      GetTensorData<uint8_t>(filter), GetTensorDims(filter), filter_offset, \
      GetTensorData<int32_t>(bias), GetTensorDims(bias), output_offset,     \
      data->output_multiplier, data->output_shift,                          \
      data->output_activation_min, data->output_activation_max,             \
      GetTensorData<output_data_type>(output), GetTensorDims(output),       \
      gemm_context)
  if (kernel_type == kReference) {
    switch (output->type) {
      case kTfLiteUInt8:
        TF_LITE_FULLY_CONNECTED(reference_ops, uint8_t);
        break;
      case kTfLiteInt16:
        TF_LITE_FULLY_CONNECTED(reference_ops, int16_t);
        break;
      default:
        context->ReportError(
            context,
            "Quantized FullyConnected expects output data type uint8 or int16");
        return kTfLiteError;
    }
  } else if (kernel_type == kPie && input->type == kTfLiteFloat32) {
    // Pie currently only supports quantized models and float inputs/outputs.
    TfLiteTensor* input_quantized =
        &context->tensors[node->temporaries->data[0]];
    return EvalPieQuantized(context, node, params, data, input, filter, bias,
                            input_quantized, output);
  } else {
    switch (output->type) {
      case kTfLiteUInt8:
        TF_LITE_FULLY_CONNECTED(optimized_ops, uint8_t);
        break;
      case kTfLiteInt16:
        TF_LITE_FULLY_CONNECTED(optimized_ops, int16_t);
        break;
      default:
        context->ReportError(
            context,
            "Quantized FullyConnected expects output data type uint8 or int16");
        return kTfLiteError;
    }
  }
#undef TF_LITE_FULLY_CONNECTED

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalShuffledQuantized(TfLiteContext* context, TfLiteNode* node,
                                   TfLiteFullyConnectedParams* params,
                                   OpData* data, const TfLiteTensor* input,
                                   const TfLiteTensor* filter,
                                   const TfLiteTensor* bias,
                                   TfLiteTensor* output,
                                   TfLiteTensor* shuffled_input_workspace) {
  gemmlowp::GemmContext* gemm_context = gemm_support::GetFromContext(context);

  // TODO(b/110697972) decide more consistently if / how / where we want
  // to perform this kind of runtime data type checks.
  if (input->type != kTfLiteUInt8 || filter->type != kTfLiteUInt8 ||
      bias->type != kTfLiteInt32 || output->type != kTfLiteInt16 ||
      shuffled_input_workspace->type != kTfLiteUInt8) {
    context->ReportError(context, "Unexpected data type");
    return kTfLiteError;
  }

#define TF_LITE_SHUFFLED_FULLY_CONNECTED(type)                  \
  type::ShuffledFullyConnected(                                 \
      GetTensorData<uint8_t>(input), GetTensorDims(input),      \
      GetTensorData<uint8_t>(filter), GetTensorDims(filter),    \
      GetTensorData<int32_t>(bias), GetTensorDims(bias),        \
      data->output_multiplier, data->output_shift,              \
      data->output_activation_min, data->output_activation_max, \
      GetTensorData<int16_t>(output), GetTensorDims(output),    \
      GetTensorData<uint8_t>(shuffled_input_workspace), gemm_context)
  if (kernel_type == kReference) {
    TF_LITE_SHUFFLED_FULLY_CONNECTED(reference_ops);
  } else {
    TF_LITE_SHUFFLED_FULLY_CONNECTED(optimized_ops);
  }
#undef TF_LITE_SHUFFLED_FULLY_CONNECTED

  return kTfLiteOk;
}

template <KernelType kernel_type>
TfLiteStatus EvalFloat(TfLiteContext* context, TfLiteNode* node,
                       TfLiteFullyConnectedParams* params, OpData* data,
                       const TfLiteTensor* input, const TfLiteTensor* filter,
                       const TfLiteTensor* bias, TfLiteTensor* output) {
  float output_activation_min, output_activation_max;
  CalculateActivationRange(params->activation, &output_activation_min,
                           &output_activation_max);
#define TF_LITE_FULLY_CONNECTED(type)                                       \
  type::FullyConnected(GetTensorData<float>(input), GetTensorDims(input),   \
                       GetTensorData<float>(filter), GetTensorDims(filter), \
                       GetTensorData<float>(bias), GetTensorDims(bias),     \
                       output_activation_min, output_activation_max,        \
                       GetTensorData<float>(output), GetTensorDims(output))
  if (kernel_type == kReference) {
    TF_LITE_FULLY_CONNECTED(reference_ops);
  } else if (kernel_type == kPie) {
    return EvalPie(context, node, params, data, input, filter, bias, output);
  } else {
    TF_LITE_FULLY_CONNECTED(optimized_ops);
  }
#undef TF_LITE_FULLY_CONNECTED

  return kTfLiteOk;
}

#undef TF_LITE_MACRO_DISPATCH

template <KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params =
      reinterpret_cast<TfLiteFullyConnectedParams*>(node->builtin_data);
  OpData* data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* filter = GetInput(context, node, kWeightsTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (filter->type) {  // Already know in/out types are same.
    case kTfLiteFloat32:
      return EvalFloat<kernel_type>(context, node, params, data, input, filter,
                                    bias, output);
    case kTfLiteUInt8:
      if (params->weights_format ==
          kTfLiteFullyConnectedWeightsFormatShuffled4x16Int8) {
        TfLiteTensor* shuffled_input_workspace =
            GetOutput(context, node, kShuffledInputWorkspaceTensor);
        return EvalShuffledQuantized<kernel_type>(context, node, params, data,
                                                  input, filter, bias, output,
                                                  shuffled_input_workspace);
      } else if (params->weights_format ==
                 kTfLiteFullyConnectedWeightsFormatDefault) {
        return EvalQuantized<kernel_type>(context, node, params, data, input,
                                          filter, bias, output);
      } else {
        context->ReportError(context,
                             "Unhandled fully-connected weights format");
        return kTfLiteError;
      }
    default:
      context->ReportError(context, "Type %d not currently supported.",
                           filter->type);
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace fully_connected

TfLiteRegistration* Register_FULLY_CONNECTED_REF() {
  static TfLiteRegistration r = {
      fully_connected::Init, fully_connected::Free, fully_connected::Prepare,
      fully_connected::Eval<fully_connected::kReference>};
  return &r;
}

TfLiteRegistration* Register_FULLY_CONNECTED_NEON_OPT() {
  static TfLiteRegistration r = {
      fully_connected::Init, fully_connected::Free, fully_connected::Prepare,
      fully_connected::Eval<fully_connected::kNeonOptimized>};
  return &r;
}

TfLiteRegistration* Register_FULLY_CONNECTED_GENERIC_OPT() {
  static TfLiteRegistration r = {
      fully_connected::Init, fully_connected::Free, fully_connected::Prepare,
      fully_connected::Eval<fully_connected::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_FULLY_CONNECTED_PIE() {
  static TfLiteRegistration r = {fully_connected::Init, fully_connected::Free,
                                 fully_connected::Prepare,
                                 fully_connected::Eval<fully_connected::kPie>};
  return &r;
}

TfLiteRegistration* Register_FULLY_CONNECTED() {
  // TODO(ahentz): We don't have a dedicated quantized version of the PIE
  // kernel. For now, the quantized version just defer to the corresponding
  // optimized MINI kernel. At some point we will allow different libraries to
  // be built with different kernels, but for now we have to pick one here.
  return Register_FULLY_CONNECTED_PIE();
#ifdef USE_NEON
  return Register_FULLY_CONNECTED_NEON_OPT();
#else
  return Register_FULLY_CONNECTED_GENERIC_OPT();
#endif
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
