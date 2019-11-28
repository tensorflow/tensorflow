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

// SVDF op that compresses a fully connected op via low-rank matrix
// factorization. See https://research.google.com/pubs/archive/43813.pdf for
// details.

#include "tensorflow/lite/kernels/internal/reference/svdf.h"

#include <cstddef>
#include <cstdint>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/internal/tensor_utils.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace svdf {

namespace {

struct OpData {
  int scratch_tensor_index;
  bool float_weights_time_initialized;
  int activation_state_tensor_index;
  int32 effective_scale_1_a;
  int effective_scale_1_b;
  int32 effective_scale_2_a;
  int effective_scale_2_b;
};

}  // namespace

// Input tensors.
constexpr int kInputTensor = 0;
constexpr int kWeightsFeatureTensor = 1;
constexpr int kWeightsTimeTensor = 2;
constexpr int kBiasTensor = 3;
// This is a variable tensor, and will be modified by this op.
constexpr int kInputActivationStateTensor = 4;

// Output tensor.
constexpr int kOutputTensor = 0;

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  op_data->float_weights_time_initialized = false;
  // Note: only needs 4 scratch tensors when is_hybrid_op, only 1 otherwise.
  context->AddTensors(context, /*tensors_to_add=*/4,
                      &op_data->scratch_tensor_index);
  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  int scratch_tensor_index = op_data->scratch_tensor_index;

  // Check we have all the inputs and outputs we need.
  TF_LITE_ENSURE_EQ(context, node->outputs->size, 1);
  TF_LITE_ENSURE_EQ(context, node->inputs->size, 5);
  op_data->activation_state_tensor_index =
      node->inputs->data[kInputActivationStateTensor];

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);

  TF_LITE_ENSURE(context,
                 input->type == kTfLiteFloat32 || input->type == kTfLiteInt8);

  // Check all the parameters of tensor match within themselves and match the
  // input configuration.
  const int rank = params->rank;
  const int batch_size = input->dims->data[0];
  const int num_filters = weights_feature->dims->data[0];
  TF_LITE_ENSURE_EQ(context, num_filters % rank, 0);
  const int num_units = num_filters / rank;
  const int memory_size = weights_time->dims->data[1];
  TF_LITE_ENSURE_EQ(context, input->dims->data[1],
                    weights_feature->dims->data[1]);
  TF_LITE_ENSURE_EQ(context, weights_time->dims->data[0], num_filters);

  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);
  if (bias) {
    TF_LITE_ENSURE_EQ(context, bias->dims->data[0], num_units);
  }

  TfLiteTensor* activation_state =
      &context->tensors[op_data->activation_state_tensor_index];
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  // Check the shape of input state tensors.
  TF_LITE_ENSURE_EQ(context, NumDimensions(activation_state), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(activation_state, 0), batch_size);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(activation_state, 1),
                    memory_size * num_filters);

  // Resize output.
  TfLiteIntArray* output_size_array = TfLiteIntArrayCreate(2);
  output_size_array->data[0] = batch_size;
  output_size_array->data[1] = num_units;
  TF_LITE_ENSURE_OK(context,
                    context->ResizeTensor(context, output, output_size_array));

  // The weights are of consistent type, so it suffices to check one.
  const bool is_hybrid_op = IsHybridOp(input, weights_feature);
  const bool is_full_integer = input->type == kTfLiteInt8;

  // Resize scratch.
  TfLiteIntArrayFree(node->temporaries);
  if (is_hybrid_op) {
    node->temporaries = TfLiteIntArrayCreate(4);
  } else if (is_full_integer) {
    node->temporaries = TfLiteIntArrayCreate(2);
  } else {
    node->temporaries = TfLiteIntArrayCreate(1);
  }
  node->temporaries->data[0] = scratch_tensor_index;

  TfLiteIntArray* scratch_size_array = TfLiteIntArrayCreate(2);
  scratch_size_array->data[0] = batch_size;
  scratch_size_array->data[1] = num_filters;

  TfLiteTensor* scratch_tensor = GetTemporary(context, node, /*index=*/0);

  // The scratch buffer is of type int32 for full integer svdf and it's of type
  // float32 for hybrid and float case.
  if (is_full_integer) {
    scratch_tensor->type = kTfLiteInt32;
  } else {
    scratch_tensor->type = kTfLiteFloat32;
  }
  scratch_tensor->allocation_type = kTfLiteArenaRw;
  TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scratch_tensor,
                                                   scratch_size_array));

  if (is_hybrid_op) {
    // Tell interpreter to allocate temporary tensors to store quantized values
    // of input tensors.
    node->temporaries->data[1] = scratch_tensor_index + 1;
    TfLiteTensor* input_quantized = GetTemporary(context, node, /*index=*/1);
    input_quantized->type = weights_feature->type;
    input_quantized->allocation_type = kTfLiteArenaRw;
    if (!TfLiteIntArrayEqual(input_quantized->dims, input->dims)) {
      TfLiteIntArray* input_quantized_size = TfLiteIntArrayCopy(input->dims);
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, input_quantized,
                                                       input_quantized_size));
    }

    // Tell interpreter to allocate temporary tensors to store scaling factors.
    node->temporaries->data[2] = scratch_tensor_index + 2;
    TfLiteTensor* scaling_factors = GetTemporary(context, node, /*index=*/2);
    scaling_factors->type = kTfLiteFloat32;
    scaling_factors->allocation_type = kTfLiteArenaRw;
    int scaling_dims[1] = {batch_size};
    if (!TfLiteIntArrayEqualsArray(scaling_factors->dims, 1, scaling_dims)) {
      TfLiteIntArray* scaling_factors_size = TfLiteIntArrayCreate(1);
      scaling_factors_size->data[0] = batch_size;
      TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, scaling_factors,
                                                       scaling_factors_size));
    }

    // Used to store dequantized weights_time matrix for hybrid computation of
    // matmul(activation_state, weights_time), which occurs in floating point.
    node->temporaries->data[3] = scratch_tensor_index + 3;
    TfLiteTensor* float_weights_time = GetTemporary(context, node, /*index=*/3);
    float_weights_time->type = kTfLiteFloat32;
    // Persistent so that we can compute the dequantized weights only once.
    float_weights_time->allocation_type = kTfLiteArenaRwPersistent;
    if (!TfLiteIntArrayEqual(float_weights_time->dims, weights_time->dims)) {
      TfLiteIntArray* float_weights_time_size =
          TfLiteIntArrayCopy(weights_time->dims);
      TF_LITE_ENSURE_OK(context,
                        context->ResizeTensor(context, float_weights_time,
                                              float_weights_time_size));
    }
  }
  if (is_full_integer) {
    // Allocated one extra tensor.
    TfLiteIntArray* output_temp_size_array = TfLiteIntArrayCreate(2);
    output_temp_size_array->data[0] = num_units;
    output_temp_size_array->data[1] = batch_size;
    node->temporaries->data[1] = scratch_tensor_index + 1;
    TfLiteTensor* output_temp = GetTemporary(context, node, /*index=*/1);
    output_temp->type = kTfLiteInt32;
    output_temp->allocation_type = kTfLiteArenaRw;
    TF_LITE_ENSURE_OK(context, context->ResizeTensor(context, output_temp,
                                                     output_temp_size_array));

    // Calculate effective scales.
    auto* input_params =
        reinterpret_cast<TfLiteAffineQuantization*>(input->quantization.params);
    auto* weights_feature_params = reinterpret_cast<TfLiteAffineQuantization*>(
        weights_feature->quantization.params);
    auto* state_params = reinterpret_cast<TfLiteAffineQuantization*>(
        activation_state->quantization.params);
    auto* weight_time_params = reinterpret_cast<TfLiteAffineQuantization*>(
        weights_time->quantization.params);
    auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
        output->quantization.params);
    const double effective_scale_1 = input_params->scale->data[0] *
                                     weights_feature_params->scale->data[0] /
                                     state_params->scale->data[0];
    const double effective_scale_2 = state_params->scale->data[0] *
                                     weight_time_params->scale->data[0] /
                                     output_params->scale->data[0];
    QuantizeMultiplier(effective_scale_1, &op_data->effective_scale_1_a,
                       &op_data->effective_scale_1_b);
    QuantizeMultiplier(effective_scale_2, &op_data->effective_scale_2_a,
                       &op_data->effective_scale_2_b);
  }
  return kTfLiteOk;
}

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteSVDFParams*>(node->builtin_data);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  const TfLiteTensor* input = GetInput(context, node, kInputTensor);
  const TfLiteTensor* weights_feature =
      GetInput(context, node, kWeightsFeatureTensor);
  const TfLiteTensor* weights_time =
      GetInput(context, node, kWeightsTimeTensor);
  const TfLiteTensor* bias = GetOptionalInputTensor(context, node, kBiasTensor);

  TfLiteTensor* scratch = GetTemporary(context, node, /*index=*/0);

  TfLiteTensor* activation_state =
      &context->tensors[op_data->activation_state_tensor_index];
  TfLiteTensor* output = GetOutput(context, node, kOutputTensor);

  switch (weights_feature->type) {
    case kTfLiteFloat32: {
      reference_ops::EvalFloatSVDF(context, node, input, weights_feature,
                                   weights_time, bias, params, scratch,
                                   activation_state, output);
      return kTfLiteOk;
      break;
    }
    case kTfLiteUInt8:
    case kTfLiteInt8: {
      if (input->type == kTfLiteFloat32) {
        TfLiteTensor* input_quantized =
            GetTemporary(context, node, /*index=*/1);
        TfLiteTensor* scaling_factors =
            GetTemporary(context, node, /*index=*/2);
        TfLiteTensor* float_weights_time =
            GetTemporary(context, node, /*index=*/3);

        // Dequantize weights time.
        // TODO(alanchiao): this dequantization initialization only needs to
        // happen once per model and should theoretically be placed in either
        // Init or Prepare. However, TFLite doesn't allocate float_weights_time
        // until the Eval function.
        // TODO(alanchiao): refactor logic out into dequantize function.
        if (!op_data->float_weights_time_initialized) {
          const float dequantization_scale = weights_time->params.scale;
          const int8_t* weights_time_ptr = GetTensorData<int8_t>(weights_time);
          float* float_weights_time_ptr =
              GetTensorData<float>(float_weights_time);
          for (int i = 0; i < NumElements(float_weights_time); ++i) {
            float_weights_time_ptr[i] =
                weights_time_ptr[i] * dequantization_scale;
          }
          op_data->float_weights_time_initialized = true;
        }
        reference_ops::EvalHybridSVDF(context, node, input, weights_feature,
                                      float_weights_time, bias, params, scratch,
                                      scaling_factors, input_quantized,
                                      activation_state, output);
        return kTfLiteOk;
      } else {
        auto* input_params = reinterpret_cast<TfLiteAffineQuantization*>(
            input->quantization.params);
        auto* output_params = reinterpret_cast<TfLiteAffineQuantization*>(
            output->quantization.params);
        TfLiteTensor* output_temp = GetTemporary(context, node, /*index=*/1);

        // Currently supports only ReLU.
        // TODO(jianlijianli): support other activations.
        TF_LITE_ENSURE_EQ(context, params->activation, kTfLiteActRelu);
        reference_ops::EvalIntegerSVDF(
            context, node, input, weights_feature, weights_time, bias, params,
            activation_state, output, scratch, output_temp,
            op_data->effective_scale_1_a, op_data->effective_scale_1_b,
            op_data->effective_scale_2_a, op_data->effective_scale_2_b,
            input_params->zero_point->data[0],
            output_params->zero_point->data[0]);
        return kTfLiteOk;
      }
      break;
    }
    default:
      context->ReportError(context, "Type %s not currently supported.",
                           TfLiteTypeGetName(weights_feature->type));
      return kTfLiteError;
  }
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
