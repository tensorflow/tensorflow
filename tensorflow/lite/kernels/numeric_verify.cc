/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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
#include <math.h>
#include <stddef.h>
#include <stdlib.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/dequantize.h"
#include "tensorflow/lite/kernels/internal/optimized/neon_check.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/dequantize.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace custom {
namespace numeric_verify {

struct OpContext {
  OpContext(TfLiteContext* context, TfLiteNode* node) {
    input = GetInput(context, node, 0);
    ref = GetInput(context, node, 1);
  }
  const TfLiteTensor* input;
  const TfLiteTensor* ref;
};

const int kTensorNotAllocated = -1;

struct OpData {
  // The percentage of the tensor value range. Must be a number less than 1.0.
  float tolerance;
  // The abstract value allowed for the floating-point value difference.
  float max_diff;
  // This boolean value is only used when the input tensor is constant.
  bool float_input_initialized;
  int cache_tensor_id = kTensorNotAllocated;
};

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  auto* op_data = new OpData();
  op_data->float_input_initialized = false;

  // Get the tolerance parameter from the buffer. Use flexbuffers asMap if there
  // multiple custom options.
  const float* buffer_t = reinterpret_cast<const float*>(buffer);
  op_data->tolerance = *buffer_t;

  return op_data;
}

void Free(TfLiteContext* context, void* buffer) {
  delete reinterpret_cast<OpData*>(buffer);
}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  TF_LITE_ENSURE_EQ(context, NumInputs(node), 2);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 0);
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);

  OpContext op_context(context, node);

  TF_LITE_ENSURE(context, op_context.input->type == kTfLiteUInt8 ||
                              op_context.input->type == kTfLiteInt8 ||
                              op_context.input->type == kTfLiteInt16 ||
                              op_context.input->type == kTfLiteFloat16);
  TF_LITE_ENSURE(context, op_context.ref->type == kTfLiteFloat32);

  op_data->max_diff = op_data->tolerance * op_context.input->params.scale;
  switch (op_context.input->type) {
    case kTfLiteUInt8:
    case kTfLiteInt8:
      op_data->max_diff *= (1 << 8);
      break;
    case kTfLiteInt16:
      op_data->max_diff *= (1 << 16);
      break;
    default:
      break;
  }

  // Allocate tensor to store the dequantized inputs.
  if (op_data->cache_tensor_id == kTensorNotAllocated) {
    TF_LITE_ENSURE_OK(
        context, context->AddTensors(context, 1, &op_data->cache_tensor_id));
  }

  TfLiteIntArrayFree(node->temporaries);
  node->temporaries = TfLiteIntArrayCreate(1);
  node->temporaries->data[0] = op_data->cache_tensor_id;

  TfLiteTensor* dequantized = GetTemporary(context, node, /*index=*/0);
  dequantized->type = op_context.ref->type;
  dequantized->allocation_type = kTfLiteDynamic;

  TF_LITE_ENSURE_OK(context, context->ResizeTensor(
                                 context, dequantized,
                                 TfLiteIntArrayCopy(op_context.input->dims)));

  return kTfLiteOk;
}

static int32_t GetQuantizedValue(const OpContext& op_context, int index) {
  switch (op_context.input->type) {
    case kTfLiteUInt8:
      return GetTensorData<uint8_t>(op_context.input)[index];
    case kTfLiteInt8:
      return GetTensorData<int8_t>(op_context.input)[index];
    case kTfLiteInt16:
      return GetTensorData<int16_t>(op_context.input)[index];
    default:
      return 0;
  }
}

template <builtin::dequantize::KernelType kernel_type>
TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  OpData* op_data = reinterpret_cast<OpData*>(node->user_data);
  OpContext op_context(context, node);
  if (IsConstantTensor(op_context.input) && op_data->float_input_initialized) {
    return kTfLiteOk;
  }

  // Dequantize the input
  TfLiteTensor* dequantized = GetTemporary(context, node, /*index=*/0);
  auto status = builtin::dequantize::DequantizeImpl<kernel_type>(
      context, node, op_context.input, dequantized);
  if (status != kTfLiteOk) {
    return status;
  }

  if (IsConstantTensor(op_context.input)) {
    op_data->float_input_initialized = true;
  }

  // If the tolerance is very small, we only display the stats of the diff.
  if (op_data->tolerance < 0.1) {
    std::vector<double> diffs, temp;
    diffs.reserve(NumElements(dequantized));
    temp.reserve(NumElements(dequantized));
    for (int i = 0; i < NumElements(op_context.ref); ++i) {
      float dequant = GetTensorData<float>(dequantized)[i];
      float reference = GetTensorData<float>(op_context.ref)[i];
      diffs.push_back(dequant - reference);
    }
    double mean =
        std::accumulate(diffs.begin(), diffs.end(), 0.0) / diffs.size();
    double max_diff = 0.0;
    std::transform(diffs.begin(), diffs.end(), temp.begin(),
                   [mean, &max_diff](double x) {
                     max_diff = std::max(max_diff, std::abs(x));
                     return x - mean;
                   });
    double sq_sum =
        std::inner_product(temp.begin(), temp.end(), temp.begin(), 0.0);
    double std = std::sqrt(sq_sum / diffs.size());
    TF_LITE_KERNEL_LOG(
        context,
        "std: %f, mean: %f, max_diff: %f (scale: %f, zero_point: %d).\n", std,
        mean, max_diff, op_context.input->params.scale,
        op_context.input->params.zero_point);
    return kTfLiteOk;
  }

  // Verify the dequantized output.
  auto max_diff = op_data->tolerance * op_context.input->params.scale;
  for (int i = 0; i < NumElements(op_context.ref); ++i) {
    int32_t value = GetQuantizedValue(op_context, i);
    float dequant = GetTensorData<float>(dequantized)[i];
    float reference = GetTensorData<float>(op_context.ref)[i];
    float diff = std::abs(reference - dequant);
    if (diff > max_diff) {
      TF_LITE_KERNEL_LOG(
          context,
          "Mismatch: %f is quantized to %d with (%f, %d). "
          "abs(%f - %f) = %f > %f (tolerance) range percentage %f.\n",
          reference, value, op_context.input->params.scale,
          op_context.input->params.zero_point, reference, dequant, diff,
          max_diff, op_data->tolerance);
      return kTfLiteError;
    }
  }
  return kTfLiteOk;
}

}  // namespace numeric_verify

TfLiteRegistration* Register_NUMERIC_VERIFY_OPT() {
  static TfLiteRegistration r = {
      numeric_verify::Init, numeric_verify::Free, numeric_verify::Prepare,
      numeric_verify::Eval<builtin::dequantize::kGenericOptimized>};
  return &r;
}

TfLiteRegistration* Register_NUMERIC_VERIFY_REF() {
  static TfLiteRegistration r = {
      numeric_verify::Init, numeric_verify::Free, numeric_verify::Prepare,
      numeric_verify::Eval<builtin::dequantize::kReference>};
  return &r;
}

TfLiteRegistration* Register_NUMERIC_VERIFY() {
#ifdef USE_NEON
  return Register_NUMERIC_VERIFY_OPT();
#else
  return Register_NUMERIC_VERIFY_REF();
#endif
}

}  // namespace custom
}  // namespace ops
}  // namespace tflite
