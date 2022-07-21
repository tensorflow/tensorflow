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

#include <stddef.h>
#include <stdint.h>

#include <map>
#include <memory>
#include <vector>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/common.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/internal/tensor_ctypes.h"
#include "tensorflow/lite/kernels/kernel_util.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace unique {

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  static const int kOutputUniqueTensor = 0;
  static const int kOutputIndexTensor = 1;

  TF_LITE_ENSURE_EQ(context, NumInputs(node), 1);
  TF_LITE_ENSURE_EQ(context, NumOutputs(node), 2);
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output_unique_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputUniqueTensor,
                                           &output_unique_tensor));
  TfLiteTensor* output_index_tensor;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, kOutputIndexTensor,
                                           &output_index_tensor));

  // The op only supports 1D input.
  TF_LITE_ENSURE_EQ(context, NumDimensions(input), 1);
  TfLiteIntArray* output_index_shape = TfLiteIntArrayCopy(input->dims);
  // The unique values are determined during evaluation, so we don't know yet
  // the size of the output tensor.
  SetTensorToDynamic(output_unique_tensor);
  return context->ResizeTensor(context, output_index_tensor,
                               output_index_shape);
}

namespace {

// Actual evaluation for the unique op.
template <typename T, typename I>
TfLiteStatus EvalImpl(TfLiteContext* context, const TfLiteTensor* input,
                      TfLiteNode* node) {
  // Map from value, to index in the unique elements vector.
  // Note that we prefer to use map than unordered_map as it showed less
  // increase in the binary size.
  std::map<T, int> unique_values;
  TfLiteTensor* output_indexes;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 1, &output_indexes));
  std::vector<T> output_values;
  I* indexes = GetTensorData<I>(output_indexes);
  const T* data = GetTensorData<T>(input);
  const int num_elements = NumElements(input);

  for (int i = 0; i < num_elements; ++i) {
    const auto element_it = unique_values.find(data[i]);
    if (element_it != unique_values.end()) {
      indexes[i] = element_it->second;
    } else {
      const int unique_index = unique_values.size();
      unique_values[data[i]] = unique_index;
      indexes[i] = unique_index;
      output_values.push_back(data[i]);
    }
  }
  // Allocate output tensor.
  TfLiteTensor* unique_output;
  TF_LITE_ENSURE_OK(context, GetOutputSafe(context, node, 0, &unique_output));
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> shape(
      TfLiteIntArrayCreate(NumDimensions(input)), TfLiteIntArrayFree);
  shape->data[0] = unique_values.size();
  TF_LITE_ENSURE_STATUS(
      context->ResizeTensor(context, unique_output, shape.release()));
  // Set the values in the output tensor.
  T* output_unique_values = GetTensorData<T>(unique_output);
  for (int i = 0; i < output_values.size(); ++i) {
    output_unique_values[i] = output_values[i];
  }
  return kTfLiteOk;
}

template <typename T>
TfLiteStatus EvalImpl(TfLiteContext* context, const TfLiteTensor* input,
                      TfLiteNode* node) {
  auto* params = reinterpret_cast<TfLiteUniqueParams*>(node->builtin_data);
  if (params == nullptr) {
    TF_LITE_KERNEL_LOG(context, "Null params passed");
    return kTfLiteError;
  }
  switch (params->index_out_type) {
    case kTfLiteInt32:
      return EvalImpl<T, int32_t>(context, input, node);
    case kTfLiteInt64:
      return EvalImpl<T, int64_t>(context, input, node);
    default:
      TF_LITE_KERNEL_LOG(
          context,
          "Unique index output array can only be Int32 or In64, requested: %s",
          TfLiteTypeGetName(params->index_out_type));
  }
  return kTfLiteError;
}

}  // namespace

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input;
  TF_LITE_ENSURE_OK(context, GetInputSafe(context, node, 0, &input));
  TfLiteTensor* output_index_tensor;
  TF_LITE_ENSURE_OK(context,
                    GetOutputSafe(context, node, 1, &output_index_tensor));
  TF_LITE_ENSURE_EQ(context, NumElements(output_index_tensor),
                    NumElements(input));

  switch (input->type) {
    case kTfLiteInt8:
      TF_LITE_ENSURE_STATUS(EvalImpl<int8_t>(context, input, node));
      break;
    case kTfLiteInt16:
      TF_LITE_ENSURE_STATUS(EvalImpl<int16_t>(context, input, node));
      break;
    case kTfLiteInt32:
      TF_LITE_ENSURE_STATUS(EvalImpl<int32_t>(context, input, node));
      break;
    case kTfLiteInt64:
      TF_LITE_ENSURE_STATUS(EvalImpl<int64_t>(context, input, node));
      break;
    case kTfLiteFloat32:
      TF_LITE_ENSURE_STATUS(EvalImpl<float>(context, input, node));
      break;
    case kTfLiteUInt8:
      TF_LITE_ENSURE_STATUS(EvalImpl<uint8_t>(context, input, node));
      break;
    default:
      TF_LITE_KERNEL_LOG(context, "Currently Unique doesn't support type: %s",
                         TfLiteTypeGetName(input->type));
      return kTfLiteError;
  }
  return kTfLiteOk;
}

}  // namespace unique

TfLiteRegistration* Register_UNIQUE() {
  static TfLiteRegistration r = {unique::Init, unique::Free, unique::Prepare,
                                 unique::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
