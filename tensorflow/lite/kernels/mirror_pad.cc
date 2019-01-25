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

#include <memory>

#include "tensorflow/lite/c/builtin_op_data.h"
#include "tensorflow/lite/c/c_api_internal.h"
#include "tensorflow/lite/kernels/internal/optimized/optimized_ops.h"
#include "tensorflow/lite/kernels/internal/quantization_util.h"
#include "tensorflow/lite/kernels/internal/reference/reference_ops.h"
#include "tensorflow/lite/kernels/internal/tensor.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"

namespace tflite {
namespace ops {
namespace builtin {
namespace mirror_pad {
namespace {

// Simple class that represents a mirror padded tensor - which is the output
// from the Op.
struct PaddedTensor {
  // If not null that means this is a scalar value.
  // Note: This is not owned by default. It will point to the value
  // in the input tensor.
  const void* value = nullptr;
  // If this tensor is not one value, then this vector will have
  // all the tensors that belongs to this tensor.
  // Pointers are owned.
  std::vector<std::unique_ptr<PaddedTensor>> values;
  // Pointers to PaddedTensors that are padded on the left of the current
  // tensor.
  std::vector<PaddedTensor*> left_pad_ptrs;
  // Pointers to PaddedTensors that are padded on the right of the current
  // tensor.
  std::vector<PaddedTensor*> right_pad_ptrs;

  // Returns mutable pointer to the tensor identified by 'indices'.
  PaddedTensor* GetMutable(const std::vector<int>& indices) {
    auto* result = this;
    for (int i = 0; i < indices.size(); ++i) {
      if (indices[i] >= result->values.size()) {
        return nullptr;
      }
      result = result->values[indices[i]].get();
      if (result == nullptr) break;
    }
    return result;
  }
};

// Util method to initialize the memory of the padded tensor.
void InitializeTensorMemory(const TfLiteIntArray* const dims, int dim_index,
                            int dims_size, PaddedTensor* padded_tensor) {
  if (dim_index >= dims_size) {
    return;
  }
  padded_tensor->values.reserve(dims->data[dim_index]);
  for (int i = 0; i < dims->data[dim_index]; ++i) {
    padded_tensor->values.emplace_back(new PaddedTensor());
    InitializeTensorMemory(dims, dim_index + 1, dims_size,
                           padded_tensor->values.back().get());
  }
}

// Returns pointer to the value at the specified index in 'data'.
inline const void* GetValuePointerAtIndex(const void* data, int index,
                                          const TfLiteType data_type) {
  switch (data_type) {
    case kTfLiteFloat32:
      return static_cast<const float*>(data) + index;
    case kTfLiteInt32:
      return static_cast<const int32_t*>(data) + index;
    case kTfLiteUInt8:
      return static_cast<const uint8_t*>(data) + index;
    case kTfLiteInt64:
      return static_cast<const int64_t*>(data) + index;
    case kTfLiteBool:
      return static_cast<const bool*>(data) + index;
    case kTfLiteInt16:
      return static_cast<const int16_t*>(data) + index;
    case kTfLiteInt8:
      return static_cast<const int8_t*>(data) + index;
    // Unsupported types ?
    default:
      return nullptr;
  }
  return nullptr;
}

// Util method that increment index in the N-d array.
void IncrementTensorIndex(const TfLiteIntArray* dims,
                          std::vector<int>* tensor_index_ptr) {
  int dimension_index = dims->size - 1;
  auto& tensor_index = *tensor_index_ptr;
  tensor_index[dimension_index]++;
  while (dimension_index >= 0 &&
         tensor_index[dimension_index] == dims->data[dimension_index]) {
    tensor_index[dimension_index] = 0;
    dimension_index--;
    if (dimension_index >= 0) tensor_index[dimension_index]++;
  }
}

// Fills the 'padded_tensor' with data from 'input_tensor'.
TfLiteStatus InitFromInputTensor(const TfLiteTensor* input_tensor,
                                 PaddedTensor* padded_tensor) {
  const auto* dims = input_tensor->dims;
  const auto data_type = input_tensor->type;
  const void* data = static_cast<const void*>(input_tensor->data.raw_const);
  // Either invalid input or unsupported type.+
  if (data == nullptr) {
    return kTfLiteError;
  }
  // Index of current processing tensor.
  std::vector<int> tensor_index(dims->size, 0);
  int flat_index = 0;
  const int num_elements = NumElements(input_tensor);
  while (flat_index < num_elements) {
    auto* tensor = padded_tensor->GetMutable(tensor_index);
    if (tensor == nullptr) {
      return kTfLiteError;
    }
    tensor->value = GetValuePointerAtIndex(data, flat_index, data_type);
    IncrementTensorIndex(dims, &tensor_index);
    ++flat_index;
  }

  return kTfLiteOk;
}

template <typename T>
inline void GetPadding(const T* data, int offset, int64_t* left_pad,
                       int64_t* right_pad) {
  *left_pad = static_cast<int64_t>(*(data + offset * 2));
  *right_pad = static_cast<int64_t>(*(data + offset * 2 + 1));
}

inline TfLiteStatus GetPadding(const TfLiteTensor* padding_matrix,
                               int dimension, int64_t* left_pad,
                               int64_t* right_pad) {
  switch (padding_matrix->type) {
    case kTfLiteInt32:
      GetPadding(padding_matrix->data.i32, dimension, left_pad, right_pad);
      break;
    case kTfLiteInt64:
      GetPadding(padding_matrix->data.i64, dimension, left_pad, right_pad);
      break;
    default:
      return kTfLiteError;
  }
  return kTfLiteOk;
}

TfLiteStatus ValidateTensor(const TfLiteTensor* padding_matrix, int offset,
                            int dimension_index, PaddedTensor* padded_tensor,
                            TfLiteContext* context) {
  if (dimension_index >= padding_matrix->dims->data[0]) {
    return kTfLiteOk;
  }

  int64_t left_pad = 0, right_pad = 0;
  TF_LITE_ENSURE_STATUS(
      GetPadding(padding_matrix, dimension_index, &left_pad, &right_pad));
  // If we are not going to include border we must have enough values
  // to use.
  if (left_pad + offset > padded_tensor->values.size()) {
    context->ReportError(
        context, "Not enough values for Mirror Pad, required %d, available %d.",
        left_pad + offset, padded_tensor->values.size());
    return kTfLiteError;
  }
  if (right_pad + offset > padded_tensor->values.size()) {
    context->ReportError(
        context, "Not enough values for Mirror Pad, required %d, available %d.",
        right_pad + offset, padded_tensor->values.size());
    return kTfLiteError;
  }
  if (!padded_tensor->values.empty()) {
    ValidateTensor(padding_matrix, offset, dimension_index + 1,
                   padded_tensor->values[0].get(), context);
  }
  return kTfLiteOk;
}

// Fills 'padded_tensor' with the padding information based on
// 'padding_matrix'.
// 'dimension_index' represents which dimension the function is operating on.
TfLiteStatus PadTensor(const TfLiteTensor* padding_matrix, int offset,
                       int dimension_index, PaddedTensor* padded_tensor,
                       TfLiteContext* context) {
  if (dimension_index >= padding_matrix->dims->data[0]) return kTfLiteOk;

  int64_t left_pad = 0, right_pad = 0;
  TF_LITE_ENSURE_STATUS(
      GetPadding(padding_matrix, dimension_index, &left_pad, &right_pad));

  for (int i = left_pad + offset - 1; i >= offset && left_pad > 0;
       --i, --left_pad) {
    padded_tensor->left_pad_ptrs.push_back(padded_tensor->values[i].get());
  }
  for (int i = padded_tensor->values.size() - (1 + offset);
       i >= 0 && right_pad > 0; --i, --right_pad) {
    padded_tensor->right_pad_ptrs.push_back(padded_tensor->values[i].get());
  }

  for (auto& tensor : padded_tensor->values) {
    TF_LITE_ENSURE_STATUS(PadTensor(padding_matrix, offset, dimension_index + 1,
                                    tensor.get(), context));
  }
  return kTfLiteOk;
}

// Fills 'output_data' with data from 'padded_tensor'.
// The function does this recursively by setting left padding first then
// original data, followed by the right padding.
template <typename T>
int FillOutput(const PaddedTensor* padded_tensor, T* output_data,
               int index_in_output) {
  if (padded_tensor == nullptr || output_data == nullptr) {
    return -1;
  }
  if (padded_tensor->value != nullptr) {
    output_data[index_in_output] = *static_cast<const T*>(padded_tensor->value);
    return index_in_output + 1;
  }
  for (const auto* tensor : padded_tensor->left_pad_ptrs) {
    index_in_output = FillOutput(tensor, output_data, index_in_output);
  }
  for (const auto& tensor : padded_tensor->values) {
    index_in_output = FillOutput(tensor.get(), output_data, index_in_output);
  }
  for (const auto* tensor : padded_tensor->right_pad_ptrs) {
    index_in_output = FillOutput(tensor, output_data, index_in_output);
  }
  return index_in_output;
}

// Returns the shape of the final output after padding.
std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> GetPaddedOutputShape(
    const TfLiteTensor* input, const TfLiteTensor* padding_matrix) {
  const int input_dims = NumDimensions(input);
  std::unique_ptr<TfLiteIntArray, void (*)(TfLiteIntArray*)> shape(
      TfLiteIntArrayCreate(input_dims), TfLiteIntArrayFree);

  int64_t left_pad = 0, right_pad = 0;
  for (int i = 0; i < input_dims; ++i) {
    GetPadding(padding_matrix, i, &left_pad, &right_pad);
    shape->data[i] = SizeOfDimension(input, i) + left_pad + right_pad;
  }
  return shape;
}

}  // namespace

TfLiteStatus Eval(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor = GetInput(context, node, 0);
  const TfLiteTensor* padding_matrix = GetInput(context, node, 1);
  auto* params =
      reinterpret_cast<TfLiteMirrorPaddingParams*>(node->builtin_data);

  if (params == nullptr) {
    return kTfLiteError;
  }
  const int input_dims = NumDimensions(input_tensor);

  TfLiteTensor* output_tensor = GetOutput(context, node, 0);
  if (IsDynamicTensor(output_tensor)) {
    auto output_size = GetPaddedOutputShape(input_tensor, padding_matrix);
    if (output_size == nullptr) {
      return kTfLiteError;
    }
    TF_LITE_ENSURE_STATUS(
        context->ResizeTensor(context, output_tensor, output_size.release()));
  }

  PaddedTensor padded_tensor;
  // Initialize memory.
  InitializeTensorMemory(input_tensor->dims, 0, input_dims, &padded_tensor);
  // Set the values from the input_tensor.
  TF_LITE_ENSURE_STATUS(InitFromInputTensor(input_tensor, &padded_tensor));

  const int offset =
      params->mode != TfLiteMirrorPaddingMode::kTfLiteMirrorPaddingReflect ? 0
                                                                           : 1;
  // Make sure padding values are sufficient and valid to use.
  TF_LITE_ENSURE_STATUS(
      ValidateTensor(padding_matrix, offset, 0, &padded_tensor, context));
  // Apply padding.
  TF_LITE_ENSURE_STATUS(
      PadTensor(padding_matrix, offset, 0, &padded_tensor, context));

  // Fill the output tensor from the padded tensor.
  TfLiteStatus status = kTfLiteOk;

#define TF_LITE_MIRROR_PAD(type) \
  FillOutput(&padded_tensor, GetTensorData<type>(output_tensor), 0);

  switch (output_tensor->type) {
    case kTfLiteFloat32: {
      TF_LITE_MIRROR_PAD(float);
      break;
    }
    case kTfLiteInt32: {
      TF_LITE_MIRROR_PAD(int32_t);
      break;
    }
    case kTfLiteUInt8: {
      TF_LITE_MIRROR_PAD(uint8_t);
      break;
    }
    case kTfLiteInt64: {
      TF_LITE_MIRROR_PAD(int64_t);
      break;
    }
    default:
      status = kTfLiteError;
      break;
  }
#undef TF_LITE_MIRROR_PAD
  return status;
}

void* Init(TfLiteContext* context, const char* buffer, size_t length) {
  return nullptr;
}

void Free(TfLiteContext* context, void* buffer) {}

TfLiteStatus Prepare(TfLiteContext* context, TfLiteNode* node) {
  const TfLiteTensor* input_tensor = GetInput(context, node, 0);
  const TfLiteTensor* padding_matrix = GetInput(context, node, 1);
  TfLiteTensor* output_tensor = GetOutput(context, node, 0);

  TF_LITE_ENSURE_EQ(context, NumDimensions(padding_matrix), 2);
  TF_LITE_ENSURE_EQ(context, SizeOfDimension(padding_matrix, 0),
                    NumDimensions(input_tensor));

  if (!IsConstantTensor(padding_matrix)) {
    SetTensorToDynamic(output_tensor);
    return kTfLiteOk;
  }
  // We have constant padding, so we can infer output size.

  auto output_size = GetPaddedOutputShape(input_tensor, padding_matrix);
  if (output_size == nullptr) {
    return kTfLiteError;
  }
  return context->ResizeTensor(context, output_tensor, output_size.release());
}

}  // namespace mirror_pad
TfLiteRegistration* Register_MIRROR_PAD() {
  static TfLiteRegistration r = {mirror_pad::Init, mirror_pad::Free,
                                 mirror_pad::Prepare, mirror_pad::Eval};
  return &r;
}

}  // namespace builtin
}  // namespace ops
}  // namespace tflite
