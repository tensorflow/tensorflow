/* Copyright 2023 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/variants/list_kernels/test_util.h"

#include <cstdint>
#include <optional>
#include <utility>

#include "tensorflow/lite/core/c/c_api_types.h"
#include "tensorflow/lite/kernels/kernel_util.h"
#include "tensorflow/lite/kernels/op_macros.h"
#include "tensorflow/lite/kernels/variants/tensor_array.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/util.h"

using ::tflite::variants::TensorArray;

namespace tflite {

// Populate tensor at given index as a TensorList.
void ListOpModel::PopulateListTensor(int index,
                                     absl::Span<const int> element_shape_data,
                                     int num_elements,
                                     TfLiteType element_type) {
  TfLiteTensor* tensor = interpreter_->tensor(index);

  TF_LITE_ASSERT_EQ(tensor->type, kTfLiteVariant);
  tensor->allocation_type = kTfLiteVariantObject;
  tensor->buffer_handle = kTfLiteNullBufferHandle;
  tensor->quantization = {kTfLiteNoQuantization};

  IntArrayUniquePtr element_shape =
      BuildTfLiteArray(element_shape_data.size(), element_shape_data.data());

  TfLiteStatus stat = TfLiteTensorVariantRealloc<TensorArray>(
      tensor, element_type, std::move(element_shape));
  TF_LITE_ASSERT_EQ(stat, kTfLiteOk);

  TensorArray* arr =
      static_cast<TensorArray*>(static_cast<VariantData*>(tensor->data.data));
  arr->Resize(num_elements);
}

// Set a list element with given data.
void ListOpModel::ListSetItem(int index, int list_index,
                              absl::Span<const int> item_dims,
                              TfLiteType item_type, const void* item_data) {
  TfLiteTensor* tensor = interpreter_->tensor(index);

  TF_LITE_ASSERT_EQ(tensor->type, kTfLiteVariant);
  TF_LITE_ASSERT_EQ(tensor->allocation_type, kTfLiteVariantObject);

  TensorArray* arr =
      static_cast<TensorArray*>(static_cast<VariantData*>(tensor->data.data));

  TF_LITE_ASSERT_EQ(arr->ElementType(), item_type);

  // Build tensor to set in list.
  TensorUniquePtr item = BuildTfLiteTensor(
      item_type, std::vector<int>(item_dims.begin(), item_dims.end()),
      kTfLiteDynamic);

  size_t item_type_size;
  TF_LITE_ASSERT_EQ(GetSizeOfType(nullptr, item_type, &item_type_size),
                    kTfLiteOk);

  // Write data to item tensor.
  const size_t item_data_size = NumElements(item.get()) * item_type_size;
  memcpy(item->data.data, item_data, item_data_size);

  TF_LITE_ASSERT_EQ(arr->Set(list_index, std::move(item)), true);
}

std::optional<size_t> TfLiteTypeSizeOf(TfLiteType type) {
  switch (type) {
    case kTfLiteFloat16:
      return sizeof(int16_t);
    case kTfLiteFloat32:
      return sizeof(float);
    case kTfLiteFloat64:
      return sizeof(double);
    case kTfLiteInt16:
      return sizeof(int16_t);
    case kTfLiteInt32:
      return sizeof(int32_t);
    case kTfLiteUInt32:
      return sizeof(uint32_t);
    case kTfLiteUInt8:
      return sizeof(uint8_t);
    case kTfLiteInt8:
      return sizeof(int8_t);
    case kTfLiteInt64:
      return sizeof(int64_t);
    case kTfLiteUInt64:
      return sizeof(uint64_t);
    case kTfLiteBool:
      return sizeof(bool);
    case kTfLiteResource:
      return sizeof(int32_t);
    case kTfLiteComplex64:
      return sizeof(float) * 2;
    case kTfLiteComplex128:
      return sizeof(double) * 2;
    case kTfLiteInt4:
      return sizeof(int8_t);
    case kTfLiteUInt16:
      return sizeof(uint16_t);
    default:
      return std::nullopt;
  }
}

std::optional<TensorType> TflToTensorType(TfLiteType tfl_type) {
  switch (tfl_type) {
    case kTfLiteInt32:
      return TensorType_INT32;
    case kTfLiteFloat32:
      return TensorType_FLOAT32;
    case kTfLiteInt64:
      return TensorType_INT64;
    case kTfLiteBool:
      return TensorType_BOOL;
    default:
      return std::nullopt;
  }
}

}  // namespace tflite
