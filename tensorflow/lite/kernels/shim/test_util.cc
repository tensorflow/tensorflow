/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/kernels/shim/test_util.h"

#include <string>

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/lite/kernels/internal/compatibility.h"
#include "tensorflow/lite/string_util.h"

namespace tflite {

using std::size_t;

TfLiteTensor* UniqueTfLiteTensor::get() { return tensor_; }

TfLiteTensor& UniqueTfLiteTensor::operator*() { return *tensor_; }

TfLiteTensor* UniqueTfLiteTensor::operator->() { return tensor_; }

const TfLiteTensor* UniqueTfLiteTensor::operator->() const { return tensor_; }

void UniqueTfLiteTensor::reset(TfLiteTensor* tensor) { tensor_ = tensor; }

UniqueTfLiteTensor::~UniqueTfLiteTensor() { TfLiteTensorFree(tensor_); }

namespace {

template <typename T>
std::string TensorValueToString(const ::TfLiteTensor* tensor,
                                const size_t idx) {
  TFLITE_DCHECK_EQ(tensor->type, ::tflite::typeToTfLiteType<T>());
  const T* val_array = reinterpret_cast<const T*>(tensor->data.raw);
  return std::to_string(val_array[idx]);
}

template <>
std::string TensorValueToString<bool>(const ::TfLiteTensor* tensor,
                                      const size_t idx) {
  TFLITE_DCHECK_EQ(tensor->type, ::tflite::typeToTfLiteType<bool>());
  const bool* val_array = reinterpret_cast<const bool*>(tensor->data.raw);
  return val_array[idx] ? "1" : "0";
}

template <typename FloatType>
std::string TensorValueToStringFloat(const ::TfLiteTensor* tensor,
                                     const size_t idx) {
  TFLITE_DCHECK_EQ(tensor->type, ::tflite::typeToTfLiteType<FloatType>());
  const FloatType* val_array =
      reinterpret_cast<const FloatType*>(tensor->data.raw);
  std::stringstream ss;
  ss << val_array[idx];
  return std::string(ss.str().data(), ss.str().length());
}

template <>
std::string TensorValueToString<float>(const ::TfLiteTensor* tensor,
                                       const size_t idx) {
  return TensorValueToStringFloat<float>(tensor, idx);
}

template <>
std::string TensorValueToString<double>(const ::TfLiteTensor* tensor,
                                        const size_t idx) {
  return TensorValueToStringFloat<double>(tensor, idx);
}

template <>
std::string TensorValueToString<StringRef>(const ::TfLiteTensor* tensor,
                                           const size_t idx) {
  TFLITE_DCHECK_EQ(tensor->type, kTfLiteString);
  const auto ref = ::tflite::GetString(tensor, idx);
  return std::string(ref.str, ref.len);
}

std::string TfliteTensorDebugStringImpl(const ::TfLiteTensor* tensor,
                                        const size_t axis,
                                        const size_t max_values,
                                        size_t* start_idx) {
  const size_t dim_size = tensor->dims->data[axis];
  if (axis == tensor->dims->size - 1) {
    std::vector<std::string> ret_list;
    ret_list.reserve(dim_size);
    int idx = *start_idx;
    for (int i = 0; i < dim_size && idx < max_values; ++i, ++idx) {
      std::string val_str;
      switch (tensor->type) {
        case kTfLiteBool: {
          val_str = TensorValueToString<bool>(tensor, idx);
          break;
        }
        case kTfLiteUInt8: {
          val_str = TensorValueToString<uint8_t>(tensor, idx);
          break;
        }
        case kTfLiteInt8: {
          val_str = TensorValueToString<int8_t>(tensor, idx);
          break;
        }
        case kTfLiteInt16: {
          val_str = TensorValueToString<int16_t>(tensor, idx);
          break;
        }
        case kTfLiteInt32: {
          val_str = TensorValueToString<int32_t>(tensor, idx);
          break;
        }
        case kTfLiteInt64: {
          val_str = TensorValueToString<int64_t>(tensor, idx);
          break;
        }
        case kTfLiteString: {
          val_str = TensorValueToString<StringRef>(tensor, idx);
          break;
        }
        case kTfLiteFloat32: {
          val_str = TensorValueToString<float>(tensor, idx);
          break;
        }
        case kTfLiteFloat64: {
          val_str = TensorValueToString<double>(tensor, idx);
          break;
        }
        default: {
          val_str = "unsupported_type";
        }
      }
      ret_list.push_back(val_str);
    }
    *start_idx = idx;
    if (idx == max_values && ret_list.size() < dim_size) {
      ret_list.push_back("...");
    }
    return absl::StrCat("[", absl::StrJoin(ret_list, ", "), "]");
  } else {
    std::vector<std::string> ret_list;
    ret_list.reserve(dim_size);
    for (int i = 0; i < dim_size && *start_idx < max_values; ++i) {
      ret_list.push_back(
          TfliteTensorDebugStringImpl(tensor, axis + 1, max_values, start_idx));
    }
    return absl::StrCat("[", absl::StrJoin(ret_list, ", "), "]");
  }
}

}  // namespace

std::string TfliteTensorDebugString(const ::TfLiteTensor* tensor,
                                    const size_t max_values) {
  if (tensor->dims->size == 0) return "";
  size_t start_idx = 0;
  return TfliteTensorDebugStringImpl(tensor, 0, max_values, &start_idx);
}

size_t NumTotalFromShape(const std::initializer_list<int>& shape) {
  size_t num_total;
  if (shape.size() > 0)
    num_total = 1;
  else
    num_total = 0;
  for (const int dim : shape) num_total *= dim;
  return num_total;
}

template <>
void PopulateTfLiteTensorValue<std::string>(
    const std::initializer_list<std::string> values, TfLiteTensor* tensor) {
  tflite::DynamicBuffer buf;
  for (const std::string& s : values) {
    buf.AddString(s.data(), s.length());
  }
  buf.WriteToTensor(tensor, /*new_shape=*/nullptr);
}

}  // namespace tflite
