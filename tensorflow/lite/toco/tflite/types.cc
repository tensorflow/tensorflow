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
#include "tensorflow/lite/toco/tflite/types.h"
#include "tensorflow/lite/string_util.h"

namespace toco {

namespace tflite {

namespace {

DataBuffer::FlatBufferOffset CopyStringToBuffer(
    const Array& array, flatbuffers::FlatBufferBuilder* builder) {
  const auto& src_data = array.GetBuffer<ArrayDataType::kString>().data;
  ::tflite::DynamicBuffer dyn_buffer;
  for (const std::string& str : src_data) {
    dyn_buffer.AddString(str.c_str(), str.length());
  }
  char* tensor_buffer;
  int bytes = dyn_buffer.WriteToBuffer(&tensor_buffer);
  std::vector<uint8_t> dst_data(bytes);
  memcpy(dst_data.data(), tensor_buffer, bytes);
  free(tensor_buffer);
  return builder->CreateVector(dst_data.data(), bytes);
}

// vector<bool> may be implemented using a bit-set, so we can't just
// reinterpret_cast, accessing its data as vector<bool> and let flatbuffer
// CreateVector handle it.
// Background: https://isocpp.org/blog/2012/11/on-vectorbool
DataBuffer::FlatBufferOffset CopyBoolToBuffer(
    const Array& array, flatbuffers::FlatBufferBuilder* builder) {
  const auto& src_data = array.GetBuffer<ArrayDataType::kBool>().data;
  return builder->CreateVector(src_data);
}

template <ArrayDataType T>
DataBuffer::FlatBufferOffset CopyBuffer(
    const Array& array, flatbuffers::FlatBufferBuilder* builder) {
  using NativeT = ::toco::DataType<T>;
  const auto& src_data = array.GetBuffer<T>().data;
  const uint8_t* dst_data = reinterpret_cast<const uint8_t*>(src_data.data());
  auto size = src_data.size() * sizeof(NativeT);
  return builder->CreateVector(dst_data, size);
}

void CopyStringFromBuffer(const ::tflite::Buffer& buffer, Array* array) {
  auto* src_data = reinterpret_cast<const char*>(buffer.data()->data());
  std::vector<std::string>* dst_data =
      &array->GetMutableBuffer<ArrayDataType::kString>().data;
  int32_t num_strings = ::tflite::GetStringCount(src_data);
  for (int i = 0; i < num_strings; i++) {
    ::tflite::StringRef str_ref = ::tflite::GetString(src_data, i);
    std::string this_str(str_ref.str, str_ref.len);
    dst_data->push_back(this_str);
  }
}

template <ArrayDataType T>
void CopyBuffer(const ::tflite::Buffer& buffer, Array* array) {
  using NativeT = ::toco::DataType<T>;
  auto* src_buffer = buffer.data();
  const NativeT* src_data =
      reinterpret_cast<const NativeT*>(src_buffer->data());
  int num_items = src_buffer->size() / sizeof(NativeT);

  std::vector<NativeT>* dst_data = &array->GetMutableBuffer<T>().data;
  for (int i = 0; i < num_items; ++i) {
    dst_data->push_back(*src_data);
    ++src_data;
  }
}
}  // namespace

::tflite::TensorType DataType::Serialize(ArrayDataType array_data_type) {
  switch (array_data_type) {
    case ArrayDataType::kFloat:
      return ::tflite::TensorType_FLOAT32;
    case ArrayDataType::kInt16:
      return ::tflite::TensorType_INT16;
    case ArrayDataType::kInt32:
      return ::tflite::TensorType_INT32;
    case ArrayDataType::kInt64:
      return ::tflite::TensorType_INT64;
    case ArrayDataType::kUint8:
      return ::tflite::TensorType_UINT8;
    case ArrayDataType::kString:
      return ::tflite::TensorType_STRING;
    case ArrayDataType::kBool:
      return ::tflite::TensorType_BOOL;
    case ArrayDataType::kComplex64:
      return ::tflite::TensorType_COMPLEX64;
    default:
      // FLOAT32 is filled for unknown data types.
      // TODO(ycling): Implement type inference in TF Lite interpreter.
      return ::tflite::TensorType_FLOAT32;
  }
}

ArrayDataType DataType::Deserialize(int tensor_type) {
  switch (::tflite::TensorType(tensor_type)) {
    case ::tflite::TensorType_FLOAT32:
      return ArrayDataType::kFloat;
    case ::tflite::TensorType_INT16:
      return ArrayDataType::kInt16;
    case ::tflite::TensorType_INT32:
      return ArrayDataType::kInt32;
    case ::tflite::TensorType_INT64:
      return ArrayDataType::kInt64;
    case ::tflite::TensorType_STRING:
      return ArrayDataType::kString;
    case ::tflite::TensorType_UINT8:
      return ArrayDataType::kUint8;
    case ::tflite::TensorType_BOOL:
      return ArrayDataType::kBool;
    case ::tflite::TensorType_COMPLEX64:
      return ArrayDataType::kComplex64;
    default:
      LOG(FATAL) << "Unhandled tensor type '" << tensor_type << "'.";
  }
}

flatbuffers::Offset<flatbuffers::Vector<uint8_t>> DataBuffer::Serialize(
    const Array& array, flatbuffers::FlatBufferBuilder* builder) {
  if (!array.buffer) return 0;  // an empty buffer, usually an output.

  switch (array.data_type) {
    case ArrayDataType::kFloat:
      return CopyBuffer<ArrayDataType::kFloat>(array, builder);
    case ArrayDataType::kInt16:
      return CopyBuffer<ArrayDataType::kInt16>(array, builder);
    case ArrayDataType::kInt32:
      return CopyBuffer<ArrayDataType::kInt32>(array, builder);
    case ArrayDataType::kInt64:
      return CopyBuffer<ArrayDataType::kInt64>(array, builder);
    case ArrayDataType::kString:
      return CopyStringToBuffer(array, builder);
    case ArrayDataType::kUint8:
      return CopyBuffer<ArrayDataType::kUint8>(array, builder);
    case ArrayDataType::kBool:
      return CopyBoolToBuffer(array, builder);
    case ArrayDataType::kComplex64:
      return CopyBuffer<ArrayDataType::kComplex64>(array, builder);
    default:
      LOG(FATAL) << "Unhandled array data type.";
  }
}

void DataBuffer::Deserialize(const ::tflite::Tensor& tensor,
                             const ::tflite::Buffer& buffer, Array* array) {
  if (tensor.buffer() == 0) return;      // an empty buffer, usually an output.
  if (buffer.data() == nullptr) return;  // a non-defined buffer.

  switch (tensor.type()) {
    case ::tflite::TensorType_FLOAT32:
      return CopyBuffer<ArrayDataType::kFloat>(buffer, array);
    case ::tflite::TensorType_INT16:
      return CopyBuffer<ArrayDataType::kInt16>(buffer, array);
    case ::tflite::TensorType_INT32:
      return CopyBuffer<ArrayDataType::kInt32>(buffer, array);
    case ::tflite::TensorType_INT64:
      return CopyBuffer<ArrayDataType::kInt64>(buffer, array);
    case ::tflite::TensorType_STRING:
      return CopyStringFromBuffer(buffer, array);
    case ::tflite::TensorType_UINT8:
      return CopyBuffer<ArrayDataType::kUint8>(buffer, array);
    case ::tflite::TensorType_BOOL:
      return CopyBuffer<ArrayDataType::kBool>(buffer, array);
    case ::tflite::TensorType_COMPLEX64:
      return CopyBuffer<ArrayDataType::kComplex64>(buffer, array);
    default:
      LOG(FATAL) << "Unhandled tensor type.";
  }
}

::tflite::Padding Padding::Serialize(PaddingType padding_type) {
  switch (padding_type) {
    case PaddingType::kSame:
      return ::tflite::Padding_SAME;
    case PaddingType::kValid:
      return ::tflite::Padding_VALID;
    default:
      LOG(FATAL) << "Unhandled padding type.";
  }
}

PaddingType Padding::Deserialize(int padding) {
  switch (::tflite::Padding(padding)) {
    case ::tflite::Padding_SAME:
      return PaddingType::kSame;
    case ::tflite::Padding_VALID:
      return PaddingType::kValid;
    default:
      LOG(FATAL) << "Unhandled padding.";
  }
}

::tflite::ActivationFunctionType ActivationFunction::Serialize(
    FusedActivationFunctionType faf_type) {
  switch (faf_type) {
    case FusedActivationFunctionType::kNone:
      return ::tflite::ActivationFunctionType_NONE;
    case FusedActivationFunctionType::kRelu:
      return ::tflite::ActivationFunctionType_RELU;
    case FusedActivationFunctionType::kRelu6:
      return ::tflite::ActivationFunctionType_RELU6;
    case FusedActivationFunctionType::kRelu1:
      return ::tflite::ActivationFunctionType_RELU_N1_TO_1;
    default:
      LOG(FATAL) << "Unhandled fused activation function type.";
  }
}

FusedActivationFunctionType ActivationFunction::Deserialize(
    int activation_function) {
  switch (::tflite::ActivationFunctionType(activation_function)) {
    case ::tflite::ActivationFunctionType_NONE:
      return FusedActivationFunctionType::kNone;
    case ::tflite::ActivationFunctionType_RELU:
      return FusedActivationFunctionType::kRelu;
    case ::tflite::ActivationFunctionType_RELU6:
      return FusedActivationFunctionType::kRelu6;
    case ::tflite::ActivationFunctionType_RELU_N1_TO_1:
      return FusedActivationFunctionType::kRelu1;
    default:
      LOG(FATAL) << "Unhandled fused activation function type.";
  }
}

}  // namespace tflite

}  // namespace toco
