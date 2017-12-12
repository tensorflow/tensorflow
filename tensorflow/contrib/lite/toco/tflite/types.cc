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
#include "tensorflow/contrib/lite/toco/tflite/types.h"

namespace toco {

namespace tflite {

namespace {
template <ArrayDataType T>
DataBuffer::FlatBufferOffset CopyBuffer(
    const Array& array, flatbuffers::FlatBufferBuilder* builder) {
  using NativeT = ::toco::DataType<T>;
  const auto& src_data = array.GetBuffer<T>().data;
  const uint8_t* dst_data = reinterpret_cast<const uint8_t*>(src_data.data());
  auto size = src_data.size() * sizeof(NativeT);
  return builder->CreateVector(dst_data, size);
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
    case ArrayDataType::kInt32:
      return ::tflite::TensorType_INT32;
    case ArrayDataType::kUint8:
      return ::tflite::TensorType_UINT8;
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
    case ::tflite::TensorType_INT32:
      return ArrayDataType::kInt32;
    case ::tflite::TensorType_UINT8:
      return ArrayDataType::kUint8;
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
    case ArrayDataType::kInt32:
      return CopyBuffer<ArrayDataType::kInt32>(array, builder);
    case ArrayDataType::kUint8:
      return CopyBuffer<ArrayDataType::kUint8>(array, builder);
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
    case ::tflite::TensorType_INT32:
      return CopyBuffer<ArrayDataType::kInt32>(buffer, array);
    case ::tflite::TensorType_UINT8:
      return CopyBuffer<ArrayDataType::kUint8>(buffer, array);
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
      return ::tflite::ActivationFunctionType_RELU1;
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
    case ::tflite::ActivationFunctionType_RELU1:
      return FusedActivationFunctionType::kRelu1;
    default:
      LOG(FATAL) << "Unhandled fused activation function type.";
  }
}

}  // namespace tflite

}  // namespace toco
