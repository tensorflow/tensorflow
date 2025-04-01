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

#ifndef TENSORFLOW_CORE_FRAMEWORK_VARIANT_ENCODE_DECODE_H_
#define TENSORFLOW_CORE_FRAMEWORK_VARIANT_ENCODE_DECODE_H_

#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/type_index.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/strings/strcat.h"
#include "tensorflow/core/platform/abi.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// Type used for tag-dispatch of the Encode/Decode Variant implementations. This
// template can determine whether the first type parameter `T` is one of the
// following:
//
// * A POD type (TypeResolver<T, true>)
// * A tensorflow::Tensor (TypeResolver<T, false, true>)
// * A protocol buffer (TypeResolver<T, false, false, true>)
// * None of the above (TypeResolver<T, false, false, false>)
//
template <
    typename T,
    bool = std::is_trivially_copyable<typename std::decay<T>::type>::value,
    bool =
        std::is_same<typename std::decay<T>::type, ::tensorflow::Tensor>::value,
    bool = std::is_base_of<protobuf::MessageLite,
                           typename std::decay<T>::type>::value>
struct TypeResolver {};

// Specialization for POD type
template <typename T>
void EncodeVariantImpl(const T& value, TypeResolver<T, true /* is_pod */>,
                       VariantTensorData* data) {
  data->set_metadata(value);
}

// Specialization for tensorflow::Tensor
template <typename T>
void EncodeVariantImpl(const T& value,
                       TypeResolver<T, false /* is_pod */, true /* Tensor */>,
                       VariantTensorData* data) {
  data->tensors_.clear();
  data->tensors_.push_back(value);
}

// Specialization for protobuf
template <typename T>
void EncodeVariantImpl(const T& value,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    true /* protobuf */>,
                       VariantTensorData* data) {
  if (!value.SerializeToString(&data->metadata_)) {
    data->metadata_.clear();
    LOG(ERROR) << "Failed to encode variant " << value.DebugString();
  }
}

// Specialization for other types
template <typename T>
void EncodeVariantImpl(const T& value,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    false /* protobuf */>,
                       VariantTensorData* data) {
  value.Encode(data);
}

// Specialization for POD type
template <typename T>
bool DecodeVariantImpl(VariantTensorData data,
                       TypeResolver<T, true /* is_pod */, false /* Tensor */,
                                    false /* protobuf */>,
                       T* value) {
  return data.get_metadata(value);
}

// Specialization for tensorflow::Tensor
template <typename T>
bool DecodeVariantImpl(VariantTensorData data,
                       TypeResolver<T, false /* is_pod */, true /* Tensor */,
                                    false /* protobuf */>,
                       T* value) {
  *value = data.tensors(0);
  return true;
}

// Specialization for protobuf
template <typename T>
bool DecodeVariantImpl(VariantTensorData data,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    true /* protobuf */>,
                       T* value) {
  std::string metadata;
  data.get_metadata(&metadata);
  return value->ParseFromString(std::move(metadata));
}

// Specialization for other types
template <typename T>
bool DecodeVariantImpl(VariantTensorData data,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    false /* protobuf */>,
                       T* value) {
  return value->Decode(std::move(data));
}

template <typename C, typename = void>
struct has_type_name : std::false_type {};

template <typename C>
struct has_type_name<
    C, typename std::enable_if<std::is_same<
           decltype(std::declval<C>().TypeName()), string>::value>::type>
    : std::true_type {};

template <typename T, bool = has_type_name<typename std::decay<T>::type>::value,
          bool = std::is_same<typename std::decay<T>::type,
                              ::tensorflow::Tensor>::value,
          bool = std::is_base_of<protobuf::MessageLite,
                                 typename std::decay<T>::type>::value>
struct TypeNameResolver {};

template <typename T>
std::string TypeNameVariantImpl(const T& value,
                                TypeNameResolver<T, true /* has_type_name */>) {
  return value.TypeName();
}

template <typename T>
std::string TypeNameVariantImpl(
    const T& value,
    TypeNameResolver<T, false /* has_type_name */, true /* Tensor */>) {
  return "tensorflow::Tensor";
}

template <typename T>
std::string TypeNameVariantImpl(
    const T& value, TypeNameResolver<T, false /* has_type_name */,
                                     false /* Tensor */, true /* protobuf */>) {
  return std::string(value.GetTypeName());
}

template <typename T>
std::string TypeNameVariantImpl(
    const T& value,
    TypeNameResolver<T, false /* has_type_name */, false /* Tensor */,
                     false /* protobuf */>) {
  return port::MaybeAbiDemangle(TypeIndex::Make<T>().name());
}

template <typename T>
std::string TypeNameVariant(const T& value) {
  return TypeNameVariantImpl(value, TypeNameResolver<T>());
}

template <typename C, typename = void>
struct has_debug_string : std::false_type {};

template <typename C>
struct has_debug_string<
    C, typename std::enable_if<std::is_same<
           decltype(std::declval<C>().DebugString()), string>::value>::type>
    : std::true_type {};

template <typename C, typename = void>
struct can_strcat : std::false_type {};

template <typename C>
struct can_strcat<
    C, typename std::enable_if<std::is_same<
           decltype(strings::StrCat(std::declval<C>())), string>::value>::type>
    : std::true_type {};

template <typename T,
          bool = has_debug_string<typename std::decay<T>::type>::value,
          bool = can_strcat<typename std::decay<T>::type>::value>
struct DebugStringResolver {};

// TODO(ebrevdo): Expand DebugStringResolver to return TypeString if
// there is no StrCat<T>() constructor.
template <typename T>
std::string DebugStringVariantImpl(
    const T& value, DebugStringResolver<T, true /* has_debug_string */>) {
  return value.DebugString();
}

template <typename T>
std::string DebugStringVariantImpl(
    const T& value, DebugStringResolver<T, false /* has_debug_string */,
                                        true /* can_strcat */>) {
  return strings::StrCat(value);
}

template <typename T>
std::string DebugStringVariantImpl(
    const T& value, DebugStringResolver<T, false /* has_debug_string */,
                                        false /* can_strcat */>) {
  return "?";
}

template <typename T>
std::string DebugStringVariant(const T& value) {
  return DebugStringVariantImpl(value, DebugStringResolver<T>());
}

template <typename T>
void EncodeVariant(const T& value, VariantTensorData* data) {
  EncodeVariantImpl(value, TypeResolver<T>(), data);
  data->set_type_name(TypeNameVariant(value));
}

template <typename T>
bool DecodeVariant(VariantTensorData* data, T* value) {
  return DecodeVariantImpl(std::move(*data), TypeResolver<T>(), value);
}

template <typename T>
void EncodeVariant(const T& value, std::string* buf) {
  VariantTensorData data;
  EncodeVariantImpl(value, TypeResolver<T>(), &data);
  data.set_type_name(TypeNameVariant(value));
  DCHECK(buf != nullptr);
  data.SerializeToString(buf);
}

template <typename T>
bool DecodeVariant(std::string* buf, T* value) {
  VariantTensorData data;
  if (!data.ParseFromString(*buf)) return false;
  if (!DecodeVariantImpl(std::move(data), TypeResolver<T>(), value)) {
    return false;
  }
  return true;
}

// Specializations for VariantTensorDataProto
template <>
std::string TypeNameVariant(const VariantTensorDataProto& value);

template <>
void EncodeVariant(const VariantTensorDataProto& value,
                   VariantTensorData* data);

template <>
bool DecodeVariant(VariantTensorData* data, VariantTensorDataProto* value);

template <>
void EncodeVariant(const VariantTensorDataProto& value, std::string* buf);

template <>
bool DecodeVariant(std::string* buf, VariantTensorDataProto* value);

// Encodes an array of Variant objects in to the given StringListEncoder.
// `variant_array` is assumed to point to an array of `n` Variant objects.
void EncodeVariantList(const Variant* variant_array, int64_t n,
                       std::unique_ptr<port::StringListEncoder> e);

// Decodes an array of Variant objects from the given StringListDecoder.
// `variant_array` is assumed to point to an array of `n` Variant objects.
bool DecodeVariantList(std::unique_ptr<port::StringListDecoder> d,
                       Variant* variant_array, int64_t n);

}  // end namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_VARIANT_ENCODE_DECODE_H_
