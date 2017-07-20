/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_FRAMEWORK_VARIANT_ENCODE_DECODE_H_
#define TENSORFLOW_FRAMEWORK_VARIANT_ENCODE_DECODE_H_

#include <iostream>
#include <type_traits>
#include <utility>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/platform/protobuf.h"

namespace tensorflow {

// The serialization format for Variant objects. Objects with references to
// other Tensors can simply store those tensors in the `tensors` field, and
// serialize other metadata content in to the `metadata` field. Objects can
// optionally set the `type_name` for type-checking before deserializing an
// object.
struct VariantTensorData {
  string type_name;
  string metadata;
  std::vector<Tensor> tensors;
  void ToProto(VariantTensorDataProto* proto) const;
  bool FromProto(const VariantTensorDataProto& proto);
};

// Type used for tag-dispatch of the Encode/Decode Variant implementations. This
// template can determine whether the first type parameter `T` is one of the
// following:
//
// * A POD type (TypeResolver<T, true>)
// * A tensorflow::Tensor (TypeResolver<T, false, true>)
// * A protocol buffer (TypeResolver<T, false, false, true>)
// * None of the above (TypeResolver<T, false, false, false>)
//
template <typename T, bool = std::is_pod<typename std::decay<T>::type>::value,
          bool = std::is_same<typename std::decay<T>::type,
                              ::tensorflow::Tensor>::value,
          bool = std::is_base_of<protobuf::MessageLite,
                                 typename std::decay<T>::type>::value>
struct TypeResolver {};

// Specialization for POD type
template <typename T>
void EncodeVariantImpl(T&& value, TypeResolver<T, true /* is_pod */>,
                       VariantTensorData* data) {
  data->metadata.assign(reinterpret_cast<const char*>(&value), sizeof(value));
}

// Specialization for tensorflow::Tensor
template <typename T>
void EncodeVariantImpl(T&& value,
                       TypeResolver<T, false /* is_pod */, true /* Tensor */>,
                       VariantTensorData* data) {
  data->tensors.clear();
  data->tensors.push_back(value);
}

// Specialization for protobuf
template <typename T>
void EncodeVariantImpl(T&& value,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    true /* protobuf */>,
                       VariantTensorData* data) {
  value.SerializeToString(&data->metadata);
}

// Specialization for other types
template <typename T>
void EncodeVariantImpl(T&& value,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    false /* protobuf */>,
                       VariantTensorData* data) {
  value.Encode(data);
}

// Specialization for POD type
template <typename T>
bool DecodeVariantImpl(const VariantTensorData& data,
                       TypeResolver<T, true /* is_pod */>, T* value) {
  std::copy_n(data.metadata.data(), sizeof(*value),
              reinterpret_cast<char*>(value));
  return true;
}

// Specialization for tensorflow::Tensor
template <typename T>
bool DecodeVariantImpl(const VariantTensorData& data,
                       TypeResolver<T, false /* is_pod */, true /* Tensor */>,
                       T* value) {
  *value = data.tensors[0];
  return true;
}

// Specialization for protobuf
template <typename T>
bool DecodeVariantImpl(const VariantTensorData& data,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    true /* protobuf */>,
                       T* value) {
  return value->ParseFromString(data.metadata);
}

// Specialization for other types
template <typename T>
bool DecodeVariantImpl(const VariantTensorData& data,
                       TypeResolver<T, false /* is_pod */, false /* Tensor */,
                                    false /* protobuf */>,
                       T* value) {
  return value->Decode(data);
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
string TypeNameVariantImpl(T&& value,
                           TypeNameResolver<T, true /* has_type_name */>) {
  return value.TypeName();
}

template <typename T>
string TypeNameVariantImpl(
    T&& value,
    TypeNameResolver<T, false /* has_type_name */, true /* Tensor */>) {
  return "tensorflow::Tensor";
}

template <typename T>
string TypeNameVariantImpl(
    T&& value, TypeNameResolver<T, false /* has_type_name */,
                                false /* Tensor */, true /* protobuf */>) {
  return value.GetTypeName();
}

template <typename T>
string TypeNameVariantImpl(
    T&& value, TypeNameResolver<T, false /* has_type_name */,
                                false /* Tensor */, false /* protobuf */>) {
  return value.TypeName();
}

template <typename T>
string TypeNameVariant(T&& value) {
  return TypeNameVariantImpl(std::forward<T>(value), TypeNameResolver<T>());
}

template <typename T>
void EncodeVariant(T&& value, VariantTensorData* data) {
  EncodeVariantImpl(std::forward<T>(value), TypeResolver<T>(), data);
}

template <typename T>
bool DecodeVariant(const VariantTensorData& data, T* value) {
  return DecodeVariantImpl(data, TypeResolver<T>(), value);
}

template <typename T>
void EncodeVariant(T&& value, string* buf) {
  VariantTensorData data;
  EncodeVariantImpl(std::forward<T>(value), TypeResolver<T>(), &data);
  VariantTensorDataProto proto;
  data.ToProto(&proto);
  proto.SerializeToString(buf);
}

template <typename T>
bool DecodeVariant(const string& buf, T* value) {
  VariantTensorDataProto proto;
  if (!proto.ParseFromString(buf)) return false;
  VariantTensorData data;
  if (!data.FromProto(proto)) return false;
  if (!DecodeVariantImpl(data, TypeResolver<T>(), value)) return false;
  return true;
}

// Specializations for VariantTensorDataProto
template <>
string TypeNameVariant(VariantTensorDataProto&& value);
template <>
void EncodeVariant(VariantTensorDataProto&& value, VariantTensorData* data);
template <>
bool DecodeVariant(const VariantTensorData& data,
                   VariantTensorDataProto* value);
template <>
void EncodeVariant(VariantTensorDataProto&& value, string* buf);
template <>
bool DecodeVariant(const string& buf, VariantTensorDataProto* value);

}  // end namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_VARIANT_ENCODE_DECODE_H_
