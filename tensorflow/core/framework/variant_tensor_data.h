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

#ifndef TENSORFLOW_CORE_FRAMEWORK_VARIANT_TENSOR_DATA_H_
#define TENSORFLOW_CORE_FRAMEWORK_VARIANT_TENSOR_DATA_H_

#include <algorithm>
#include <vector>

#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class VariantTensorDataProto;

// The serialization format for Variant objects. Objects with references to
// other Tensors can simply store those tensors in the `tensors` field, and
// serialize other metadata content in to the `metadata` field. Objects can
// optionally set the `type_name` for type-checking before deserializing an
// object.
//
// This is the native C++ class equivalent of VariantTensorDataProto. They are
// separate so that kernels do not need to depend on protos.
class VariantTensorData {
 public:
  VariantTensorData();
  // TODO(b/118823936): This silently returns if the proto is invalid.
  // Consider calling FromProto explicitly instead.
  VariantTensorData(VariantTensorDataProto proto);
  ~VariantTensorData();

  // Name of the type of objects being serialized.
  const string& type_name() const { return type_name_; }
  void set_type_name(const string& type_name) { type_name_ = type_name; }

  template <typename T, bool = std::is_pod<typename std::decay<T>::type>::value>
  struct PODResolver {};

  // Portions of the object that are not Tensors.
  // Directly supported types include string POD types.
  template <typename T>
  void set_metadata(const T& value) {
    SetMetadata<T>(value, PODResolver<T>());
  }

  template <typename T>
  bool get_metadata(T* value) const {
    return GetMetadata<T>(value, PODResolver<T>());
  }

  // Tensors contained within objects being serialized.
  int tensors_size() const;
  const Tensor& tensors(int index) const;
  const std::vector<Tensor>& tensors() const;
  Tensor* add_tensors();

  // Conversion to and from VariantTensorDataProto
  void ToProto(VariantTensorDataProto* proto) const;
  // This allows optimizations via std::move.
  bool FromProto(VariantTensorDataProto proto);
  bool FromConstProto(const VariantTensorDataProto& proto);

  // Serialization via VariantTensorDataProto
  string SerializeAsString() const;
  bool SerializeToString(string* buf);
  bool ParseFromString(string s);

  string DebugString() const;

 public:
  string type_name_;
  string metadata_;
  std::vector<Tensor> tensors_;

 private:
  template <typename T>
  void SetMetadata(const string& value, PODResolver<T, false /* is_pod */>) {
    metadata_ = value;
  }

  template <typename T>
  bool GetMetadata(string* value, PODResolver<T, false /* is_pod */>) const {
    *value = metadata_;
    return true;
  }

  template <typename T>
  void SetMetadata(const T& value, PODResolver<T, true /* is_pod */>) {
    metadata_.assign(reinterpret_cast<const char*>(&value), sizeof(T));
  }

  template <typename T>
  bool GetMetadata(T* value, PODResolver<T, true /* is_pod */>) const {
    if (metadata_.size() != sizeof(T)) return false;
    std::copy_n(metadata_.data(), sizeof(T), reinterpret_cast<char*>(value));
    return true;
  }
};

// For backwards compatibility for when this was a proto
string ProtoDebugString(const VariantTensorData& object);

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_VARIANT_TENSOR_DATA_H_
