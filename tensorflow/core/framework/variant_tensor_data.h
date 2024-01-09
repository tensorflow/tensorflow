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
  VariantTensorData() = default;

  // TODO(b/118823936): This silently returns if the proto is invalid.
  // Consider calling FromProto explicitly instead.
  VariantTensorData(VariantTensorDataProto proto);

  // Name of the type of objects being serialized.
  const std::string& type_name() const { return type_name_; }
  void set_type_name(const std::string& type_name) { type_name_ = type_name; }

  template <typename T, bool = std::is_pod<typename std::decay<T>::type>::value>
  struct PODResolver {};

  // Portions of the object that are not Tensors.
  // Directly supported types include string POD types.
  template <typename T>
  void set_metadata(const T& value) {
    SetMetadata(value, PODResolver<T>());
  }

  template <typename T>
  bool get_metadata(T* value) const {
    return GetMetadata(value, PODResolver<T>());
  }

  std::string& metadata_string() { return metadata_; }

  const std::string& metadata_string() const { return metadata_; }

  // Tensors contained within objects being serialized.
  int tensors_size() const;
  const Tensor& tensors(int index) const;
  const std::vector<Tensor>& tensors() const;
  Tensor* add_tensors();

  // A more general version of add_tensors. Parameters are perfectly forwarded
  // to the constructor of the tensor added here.
  template <typename... TensorConstructorArgs>
  Tensor* add_tensor(TensorConstructorArgs&&... args);

  // Conversion to and from VariantTensorDataProto
  void ToProto(VariantTensorDataProto* proto) const;
  // This allows optimizations via std::move.
  bool FromProto(VariantTensorDataProto proto);
  bool FromConstProto(const VariantTensorDataProto& proto);

  // Serialization via VariantTensorDataProto
  std::string SerializeAsString() const;
  bool SerializeToString(std::string* buf);
  bool ParseFromString(std::string s);

  std::string DebugString() const;

 public:
  std::string type_name_;
  std::string metadata_;
  std::vector<Tensor> tensors_;

 private:
  void SetMetadata(const std::string& value,
                   PODResolver<std::string, false /* is_pod */>) {
    metadata_ = value;
  }

  bool GetMetadata(std::string* value,
                   PODResolver<std::string, false /* is_pod */>) const {
    *value = metadata_;
    return true;
  }

  // Specialize for bool, it is undefined behvaior to assign a non 0/1 value to
  // a bool. Now we coerce a non-zero value to true.
  bool GetMetadata(bool* value, PODResolver<bool, true /* is_pod */>) const {
    if (metadata_.size() != sizeof(bool)) return false;
    *value = false;
    for (size_t i = 0; i < sizeof(bool); ++i)
      *value = *value || (metadata_.data()[i] != 0);
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
std::string ProtoDebugString(const VariantTensorData& object);

template <typename... TensorConstructorArgs>
Tensor* VariantTensorData::add_tensor(TensorConstructorArgs&&... args) {
  tensors_.emplace_back(std::forward<TensorConstructorArgs>(args)...);
  return &tensors_.back();
}

}  // namespace tensorflow

#endif  // TENSORFLOW_CORE_FRAMEWORK_VARIANT_TENSOR_DATA_H_
