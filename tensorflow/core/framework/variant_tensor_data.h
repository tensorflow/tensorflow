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

#ifndef TENSORFLOW_FRAMEWORK_VARIANT_TENSOR_DATA_H
#define TENSORFLOW_FRAMEWORK_VARIANT_TENSOR_DATA_H

#include <vector>

#include "tensorflow/core/platform/types.h"

namespace tensorflow {

class VariantTensorDataProto;
class Tensor;

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
  VariantTensorData(const VariantTensorDataProto& proto);
  ~VariantTensorData();

  // Name of the type of objects being serialized.
  const string& type_name() const { return type_name_; }
  void set_type_name(const string& type_name) { type_name_ = type_name; }

  // Portions of the object that are not Tensors.
  const string& metadata() const { return metadata_; }
  void set_metadata(const string& metadata) { metadata_ = metadata; }

  // Tensors contained within objects being serialized.
  int tensors_size();
  const Tensor& tensors(int index) const;
  std::vector<Tensor> tensors();
  Tensor* add_tensors();

  // Conversion to and from VariantTensorDataProto
  void ToProto(VariantTensorDataProto* proto) const;
  bool FromProto(const VariantTensorDataProto& proto);

  // Serialization via VariantTensorDataProto
  string SerializeAsString() const;
  bool SerializeToString(string* buf);
  bool ParseFromString(const string& s);

  string DebugString() const;

 public:
  string type_name_;
  string metadata_;
  std::vector<Tensor> tensors_;
};

// For backwards compatibility for when this was a proto
string ProtoDebugString(const VariantTensorData& object);

}  // namespace tensorflow

#endif  // TENSORFLOW_FRAMEWORK_VARIANT_TENSOR_DATA_H
