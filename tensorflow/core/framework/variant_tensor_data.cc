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

#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {

VariantTensorData::VariantTensorData(VariantTensorDataProto proto) {
  FromProto(std::move(proto));
}

int VariantTensorData::tensors_size() const { return tensors_.size(); }

const Tensor& VariantTensorData::tensors(int index) const {
  return tensors_[index];
}

const std::vector<Tensor>& VariantTensorData::tensors() const {
  return tensors_;
}

Tensor* VariantTensorData::add_tensors() {
  tensors_.emplace_back();
  return &(tensors_[tensors_.size() - 1]);
}

template <typename... TensorConstructorArgs>
Tensor* VariantTensorData::add_tensor(TensorConstructorArgs&&... args) {
  tensors_.emplace_back(std::forward<TensorConstructorArgs>(args)...);
  return &tensors_.back();
}

void VariantTensorData::ToProto(VariantTensorDataProto* proto) const {
  proto->set_type_name(type_name());
  proto->set_metadata(metadata_);
  proto->clear_tensors();
  for (const auto& tensor : tensors_) {
    tensor.AsProtoField(proto->mutable_tensors()->Add());
  }
}

bool VariantTensorData::FromProto(VariantTensorDataProto proto) {
  // TODO(ebrevdo): Do this lazily.
  set_type_name(proto.type_name());
  set_metadata(proto.metadata());
  for (const auto& tensor : proto.tensors()) {
    Tensor tmp;
    if (!tmp.FromProto(tensor)) return false;
    tensors_.push_back(tmp);
  }
  return true;
}

bool VariantTensorData::FromConstProto(const VariantTensorDataProto& proto) {
  set_type_name(proto.type_name());
  set_metadata(proto.metadata());
  for (const auto& tensor : proto.tensors()) {
    Tensor tmp;
    if (!tmp.FromProto(tensor)) return false;
    tensors_.push_back(tmp);
  }
  return true;
}

string VariantTensorData::SerializeAsString() const {
  VariantTensorDataProto proto;
  ToProto(&proto);
  return proto.SerializeAsString();
}

bool VariantTensorData::SerializeToString(string* buf) {
  VariantTensorDataProto proto;
  ToProto(&proto);
  return proto.SerializeToString(buf);
}

bool VariantTensorData::ParseFromString(string s) {
  VariantTensorDataProto proto;
  const bool status = proto.ParseFromString(s);
  if (status) FromProto(std::move(proto));
  return status;
}

string VariantTensorData::DebugString() const {
  string repeated_field = "";
  for (const auto& t : tensors_) {
    repeated_field =
        strings::StrCat(repeated_field, " tensors: ", t.DebugString());
  }
  return strings::StrCat("type_name: ", type_name(), " metadata: ", metadata_,
                         repeated_field);
}

string ProtoDebugString(const VariantTensorData& object) {
  return object.DebugString();
}

}  // namespace tensorflow
