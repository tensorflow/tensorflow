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

#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

template <>
void* Variant::get() {
  if (is_empty()) {
    return nullptr;
  }
  return value_->RawPtr();
}

template <>
const void* Variant::get() const {
  if (is_empty()) {
    return nullptr;
  }
  return value_->RawPtr();
}

void VariantTensorData::ToProto(VariantTensorDataProto* proto) const {
  proto->set_type_name(type_name);
  proto->set_metadata(metadata);
  proto->clear_tensors();
  for (int i = 0; i < tensors.size(); ++i) {
    tensors[i].AsProtoField(proto->mutable_tensors()->Add());
  }
}

bool VariantTensorData::FromProto(const VariantTensorDataProto& proto) {
  type_name = proto.type_name();
  metadata = proto.metadata();
  tensors.clear();
  for (int i = 0; i < proto.tensors_size(); ++i) {
    Tensor tmp;
    if (!tmp.FromProto(proto.tensors(i))) return false;
    tensors.push_back(tmp);
  }
  return true;
}

template <>
string TypeNameVariant(const VariantTensorDataProto& value) {
  return value.GetTypeName();
}

template <>
void EncodeVariant(const VariantTensorDataProto& value, VariantTensorData* data) {
  data->FromProto(value);
}

template <>
bool DecodeVariant(const VariantTensorData& data,
                   VariantTensorDataProto* value) {
  data.ToProto(value);
  return true;
}

template <>
void EncodeVariant(const VariantTensorDataProto& value, string* buf) {
  value.SerializeToString(buf);
}

template <>
bool DecodeVariant(const string& buf, VariantTensorDataProto* value) {
  return value->ParseFromString(buf);
}

}  // end namespace tensorflow
