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

#include "tensorflow/core/framework/variant.h"
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/variant_encode_decode.h"
#include "tensorflow/core/framework/variant_tensor_data.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/gtl/map_util.h"

namespace tensorflow {

bool Variant::TryDecode(Variant* out) const {
  const VariantTensorDataProto* p = get<VariantTensorDataProto>();
  if (p == nullptr) return false;
  VariantTensorData data(*p);
  return out->Decode(data);
}

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

template <>
string TypeNameVariant(const VariantTensorDataProto& value) {
  return value.type_name();
}

template <>
void EncodeVariant(const VariantTensorDataProto& value,
                   VariantTensorData* data) {
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
