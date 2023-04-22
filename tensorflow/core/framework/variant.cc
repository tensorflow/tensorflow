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
#include "tensorflow/core/framework/variant_op_registry.h"

namespace tensorflow {

Variant::~Variant() { ResetMemory(); }

bool Variant::Decode(VariantTensorData data) {
  if (!is_empty()) {
    return GetValue()->Decode(std::move(data));
  }
  return true;
}

template <>
void* Variant::get() {
  if (is_empty()) {
    return nullptr;
  }
  return GetValue()->RawPtr();
}

template <>
const void* Variant::get() const {
  if (is_empty()) {
    return nullptr;
  }
  return GetValue()->RawPtr();
}

template <>
string TypeNameVariant(const VariantTensorDataProto& value) {
  return value.type_name();
}

template <>
void EncodeVariant(const VariantTensorDataProto& value,
                   VariantTensorData* data) {
  data->FromConstProto(value);
}

template <>
bool DecodeVariant(VariantTensorData* data, VariantTensorDataProto* value) {
  data->ToProto(value);
  return true;
}

template <>
void EncodeVariant(const VariantTensorDataProto& value, string* buf) {
  value.SerializeToString(buf);
}

template <>
bool DecodeVariant(string* buf, VariantTensorDataProto* value) {
  return value->ParseFromString(*buf);
}

void EncodeVariantList(const Variant* variant_array, int64 n,
                       std::unique_ptr<port::StringListEncoder> e) {
  for (int i = 0; i < n; ++i) {
    string s;
    variant_array[i].Encode(&s);
    e->Append(s);
  }
  e->Finalize();
}

bool DecodeVariantList(std::unique_ptr<port::StringListDecoder> d,
                       Variant* variant_array, int64 n) {
  std::vector<uint32> sizes(n);
  if (!d->ReadSizes(&sizes)) return false;

  for (int i = 0; i < n; ++i) {
    if (variant_array[i].is_empty()) {
      variant_array[i] = VariantTensorDataProto();
    }
    // TODO(ebrevdo): Replace with StringPiece?  Any way to make this a
    // zero-copy operation that keeps a reference to the data in d?
    string str(d->Data(sizes[i]), sizes[i]);
    if (!variant_array[i].Decode(std::move(str))) return false;
    if (!DecodeUnaryVariant(&variant_array[i])) {
      LOG(ERROR) << "Could not decode variant with type_name: \""
                 << variant_array[i].TypeName()
                 << "\".  Perhaps you forgot to register a "
                    "decoder via REGISTER_UNARY_VARIANT_DECODE_FUNCTION?";
      return false;
    }
  }
  return true;
}

}  // end namespace tensorflow
