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

#include "tensorflow/core/platform/variant_coding.h"

#include <vector>
#include "tensorflow/core/framework/tensor.pb.h"
#include "tensorflow/core/framework/variant_op_registry.h"
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/strings/strcat.h"

namespace tensorflow {
namespace port {

void EncodeVariantList(const Variant* variant_array, int64 n, string* out) {
  out->clear();
  string rest;
  for (int i = 0; i < n; ++i) {
    string s;
    variant_array[i].Encode(&s);
    core::PutVarint32(out, s.length());
    strings::StrAppend(&rest, s);
  }
  strings::StrAppend(out, rest);
}

bool DecodeVariantList(const string& in, Variant* variant_array, int64 n) {
  std::vector<uint32> sizes(n);
  StringPiece reader(in);
  int64 total = 0;
  for (auto& size : sizes) {
    if (!core::GetVarint32(&reader, &size)) return false;
    total += size;
  }
  if (total != static_cast<int64>(reader.size())) {
    return false;
  }

  for (int i = 0; i < n; ++i) {
    if (variant_array[i].is_empty()) {
      variant_array[i] = VariantTensorDataProto();
    }
    string str(reader.data(), sizes[i]);
    if (!variant_array[i].Decode(str)) return false;
    if (!DecodeUnaryVariant(&variant_array[i])) {
      LOG(ERROR) << "Could not decode variant with type_name: \""
                 << variant_array[i].TypeName()
                 << "\".  Perhaps you forgot to register a "
                    "decoder via REGISTER_UNARY_VARIANT_DECODE_FUNCTION?";
      return false;
    }
    reader.remove_prefix(sizes[i]);
  }
  return true;
}

}  // end namespace port
}  // end namespace tensorflow
