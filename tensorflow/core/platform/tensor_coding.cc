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

#include "tensorflow/core/platform/tensor_coding.h"

#include <vector>
#include "tensorflow/core/lib/core/coding.h"
#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {
namespace port {

void AssignRefCounted(StringPiece src, core::RefCounted* obj, string* out) {
  out->assign(src.data(), src.size());
}

void EncodeStringList(const string* strings, int64 n, string* out) {
  out->clear();
  for (int i = 0; i < n; ++i) {
    core::PutVarint32(out, strings[i].size());
  }
  for (int i = 0; i < n; ++i) {
    out->append(strings[i]);
  }
}

bool DecodeStringList(const string& src, string* strings, int64 n) {
  std::vector<uint32> sizes(n);
  StringPiece reader(src);
  int64 tot = 0;
  for (auto& v : sizes) {
    if (!core::GetVarint32(&reader, &v)) return false;
    tot += v;
  }
  if (tot != static_cast<int64>(reader.size())) {
    return false;
  }

  string* data = strings;
  for (int64 i = 0; i < n; ++i, ++data) {
    auto size = sizes[i];
    if (size > reader.size()) {
      return false;
    }
    data->assign(reader.data(), size);
    reader.remove_prefix(size);
  }

  return true;
}

void CopyFromArray(string* s, const char* base, size_t bytes) {
  s->assign(base, bytes);
}

}  // namespace port
}  // namespace tensorflow
