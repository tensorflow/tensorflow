/* Copyright 2015 Google Inc. All Rights Reserved.

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

#include "tensorflow/core/graph/tensor_id.h"

#include <string>

#include "tensorflow/core/lib/core/stringpiece.h"

namespace tensorflow {

TensorId ParseTensorName(const string& name) {
  return ParseTensorName(StringPiece(name.data(), name.size()));
}

TensorId ParseTensorName(StringPiece name) {
  // Parse either a name, or a name:digits.  To do so, we go backwards
  // from the end of the string, skipping over a run of digits.  If
  // we hit a ':' character, then we know we are in the 'name:digits'
  // regime.  Otherwise, the output index is implicitly 0, and the whole
  // name string forms the first part of the tensor name.
  //
  // Equivalent to matching with this regexp: ([^:]+):(\\d+)
  const char* base = name.data();
  const char* p = base + name.size() - 1;
  int index = 0;
  int mul = 1;
  while (p > base && (*p >= '0' && *p <= '9')) {
    index += ((*p - '0') * mul);
    mul *= 10;
    p--;
  }
  TensorId id;
  if (p > base && *p == ':' && mul > 1) {
    id.first = StringPiece(base, p - base);
    id.second = index;
  } else {
    id.first = name;
    id.second = 0;
  }
  return id;
}

}  // namespace tensorflow
