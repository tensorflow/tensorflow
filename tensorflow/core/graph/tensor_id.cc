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

#include "tensorflow/core/graph/tensor_id.h"

#include <cstddef>
#include <cstdint>
#include <string>

#include "absl/strings/string_view.h"
#include "absl/strings/strip.h"
#include "tensorflow/core/graph/graph.h"
#include "tensorflow/core/platform/str_util.h"

namespace tensorflow {

TensorId::TensorId(const SafeTensorId& id) : TensorId(id.first, id.second) {}

SafeTensorId::SafeTensorId(const TensorId& id)
    : SafeTensorId(std::string(id.first), id.second) {}

TensorId ParseTensorName(absl::string_view name) {
  // Parse either a name, ^name, or name:digits.  To do so, we go backwards from
  // the end of the string, skipping over a run of digits.  If we hit a ':'
  // character, then we know we are in the 'name:digits' regime.  Otherwise, we
  // see if the name starts with '^', indicating a control edge. If we find
  // neither ':' nor '^' characters, the output index is implicitly 0, and the
  // whole name string forms the first part of the tensor name.
  size_t colon_pos = name.rfind(':');
  if (colon_pos != absl::string_view::npos) {
    absl::string_view prefix = name.substr(0, colon_pos);
    absl::string_view suffix = name.substr(colon_pos + 1);
    uint64_t index;
    if (str_util::ConsumeLeadingDigits(&suffix, &index) && suffix.empty()) {
      return TensorId(prefix, index);
    }
  }
  if (absl::ConsumePrefix(&name, "^")) {
    return TensorId(name, Graph::kControlSlot);
  }
  return TensorId(name, 0);
}

bool IsTensorIdControl(const TensorId& tensor_id) {
  return tensor_id.index() == Graph::kControlSlot;
}

}  // namespace tensorflow
