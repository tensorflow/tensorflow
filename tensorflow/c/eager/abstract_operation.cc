/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/c/eager/abstract_operation.h"

#include <string>

namespace tensorflow {
Status AbstractOperation::SetAttrShape(const char* attr_name,
                                       const PartialTensorShape shape) {
  return SetAttrShape(attr_name, shape.dim_sizes().data(), shape.dims());
}

Status AbstractOperation::SetAttrStringList(const char* attr_name,
                                            absl::Span<string const> values) {
  std::vector<const char*> raw_strs;
  std::vector<size_t> lengths;
  raw_strs.reserve(values.size());
  lengths.reserve(values.size());
  for (const auto& s : values) {
    raw_strs.emplace_back(s.data());
    lengths.emplace_back(s.size());
  }
  return SetAttrStringList(attr_name,
                           reinterpret_cast<const void**>(raw_strs.data()),
                           lengths.data(), values.size());
}
}  // namespace tensorflow
