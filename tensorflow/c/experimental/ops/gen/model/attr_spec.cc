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
#include "tensorflow/c/experimental/ops/gen/model/attr_spec.h"

#include "absl/strings/match.h"
#include "tensorflow/core/framework/op_def.pb.h"

namespace tensorflow {
namespace generator {

AttrSpec AttrSpec::Create(const OpDef::AttrDef& attr_def) {
  return AttrSpec(attr_def);
}

AttrSpec::AttrSpec(const OpDef::AttrDef& attr_def) {
  name_ = attr_def.name();
  description_ = attr_def.description();
  full_type_ = attr_def.type();
  default_value_ = attr_def.default_value();
  if (absl::StartsWith(full_type_, "list(")) {
    is_list_ = true;
    // strip surrounding "list(%s)"
    base_type_ = full_type_.substr(5, full_type_.length() - 6);
  } else {
    is_list_ = false;
    base_type_ = full_type_;
  }
}

}  // namespace generator
}  // namespace tensorflow
