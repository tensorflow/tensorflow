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
#include "tensorflow/c/experimental/ops/gen/model/arg_type.h"

#include "tensorflow/core/framework/op_def.pb.h"

namespace tensorflow {
namespace generator {

ArgType ArgType::CreateInput(const OpDef::ArgDef& arg_def) {
  return ArgType(arg_def, kInput);
}

ArgType ArgType::CreateInputRef(const OpDef::ArgDef& arg_def) {
  return ArgType(arg_def, kInputRef);
}

ArgType ArgType::CreateOutput(const OpDef::ArgDef& arg_def) {
  return ArgType(arg_def, kOutput);
}

ArgType::ArgType(const OpDef::ArgDef& arg_def, Kind kind)
    : kind_(kind), data_type_(arg_def.type()) {
  if (!arg_def.type_attr().empty()) {
    type_attr_name_ = arg_def.type_attr();
  }
  if (!arg_def.type_list_attr().empty()) {
    type_attr_name_ = arg_def.type_list_attr();
  }

  is_list_ =
      !arg_def.type_list_attr().empty() || !arg_def.number_attr().empty();
}
}  // namespace generator
}  // namespace tensorflow
