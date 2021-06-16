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
#include "tensorflow/c/experimental/ops/gen/model/op_spec.h"

namespace tensorflow {
namespace generator {

OpSpec OpSpec::Create(const OpDef& op_def, const ApiDef& api_def) {
  return OpSpec(op_def, api_def);
}

OpSpec::OpSpec(const OpDef& op_def, const ApiDef& api_def)
    : name_(op_def.name()),
      summary_(api_def.summary()),
      description_(api_def.description()) {
  // Parse the attributes.
  for (const OpDef::AttrDef& attr_def : op_def.attr()) {
    AttrSpec attr = AttrSpec::Create(attr_def);
    // Typed attributes could be determined based on which attribute names were
    // the types of the arguments above, but this is easier:
    if (attr_def.type() == "type" || attr_def.type() == "list(type)") {
      type_attrs_[attr.name()] = attr;
    } else {
      argument_attrs_.push_back(attr);
    }
  }

  // Parse the arguments
  for (const OpDef::ArgDef& arg_def : op_def.input_arg()) {
    ArgSpec arg = ArgSpec::CreateInput(arg_def, input_args_.size());
    input_args_.push_back(arg);
  }
  for (const OpDef::ArgDef& arg_def : op_def.output_arg()) {
    ArgSpec arg = ArgSpec::CreateOutput(arg_def, output_args_.size());
    output_args_.push_back(arg);
  }
}

}  // namespace generator
}  // namespace tensorflow
