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

#include <string>

#include "absl/container/flat_hash_set.h"
#include "tensorflow/c/experimental/ops/gen/model/arg_spec.h"
#include "tensorflow/c/experimental/ops/gen/model/attr_spec.h"
#include "tensorflow/core/framework/api_def.pb.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {

OpSpec OpSpec::Create(const OpDef& op_def, const ApiDef& api_def) {
  return OpSpec(op_def, api_def);
}

OpSpec::OpSpec(const OpDef& op_def, const ApiDef& api_def)
    : name_(op_def.name()),
      summary_(api_def.summary()),
      description_(api_def.description()) {
  absl::flat_hash_set<string> inferred_attrs;
  // Parse the arguments
  for (const OpDef::ArgDef& arg_def : op_def.input_arg()) {
    ArgSpec arg = ArgSpec::CreateInput(arg_def, input_args_.size());
    input_args_.push_back(arg);
    if (!arg_def.type_attr().empty()) {
      inferred_attrs.insert(arg_def.type_attr());
      if (!arg_def.number_attr().empty()) {
        inferred_attrs.insert(arg_def.number_attr());
      }
    } else if (!arg_def.type_list_attr().empty()) {
      inferred_attrs.insert(arg_def.type_list_attr());
    }
  }
  for (const OpDef::ArgDef& arg_def : op_def.output_arg()) {
    ArgSpec arg = ArgSpec::CreateOutput(arg_def, output_args_.size());
    output_args_.push_back(arg);
  }
  // Parse the attributes.
  for (const OpDef::AttrDef& attr_def : op_def.attr()) {
    AttrSpec attr = AttrSpec::Create(attr_def);
    // Only non-inferred args are added as arguments.
    if (inferred_attrs.find(attr_def.name()) == inferred_attrs.end()) {
      argument_attrs_.push_back(attr);
    }
  }
}

}  // namespace generator
}  // namespace tensorflow
