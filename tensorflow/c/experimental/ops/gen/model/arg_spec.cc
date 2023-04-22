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
#include "tensorflow/c/experimental/ops/gen/model/arg_spec.h"

namespace tensorflow {
namespace generator {

ArgSpec::ArgSpec(const OpDef::ArgDef& arg_def, ArgType arg_type, int position)
    : name_(arg_def.name()),
      description_(arg_def.description()),
      arg_type_(arg_type),
      position_(position) {}

ArgSpec ArgSpec::CreateInput(const OpDef::ArgDef& arg_def, int position) {
  if (arg_def.is_ref()) {
    return ArgSpec(arg_def, ArgType::CreateInputRef(arg_def), position);
  } else {
    return ArgSpec(arg_def, ArgType::CreateInput(arg_def), position);
  }
}

ArgSpec ArgSpec::CreateOutput(const OpDef::ArgDef& arg_def, int position) {
  return ArgSpec(arg_def, ArgType::CreateOutput(arg_def), position);
}

}  // namespace generator
}  // namespace tensorflow
