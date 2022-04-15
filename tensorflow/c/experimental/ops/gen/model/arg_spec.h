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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_MODEL_ARG_SPEC_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_MODEL_ARG_SPEC_H_

#include "tensorflow/c/experimental/ops/gen/model/arg_type.h"
#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {

// An input or output argument to an Op.
//
// Essentially, this represents an OpDef::ArgDef and its context within the Op.
class ArgSpec {
 public:
  ArgSpec() = default;
  ArgSpec(const ArgSpec& other) = default;
  static ArgSpec CreateInput(const OpDef::ArgDef& arg_def, int position);
  static ArgSpec CreateOutput(const OpDef::ArgDef& arg_def, int position);

  const string& name() const { return name_; }
  const string& description() const { return description_; }
  const ArgType arg_type() const { return arg_type_; }
  const int position() const { return position_; }

 private:
  explicit ArgSpec(const OpDef::ArgDef& arg_def, ArgType arg_type,
                   int position);

  string name_;
  string description_;
  ArgType arg_type_;
  int position_;
};

}  // namespace generator
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_MODEL_ARG_SPEC_H_
