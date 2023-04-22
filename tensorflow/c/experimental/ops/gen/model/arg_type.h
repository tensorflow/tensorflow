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
#ifndef TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_MODEL_ARG_TYPE_H_
#define TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_MODEL_ARG_TYPE_H_

#include "tensorflow/core/framework/op_def.pb.h"
#include "tensorflow/core/platform/types.h"

namespace tensorflow {
namespace generator {

// Type information of an Op argument (ArgSpec)..
//
// This represents the type information with OpDef::ArgDef and any type-related
// context.
class ArgType {
 public:
  ArgType() = default;
  ArgType(const ArgType& other) = default;
  static ArgType CreateInput(const OpDef::ArgDef& arg_def);
  static ArgType CreateInputRef(const OpDef::ArgDef& arg_def);
  static ArgType CreateOutput(const OpDef::ArgDef& arg_def);

  const tensorflow::DataType data_type() const { return data_type_; }
  const string type_attr_name() const { return type_attr_name_; }
  const bool is_read_only() const { return kind_ == kInput; }
  const bool is_list() const { return is_list_; }

 private:
  enum Kind { kInput = 0, kInputRef, kOutput };

  explicit ArgType(const OpDef::ArgDef& arg_def, Kind kind);

  Kind kind_;
  tensorflow::DataType data_type_;
  string type_attr_name_;
  bool is_list_;
};

}  // namespace generator
}  // namespace tensorflow

#endif  // TENSORFLOW_C_EXPERIMENTAL_OPS_GEN_MODEL_ARG_TYPE_H_
