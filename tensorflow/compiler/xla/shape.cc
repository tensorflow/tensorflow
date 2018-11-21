/* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/compiler/xla/shape.h"

#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "tensorflow/compiler/xla/shape_util.h"

namespace xla {

ProgramShape::ProgramShape(const ProgramShapeProto& program_shape_proto) {
  for (const Shape& shape : program_shape_proto.parameters()) {
    *add_parameters() = shape;
  }
  *mutable_result() = program_shape_proto.result();
  for (const string& name : program_shape_proto.parameter_names()) {
    add_parameter_names(name);
  }
}

ProgramShapeProto ProgramShape::ToProto() const {
  ProgramShapeProto proto;
  for (const Shape& shape : parameters()) {
    *proto.add_parameters() = shape;
  }
  *proto.mutable_result() = result();
  for (const string& name : parameter_names()) {
    proto.add_parameter_names(name);
  }
  return proto;
}

string ProgramShape::ToString() const {
  std::vector<string> parameter_strings(parameters_size());
  for (int i = 0; i < parameters_size(); ++i) {
    parameter_strings[i] = absl::StrCat(
        i < parameter_names_size() ? parameter_names(i) : "(unknown)", ": ",
        ShapeUtil::HumanString(parameters(i)));
  }
  return absl::StrCat("(", absl::StrJoin(parameter_strings, ", "), ") -> ",
                      ShapeUtil::HumanString(result()));
}

std::ostream& operator<<(std::ostream& out, const ProgramShape& program_shape) {
  out << program_shape.ToString() << "\n";
  return out;
}

}  // namespace xla
