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

#ifndef TENSORFLOW_COMPILER_XLA_SHAPE_H_
#define TENSORFLOW_COMPILER_XLA_SHAPE_H_

#include <string>
#include <vector>

#include "absl/types/optional.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Shape of the parameters and output of an XLA computation. This is analogous
// to a traditional function signature.
class ProgramShape {
 public:
  ProgramShape() = default;

  // Creates a ProgramShape from a ProgramShapeProto protobuf.
  explicit ProgramShape(const ProgramShapeProto& program_shape_proto);

  // Returns a proto representation of the object.
  ProgramShapeProto ToProto() const;

  string ToString() const;

  // The following methods mirror the protobuf generated code interface for the
  // message ProgramShapeProto. This enabled easy migration of this data
  // structure from a proto to a proper C++ class.
  // TODO(b/29771030): Replace or augment these methods with a more ergonomic
  // interface.

  // Methods for accessing and manipulating the Shape of the parameters.
  int parameters_size() const { return parameters_.size(); }
  const Shape& parameters(int index) const { return parameters_.at(index); }
  Shape* mutable_parameters(int index) { return &parameters_.at(index); }
  Shape* add_parameters() {
    parameters_.emplace_back();
    return &parameters_.back();
  }
  void clear_parameters() { parameters_.clear(); }
  const std::vector<Shape>& parameters() const { return parameters_; }
  std::vector<Shape>* mutable_parameters() { return &parameters_; }

  // Methods for accessing and manipulating the Shape of the result.
  const Shape& result() const { return result_; }
  Shape* mutable_result() { return &result_; }
  void clear_result() { result_.Clear(); }

  // Methods for accessing and manipulating the names of the parameters.
  int parameter_names_size() const { return parameter_names_.size(); }
  const string& parameter_names(int index) const {
    return parameter_names_.at(index);
  }
  void set_parameter_names(int index, const string& value) {
    parameter_names_.at(index) = value;
  }
  string* mutable_parameter_names(int index) {
    return &parameter_names_.at(index);
  }
  void add_parameter_names(const string& value) {
    parameter_names_.push_back(value);
  }
  string* add_parameter_names() {
    parameter_names_.push_back("");
    return &parameter_names_.back();
  }
  void clear_parameter_names() { parameter_names_.clear(); }
  const std::vector<string>& parameter_names() const {
    return parameter_names_;
  }
  std::vector<string>* mutable_parameter_names() { return &parameter_names_; }

  string ShortDebugString() const { return ToProto().ShortDebugString(); }
  string DebugString() const { return ToProto().DebugString(); }

 private:
  // The shapes of the parameters of the computation represented by this object.
  std::vector<Shape> parameters_;

  // The names of the parameters of the computation represented by this object.
  std::vector<string> parameter_names_;

  // The shape of the result of the computation represented by this object.
  Shape result_;
};

std::ostream& operator<<(std::ostream& out, const ProgramShape& program_shape);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SHAPE_H_
