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
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// A shape describes the number of dimensions in a array, the bounds of each
// dimension, and the primitive component type. For tuples, shape describes the
// structure (number of elements and nesting).
class Shape {
 public:
  Shape() = default;

  // Construct a shape from a ShapeProto.
  explicit Shape(const ShapeProto& shape_proto);

  // Returns a ShapeProto representation of the Shape.
  ShapeProto ToProto() const;

  // Returns a human-readable string that represents the given shape, with or
  // without layout. e.g. "F32[42,12] {0, 1}" or "F32[64]".
  string ToString(bool print_layout = false) const;

  // Returns the rank (number of dimensions) of the given shape. Shape must be
  // an array.
  int64 rank() const {
    CHECK(IsArray()) << "Non-arrays do not have a rank, shape: " << ToString();
    return dimensions_.size();
  }

  // Returns whether the shape is of the specified type (array, tuple, etc).
  bool IsArray() const { return primitive_util::IsArrayType(element_type()); }
  bool IsTuple() const { return element_type() == TUPLE; }
  bool IsToken() const { return element_type() == TOKEN; }
  bool IsOpaque() const { return element_type() == OPAQUE; }

  // The following methods mirror the protobuf generated code interface for the
  // message ShapeProto. This enabled easy migration of this data structure
  // from a proto to a proper C++ class.
  // TODO(b/29771030): Replace or augment these methods with a more ergonomic
  // interface.

  // Methods for accessing the primitive type.
  PrimitiveType element_type() const { return element_type_; }
  void set_element_type(PrimitiveType value) { element_type_ = value; }

  // Methods for accessing the dimensions array.
  int dimensions_size() const { return dimensions_.size(); }
  int64 dimensions(int index) const { return dimensions_.at(index); }
  void set_dimensions(int index, int64 value) { dimensions_.at(index) = value; }
  void add_dimensions(int64 value) { dimensions_.push_back(value); }
  void clear_dimensions() { dimensions_.clear(); }
  const std::vector<int64>& dimensions() const { return dimensions_; }
  std::vector<int64>* mutable_dimensions() { return &dimensions_; }

  // Methods for accessing the tuple subshapes. This field only non-empty for
  // tuple shapes.
  int tuple_shapes_size() const { return tuple_shapes_.size(); }
  const Shape& tuple_shapes(int index) const { return tuple_shapes_.at(index); }
  Shape* mutable_tuple_shapes(int index) { return &tuple_shapes_.at(index); }
  Shape* add_tuple_shapes() {
    tuple_shapes_.push_back(Shape());
    return &tuple_shapes_.back();
  }
  void clear_tuple_shapes() { tuple_shapes_.clear(); }
  const std::vector<Shape>& tuple_shapes() const { return tuple_shapes_; }
  std::vector<Shape>* mutable_tuple_shapes() { return &tuple_shapes_; }

  // Methods for accessing the layout field.
  bool has_layout() const { return layout_.format() != INVALID_FORMAT; }
  const Layout& layout() const { return layout_; }
  Layout* mutable_layout() { return &layout_; }
  void clear_layout() { layout_.Clear(); }

  void Swap(Shape* other) {
    using std::swap;
    swap(*this, *other);
  }

  void Clear() {
    element_type_ = PRIMITIVE_TYPE_INVALID;
    dimensions_.clear();
    tuple_shapes_.clear();
    clear_layout();
  }

  string SerializeAsString() const { return ToProto().SerializeAsString(); }
  string ShortDebugString() const { return ToProto().ShortDebugString(); }
  string DebugString() const { return ToProto().DebugString(); }

 public:
  // The element type of this shape (tuple, array, etc).
  PrimitiveType element_type_ = PRIMITIVE_TYPE_INVALID;

  // The array bounds of the dimensions. This is nonempty only for array shapes.
  std::vector<int64> dimensions_;

  // The tuple element subshapes. This is nonempty only for tuple shapes.
  std::vector<Shape> tuple_shapes_;

  // The layout of the shape. Only relevant for arrays.
  Layout layout_;
};

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

std::ostream& operator<<(std::ostream& out, const Shape& shape);
std::ostream& operator<<(std::ostream& out, const ProgramShape& program_shape);

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SHAPE_H_
