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

Shape::Shape(const ShapeProto& shape_proto) {
  set_element_type(shape_proto.element_type());
  dimensions_.reserve(shape_proto.dimensions_size());
  for (const int64 dimension : shape_proto.dimensions()) {
    add_dimensions(dimension);
  }
  // A malformed proto may have different is_dynamic_dimension_size and
  // dimensions_size. Since C++ is evil, and we have no good way of bailing out
  // in a constructor, conservatively trim the is_dynamic_dimension size.
  // TODO(b/120111794): Make this a hard error when we have a factory method
  // instead of a constructor.
  if (shape_proto.dimensions_size() !=
      shape_proto.is_dynamic_dimension_size()) {
    if (shape_proto.is_dynamic_dimension_size() != 0) {
      LOG(ERROR) << "Malformed shape proto: number of is_dynamic_dimension "
                    "fields does not match number of dimension fields";
    } else {
      LOG(WARNING) << "Malformed shape proto: is_dynamic_dimension is empty";
    }
  }
  int64 num_dynamic_dimension_fields = std::min(
      shape_proto.dimensions_size(), shape_proto.is_dynamic_dimension_size());
  for (int i = 0; i < num_dynamic_dimension_fields; i++) {
    dynamic_dimensions_[i] = shape_proto.is_dynamic_dimension(i);
  }
  tuple_shapes_.reserve(shape_proto.tuple_shapes_size());
  for (const ShapeProto& element_shape : shape_proto.tuple_shapes()) {
    tuple_shapes_.emplace_back(element_shape);
  }
  if (shape_proto.has_layout()) {
    *mutable_layout() = Layout::CreateFromProto(shape_proto.layout());
  }
}

ShapeProto Shape::ToProto() const {
  ShapeProto proto;
  proto.set_element_type(element_type_);
  proto.mutable_dimensions()->Reserve(dimensions_size());
  for (const int64 dimension : dimensions()) {
    proto.add_dimensions(dimension);
  }
  for (const bool dynamic : dynamic_dimensions_) {
    proto.add_is_dynamic_dimension(dynamic);
  }
  proto.mutable_tuple_shapes()->Reserve(tuple_shapes_size());
  for (const Shape& shape : tuple_shapes()) {
    *proto.add_tuple_shapes() = shape.ToProto();
  }
  if (has_layout()) {
    *proto.mutable_layout() = layout().ToProto();
  }
  return proto;
}

string Shape::ToString(bool print_layout) const {
  if (print_layout) {
    return ShapeUtil::HumanStringWithLayout(*this);
  } else {
    return ShapeUtil::HumanString(*this);
  }
}

bool Shape::IsInteger() const {
  switch (element_type()) {
    case PrimitiveType::S8:
    case PrimitiveType::S16:
    case PrimitiveType::S32:
    case PrimitiveType::S64:
    case PrimitiveType::U8:
    case PrimitiveType::U16:
    case PrimitiveType::U32:
    case PrimitiveType::U64:
      return true;
    case PrimitiveType::TUPLE:
      return absl::c_any_of(tuple_shapes_,
                            [](const Shape& s) { return s.IsInteger(); });
    default:
      return false;
  }
}

bool Shape::is_static() const {
  if (IsTuple()) {
    for (const Shape& subshape : tuple_shapes_) {
      if (!subshape.is_static()) {
        return false;
      }
    }
  }
  return !absl::c_any_of(dynamic_dimensions_, [](bool b) { return b; });
}

void Shape::DeleteDimension(int64 dim_to_delete) {
  CHECK(IsArray());
  CHECK_GE(dim_to_delete, 0);
  CHECK_LT(dim_to_delete, dimensions_.size());
  dimensions_.erase(dimensions_.begin() + dim_to_delete);
  dynamic_dimensions_.erase(dynamic_dimensions_.begin() + dim_to_delete);
  if (LayoutUtil::HasLayout(*this)) {
    layout_.set_format(DENSE);
    for (int64 i = 0; i < layout_.minor_to_major().size();) {
      if (layout_.minor_to_major(i) == dim_to_delete) {
        layout_.mutable_minor_to_major()->erase(
            layout_.mutable_minor_to_major()->begin() + i);
        continue;
      }
      if (layout_.minor_to_major(i) > dim_to_delete) {
        (*layout_.mutable_minor_to_major())[i] -= 1;
      }
      ++i;
    }
  }
}

bool Shape::Equal::operator()(const Shape& lhs, const Shape& rhs) {
  if (lhs.IsTuple()) {
    return rhs.IsTuple() &&
           absl::c_equal(
               lhs.tuple_shapes(), rhs.tuple_shapes(),
               [=](const Shape& l, const Shape& r) { return (*this)(l, r); });
  } else if (!lhs.IsArray()) {
    // Non-tuple, non-array tupes such as opaque and token types are trivially
    // the same.
    return lhs.element_type() == rhs.element_type();
  }

  if (!rhs.IsArray()) {
    return false;
  }

  if (!ignore_element_type_) {
    if ((ignore_fp_precision_ &&
         !ShapeUtil::SameElementTypeIgnoringFpPrecision(lhs, rhs)) ||
        (!ignore_fp_precision_ && !ShapeUtil::SameElementType(lhs, rhs))) {
      VLOG(3) << "CompareShapes: lhs element type != rhs element type";
      return false;
    }
  }

  if (!ignore_dimensions_) {
    if (!ShapeUtil::SameDimensions(lhs, rhs)) {
      VLOG(3) << "CompareShapes: lhs dimensions != rhs dimensions";
      return false;
    }
  } else {
    if (!ShapeUtil::SameRank(lhs, rhs)) {
      VLOG(3) << "CompareShapes: lhs rank != rhs rank";
      return false;
    }
  }

  if (!ignore_layout_) {
    if (lhs.layout().format() != rhs.layout().format()) {
      VLOG(3) << "CompareShapes: lhs layout format != rhs layout format";
      return false;
    }
    if (LayoutUtil::IsDenseArray(lhs)) {
      Layout::Equal equal;
      if (ignore_tiles_in_layout_) {
        equal.IgnoreTiles();
      }
      if (ignore_element_size_in_layout_) {
        equal.IgnoreElementSize();
      }
      if (ignore_memory_space_in_layout_) {
        equal.IgnoreMemorySpace();
      }
      if (!equal(lhs.layout(), rhs.layout())) {
        VLOG(3) << "CompareShapes: lhs layout != rhs layout";
        return false;
      }
    }
  }

  if (!ignore_dynamic_dimension_) {
    for (int i = 0; i < lhs.rank(); ++i) {
      if (lhs.is_dynamic_dimension(i) != rhs.is_dynamic_dimension(i)) {
        VLOG(3)
            << "CompareShapes: lhs and rhs have different dynamic dimensions.";
        return false;
      }
    }
  }
  return true;
}

std::ostream& operator<<(std::ostream& out, const Shape& shape) {
  out << shape.ToString(/*print_layout=*/true);
  return out;
}

ProgramShape::ProgramShape(const ProgramShapeProto& program_shape_proto) {
  for (const ShapeProto& shape_proto : program_shape_proto.parameters()) {
    *add_parameters() = Shape(shape_proto);
  }
  *mutable_result() = Shape(program_shape_proto.result());
  for (const string& name : program_shape_proto.parameter_names()) {
    add_parameter_names(name);
  }
}

ProgramShapeProto ProgramShape::ToProto() const {
  ProgramShapeProto proto;
  for (const Shape& shape : parameters()) {
    *proto.add_parameters() = shape.ToProto();
  }
  *proto.mutable_result() = result().ToProto();
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
