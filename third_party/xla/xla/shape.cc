/* Copyright 2018 The OpenXLA Authors.

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

#include "xla/shape.h"

#include <algorithm>
#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"
#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/log/log.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/primitive_util.h"
#include "xla/printer.h"
#include "xla/shape_util.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// Defined in .cc file to avoid inlining these large routines
Shape::Shape() = default;
Shape::~Shape() = default;
Shape::Shape(const Shape&) = default;
Shape::Shape(Shape&&) noexcept = default;
Shape& Shape::operator=(const Shape&) = default;
Shape& Shape::operator=(Shape&&) noexcept = default;

Shape::Shape(const PrimitiveType element_type) {
  CHECK(element_type == TOKEN || element_type == OPAQUE_TYPE)
      << "Invalid element type for token or opaque shape: " << element_type_;
  set_element_type(element_type);
}

Shape::Shape(const PrimitiveType element_type,
             const absl::Span<const int64_t> dimensions,
             const absl::Span<const bool> dynamic_dimensions) {
  CHECK(primitive_util::IsArrayType(element_type))
      << "Invalid element type for array shape: " << element_type;
  if (!dynamic_dimensions.empty()) {
    CHECK_EQ(dimensions.size(), dynamic_dimensions.size())
        << "If dynamic_dimensions is provided, it must have the same size as "
           "dimensions.";
  }

  set_element_type(element_type);
  auto& state = array_state();
  state.dimensions = {dimensions.begin(), dimensions.end()};
  if (dynamic_dimensions.empty()) {
    // Assume all dimensions are static.
    state.dynamic_dimensions.resize(dimensions.size(), false);
  } else {
    state.dynamic_dimensions = absl::InlinedVector<bool, InlineRank()>(
        dynamic_dimensions.begin(), dynamic_dimensions.end());
  }
}

Shape::Shape(std::vector<Shape> tuple_shapes) {
  set_element_type(TUPLE);
  tuple_state().tuple_shapes = std::move(tuple_shapes);
}

Shape::Shape(const ShapeProto& shape_proto) {
  set_element_type(shape_proto.element_type());
  if (auto* const state = if_array_state()) {
    state->dimensions.reserve(shape_proto.dimensions_size());
    for (const int64_t dimension : shape_proto.dimensions()) {
      add_dimensions(dimension);
    }
    // A malformed proto may have different is_dynamic_dimension_size and
    // dimensions_size. Since C++ is evil, and we have no good way of bailing
    // out in a constructor, conservatively trim the is_dynamic_dimension size.
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
    const int64_t num_dynamic_dimension_fields = std::min(
        shape_proto.dimensions_size(), shape_proto.is_dynamic_dimension_size());
    for (int i = 0; i < num_dynamic_dimension_fields; i++) {
      state->dynamic_dimensions[i] = shape_proto.is_dynamic_dimension(i);
    }
  } else if (auto* const state = if_tuple_state()) {
    state->tuple_shapes.reserve(shape_proto.tuple_shapes_size());
    for (const ShapeProto& element_shape : shape_proto.tuple_shapes()) {
      state->tuple_shapes.emplace_back(element_shape);
    }
  }
  if (shape_proto.has_layout()) {
    if (!IsArray()) {
      LOG(ERROR) << "Malformed shape proto: element_type "
                 << PrimitiveType_Name(element_type())
                 << " should not have a layout.";
    } else {
      *mutable_layout() = Layout::CreateFromProto(shape_proto.layout());
    }
  }
}

void Shape::SetProto(ShapeProto& proto) const {
  proto.Clear();
  proto.set_element_type(element_type_);

  if (const auto* const state = if_array_state()) {
    proto.mutable_dimensions()->Reserve(state->dimensions.size());
    for (const int64_t dimension : state->dimensions) {
      proto.add_dimensions(dimension);
    }
    for (const bool dynamic : state->dynamic_dimensions) {
      proto.add_is_dynamic_dimension(dynamic);
    }
    if (state->layout.has_value()) {
      state->layout->SetProto(*proto.mutable_layout());
    }
  } else if (const auto* const state = if_tuple_state()) {
    proto.mutable_tuple_shapes()->Reserve(state->tuple_shapes.size());
    for (const Shape& shape : state->tuple_shapes) {
      shape.SetProto(*proto.add_tuple_shapes());
    }
  }
}

ShapeProto Shape::ToProto() const {
  ShapeProto proto;
  SetProto(proto);
  return proto;
}

void Shape::Print(Printer* printer, bool print_layout) const {
  if (print_layout) {
    ShapeUtil::PrintHumanStringWithLayout(printer, *this);
  } else {
    ShapeUtil::PrintHumanString(printer, *this);
  }
}

std::string Shape::ToString(bool print_layout) const {
  if (print_layout) {
    return ShapeUtil::HumanStringWithLayout(*this);
  } else {
    return ShapeUtil::HumanString(*this);
  }
}

bool Shape::AreAllLeavesIntegers() const {
  if (const auto* const state = if_tuple_state()) {
    return absl::c_all_of(state->tuple_shapes, [](const Shape& s) {
      return s.AreAllLeavesIntegers();
    });
  }
  return primitive_util::IsIntegralType(element_type());
}

bool Shape::is_static() const {
  if (const auto* const state = if_tuple_state()) {
    return absl::c_all_of(state->tuple_shapes,
                          [](const Shape& s) { return s.is_static(); });
  }
  if (const auto* const state = if_array_state()) {
    return !absl::c_any_of(state->dynamic_dimensions, [](bool b) { return b; });
  }
  return true;
}

bool Shape::is_unbounded_dynamic() const {
  if (const auto* const state = if_tuple_state()) {
    return absl::c_any_of(state->tuple_shapes, [](const Shape& subshape) {
      return subshape.is_unbounded_dynamic();
    });
  }
  if (const auto* const state = if_array_state()) {
    return absl::c_any_of(state->dimensions,
                          [](int64_t dim) { return dim == kUnboundedSize; });
  }
  return false;
}

bool Shape::is_bounded_dynamic() const {
  if (const auto* const state = if_tuple_state()) {
    return absl::c_any_of(state->tuple_shapes, [](const Shape& subshape) {
      return subshape.is_bounded_dynamic();
    });
  }
  if (const auto* const state = if_array_state()) {
    for (auto i = 0; i < state->dimensions.size(); ++i) {
      if (is_bounded_dynamic_dimension(i)) return true;
    }
    return false;
  }
  return false;
}

void Shape::DeleteDimension(int64_t dim_to_delete) {
  auto& state = array_state();
  CHECK_GE(dim_to_delete, 0);
  CHECK_LT(dim_to_delete, state.dimensions.size());
  state.dimensions.erase(state.dimensions.begin() + dim_to_delete);
  state.dynamic_dimensions.erase(state.dynamic_dimensions.begin() +
                                 dim_to_delete);
  if (LayoutUtil::HasLayout(*this)) {
    state.layout->DeleteDimension(dim_to_delete);  // NOLINT: optional-access
  }
}

void Shape::DeleteDimensions(absl::Span<const int64_t> sorted_dims_to_delete) {
  auto& state = array_state();
  CHECK(absl::c_is_sorted(sorted_dims_to_delete));
  state.dimensions = RemoveElements(sorted_dims_to_delete, state.dimensions);
  state.dynamic_dimensions =
      RemoveElements(sorted_dims_to_delete, state.dynamic_dimensions);
  if (LayoutUtil::HasLayout(*this)) {
    for (auto it = sorted_dims_to_delete.rbegin();
         it != sorted_dims_to_delete.rend(); ++it) {
      state.layout->DeleteDimension(*it);  // NOLINT: optional-access
    }
  }
}

void Shape::CheckStateIsEmpty() const {
  if (const auto* const state = if_array_state()) {
    CHECK(state->dimensions.empty()) << ToString();
    CHECK(state->dynamic_dimensions.empty()) << ToString();
    CHECK(!state->layout.has_value()) << ToString();
  } else if (const auto* const state = if_tuple_state()) {
    CHECK(state->tuple_shapes.empty()) << ToString();
  }
}

const std::vector<Shape>& Shape::tuple_shapes() const {
  return tuple_state().tuple_shapes;
}

void Shape::Clear() {
  // Before setting the element type to invalid, we need to clear the state
  // because the state may be non-empty if the shape was previously valid.
  // Without this step, set_element_type() may CHECK-fail.
  if (auto* const state = if_array_state()) {
    *state = ArrayState();
  } else if (auto* const state = if_tuple_state()) {
    *state = TupleState();
  }
  set_element_type(PRIMITIVE_TYPE_INVALID);
}

void Shape::set_element_type(const PrimitiveType value) {
  element_type_ = value;

  // Make sure the variant state matches the element type.
  // If we have to change the case of the variant, and the current case is not
  // empty, it's likely a programmer error - we CHECK-fail to catch it.
  if (element_type_ == TOKEN) {
    if (!if_token_state()) {
      CheckStateIsEmpty();
      state_ = TokenState();
    }
    return;
  }
  if (element_type_ == OPAQUE_TYPE) {
    if (!if_opaque_state()) {
      CheckStateIsEmpty();
      state_ = OpaqueState();
    }
    return;
  }
  if (element_type_ == TUPLE) {
    if (!if_tuple_state()) {
      CheckStateIsEmpty();
      state_ = TupleState();
    }
    return;
  }
  if (primitive_util::IsArrayType(element_type_)) {
    if (!if_array_state()) {
      CheckStateIsEmpty();
      state_ = ArrayState();
    }
    return;
  }
  // Treat all other types as invalid.
  if (element_type_ != PRIMITIVE_TYPE_INVALID) {
    LOG(ERROR) << "Unsupported element type: " << element_type_;
    element_type_ = PRIMITIVE_TYPE_INVALID;
  }
  if (!if_invalid_state()) {
    CheckStateIsEmpty();
    state_ = InvalidState();
  }
}

const Shape& Shape::tuple_shapes(int index) const {
  return tuple_state().tuple_shapes[index];
}

Shape* Shape::add_tuple_shapes() {
  auto& state = tuple_state();
  state.tuple_shapes.push_back(Shape());
  return &state.tuple_shapes.back();
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
    if (!ShapeUtil::SameRank(lhs, rhs)) {
      VLOG(3) << "CompareShapes: lhs rank != rhs rank";
      return false;
    }
    for (int i = 0; i < lhs.dimensions().size(); ++i) {
      if (ignore_dynamic_dimension_ &&
          (lhs.is_unbounded_dynamic_dimension(i) ||
           rhs.is_unbounded_dynamic_dimension(i))) {
        continue;
      }
      if (lhs.dimensions(i) != rhs.dimensions(i)) {
        VLOG(3) << "CompareShapes: lhs dimensions != rhs dimensions";
        return false;
      }
    }
  } else {
    if (!ShapeUtil::SameRank(lhs, rhs)) {
      VLOG(3) << "CompareShapes: lhs rank != rhs rank";
      return false;
    }
  }

  if (!ignore_layout_) {
    if (lhs.IsArray()) {
      Layout::Equal equal;
      if (lhs.has_layout() || rhs.has_layout()) {
        if (!lhs.has_layout() || !rhs.has_layout()) {
          VLOG(3) << "CompareShapes: both shapes do not have layouts";
          return false;
        }
        if (ignore_tiles_in_layout_) {
          equal.IgnoreTiles();
        }
        if (ignore_element_size_in_layout_) {
          equal.IgnoreElementSize();
        }
        if (ignore_memory_space_in_layout_) {
          equal.IgnoreMemorySpace();
        }
        if (ignore_tail_padding_alignment_in_elements_in_layout_) {
          equal.IgnoreTailPaddingAlignmentInElements();
        }
        if (ignore_split_config_in_layout_) {
          equal.IgnoreSplitConfigs();
        }
        if (!equal(lhs.layout(), rhs.layout())) {
          VLOG(3) << "CompareShapes: lhs layout != rhs layout";
          return false;
        }
      }
    }
  }

  if (!ignore_dynamic_dimension_) {
    for (int i = 0; i < lhs.dimensions().size(); ++i) {
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

ProgramShape::ProgramShape() = default;
ProgramShape::~ProgramShape() = default;
ProgramShape::ProgramShape(const ProgramShape&) = default;
ProgramShape::ProgramShape(ProgramShape&&) = default;
ProgramShape& ProgramShape::operator=(const ProgramShape&) = default;
ProgramShape& ProgramShape::operator=(ProgramShape&&) = default;

ProgramShape::ProgramShape(const ProgramShapeProto& program_shape_proto) {
  for (const ShapeProto& shape_proto : program_shape_proto.parameters()) {
    *add_parameters() = Shape(shape_proto);
  }
  *mutable_result() = Shape(program_shape_proto.result());
  for (const std::string& name : program_shape_proto.parameter_names()) {
    add_parameter_names(name);
  }
}

ProgramShapeProto ProgramShape::ToProto() const {
  ProgramShapeProto proto;
  for (const Shape& shape : parameters()) {
    *proto.add_parameters() = shape.ToProto();
  }
  *proto.mutable_result() = result().ToProto();
  for (const std::string& name : parameter_names()) {
    proto.add_parameter_names(name);
  }
  return proto;
}

void ProgramShape::Print(Printer* printer) const {
  ShapeUtil::PrintHumanString(printer, *this);
}

std::string ProgramShape::ToString() const {
  return ShapeUtil::HumanString(*this);
}

std::ostream& operator<<(std::ostream& out, const ProgramShape& program_shape) {
  out << program_shape.ToString() << "\n";
  return out;
}

}  // namespace xla
