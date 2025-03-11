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

#ifndef XLA_SHAPE_H_
#define XLA_SHAPE_H_

#include <cstdint>
#include <limits>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/primitive_util.h"
#include "xla/printer.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/util.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A shape describes the number of dimensions in a array, the bounds of each
// dimension, and the primitive component type. For tuples, shape describes the
// structure (number of elements and nesting).
class Shape {
 public:
  Shape();
  ~Shape();
  Shape(const Shape&);
  Shape(Shape&&) noexcept;
  Shape& operator=(const Shape&);
  Shape& operator=(Shape&&) noexcept;

  // Construct a shape from a ShapeProto.
  explicit Shape(const ShapeProto& shape_proto);

  Shape(PrimitiveType element_type, absl::Span<const int64_t> dimensions,
        absl::Span<const bool> dynamic_dimensions,
        std::vector<Shape> tuple_shapes)
      : element_type_(element_type),
        dimensions_(dimensions.begin(), dimensions.end()),
        dynamic_dimensions_(dynamic_dimensions.begin(),
                            dynamic_dimensions.end()),
        tuple_shapes_(std::move(tuple_shapes)) {}

  // Returns a ShapeProto representation of the Shape.
  ShapeProto ToProto() const;
  // Sets a ShapeProto to the representation of the Shape.
  void SetProto(ShapeProto& proto) const;

  // Prints a human-readable string that represents the given shape, with or
  // without layout. e.g. "F32[42,12] {0, 1}" or "F32[64]".
  void Print(Printer* printer, bool print_layout = false) const;

  // Returns a human-readable string that represents the given shape, with or
  // without layout. e.g. "F32[42,12] {0, 1}" or "F32[64]".
  std::string ToString(bool print_layout = false) const;

  // Returns the rank (number of dimensions) of the given shape. Returns 0 for
  // non-array shapes.
  int64_t rank() const {
    return dimensions_.size();
  }

  // Returns whether the shape is of the specified type (array, tuple, etc).
  bool IsArray() const { return primitive_util::IsArrayType(element_type()); }
  bool IsTuple() const { return element_type() == TUPLE; }
  bool IsToken() const { return element_type() == TOKEN; }
  bool IsOpaque() const { return element_type() == OPAQUE_TYPE; }

  // Returns whether all elements in the shape are integer.
  // A nested tuple of integers is considered as integer.
  bool IsInteger() const;

  // Returns true if no array dimension in the shape is dynamically sized. Tuple
  // shapes are traversed recursively.
  bool is_static() const;

  bool is_dynamic() const { return !is_static(); }

  // Unbounded dynamism.
  // If `dimensions(axis) == kUnboundedSize && is_dynamic_dimension(axis)`,
  // this means that the axis has unbounded dynamic size.
  // The sentinel value for kUnboundedSize is chosen to be exactly the same
  // as the sentinel value mlir::ShapedType::kDynamic.
  static constexpr int64_t kUnboundedSize = std::numeric_limits<int64_t>::min();

  // Returns true if the shape has one or more dimensions with unbounded sizes.
  // Tuple shapes are traversed recursively, returns true if any element is
  // unbounded dynamic.
  bool is_unbounded_dynamic() const;

  // Returns true if the given dimension is unbounded dynamic.
  bool is_unbounded_dynamic_dimension(int dimension) const {
    return dimensions_[dimension] == kUnboundedSize;
  }

  // Sets a given dimension as unbounded dynamic.
  void set_unbounded_dynamic_dimension(int dimension) {
    dynamic_dimensions_[dimension] = true;
    dimensions_[dimension] = kUnboundedSize;
  }

  // Returns true if the shape has one or more dimensions with bounded sizes.
  // Tuple shapes are traversed recursively, returns true if any element is
  // bounded dynamic.
  bool is_bounded_dynamic() const;

  // Returns true if the given dimension is bounded dynamic.
  bool is_bounded_dynamic_dimension(int dimension) const {
    return is_dynamic_dimension(dimension) &&
           !is_unbounded_dynamic_dimension(dimension);
  }

  // Returns true if the given dimension is dynamically-sized.
  bool is_dynamic_dimension(int dimension) const {
    return dynamic_dimensions_[dimension];
  }

  // Returns true if the given dimension is statically-sized.
  bool is_static_dimension(int dimension) const {
    return !dynamic_dimensions_[dimension];
  }

  // Sets whether or not the given dimension is dynamically-sized.
  void set_dynamic_dimension(int dimension, bool is_dynamic) {
    dynamic_dimensions_[dimension] = is_dynamic;
  }

  absl::Span<const bool> dynamic_dimensions() const {
    return dynamic_dimensions_;
  }

  absl::Span<bool> mutable_dynamic_dimensions() {
    return absl::MakeSpan(dynamic_dimensions_);
  }

  // Removes the given dimension from the shape. Layout, if it exists, is
  // adjusted to match the modified shape.
  void DeleteDimension(int64_t dim_to_delete);
  void DeleteDimensions(absl::Span<const int64_t> sorted_dims_to_delete);

  // Methods for accessing the primitive type.
  PrimitiveType element_type() const { return element_type_; }
  void set_element_type(PrimitiveType value) { element_type_ = value; }

  // Methods for accessing the dimensions array.
  int dimensions_size() const { return dimensions_.size(); }
  int64_t dimensions(int index) const { return dimensions_[index]; }

  int64_t dimensions_minor(int index) const {
    CHECK(has_layout());
    return dimensions_[layout_->minor_to_major(index)];
  }
  void set_dimensions(int index, int64_t value) { dimensions_[index] = value; }
  void set_dimensions_minor(int index, int64_t value) {
    CHECK(has_layout());
    dimensions_[layout_->minor_to_major(index)] = value;
  }
  void add_dimensions(int64_t value) {
    dimensions_.push_back(value);
    dynamic_dimensions_.push_back(false);
  }
  void clear_dimensions() {
    dimensions_.clear();
    dynamic_dimensions_.clear();
  }
  absl::Span<const int64_t> dimensions() const { return dimensions_; }
  absl::Span<int64_t> mutable_dimensions() {
    return absl::MakeSpan(dimensions_);
  }

  // Methods for accessing the tuple subshapes. This field only non-empty for
  // tuple shapes.
  int tuple_shapes_size() const { return tuple_shapes_.size(); }
  const Shape& tuple_shapes(int index) const;
  Shape* mutable_tuple_shapes(int index) { return &tuple_shapes_[index]; }
  Shape* add_tuple_shapes();
  void clear_tuple_shapes() { tuple_shapes_.clear(); }
  const std::vector<Shape>& tuple_shapes() const { return tuple_shapes_; }
  std::vector<Shape>* mutable_tuple_shapes() { return &tuple_shapes_; }

  // Methods for accessing the layout field.
  bool has_layout() const { return layout_ != std::nullopt; }
  const Layout& layout() const {
    CHECK(has_layout()) << ShortDebugString();
    return *layout_;
  }
  Layout* mutable_layout() {
    CHECK(IsArray()) << ShortDebugString();
    if (layout_ == std::nullopt) {
      layout_.emplace();
    }
    return &(*layout_);
  }
  void clear_layout() { layout_ = std::nullopt; }

  // Recursively clear all dynamic dimension of a shape, including bounded and
  // unbounded dynamic dimensions.
  void clear_dynamic_dimensions() {
    if (!IsTuple()) {
      if (is_dynamic()) {
        mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
      }
      for (int64_t i = 0; i < dynamic_dimensions_.size(); ++i) {
        dynamic_dimensions_[i] = false;
      }
      return;
    }
    for (auto& subshape : tuple_shapes_) {
      subshape.clear_dynamic_dimensions();
    }
  }

  void Clear() {
    element_type_ = PRIMITIVE_TYPE_INVALID;
    clear_dimensions();
    tuple_shapes_.clear();
    clear_layout();
  }

  std::string SerializeAsString() const {
    return ToProto().SerializeAsString();
  }
  std::string ShortDebugString() const { return ToProto().ShortDebugString(); }
  std::string DebugString() const { return ToProto().DebugString(); }

  // Equal is a configurable functor to check the equality of two shapes.
  //
  // Examples:
  //
  // - Comparing two shapes ignoring their layout difference:
  //   Equal().IgnoreLayout()(shape1, shape2);
  //
  // - Comparing two shapes ignoring their layout and element type difference:
  //   Equal().IgnoreLayout().IgnoreElementType()(shape1, shape2);
  class Equal {
   public:
    Equal() = default;

    bool operator()(const Shape& lhs, const Shape& rhs);

    Equal& IgnoreLayout(bool ignore_layout = true) {
      ignore_layout_ = ignore_layout;
      return *this;
    }
    Equal& IgnoreTilesInLayout() {
      ignore_tiles_in_layout_ = true;
      return *this;
    }
    Equal& IgnoreElementSizeInLayout() {
      ignore_element_size_in_layout_ = true;
      return *this;
    }
    Equal& IgnoreMemorySpaceInLayout() {
      ignore_memory_space_in_layout_ = true;
      return *this;
    }
    Equal& MinorToMajorOnlyInLayout() {
      ignore_tiles_in_layout_ = true;
      ignore_element_size_in_layout_ = true;
      ignore_memory_space_in_layout_ = true;
      ignore_tail_padding_alignment_in_elements_in_layout_ = true;
      ignore_split_config_in_layout_ = true;
      return *this;
    }
    Equal& IgnoreElementType() {
      ignore_element_type_ = true;
      return *this;
    }
    Equal& IgnoreFpPrecision() {
      ignore_fp_precision_ = true;
      return *this;
    }
    Equal& IgnoreDynamicDimension() {
      ignore_dynamic_dimension_ = true;
      return *this;
    }
    Equal& IgnoreDimensions() {
      ignore_dimensions_ = true;
      return *this;
    }
    Equal& IgnoreTailPaddingAlignmentInElements() {
      ignore_tail_padding_alignment_in_elements_in_layout_ = true;
      return *this;
    }
    Equal& IgnoreSplitConfigInLayout() {
      ignore_split_config_in_layout_ = true;
      return *this;
    }

   private:
    bool ignore_layout_ = false;
    bool ignore_tiles_in_layout_ = false;
    bool ignore_element_size_in_layout_ = false;
    bool ignore_memory_space_in_layout_ = false;
    bool ignore_element_type_ = false;
    bool ignore_fp_precision_ = false;
    bool ignore_dynamic_dimension_ = false;
    bool ignore_dimensions_ = false;
    bool ignore_tail_padding_alignment_in_elements_in_layout_ = false;
    bool ignore_split_config_in_layout_ = false;
  };

  // Test that all fields of the shape are the same, equivalent to Equal().
  bool operator==(const Shape& other) const { return Equal()(*this, other); }
  bool operator!=(const Shape& other) const { return !(*this == other); }

  template <typename H, bool kIsLayoutSensitive = true>
  static H Hash(H h, const Shape& s) {
    if (s.IsTuple()) {
      for (const Shape& subshape : s.tuple_shapes_) {
        h = Shape::Hash<H, kIsLayoutSensitive>(std::move(h), subshape);
      }
      return H::combine(std::move(h), s.tuple_shapes_size());
    }
    h = H::combine(std::move(h), s.element_type_, s.dimensions_,
                   s.dynamic_dimensions_);
    if (kIsLayoutSensitive) {
      h = H::combine(std::move(h), s.layout_);
    }
    return std::move(h);
  }

  template <typename H>
  friend H AbslHashValue(H h, const Shape& s) {
    return Shape::Hash(std::move(h), s);
  }

 private:
  // The element type of this shape (tuple, array, etc).
  PrimitiveType element_type_ = PRIMITIVE_TYPE_INVALID;

  // The array bounds of the dimensions. This is nonempty only for array
  // shapes. For a dynamically-sized dimension, the respective value in this
  // vector is an inclusive upper limit of the array bound.
  DimensionVector dimensions_;

  // This vector is the same size as 'dimensions_' and indicates whether the
  // respective dimension is dynamically sized.
  absl::InlinedVector<bool, InlineRank()> dynamic_dimensions_;

  // The tuple element subshapes. This is nonempty only for tuple shapes.
  std::vector<Shape> tuple_shapes_;

  // The layout of the shape. Only relevant for arrays.
  std::optional<Layout> layout_;
};

// Shape of the parameters and output of an XLA computation. This is analogous
// to a traditional function signature.
class ProgramShape {
 public:
  ProgramShape();
  ~ProgramShape();
  ProgramShape(const ProgramShape&);
  ProgramShape(ProgramShape&&);
  ProgramShape& operator=(const ProgramShape&);
  ProgramShape& operator=(ProgramShape&&);

  // Creates a ProgramShape from a ProgramShapeProto protobuf.
  explicit ProgramShape(const ProgramShapeProto& program_shape_proto);

  // Returns a proto representation of the object.
  ProgramShapeProto ToProto() const;

  void Print(Printer* printer) const;

  std::string ToString() const;

  // The following methods mirror the protobuf generated code interface for the
  // message ProgramShapeProto. This enabled easy migration of this data
  // structure from a proto to a proper C++ class.
  // TODO(b/29771030): Replace or augment these methods with a more ergonomic
  // interface.

  // Methods for accessing and manipulating the Shape of the parameters.
  int parameters_size() const { return parameters_.size(); }
  const Shape& parameters(int index) const { return parameters_[index]; }
  Shape* mutable_parameters(int index) { return &parameters_[index]; }
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
  const std::string& parameter_names(int index) const {
    return parameter_names_[index];
  }
  void set_parameter_names(int index, const std::string& value) {
    parameter_names_[index] = value;
  }
  std::string* mutable_parameter_names(int index) {
    return &parameter_names_[index];
  }
  void add_parameter_names(const std::string& value) {
    parameter_names_.push_back(value);
  }
  std::string* add_parameter_names() {
    parameter_names_.push_back("");
    return &parameter_names_.back();
  }
  void clear_parameter_names() { parameter_names_.clear(); }
  const std::vector<std::string>& parameter_names() const {
    return parameter_names_;
  }
  std::vector<std::string>* mutable_parameter_names() {
    return &parameter_names_;
  }

  std::string ShortDebugString() const { return ToProto().ShortDebugString(); }
  std::string DebugString() const { return ToProto().DebugString(); }

 private:
  // The shapes of the parameters of the computation represented by this object.
  std::vector<Shape> parameters_;

  // The names of the parameters of the computation represented by this object.
  std::vector<std::string> parameter_names_;

  // The shape of the result of the computation represented by this object.
  Shape result_;
};

std::ostream& operator<<(std::ostream& out, const Shape& shape);
std::ostream& operator<<(std::ostream& out, const ProgramShape& program_shape);

}  // namespace xla

#endif  // XLA_SHAPE_H_
