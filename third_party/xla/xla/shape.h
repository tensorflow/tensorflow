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

#include <stdbool.h>

#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/macros.h"
#include "absl/base/optimization.h"
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
//
// Depending on the element type, the shape falls into one of the following
// categories:
//
// - Invalid: element_type == PRIMITIVE_TYPE_INVALID
// - Token: element_type == TOKEN
// - Opaque: element_type == OPAQUE_TYPE
// - Array: element_type is an array type
// - Tuple: element_type == TUPLE
//
// These categories are mutually exclusive, i.e. a shape can only be one of
// them.
class Shape {
 public:
  // Returns true if the given dimension size is valid.
  [[nodiscard]] static bool IsValidDimensionSize(int64_t size,
                                                 bool is_dynamic) {
    return size >= 0 || (is_dynamic && size == kUnboundedSize);
  }

  // Creates an invalid shape, with element type PRIMITIVE_TYPE_INVALID and the
  // other fields empty.
  Shape();

  ~Shape();

  Shape(const Shape&);
  Shape(Shape&&) noexcept;
  Shape& operator=(const Shape&);
  Shape& operator=(Shape&&) noexcept;

  // Constructs a shape from a ShapeProto. Results in an invalid shape (as
  // opposed to crashing) if the proto has logically invalid fields.
  ABSL_DEPRECATE_AND_INLINE()
  explicit Shape(const ShapeProto& shape_proto)
      : Shape(FromProto(shape_proto).value_or(Shape())) {}

  // Creates a token, opaque or buffer shape.
  // Precondition:
  //  - `element_type` must be TOKEN, OPAQUE_TYPE or BUFFER.
  explicit Shape(PrimitiveType element_type);

  // Creates an array shape. `dimensions` can be empty, in which case the shape
  // is a scalar (degenerated array).
  // Precondition:
  //  - `element_type` must be a valid array type.
  //  - `dynamic_dimensions` must be either empty or have the same size as
  //    `dimensions`. If it's empty (the default), all dimensions are static.
  //    Otherwise, `dynamic_dimensions[i]` is true if the `i`th dimension is
  //    dynamic.
  Shape(PrimitiveType element_type, absl::Span<const int64_t> dimensions,
        absl::Span<const bool> dynamic_dimensions = {});

  // Creates a tuple shape. `tuple_shapes` can be empty, in which case the
  // shape is a nil shape (empty tuple).
  explicit Shape(std::vector<Shape> tuple_shapes);

  // Constructs a shape from a ShapeProto. Results in an invalid shape (as
  // opposed to crashing) if the proto has logically invalid fields.
  static absl::StatusOr<Shape> FromProto(const ShapeProto& shape_proto);

  // Returns a ShapeProto representation of the Shape.
  ShapeProto ToProto() const;

  // Prints a human-readable string that represents the given shape, with or
  // without layout. e.g. "F32[42,12] {0, 1}" or "F32[64]".
  void Print(Printer* printer, bool print_layout = false) const;

  // Returns a human-readable string that represents the given shape, with or
  // without layout. e.g. "F32[42,12] {0, 1}" or "F32[64]".
  std::string ToString(bool print_layout = false) const;

  // Returns whether the shape is an array primitive type, that is, whether the
  // state of the shape is an ArrayState.
  bool IsArrayExcludingBuffer() const {
    const bool result =
        primitive_util::IsArrayType(element_type_including_buffer());
    // We do this check in debug mode only to avoid performance regressions.
    DCHECK_EQ(result, if_array_state() != nullptr)
        << "Shape " << ToString()
        << " has inconsistent element_type and state.";
    return result;
  }
  // Returns whether the shape is a tuple primitive type, that is, whether the
  // state of the shape is a TupleState.
  bool IsTuple() const {
    const bool result = element_type_including_buffer() == TUPLE;
    // We do this check in debug mode only to avoid performance regressions.
    DCHECK_EQ(result, if_tuple_state() != nullptr)
        << "Shape " << ToString()
        << " has inconsistent element_type and state.";
    return result;
  }
  // Returns whether the shape is a buffer primitive type, that is, whether the
  // state of the shape is a BufferState.
  bool IsBuffer() const {
    const bool result = element_type_including_buffer() == BUFFER;
    // We do this check in debug mode only to avoid performance regressions.
    DCHECK_EQ(result, if_buffer_state() != nullptr)
        << "Shape " << ToString()
        << " has inconsistent element_type and state.";
    return result;
  }
  // Returns whether the shape is a token primitive type, that is, whether the
  // state of the shape is a TokenState.
  bool IsToken() const {
    const bool result = element_type_including_buffer() == TOKEN;
    // We do this check in debug mode only to avoid performance regressions.
    DCHECK_EQ(result, if_token_state() != nullptr)
        << "Shape " << ToString()
        << " has inconsistent element_type and state.";
    return result;
  }
  // Returns whether the shape is an opaque primitive type, that is, whether the
  // state of the shape is an OpaqueState.
  bool IsOpaque() const {
    const bool result = element_type_including_buffer() == OPAQUE_TYPE;
    // We do this check in debug mode only to avoid performance regressions.
    DCHECK_EQ(result, if_opaque_state() != nullptr)
        << "Shape " << ToString()
        << " has inconsistent element_type and state.";
    return result;
  }
  // Returns true if the shape is an array or a buffer.
  bool IsArray() const { return IsArrayExcludingBuffer() || IsBuffer(); }

  // Returns whether all elements in the shape are integers.
  // Tuple shapes are traversed recursively.
  bool AreAllLeavesIntegers() const;

  // Returns true if no array dimension in the shape is dynamically sized. Tuple
  // shapes are traversed recursively.
  bool is_static() const;

  // Returns true if the shape contains at least one dynamic dimension. Tuple
  // shapes are traversed recursively.
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
  // Precondition: this is an array shape and `dimension` is a valid dimension
  // index.
  bool is_unbounded_dynamic_dimension(int dimension) const {
    return array_state_maybe_underneath_buffer().dimensions[dimension] ==
           kUnboundedSize;
  }

  // Sets a given dimension as unbounded dynamic.
  // Precondition: this is an array shape and `dimension` is a valid dimension
  // index.
  void set_unbounded_dynamic_dimension(int dimension) {
    ArrayState& state = array_state_maybe_underneath_buffer();
    state.dynamic_dimensions[dimension] = true;
    state.dimensions[dimension] = kUnboundedSize;
  }

  // Returns true if the shape has one or more dimensions with bounded sizes.
  // Tuple shapes are traversed recursively, returns true if any element is
  // bounded dynamic.
  bool is_bounded_dynamic() const;

  // Returns true if the given dimension is bounded dynamic.
  // Precondition: this is an array shape and `dimension` is a valid dimension
  // index.
  bool is_bounded_dynamic_dimension(int dimension) const {
    return is_dynamic_dimension(dimension) &&
           !is_unbounded_dynamic_dimension(dimension);
  }

  // Returns true if the given dimension is dynamically-sized.
  // Precondition: this is an array shape and `dimension` is a valid dimension
  // index.
  bool is_dynamic_dimension(int dimension) const {
    return array_state_maybe_underneath_buffer().dynamic_dimensions[dimension];
  }

  // Returns true if the given dimension is statically-sized.
  // Precondition: this is an array shape and `dimension` is a valid dimension
  // index.
  bool is_static_dimension(int dimension) const {
    return !array_state_maybe_underneath_buffer().dynamic_dimensions[dimension];
  }

  // Sets whether or not the given dimension is dynamically-sized.
  // Precondition:
  //   - This is an array shape.
  //   - `dimension` is a valid dimension index.
  //   - The dimension's size is valid for the given dynamic-ness.
  void set_dynamic_dimension(int dimension, bool is_dynamic);

  // Returns a span to indicate whether each dimension is dynamic.
  // Precondition: this is an array shape.
  absl::Span<const bool> dynamic_dimensions() const {
    return array_state_maybe_underneath_buffer().dynamic_dimensions;
  }

  // Removes the given dimension from the shape. Layout, if it exists, is
  // adjusted to match the modified shape.
  // Precondition: this is an array shape, and the input dimension indices are
  // valid.
  void DeleteDimension(int64_t dim_to_delete);
  // Like the above, but deletes multiple dimensions at once. The dimensions
  // must not contain duplicates.
  void DeleteDimensions(absl::Span<const int64_t> dims_to_delete);

  // Returns the primitive type of the shape.
  PrimitiveType element_type_including_buffer() const { return element_type_; }

  // Returns the primitive type of the array or buffer shape.
  // Precondition: this is an array shape or a buffer shape.
  PrimitiveType element_type() const {
    if (const auto* const state = if_buffer_state()) {
      return state->buffer_shape->element_type();
    }
    return element_type_;
  }

  // Sets the primitive type of the shape. If the new type and the old type
  // are in different categories (e.g. array vs. tuple), the state is reset
  // to the default (empty) state for the new type; otherwise, the state is
  // preserved. This behavior ensures that the state is always consistent with
  // the element type.
  void set_element_type(PrimitiveType value);

  // Returns the number of dimensions in the shape.
  // Precondition: this is an array shape.
  ABSL_DEPRECATE_AND_INLINE()
  inline int dimensions_size() const { return dimensions().size(); }

  // Returns the size of the given dimension if it's static, or the upper bound
  // of the dimension size if it's dynamic.
  // Precondition: this is an array shape and `index` is a valid dimension
  // index.
  int64_t dimensions(int index) const {
    return array_state_maybe_underneath_buffer().dimensions[index];
  }

  // Returns the size of the index-th minor dimension.
  // Precondition: this is an array shape, `index` is a valid dimension
  // index, and the shape has a layout.
  int64_t dimensions_minor(int index) const {
    CHECK(has_layout());
    const ArrayState& state = array_state_maybe_underneath_buffer();
    return state.dimensions[state.layout->minor_to_major(index)];
  }

  // Sets the size of the given dimension if it's static, or sets the upper
  // bound of the dimension size if it's dynamic.
  // Arguments:
  //   - `index` is the index of the dimension.
  //   - `size` is the size of the dimension if it is static, or the upper
  //      bound of the dimension size if it is dynamic.
  //   - `is_dynamic` is the dynamic-ness of the dimension:
  //     - false: the dimension is static.
  //     - true: the dimension is dynamic.
  //     - nullopt: don't change the dynamic-ness of the dimension.
  // Precondition:
  //   - This is an array shape.
  //   - `index` is a valid dimension index
  //   - `size` is either >= 0 or, when the dimension is dynamic,
  //     kUnboundedSize.
  void set_dimensions(int index, int64_t size,
                      std::optional<bool> is_dynamic = std::nullopt);

  // Like set_dimensions, but sets the index-th minor dimension instead of
  // the index-th dimension.
  void set_dimensions_minor(int index, int64_t size,
                            std::optional<bool> is_dynamic = std::nullopt);

  // Appends a new dimension with the given size.
  // Arguments:
  //   - `value` is the size of the dimension if it is static, or the upper
  //      bound of the dimension size if it is dynamic.
  //   - `is_dynamic` is the dynamic-ness of the dimension:
  //     - false: the dimension is static.
  //     - true: the dimension is dynamic.
  // Precondition:
  //   - This is an array shape.
  //   - Either `value` is >= 0, or `is_dynamic` is true and `value` is
  //     kUnboundedSize.
  void add_dimensions(int64_t value, bool is_dynamic = false);

  // Clears all dimensions (i.e. makes this shape a scalar).
  // Precondition: this is an array shape.
  void clear_dimensions() {
    ArrayState& state = array_state_maybe_underneath_buffer();
    state.dimensions.clear();
    state.dynamic_dimensions.clear();
  }

  // Returns a span to indicate the size of each dimension.
  // Precondition: this is an array shape.
  absl::Span<const int64_t> dimensions() const {
    return array_state_maybe_underneath_buffer().dimensions;
  }

  // Returns the number of top-level tuple components in this shape.
  // Precondition: this is a tuple shape.
  ABSL_DEPRECATE_AND_INLINE()
  inline int tuple_shapes_size() const { return tuple_shapes().size(); }

  // Returns the shape of the i-th tuple component.
  // Precondition: this is a tuple shape and `index` is a valid tuple component
  // index.
  const Shape& tuple_shapes(int index) const;
  Shape* mutable_tuple_shapes(int index);

  // Appends a new invalid shape to the tuple and returns a pointer to it.
  // Precondition: this is a tuple shape.
  // Postcondition: the returned pointer is not null, and the pointee is owned
  // by this shape.
  Shape* add_tuple_shapes();

  // Clears all tuple components (i.e. makes this shape a 0-tuple).
  // Precondition: this is a tuple shape.
  void clear_tuple_shapes() { tuple_state().tuple_shapes.clear(); }

  // Returns a vector of all tuple component shapes.
  // Precondition: this is a tuple shape.
  const std::vector<Shape>& tuple_shapes() const;
  std::vector<Shape>* mutable_tuple_shapes() {
    return &tuple_state().tuple_shapes;
  }

  // Returns the underlying shape of the buffer.
  const Shape& buffer_shape() const;

  // Returns true if the shape is an array storage and has a layout. Both
  // if_array_state and if_buffer_state correspond to an array storage.
  bool has_layout() const {
    if (!if_array_state() && !if_buffer_state()) {
      return false;
    }
    const ArrayState& state = array_state_maybe_underneath_buffer();
    return state.layout != std::nullopt;
  }

  // Returns the layout of the shape.
  // Precondition: has_layout() is true.
  const Layout& layout() const {
    CHECK(has_layout()) << ToString();
    return *array_state_maybe_underneath_buffer().layout;
  }

  // Returns a pointer to the layout of the shape. If the shape does not have a
  // layout, an empty layout is created.
  // Precondition: this is an array shape or a buffer shape.
  // Postcondition: the returned pointer is not null, and the pointee is owned
  // by this shape.
  Layout* mutable_layout() {
    ArrayState& state = array_state_maybe_underneath_buffer();
    if (state.layout == std::nullopt) {
      state.layout.emplace();
    }
    return &(*state.layout);
  }

  // Removes the layout of the shape, if any.
  // Precondition: this is an array shape.
  void clear_layout() {
    array_state_maybe_underneath_buffer().layout = std::nullopt;
  }

  // Recursively clear all dynamic dimension of a shape, including bounded and
  // unbounded dynamic dimensions. Clearing a dynamic dimension means
  // changing the dimension to static and setting its size as the dynamic
  // dimension's size upper bound.
  void clear_dynamic_dimensions() {
    if (if_array_state() || if_buffer_state()) {
      ArrayState& state = array_state_maybe_underneath_buffer();
      if (is_dynamic()) {
        mutable_layout()->set_dynamic_shape_metadata_prefix_bytes(0);
      }
      for (int64_t i = 0; i < state.dynamic_dimensions.size(); ++i) {
        state.dynamic_dimensions[i] = false;
      }
      return;
    }
    if (auto* const state = if_tuple_state()) {
      for (auto& subshape : state->tuple_shapes) {
        subshape.clear_dynamic_dimensions();
      }
    }
  }

  // Resets this to the default state (an invalid shape).
  void Clear();

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
    Equal& IgnoreBuffer(bool ignore_buffer = true) {
      ignore_buffer_ = ignore_buffer;
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
    bool ignore_buffer_ = false;
  };

  // Test that all fields of the shape are the same, equivalent to Equal().
  bool operator==(const Shape& other) const { return Equal()(*this, other); }
  bool operator!=(const Shape& other) const { return !(*this == other); }

  template <typename H, bool kIsLayoutSensitive = true>
  static H Hash(H h, const Shape& s) {
    if (const auto* const state = s.if_tuple_state()) {
      for (const Shape& subshape : state->tuple_shapes) {
        h = Shape::Hash<H, kIsLayoutSensitive>(std::move(h), subshape);
      }
      return H::combine(std::move(h), state->tuple_shapes.size());
    }
    if (const auto* const state = s.if_array_state()) {
      h = H::combine(std::move(h), s.element_type_, state->dimensions,
                     state->dynamic_dimensions);
      if (kIsLayoutSensitive) {
        h = H::combine(std::move(h), state->layout);
      }
      return h;
    }
    if (const auto* const state = s.if_buffer_state()) {
      return H::combine(std::move(h), s.element_type_, state->buffer_shape);
    }
    return H::combine(std::move(h), s.element_type_);
  }

  template <typename H>
  friend H AbslHashValue(H h, const Shape& s) {
    return Shape::Hash(std::move(h), s);
  }

 private:
  friend class ShapeUtil;
  friend absl::Status ValidateNonLayoutProperties(const Shape& shape);

  // Define one state struct for each shape category. Depending on the element
  // type, the state_ variant will be set to exactly one of these structs.
  // This design has several benefits:
  //   - It prevents (by construction) bugs where the shape's state has
  //     non-empty fields that don't match the shape's element type.
  //   - It prevents (by construction) bugs where the code accesses a field
  //     of a shape's state that doesn't match the shape's element type (e.g.
  //     accessing the tuple_shapes field of an array shape).
  //   - It simplifies the code by eliminating the need for runtime handling of
  //     fields that are irrelevant to the shape's category.
  //   - It reduces the size of the Shape class as the variant doesn't need to
  //     store the fields for all shape categories at once.
  struct InvalidState {};
  struct TokenState {};
  struct OpaqueState {};
  struct ArrayState {
    // The array bounds of the dimensions. For a dynamically-sized dimension,
    // the respective value in this vector is an inclusive upper limit of the
    // array bound.
    DimensionVector dimensions;

    // This vector has the same size as 'dimensions' and indicates whether the
    // respective dimension is dynamically sized.
    absl::InlinedVector<bool, InlineRank()> dynamic_dimensions;

    // The layout of the shape.
    std::optional<Layout> layout;
  };
  struct TupleState {
    // The tuple element subshapes.
    std::vector<Shape> tuple_shapes;
  };
  struct BufferState {
    // Creates a buffer state with an empty buffer shape.
    BufferState();

    // Supports copying.
    BufferState(const BufferState& state);
    BufferState& operator=(const BufferState& state);

    // The underlying array shape for the buffer type. Not null.
    // Using Shape directly results in a circular dependency.
    std::unique_ptr<Shape> buffer_shape;
  };
  using State = std::variant<InvalidState, TokenState, OpaqueState, ArrayState,
                             TupleState, BufferState>;

  // CHECKs that the dimension size is valid.
  void CheckDimensionSize(int dim_index, int64_t size, bool is_dynamic);

  // Like add_dimensions(), but does not CHECK that the arguments are valid.
  // Instead, we rely on validation down the road to catch invalid shapes.
  // This is useful for code that should not crash, such as constructing a
  // Shape from an unvalidated proto.
  void UnsafeAddDimension(int64_t value, bool is_dynamic);

  // Convenience accessors for the state_ variant. Each if_*_state() accessor
  // returns a pointer to the corresponding state struct, or nullptr if the
  // shape is not of the corresponding category. The version without the `if_`
  // prefix is similar, but will CHECK-fail if the shape is not of the
  // corresponding category. I.e. if_foo_state() vs foo_state() is analogous to
  // std::get_if() vs std::get().
  //
  // In general, prefer foo_state() over if_foo_state() as the former catches
  // programmer errors earlier and generates a more informative error message.
  // However, if_foo_state() is useful in cases where it's not a programmer
  // error if the shape is not of the corresponding category.

  const InvalidState* if_invalid_state() const {
    return std::get_if<InvalidState>(&state_);
  }
  const TokenState* if_token_state() const {
    return std::get_if<TokenState>(&state_);
  }
  const OpaqueState* if_opaque_state() const {
    return std::get_if<OpaqueState>(&state_);
  }
  const ArrayState* if_array_state() const {
    return std::get_if<ArrayState>(&state_);
  }
  ArrayState* if_array_state() { return std::get_if<ArrayState>(&state_); }
  const TupleState* if_tuple_state() const {
    return std::get_if<TupleState>(&state_);
  }
  TupleState* if_tuple_state() { return std::get_if<TupleState>(&state_); }
  const BufferState* if_buffer_state() const {
    return std::get_if<BufferState>(&state_);
  }
  BufferState* if_buffer_state() { return std::get_if<BufferState>(&state_); }

  const InvalidState& invalid_state() const;
  const TokenState& token_state() const;
  const OpaqueState& opaque_state() const;

  // Returns the array state of the array state of the buffer shape, assuming
  // that the shape is an array or a buffer shape.
  const ArrayState& array_state_maybe_underneath_buffer() const;
  ArrayState& array_state_maybe_underneath_buffer();

  const ArrayState& array_state() const;
  ArrayState& array_state();

  const TupleState& tuple_state() const;
  TupleState& tuple_state();

  const BufferState& buffer_state() const;
  BufferState& buffer_state();

  // CHECK-fails if this shape's state is not empty.
  void CheckStateIsEmpty() const;

  // The element type of this shape (tuple, array, etc).
  PrimitiveType element_type_ = PRIMITIVE_TYPE_INVALID;

  // The state of this shape.
  // Invariant: element_type_ always matches the type held in this variant.
  State state_;
};

// Shape of the parameters and output of an XLA computation. This is analogous
// to a traditional function signature.
class ProgramShape {
 public:
  // Constructs an empty ProgramShape, which has 0 parameters and an empty
  // (invalid) result shape.
  ProgramShape();
  ~ProgramShape();

  ProgramShape(const ProgramShape&);
  ProgramShape(ProgramShape&&);
  ProgramShape& operator=(const ProgramShape&);
  ProgramShape& operator=(ProgramShape&&);

  // Constructs a ProgramShape from a ProgramShapeProto protobuf. If the
  // ProgramShapeProto is invalid, an empty ProgramShape is constructed.
  ABSL_DEPRECATE_AND_INLINE()
  explicit ProgramShape(const ProgramShapeProto& program_shape_proto)
      : ProgramShape(FromProto(program_shape_proto).value_or(ProgramShape())) {}

  // Creates a ProgramShape from a ProgramShapeProto protobuf.
  static absl::StatusOr<ProgramShape> FromProto(
      const ProgramShapeProto& program_shape_proto);

  // Returns a proto representation of the object.
  ProgramShapeProto ToProto() const;

  void Print(Printer* printer) const;

  std::string ToString() const;

  // Methods for accessing and manipulating the Shape of the parameters.
  int parameters_size() const { return parameters_.size(); }
  const Shape& parameters(int index) const { return parameters_[index]; }
  Shape* mutable_parameters(int index) { return &parameters_[index]; }
  void AddParameter(Shape shape, std::string name) {
    parameters_.push_back(std::move(shape));
    parameter_names_.push_back(std::move(name));
  }

  void clear_parameters() {
    parameters_.clear();
    parameter_names_.clear();
  }
  const std::vector<Shape>& parameters() const { return parameters_; }

  // Methods for accessing and manipulating the Shape of the result.
  const Shape& result() const { return result_; }
  Shape* mutable_result() { return &result_; }

  // Methods for accessing and manipulating the names of the parameters.
  const std::string& parameter_names(int index) const {
    return parameter_names_[index];
  }
  void set_parameter_names(int index, const std::string& value) {
    parameter_names_[index] = value;
  }
  void clear_parameter_names() {
    for (auto& name : parameter_names_) {
      name.clear();
    }
  }
  const std::vector<std::string>& parameter_names() const {
    return parameter_names_;
  }

 private:
  // Invariant: parameters_ and parameter_names_ have the same size.

  // The shapes of the parameters of the computation represented by this object.
  std::vector<Shape> parameters_;

  // The names of the parameters of the computation represented by this object.
  std::vector<std::string> parameter_names_;

  // The shape of the result of the computation represented by this object.
  Shape result_;
};

std::ostream& operator<<(std::ostream& out, const Shape& shape);
std::ostream& operator<<(std::ostream& out, const ProgramShape& program_shape);

// We prefer to keep small functions that are on a hot path of various ShapeUtil
// traversal functions in the header file, to avoid the overhead of function
// call indirection. We do it only for small functions, to avoid the code bloat.

inline const Shape::InvalidState& Shape::invalid_state() const {
  const auto* const state = if_invalid_state();
  CHECK(state) << "Expected an invalid shape. Got " << ToString();
  return *state;
}

inline const Shape::TokenState& Shape::token_state() const {
  const auto* const state = if_token_state();
  CHECK(state) << "Expected a token shape. Got " << ToString();
  return *state;
}

inline const Shape::OpaqueState& Shape::opaque_state() const {
  const auto* const state = if_opaque_state();
  CHECK(state) << "Expected an opaque shape. Got " << ToString();
  return *state;
}

inline const Shape::ArrayState& Shape::array_state() const {
  const auto* const state = if_array_state();
  CHECK(state) << "Expected an array shape. Got " << ToString()
               << "\nThis is a programmer error. Please read "
                  "the Shape object's array properties (e.g. dimensions) "
                  "only when it's an array shape.";
  return *state;
}

inline Shape::ArrayState& Shape::array_state() {
  auto* const state = if_array_state();
  CHECK(state) << "Expected an array shape. Got " << ToString()
               << "\nThis is a programmer error. Please mutate "
                  "the Shape object's array properties (e.g. dimensions) "
                  "only when it's an array shape.";
  return *state;
}

inline const Shape::TupleState& Shape::tuple_state() const {
  const auto* const state = if_tuple_state();
  CHECK(state) << "Expected a tuple shape. Got " << ToString()
               << "\nThis is a programmer error. Please read "
                  "the Shape object's tuple properties (e.g. tuple_shapes) "
                  "only when it's a tuple shape.";
  return *state;
}

inline Shape::TupleState& Shape::tuple_state() {
  auto* const state = if_tuple_state();
  CHECK(state) << "Expected a tuple shape. Got " << ToString()
               << "\nThis is a programmer error. Please mutate "
                  "the Shape object's tuple properties (e.g. tuple_shapes) "
                  "only when it's a tuple shape.";
  return *state;
}

inline const Shape::BufferState& Shape::buffer_state() const {
  const auto* const state = if_buffer_state();
  CHECK(state) << "Expected a buffer shape. Got " << ToString()
               << "\nThis is a programmer error. Please read "
                  "the Shape object's buffer properties (e.g. buffer_shape) "
                  "only when it's a buffer shape.";
  return *state;
}

inline Shape::BufferState& Shape::buffer_state() {
  auto* const state = if_buffer_state();
  CHECK(state) << "Expected a buffer shape. Got " << ToString()
               << "\nThis is a programmer error. Please mutate "
                  "the Shape object's buffer properties (e.g. buffer_shape) "
                  "only when it's a buffer shape.";
  return *state;
}

inline const Shape::ArrayState& Shape::array_state_maybe_underneath_buffer()
    const {
  if (const ArrayState* array = if_array_state(); ABSL_PREDICT_TRUE(array)) {
    return *array;
  }
  const BufferState* buffer = if_buffer_state();
  CHECK_NE(buffer, nullptr);
  return *buffer->buffer_shape->if_array_state();
}

inline Shape::ArrayState& Shape::array_state_maybe_underneath_buffer() {
  if (ArrayState* array = if_array_state(); ABSL_PREDICT_TRUE(array)) {
    return *array;
  }
  BufferState* buffer = if_buffer_state();
  CHECK_NE(buffer, nullptr);
  return *buffer->buffer_shape->if_array_state();
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE const Shape& Shape::tuple_shapes(
    int index) const {
  return tuple_state().tuple_shapes[index];
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE Shape* Shape::mutable_tuple_shapes(
    int index) {
  return &tuple_state().tuple_shapes[index];
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE const std::vector<Shape>&
Shape::tuple_shapes() const {
  return tuple_state().tuple_shapes;
}

inline ABSL_ATTRIBUTE_ALWAYS_INLINE const Shape& Shape::buffer_shape() const {
  return *buffer_state().buffer_shape;
}

}  // namespace xla

#endif  // XLA_SHAPE_H_
