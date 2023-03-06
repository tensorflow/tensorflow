/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_COMPILER_XLA_LITERAL_H_
#define TENSORFLOW_COMPILER_XLA_LITERAL_H_

#include <algorithm>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <ostream>
#include <string>
#include <type_traits>
#include <utility>
#include <variant>
#include <vector>

#include "absl/functional/function_ref.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/printer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/tsl/lib/core/bitmap.h"
#include "tensorflow/tsl/platform/cpu_info.h"
#include "tensorflow/tsl/platform/logging.h"
#include "tensorflow/tsl/platform/protobuf.h"
#include "tensorflow/tsl/platform/status.h"

namespace xla {

// Forward declare Literal and LiteralSlice class to be used by the creation
// methods in the base class.
class Literal;
class LiteralSlice;

// Abstract base class for literals.
class LiteralBase {
 public:
  virtual ~LiteralBase() = 0;

  // Literals are equal if they have compatible shapes and the same data
  // values. Layout is not compared.
  bool operator==(const LiteralBase& other) const;
  bool operator!=(const LiteralBase& other) const { return !(*this == other); }

  // Returns the shape of the literal.
  const Shape& shape() const;

  // Serialize to proto.
  LiteralProto ToProto() const;

  // Returns a Span of the array for this literal for the given NativeT
  // (e.g., float). CHECKs if the subshape of the literal at the given
  // ShapeIndex is not array. See primitive_util.h for the mapping from XLA type
  // to native type.
  template <typename NativeT>
  absl::Span<const NativeT> data(const ShapeIndex& shape_index = {}) const;

  // Returns a const pointer to (or size of) the underlying buffer holding the
  // array at the given shape index. CHECKs if the subshape of the literal at
  // the given ShapeIndex is not array.
  const void* untyped_data(const ShapeIndex& shape_index = {}) const;
  int64_t size_bytes(const ShapeIndex& shape_index = {}) const;

  // Returns this literal's data as a string. This literal must be a rank-1 U8
  // array.
  std::string GetR1U8AsString() const;

  // Prints a string representation of the literal value. The Shape of the
  // literal is a prefix of the literal value in the string.
  //
  // Warning: this function can take minutes for multi-million element Literals.
  void Print(Printer* printer) const;

  // Similar to Print, but prints the result in a compact one-line form.
  void PrintOneline(Printer* printer) const;

  // Prints a string representation of the literal value which does *not*
  // include the shape string.
  void PrintWithoutShape(Printer* printer) const;

  // Similar to PrintWithoutShape, but prints the result in a compact one-line
  // form.
  void PrintWithoutShapeOneline(Printer* printer) const;

  // Prints a string representation of the literal value which includes the
  // shape string with its layout.does *not* include the shape string.
  void PrintWithLayout(Printer* printer) const;

  // Similar to PrintWithLayout, but prints the result in a compact one-line
  // form.
  void PrintWithLayoutOneline(Printer* printer) const;

  // Returns a string representation of the literal value. The Shape of the
  // literal is a prefix of the literal value in the string.
  //
  // Warning: this function can take minutes for multi-million element Literals.
  std::string ToString() const;

  // Similar to ToString, but return the result in a compact one-line form.
  std::string ToStringOneline() const;

  // Returns a string representation of the literal value which does *not*
  // include the shape string.
  std::string ToStringWithoutShape() const;

  // Similar to ToStringWithoutShape, but return the result in a compact
  // one-line form.
  std::string ToStringWithoutShapeOneline() const;

  // Returns a string representation of the literal value which includes the
  // shape string with its layout.does *not* include the shape string.
  std::string ToStringWithLayout() const;

  // Similar to ToStringWithLayout, but return the result in a compact one-line
  // form.
  std::string ToStringWithLayoutOneline() const;

  // Gets an element in the literal at the given index. The multi_index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  NativeT Get(absl::Span<const int64_t> multi_index,
              const ShapeIndex& shape_index) const;
  // Overloads of Get for array literals. CHECKs if the literal is not
  // array-shaped and dense.
  template <typename NativeT>
  NativeT Get(absl::Span<const int64_t> multi_index) const;

  // Get the dynamic size on dim_index in the literal at the given shape_index.
  int32_t GetDynamicSize(int64_t dim_index,
                         const ShapeIndex& shape_index) const;
  int32_t GetDynamicSize(int64_t dim_index) const;

  // Returns the element value at index (0, ..., 0), however many zeroes are
  // required for that index.
  template <typename NativeT>
  NativeT GetFirstElement() const;

  // As above but returns any integer type casted to an int64_t.
  std::optional<int64_t> GetFirstInteger() const;

  // As Get(), but determines the correct type and converts the value
  // into text.
  std::string GetAsString(absl::Span<const int64_t> multi_index,
                          const ShapeIndex& shape_index = {}) const;

  // Return whether the value at the specified index is equal to the provided
  // generic `value` (T must be an arithmetic type).
  //
  // Precondition: must be an array.
  template <typename T>
  typename std::enable_if<(std::is_arithmetic<T>::value ||
                           std::is_same<T, Eigen::half>::value ||
                           std::is_same<T, bfloat16>::value ||
                           std::is_same<T, tsl::float8_e5m2>::value ||
                           std::is_same<T, tsl::float8_e4m3fn>::value),
                          bool>::type
  IsEqualAt(absl::Span<const int64_t> multi_index, T value) const {
    if (auto as_s64 = GetIntegralAsS64(multi_index)) {
      return *as_s64 == value;
    }
    complex128 as_complex128 = *GetAsComplex128(multi_index);
    return as_complex128.imag() == 0 && as_complex128.real() == value;
  }

  bool IsEqualAt(absl::Span<const int64_t> multi_index,
                 complex128 value) const {
    if (auto as_s64 = GetIntegralAsS64(multi_index)) {
      return *as_s64 == value.real() && value.imag() == 0;
    }
    auto as_complex128 = GetAsComplex128(multi_index);
    return *as_complex128 == value;
  }

  // As Get(), but determines the correct type and converts the value into
  // int64_t.  This literal must be an array.
  std::optional<int64_t> GetIntegralAsS64(
      absl::Span<const int64_t> multi_index) const;

  // As Get(), but determines the correct type, and converts the value into
  // double. This literal must be an array.
  std::optional<double> GetAsDouble(
      absl::Span<const int64_t> multi_index) const;

  // As Get(), but determines the correct type, and converts the value into
  // complex128. All floating point types can be converted into complex128.
  //
  // This literal must be an array.
  std::optional<complex128> GetAsComplex128(
      absl::Span<const int64_t> multi_index) const;

  // Convert each element whose *linear* index is listed in "linear_indices"
  // to a double and return the sum of all of these elements.
  std::optional<double> GetSumAsDouble(
      absl::Span<const int64_t> linear_indices) const;

  // Invokes the "per cell" callback for each element in the provided
  // literal with the element's indices and a string representation of
  // the element's value.
  //
  // This function is useful if you want a polymorphic representation
  // of the tensor's elements (turning it to a string for something
  // like representation in a protobuf).
  //
  // This literal must have a dense layout.
  void EachCellAsString(
      absl::FunctionRef<void(absl::Span<const int64_t> indices,
                             const std::string& value)>
          per_cell) const;
  template <typename NativeT>
  void EachCell(
      absl::FunctionRef<void(absl::Span<const int64_t> indices, NativeT value)>
          per_cell) const;

  // Checks whether all of this literal's values are equal to the given scalar
  // literal.
  //
  // If `this` is not an array (e.g. it's a tuple), returns false.  This is
  // simpler than trying to handle subshapes here, and it's almost always what
  // you want.
  //
  // Preconditions:
  //  - `scalar` is a scalar.
  //  - `scalar` has the same element-type as `this`.
  bool IsAll(const Literal& scalar) const;

  // Returns whether every element in this literal is equal to value.
  //
  // value is an int8_t because we expect this to be called with small
  // compile-time constants (0, -1, etc.) and so that whatever value you pass
  // can be represented exactly by floating-point types as small as 16 bits.
  //
  // If value doesn't fit in this literal's type, returns false.  Values of 1/0
  // are considered equal to true/false; other values are not considered equal
  // to true.
  //
  // Returns false if this literal is not array-shaped.
  bool IsAll(int8_t value) const;

  // Like IsAll(int8_t), except we check whether the literal is equal to a
  // particular floating-point or complex number.
  //
  // Returns false if this literal is not a floating-point / complex value, or
  // if it's not an array.
  //
  // This casts value to the type of literal, then compares using ==, with the
  // caveat that NaNs are considered equal. Unlike IsAll, this does not
  // necessarily return false if the value does not fit in this literal's type.
  bool IsAllFloat(float value) const;
  bool IsAllComplex(complex64 value) const;

  // Deetermines if this literal consists entirely of the first element of the
  // literal.
  //
  // Returns false if this literal is not an array.
  bool IsAllFirst() const;

  // Literal consists entirely of an iota.
  bool IsR1Iota() const;

  // Returns the stride if the literal is a strided iota.
  std::optional<int64_t> IsR1StridedIota() const;

  // Returns whether this literal is zero at the specified index. This literal
  // must be an array with a dense layout.
  bool IsZero(absl::Span<const int64_t> indices) const;

  // Returns the count of the elements in the array at the given shape index in
  // this literal.
  int64_t element_count(const ShapeIndex& index = {}) const {
    if (index.empty()) {
      // Common case, avoid GetSubshape().
      return ShapeUtil::ElementsIn(shape());
    }
    return ShapeUtil::ElementsIn(ShapeUtil::GetSubshape(shape(), index));
  }

  // Compute a hash for this literal.
  template <typename H>
  friend H AbslHashValue(H state, const LiteralBase& value) {
    return LiteralBase::Hash(std::move(state), value);
  }

  template <typename H, bool kIsLayoutSensitive = true,
            int64_t kByteLimit = std::numeric_limits<int64_t>::max()>
  static H Hash(H state, const LiteralBase& literal) {
    state =
        Shape::Hash<H, kIsLayoutSensitive>(std::move(state), literal.shape());

    ShapeUtil::ForEachSubshape(
        literal.shape(), [&](const Shape& subshape, const ShapeIndex& index) {
          if (!subshape.IsArray()) {
            return;
          }

          CHECK(LayoutUtil::IsDenseArray(subshape));
          auto data = absl::MakeConstSpan(
              static_cast<const char*>(literal.untyped_data(index)),
              std::min(kByteLimit, literal.size_bytes(index)));
          state = H::combine(std::move(state), data);
        });

    return std::move(state);
  }

  // Converts this literal to the given shape. Returns an error is the
  // conversion is not possible.
  StatusOr<Literal> ConvertToShape(const Shape& dest_shape) const;

  // Converts this literal to another primitive type using a bitcast
  // conversion. Returns an error if the conversion is not possible. This
  // literal must be array-shaped.
  StatusOr<Literal> BitcastConvert(const Shape& dest_shape) const;

  // Converts this literal to another primitive type. Returns an error if the
  // conversion is not possible. This literal must be array-shaped.
  StatusOr<Literal> Convert(PrimitiveType primitive_dest_type) const;

  // Clones the underlying buffers into a new Literal.
  Literal Clone() const;
  std::unique_ptr<Literal> CloneToUnique() const;

  // TODO(b/67651157): The methods below which perform computation on Literals
  // (Reshape, Slice, etc) should be moved elsewhere, and perhaps combined with
  // evaluator code which operates on Literals.
  //
  // Creates a new value that has the equivalent value as this
  // literal, but conforms to new_layout; e.g. a literal matrix that was in {0,
  // 1} minor-to-major dimension layout can be re-layed-out as {1, 0}
  // minor-to-major dimension layout and the value in the cell at any given
  // logical index (i0, i1) will be the same.
  //
  // For tuple shaped literals, shape_index should be used to select the inner
  // array that the new layout applies to.
  //
  // Note: this is useful when the client wants to ensure that a value placed in
  // the XLA allocation tracker has a particular layout; for efficiency
  // purposes or avoiding unimplemented operation/layout combinations.
  Literal Relayout(const Layout& new_layout,
                   const ShapeIndex& shape_index = {}) const;

  // An overload of Relayout which changes the layout of the entire shape rather
  // than being limited to a single array within the shape.
  Literal Relayout(const Shape& shape_with_layout) const;

  // Generate a new literal whose static sizes are equal to the previous
  // literal's dynamic sizes.
  Literal ToStatic() const;

  // Expand a static literal into a new one with a bounded dyanmic literal. The
  // static dimensions of the original literal becomes dynamic dimensions of the
  // new literal, where the argument `bounded_shape` becomes the bounded shape
  // of the new literal.
  //
  // Precondition: bounded_shape.is_dynamic()
  Literal ToBoundedDynamic(const Shape& bounded_shape) const;

  // Creates a new literal by reshaping this literal to have the given
  // dimensions. The total number of elements must not change; The
  // implementation currently only supports monotonic dim0-major layouts.
  // This literal must be an array.
  StatusOr<Literal> Reshape(absl::Span<const int64_t> dimensions) const;

  // Creates a new literal by broadcasting this literal with `dimensions` to
  // yield a literal of shape `result_shape`.
  StatusOr<Literal> Broadcast(const Shape& result_shape,
                              absl::Span<const int64_t> dimensions) const;

  // Creates a new literal by reordering the dimensions of this literal.
  // The given `permutation` must be a permutation of the dimension numbers
  // in the original literal, and it specifies the order of the new dimensions
  // in the result literal (i.e., new_order[i] = old_order[permutation[i]]).
  // For example, a transpose call on a literal of shape [3 x 8 x 4] and
  // `permutation` = {2, 0, 1} returns a new literal of shape [4 x 3 x 8].
  // This literal must be an array.
  Literal Transpose(absl::Span<const int64_t> permutation) const;

  // Creates a sub-array from this literal by extracting the indices
  // [start_index, limit_index) of each dimension. The result literal has the
  // same rank and layout as for the given literal. The number of indices in
  // start_indices and limit_indices must be the rank of the literal, and the
  // indices follow the order of the dimensions.
  // This literal must be an array.
  Literal Slice(absl::Span<const int64_t> start_indices,
                absl::Span<const int64_t> limit_indices) const;

  // Creates a literal with a prepended dimension with bound "times"; e.g. a
  // f32[3x2] with times=4 will produce a f32[4x3x2] with the 3x2 from this
  // literal replicated four times.
  // This literal must be an array.
  template <typename NativeT>
  Literal Replicate(int64_t times) const;

  // Returns true if the leaf arrays of the literal within the given shape index
  // are all determined.
  // See comments on ArrayValueState for detailed explanation.
  bool IsDetermined(const ShapeIndex& shape_index = {}) const;

  // Returns true if the leaf arrays of the literal within the given shape index
  // are all known.
  // See comments on ArrayValueState for detailed explanation.
  bool IsKnown(const ShapeIndex& shape_index = {}) const;

  // Creates a new Literal object with the shape specified as parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  //
  // Note: It's an antipattern to use this method then immediately call
  // MutableLiteralBase::Populate on the result (since that results in zero
  // initialization, then reinitialization. Consider if a call to
  // std::make_unique<Literal>(shape), followed by the call to
  // MutableLiteralBase::Populate can be used instead.
  static Literal CreateFromShape(const Shape& shape);

  // WARNING: These two functions are only supposed to be used by HloEvaluator.
  // The rest of XLA assumes all literals are known.
  // Similar to CreateFromShape() but marks all leaf arrays as unknown.
  static Literal CreateFromShapeWithUnknownLeafArrays(const Shape& shape);
  // Similar to CreateFromShape() but marks all leaf arrays as undetermined.
  static Literal CreateFromShapeWithUndeterminedLeafArrays(const Shape& shape);

 protected:
  // Array literals could be in one of the following three states:
  //   1) Known: we have evaluated and known the value of the array literal.
  //   2) Unknown: we have tried to evaluate the array literal, but its value
  //               cannot be evaluated statically.
  //   3) Undetermined: we haven't tried to evaluate the array literal.
  //  Unknown and Undetermined states are only meant to be used within
  //  HloEvaluator. The rest of XLA assumes array literals are all known.
  //  Literals that are unknown or undetermined can be copied from, using
  //  CopyFrom and Clone, or moved from using move constructor. Accessing values
  //  of such literals causes undefined behavior.
  enum class ArrayValueState { kKnown = 0, kUnknown = 1, kUndetermined = 2 };

  // A data structure representing a subshape at a particular ShapeIndex within
  // the literal. For array-shaped ShapeIndexes, this data structure holds the
  // pointer to the memory allocated for the array data.
  class Piece {
   public:
    ArrayValueState get_array_value_state() const;
    void set_array_value_state(ArrayValueState state);
    // Returns the buffer holding the array data for this piece as an array
    // slice. This piece must be array-shaped.
    template <typename NativeT>
    absl::Span<const NativeT> data() const;
    template <typename NativeT>
    absl::Span<NativeT> data();

    // Returns the buffer holding the array data for this piece as a void*. This
    // piece must be array-shaped.
    void* untyped_data();
    const void* untyped_data() const;

    // Gets or sets an element in the array at the given index. The multi_index
    // is CHECKed against the dimension sizes of the array.  This piece must be
    // array-shaped.
    template <typename NativeT>
    NativeT Get(absl::Span<const int64_t> index) const;
    template <typename NativeT>
    void Set(absl::Span<const int64_t> index, NativeT value);

    int32_t GetDynamicSize(int64_t dim_index) const;
    void SetDynamicSize(int64_t dim_index, int32_t size);
    void AllocateBuffers();
    void DeallocateBuffers();
    // Gets/sets the buffer holding the array data.
    const char* buffer() const;
    char* buffer() {
      return const_cast<char*>(const_cast<const Piece*>(this)->buffer());
    }
    void set_buffer(char* buffer) {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      auto* dense_rep = std::holds_alternative<Uninitialized>(rep_)
                            ? &rep_.emplace<DenseRep>()
                            : GetDenseRep();
      DCHECK(dense_rep);
      dense_rep->data = buffer;
    }
    void MoveDataFrom(Piece& from) {
      DCHECK(!std::holds_alternative<DenseRep>(rep_));
      DCHECK(!std::holds_alternative<TupleRep>(rep_));
      if (auto* dense_rep = from.GetDenseRep()) {
        rep_.emplace<DenseRep>().data = dense_rep->data;
      } else if (auto* inlined_rep = from.GetDenseInlinedRep()) {
        std::memcpy(rep_.emplace<DenseInlinedRep>().data, inlined_rep->data,
                    from.total_bytes_dense());
      }
      from.rep_.emplace<Uninitialized>();
    }

    // Gets/sets the buffer holding dynamic sizes.
    const int32_t* dynamic_size_buffer() const {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      return reinterpret_cast<const int32_t*>(buffer() + size_bytes_dense());
    }
    int32_t* dynamic_size_buffer() {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      return const_cast<int32_t*>(
          const_cast<const Piece*>(this)->dynamic_size_buffer());
    }

    int64_t dynamic_size_buffer_bytes() const {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      return subshape().dimensions_size() * sizeof(int32_t);
    }

    // Gets or sets the subshape of this piece. This reference points to a
    // subshape within the shape in the containing Literal (Literal::shape_).
    const Shape& subshape() const { return *subshape_; }
    void set_subshape(const Shape* subshape) {
      subshape_ = subshape;
      if (std::holds_alternative<Uninitialized>(rep_)) {
        if (subshape_->IsTuple()) {
          rep_.emplace<TupleRep>();
        }
      }
    }

    // Returns the size in bytes of the buffer holding the dense array data.
    int64_t size_bytes_dense() const {
      DCHECK(LayoutUtil::IsDenseArray(*subshape_));
      return ShapeUtil::ByteSizeOf(subshape());
    }

    // Total size in bytes, including the dynamic size addition.
    //
    // The shape can become dynamic after this literal is allocated, so we
    // over-allocate the margin for the dynamic shape description in case we
    // need it.
    int64_t total_bytes_dense() const {
      return size_bytes_dense() + dynamic_size_buffer_bytes();
    }

    // Returns the number of elements in this piece's array.
    int64_t element_count() const { return ShapeUtil::ElementsIn(subshape()); }

    // Returns the child piece at 'index' of this piece.
    Piece& child(int64_t index) {
      return const_cast<Piece&>(const_cast<const Piece*>(this)->child(index));
    }
    const Piece& child(int64_t index) const {
      auto* tuple_rep = GetTupleRep();
      DCHECK(tuple_rep);
      return tuple_rep->children[index];
    }

    // Adds a child piece to this piece's children.
    void emplace_back(Piece child_piece) {
      auto* tuple_rep = GetTupleRep();
      DCHECK(tuple_rep);
      tuple_rep->children.emplace_back(std::move(child_piece));
    }

    // Returns the size of children pieces of this piece.
    int64_t children_size() {
      if (auto* tuple_rep = GetTupleRep()) {
        return tuple_rep->children.size();
      }
      return 0;
    }

    // Visitor functions that recursively traverses the piece and calls the
    // given function at each child piece. The function has the type:
    //    void (const ShapeIndex& index, const Piece& piece)
    template <typename Fn>
    void ForEachSubpiece(const Fn& func) const {
      ShapeIndex index;
      return ForEachHelper(
                 [&func](const ShapeIndex& index, const Piece& piece) {
                   func(index, piece);
                   return OkStatus();
                 },
                 *this, &index)
          .IgnoreError();
    }
    // Same as above, but the function has the type:
    //    Status (const ShapeIndex& index, const Piece& piece)
    // The first non-OK return value is returned by the function.
    template <typename Fn>
    Status ForEachSubpieceWithStatus(const Fn& func) const {
      ShapeIndex index;
      return ForEachHelper(func, *this, &index);
    }
    // Same as above, but the function has the type:
    //    Bool (const ShapeIndex& index, const Piece& piece)
    // The first non-true return value is returned by the function.
    template <typename Fn>
    bool ForEachSubpieceWithBool(const Fn& func) const {
      ShapeIndex index;
      return ForEachHelperBool(func, *this, &index);
    }
    // Same as above, but the function has the type:
    //    Void (const ShapeIndex& index, Piece& piece)
    template <typename Fn>
    void ForEachMutableSubpiece(const Fn& func) {
      ShapeIndex index;
      return ForEachMutableHelper(
                 [&func](const ShapeIndex& index, Piece* piece) {
                   func(index, piece);
                   return OkStatus();
                 },
                 const_cast<xla::LiteralBase::Piece*>(this), &index)
          .IgnoreError();
    }
    // Same as above, but the function has the type:
    //    Status (const ShapeIndex& index, Piece& piece)
    // The first non-OK return value is returned by the function.
    template <typename Fn>
    Status ForEachMutableSubpieceWithStatus(const Fn& func) {
      ShapeIndex index;
      return ForEachMutableHelper(
          func, const_cast<xla::LiteralBase::Piece*>(this), &index);
    }

    // Checks whether all elements of this Piece are equal to the given literal.
    //
    // Returns false if this Piece is not an array.
    //
    // Preconditions:
    //  - `scalar` is a scalar.
    //  - `scalar`'s type matches that of `this`.
    bool IsAll(const Literal& scalar) const;

    // Returns true if this piece and 'other' contain the same data. This piece
    // and 'other' must be array-shaped and compatible. If a literal has dynamic
    // shape, comparison is done only for the valid elements.
    bool EqualElements(const Piece& other) const;

    // Returns true if this piece and other pieces have the same dynamic
    // dimension sizes.
    bool EqualDynamicSize(const Piece& other) const;

    // Writes the shape and data (if array-shaped) into the given proto.
    void WriteToProto(LiteralProto* proto) const;

    // Copy the data from 'src' into this piece's buffer. Shapes of this piece
    // and src must be compatible. If only_dynamic_bound is true, only elements
    // within dynamic bounds will be copied.
    Status CopyFrom(const Piece& src, bool only_dynamic_bound);

    // Copies the data from the given proto into this piece. The shape of this
    // piece must be equal (not just compatible) to the shape of the proto.
    Status CopyFromProto(const LiteralProto& proto);

    // See comments on ArrayValueState for detailed explanation.
    bool IsDetermined() const;

    bool IsKnown() const;

   private:
    // Uninitialized state representation.
    struct Uninitialized {};
    // Out of line dense array storage.
    union DenseRep {
      char* data;
    };
    struct TupleRep {
      // Children pieces for tuple shaped pieces.
      std::vector<Piece> children = {};
    };

    // Use just so many bytes that we don't increase the sizeof(Piece).
    static inline constexpr size_t kMaxInlinedBytes =
        std::max(sizeof(DenseRep), sizeof(TupleRep));

    // Inlined dense array storage.
    struct DenseInlinedRep {
      char data[kMaxInlinedBytes];
    };

    // Helper visiter to access the buffer in the representation variant.
    struct BufferVisitor {
      char* operator()(Uninitialized&) { return nullptr; }
      const char* operator()(const Uninitialized&) const { return nullptr; }
      char* operator()(TupleRep&) { return nullptr; }
      const char* operator()(const TupleRep&) const { return nullptr; }
      char* operator()(DenseInlinedRep& rep) { return rep.data; }
      const char* operator()(const DenseInlinedRep& rep) const {
        return rep.data;
      }
      char* operator()(DenseRep& rep) { return rep.data; }
      const char* operator()(const DenseRep& rep) const { return rep.data; }
    };

    const DenseInlinedRep* GetDenseInlinedRep() const {
      return std::get_if<DenseInlinedRep>(&rep_);
    }
    DenseInlinedRep* GetDenseInlinedRep() {
      return std::get_if<DenseInlinedRep>(&rep_);
    }

    const DenseRep* GetDenseRep() const { return std::get_if<DenseRep>(&rep_); }
    DenseRep* GetDenseRep() { return std::get_if<DenseRep>(&rep_); }

    const TupleRep* GetTupleRep() const { return std::get_if<TupleRep>(&rep_); }
    TupleRep* GetTupleRep() { return std::get_if<TupleRep>(&rep_); }
    // Helpers for traversing the piece via ForEachSubpiece rooted at 'index'.
    // The first non-OK (or non-true) value is returned by the function.
    // The callable 'func' has the same signature as described above in
    // ForEachSubpiece*.
    template <typename Fn>
    Status ForEachHelper(const Fn& func, const Piece& piece,
                         ShapeIndex* index) const {
      TF_RETURN_IF_ERROR(func(*index, piece));
      if (auto* tuple_rep = piece.GetTupleRep()) {
        for (int64_t i = 0; i < tuple_rep->children.size(); ++i) {
          index->push_back(i);
          TF_RETURN_IF_ERROR(
              ForEachHelper(func, tuple_rep->children[i], index));
          index->pop_back();
        }
      }
      return OkStatus();
    }
    template <typename Fn>
    bool ForEachHelperBool(const Fn& func, const Piece& piece,
                           ShapeIndex* index) const {
      if (!func(*index, piece)) {
        return false;
      }
      if (auto* tuple_rep = piece.GetTupleRep()) {
        for (int64_t i = 0; i < tuple_rep->children.size(); ++i) {
          index->push_back(i);
          if (!ForEachHelperBool(func, tuple_rep->children[i], index)) {
            return false;
          }
          index->pop_back();
        }
      }
      return true;
    }
    template <typename Fn>
    Status ForEachMutableHelper(const Fn& func, Piece* piece,
                                ShapeIndex* index) {
      TF_RETURN_IF_ERROR(func(*index, piece));
      if (auto* tuple_rep = piece->GetTupleRep()) {
        for (int64_t i = 0; i < tuple_rep->children.size(); ++i) {
          index->push_back(i);
          TF_RETURN_IF_ERROR(
              ForEachMutableHelper(func, &tuple_rep->children[i], index));
          index->pop_back();
        }
      }
      return OkStatus();
    }

    // Recursive helper for EqualElements.
    template <typename NativeT>
    bool EqualElementsInternal(const Piece& other,
                               std::vector<int64_t>* multi_index) const;

    // Internal helper to copy elements from another given piece
    template <typename NativeT>
    void CopyElementsWithDynamicBound(const LiteralBase::Piece& src);

    // Storage representation of this piece.
    std::variant<Uninitialized, DenseInlinedRep, DenseRep, TupleRep> rep_;

    // The shape of piece. This points into the shape of the containing Literal
    // (Literal::shape_).
    const Shape* subshape_ = nullptr;

    ArrayValueState array_value_state_ = ArrayValueState::kKnown;
  };  // class Piece

  const Piece& piece(const ShapeIndex& shape_index) const;

  // Returns the piece at the root of the shape.
  virtual const Piece& root_piece() const = 0;

  // LiteralSlice and Literal must access Pieces of other Literals.
  friend class MutableLiteralBase;
  friend class LiteralSlice;
  friend class BorrowingLiteral;

 private:
  template <typename NativeT>
  Literal SliceInternal(const Shape& result_shape,
                        absl::Span<const int64_t> start_indices) const;

  // Like IsAllFloat, but if round_value is false and the value is not
  // representable with the literal's type (e.g., due to rounding error or
  // overflow/underflow when casting the value to the literal's type), returns
  // false.
  bool IsAllFloatImpl(float value, bool round_value) const;
};

// Abstract base class representing a mutable literal in XLA.
class MutableLiteralBase : public LiteralBase {
 public:
  ~MutableLiteralBase() override = 0;

  // Returns a Span view of the array for this literal for the
  // given NativeT (e.g., float). CHECKs if the subshape of the literal at the
  // given ShapeIndex is not array. See primitive_util.h for the mapping from
  // XLA type to native type.
  template <typename NativeT>
  absl::Span<NativeT> data(const ShapeIndex& shape_index = {});
  // Unhide const method from parent class.
  using LiteralBase::data;

  // TODO(b/67651157): Remove this accessor. Literal users should not be able to
  // mutate the shape as this can produce malformed Literals.
  Shape* mutable_shape_do_not_use();

  // Set the dynamic size on dim_index in the literal at the given shape_index.
  void SetDynamicSize(int64_t dim_index, const ShapeIndex& shape_index,
                      int32_t size);
  void SetDynamicSize(int64_t dim_index, int32_t size);

  // Returns a pointer to the underlying buffer holding the array at the given
  // shape index. CHECKs if the subshape of the literal at the given ShapeIndex
  // is not array.
  void* untyped_data(const ShapeIndex& shape_index = {});
  // Unhide const method from parent class.
  using LiteralBase::untyped_data;

  template <typename NativeT>
  void MutableEachCell(absl::FunctionRef<NativeT(
                           absl::Span<const int64_t> indices, NativeT value)>
                           per_cell);

  // Copy values from 'src_literal' rooted at 'src_shape_index' into this
  // literal rooted at 'dest_shape_index'. The subshape of this literal rooted
  // at 'dest_shape_index' must be compatible with the subshape of 'src_literal'
  // rooted at 'src_shape_index', but need not be arrays. If only_dynamic_bound
  // is true, only elements within dynamic bounds will be copied.
  Status CopyFrom(const LiteralSlice& src_literal,
                  const ShapeIndex& dest_shape_index = {},
                  const ShapeIndex& src_shape_index = {},
                  bool only_dynamic_bound = false);

  // Copies the values from src_literal, starting at src_base shape indexes,
  // to this literal, starting at dest_base, where the copy size in each
  // dimension is specified by copy_size.
  // The src_literal and this literal must have the same primitive type,
  // src_base+copy_size must fit the source literal dimensions, as well as
  // dest_base+copy_size must fit the destination literal dimensions.
  // Note: if either src_literal or this literal contains dimensions with zero
  // element, then copy_size must be 0 in these dimensions while the
  // corresponding base indices being 0.
  // This literal and 'src_literal' must be arrays.
  Status CopySliceFrom(const LiteralSlice& src_literal,
                       absl::Span<const int64_t> src_base,
                       absl::Span<const int64_t> dest_base,
                       absl::Span<const int64_t> copy_size);

  // Copies one element from src_literal[src_index] to (*this)[dest_index].
  Status CopyElementFrom(const LiteralSlice& src_literal,
                         absl::Span<const int64_t> src_index,
                         absl::Span<const int64_t> dest_index);

  // Sets an element in the literal at the given index. The multi_index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  void Set(absl::Span<const int64_t> multi_index, const ShapeIndex& shape_index,
           NativeT value);
  // Overloads of Set for array literals. CHECKs if the literal is not
  // array-shaped and dense.
  template <typename NativeT>
  void Set(absl::Span<const int64_t> multi_index, NativeT value);

  // As Set(), but truncates `value` to the literal element type before storing.
  // This literal must be an array.
  Status SetIntegralAsS64(absl::Span<const int64_t> multi_index, int64_t value);

  // As Set(), but truncates `value` to the literal element type before storing.
  // This literal must be an array.
  Status SetFromDouble(absl::Span<const int64_t> multi_index, double value);

  // Populate this literal with the given values. Examples:
  //
  //   // Populate with floats.
  //   Array2D<float> float_values = ...
  //   literal.PopulateR2FromArray2D(values);
  //
  //   // Populate with int32s.
  //   literal.PopulateR2<int32_t>({{1, 2}, {3, 4}});
  //
  // The shape and element type of this literal must match given values. For
  // example, in the call above to literal.PopulateR2(), 'literal' must be a 2x2
  // array of S32.
  template <typename NativeT>
  void PopulateR1(absl::Span<const NativeT> values);
  void PopulateR1(const tsl::core::Bitmap& values);
  template <typename NativeT>
  void PopulateR2(std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  void PopulateFromArray(const Array<NativeT>& values);
  template <typename NativeT>
  void PopulateR2FromArray2D(const Array2D<NativeT>& values);
  template <typename NativeT>
  void PopulateR3FromArray3D(const Array3D<NativeT>& values);
  template <typename NativeT>
  void PopulateR4FromArray4D(const Array4D<NativeT>& values);

  // Populates literal values by calling the generator function for every cell
  // in this literal object.
  //
  // generator must be a callable of the type
  // NativeT(absl::Span<const int64_t> indexes) or compatible.
  //
  // This literal must have a dense layout.
  template <typename NativeT>
  Status Populate(
      absl::FunctionRef<NativeT(absl::Span<const int64_t>)> generator);

  // A parallel version of Populate(). This can be used if the generator is
  // thread-safe and the values for the shape's different elements are
  // independent.
  template <typename NativeT>
  Status PopulateParallel(
      absl::FunctionRef<NativeT(absl::Span<const int64_t>, int)> generator);

  // Fills this literal with the given value.
  template <typename NativeT>
  void PopulateWithValue(NativeT value);

  // This operation is the inverse of DecomposeTuple. The given elements are
  // moved into the tuple elements of a new tuple-shaped Literal which is
  // returned. Upon return, each of the Literals in 'elements' is set to a nil
  // shape (empty tuple).
  static Literal MoveIntoTuple(absl::Span<Literal> elements);

  // Serialize from a proto.
  static StatusOr<Literal> CreateFromProto(const LiteralProto& proto,
                                           bool prohibit_empty_literal = true);

 protected:
  // Returns the piece at the given ShapeIndex.
  Piece& piece(const ShapeIndex& shape_index) {
    return const_cast<Piece&>(LiteralBase::piece(shape_index));
  }

  Piece& mutable_root_piece() { return const_cast<Piece&>(root_piece()); }

  // Internal template helper for the Literal::CopySliceFrom(), matching its
  // arguments one by one.
  template <typename NativeT>
  Status CopySliceFromInternal(const LiteralBase& src_literal,
                               absl::Span<const int64_t> src_base,
                               absl::Span<const int64_t> dest_base,
                               absl::Span<const int64_t> copy_size);

  // Utility structure which is used to create the optimal configuration for
  // a ShapeUtil::ForEachIndex() scan across two literals.
  struct StrideConfig {
    StrideConfig(const Shape& source_shape, const Shape& dest_shape,
                 absl::Span<const int64_t> dimensions);

    // The dimensions of the stride operation. Essentially every dimension
    // will be iterated from base[i] to base[i]+dimensions[i], in step[i]
    // steps.
    absl::Span<const int64_t> dimensions;
    DimensionVector base;
    DimensionVector step;
    int64_t minor_dimension = 0;
    // The size of the strides for source and destination. One of the two
    // (the one looping through its most minor dimension) will be 1, while
    // the other will be the stride size at the dimension matching the other
    // shape most minor dimension being scanned.
    int64_t dest_stride = 1;
    int64_t source_stride = 1;
    // The size of the inner loop on the most minor dimension.
    int64_t minor_loop_size = 1;
  };

  // A unique_ptr like class which may or may not have ownership of its pointer.
  // The literal may or may not own the storage of the shape. Creating/copying a
  // shape can incur significant overhead which in many case we'd like to avoid,
  // esp. for small literals.
  class MaybeOwningShapePtr {
   public:
    MaybeOwningShapePtr() = default;
    explicit MaybeOwningShapePtr(std::unique_ptr<Shape> unique)
        : ptr_and_owning_bit_(TakeUnique(std::move(unique))) {}

    explicit MaybeOwningShapePtr(const Shape* borrowed)
        : ptr_and_owning_bit_(Borrow(borrowed)) {}

    ~MaybeOwningShapePtr() { MaybeDeleteOwned(); }

    const Shape* get() const {
      return reinterpret_cast<const Shape*>(ptr_and_owning_bit_ & kPointerMask);
    }
    Shape* get_mutable(bool ensure_owned = false) {
      const Shape* const_ptr = get();
      // TODO(b/67651157): Remove this copy on write logic and combine get() and
      // get_mutable() once we remove mutable_shape_do_not_use().
      if (const_ptr && !OwnsPtr()) {
        ptr_and_owning_bit_ = TakeUnique(std::make_unique<Shape>(*const_ptr));
        const_ptr = get();
      }
      DCHECK(OwnsPtr());
      return const_cast<Shape*>(const_ptr);
    }
    const Shape* operator->() const { return get(); }
    const Shape& operator*() const { return *get(); }

    MaybeOwningShapePtr& operator=(std::unique_ptr<Shape> unique) {
      MaybeDeleteOwned();
      ptr_and_owning_bit_ = TakeUnique(std::move(std::move(unique)));
      return *this;
    }

    MaybeOwningShapePtr& operator=(const Shape* borrowed) {
      MaybeDeleteOwned();
      ptr_and_owning_bit_ = Borrow(borrowed);
      return *this;
    }

    MaybeOwningShapePtr& operator=(MaybeOwningShapePtr&& other) {
      using std::swap;
      swap(ptr_and_owning_bit_, other.ptr_and_owning_bit_);
      return *this;
    }

    MaybeOwningShapePtr(const MaybeOwningShapePtr&) = delete;
    MaybeOwningShapePtr(MaybeOwningShapePtr&& other)
        : ptr_and_owning_bit_(other.ptr_and_owning_bit_) {
      other.ptr_and_owning_bit_ = 0;
    }

    MaybeOwningShapePtr Clone() const {
      const Shape* ptr = get();
      if (ptr && OwnsPtr()) {
        return MaybeOwningShapePtr(std::make_unique<Shape>(*ptr));
      }
      return MaybeOwningShapePtr(ptr);
    }

   private:
    enum : uint64_t {
      kOwningBitMask = 1UL,
      kPointerMask = ~kOwningBitMask,
    };
    static intptr_t TakeUnique(std::unique_ptr<Shape> unique) {
      Shape* released = unique.release();
      DCHECK_EQ(reinterpret_cast<intptr_t>(released) & kOwningBitMask, 0);
      return reinterpret_cast<intptr_t>(released) | kOwningBitMask;
    }

    static intptr_t Borrow(const Shape* borrowed) {
      DCHECK_EQ(reinterpret_cast<intptr_t>(borrowed) & kOwningBitMask, 0);
      return reinterpret_cast<intptr_t>(borrowed);
    }

    bool OwnsPtr() const { return kOwningBitMask & ptr_and_owning_bit_; }

    void MaybeDeleteOwned() {
      if (OwnsPtr()) {
        delete get();
      }
    }

    intptr_t ptr_and_owning_bit_ = 0;
  };

  // The parent class borrows this shape.
  MaybeOwningShapePtr shape_;

  // Implementation details shared between Populate() and PopulateParallel()
  //  template <typename NativeT, typename FnType>
  //  Status PopulateInternal(const FnType& generator, bool parallel);
  template <typename NativeT>
  Status PopulateInternal(
      absl::FunctionRef<NativeT(absl::Span<const int64_t>, int)> generator,
      bool parallel);

  friend class LiteralBase;
  friend class MutableBorrowingLiteral;
};
std::ostream& operator<<(std::ostream& out, const Literal& literal);

// The underlying buffer and shape is always owned by this class.
class Literal : public MutableLiteralBase {
 public:
  Literal();

  // Create a literal of the given shape. The literal is allocated sufficient
  // memory to hold the shape. Memory is uninitialized.
  explicit Literal(const Shape& shape);
  ~Literal() override;

  // Literals are moveable, but not copyable. To copy a literal use
  // Literal::Clone or Literal::CloneToUnique. This prevents inadvertent copies
  // of literals which can be expensive.
  Literal(const Literal& other) = delete;
  Literal& operator=(const Literal& other) = delete;
  Literal(Literal&& other);
  // 'allocate_arrays' indicates whether to allocate memory for the arrays in
  // the shape. If false, buffer pointers inside of the Literal::Pieces are set
  // to nullptr.
  Literal(const Shape& shape, bool allocate_arrays,
          ArrayValueState leaf_array_value_state = ArrayValueState::kKnown);
  Literal& operator=(Literal&& other);

  // Similar to CopyFrom, but with move semantics. The subshape of this literal
  // rooted at 'dest_shape_index' must be *equal* to the shape 'src_literal'
  // (layouts and shapes must match), but need not be arrays. The memory
  // allocated in this literal for the subshape at dest_shape_index is
  // deallocated, and the respective buffers are replaced with those in
  // src_literal. Upon return, src_literal is set to a nil shape (empty tuple).
  virtual Status MoveFrom(Literal&& src_literal,
                          const ShapeIndex& dest_shape_index);
  Status MoveFrom(Literal&& src_literal) {
    return MoveFrom(std::move(src_literal), /*dest_shape_index=*/{});
  }

  // Returns a vector containing the tuple elements of this Literal as separate
  // Literals. This Literal must be tuple-shaped and can be a nested tuple. The
  // elements are moved into the new Literals; no data is copied. Upon return
  // this Literal is set to a nil shape (empty tuple)
  //
  // TODO(jlebar): Because this function invalidates `this`, it should be
  // ref-qualified with &&.
  std::vector<Literal> DecomposeTuple();

  // Returns a subliteral specified by given shape_index. No data is copied, the
  // current literal becomes invalid after this function call.
  //
  // TODO(jlebar): Because this function invalidates `this`, it should be
  // ref-qualified with &&.
  Literal SubLiteral(ShapeIndexView shape_index);

 private:
  friend class LiteralBase;
  friend class MutableLiteralBase;
  const Piece& root_piece() const override { return root_piece_; };
  // Deallocate the buffers held by this literal.
  void DeallocateBuffers();

  // Recursively sets the subshapes and buffers of all subpieces rooted at
  // 'piece'. If 'allocate_array' is true, memory is allocated for the arrays in
  // the shape.
  void SetPiece(
      const Shape& shape, Piece* piece, bool allocate_arrays,
      ArrayValueState leaf_array_value_state = ArrayValueState::kKnown);
  Piece root_piece_;
};

// The underlying buffer is not owned by this class and is always owned by
// others. The shape is not owned by this class and not mutable.
class MutableBorrowingLiteral : public MutableLiteralBase {
 public:
  ~MutableBorrowingLiteral() override;

  MutableBorrowingLiteral() : MutableLiteralBase() {}

  MutableBorrowingLiteral(const MutableBorrowingLiteral& literal);
  MutableBorrowingLiteral& operator=(const MutableBorrowingLiteral& literal);

  // Implicit conversion constructors.
  // NOLINTNEXTLINE(google-explicit-constructor)
  MutableBorrowingLiteral(MutableLiteralBase* literal);
  MutableBorrowingLiteral(MutableBorrowingLiteral literal,
                          const ShapeIndex& view_root);
  MutableBorrowingLiteral(const char* src_buf_ptr, const Shape& shape);

  // Create a literal from a list of buffers and a shape.
  // Returns a tuple literal if `shape` is a tuple type.
  MutableBorrowingLiteral(absl::Span<char*> src_buf_ptrs, const Shape& shape);

 private:
  const Piece& root_piece() const override { return *root_piece_; };
  // Recursively copies the subtree from the `src_piece` at the given child
  // index to the `dest_piece`. For buffers only the pointers are copied, but
  // not the content.
  void CopyPieceSubtree(const Shape& shape, const Piece* src_piece,
                        Piece* dest_piece);
  Piece* root_piece_ = nullptr;
};

// A read-only view of a Literal. A LiteralSlice contains pointers to shape and
// literal buffers always owned by others.
class LiteralSlice : public LiteralBase {
 public:
  LiteralSlice() : LiteralBase() {}

  // Implicit conversion constructors.
  // NOLINTNEXTLINE(google-explicit-constructor)
  LiteralSlice(const LiteralBase& literal);
  LiteralSlice(const LiteralBase& literal, const ShapeIndex& view_root);

 private:
  const Piece& root_piece() const override { return *root_piece_; };

  const Piece* root_piece_;  // Not owned.
};

// A read-only Literal where the underlying buffers are never owned by this
// class.
class BorrowingLiteral : public LiteralBase {
 public:
  BorrowingLiteral() : LiteralBase() {}

  // 'src_buf_ptr' is not owned by this class and must outlive the
  // lifetime of this class. It points to an appropriately sized buffer with
  // data interpretered as indicated by 'shape'.
  // This constructor is only used for array shapes.
  BorrowingLiteral(const char* src_buf_ptr, const Shape& shape);
  // Similar as above, except to be used for constructing non-nested tuples.
  BorrowingLiteral(absl::Span<const char* const> src_buf_ptrs,
                   const Shape& shape);
  // TODO(b/79707221): adding constructors for nested tuples as well.

 private:
  // Recursively builds the subtree for the given piece and sets the subshapes
  // of the given piece with the given shape.
  void BuildPieceSubtree(const Shape& shape, Piece* piece);

  // Accessor for the root piece of this literal.
  const Piece& root_piece() const override { return root_piece_; };
  Piece root_piece_;

  // Shape of this literal. Stored as unique_ptr such that the (default) move
  // construction of this class would be trivially correct: the pointer to Shape
  // root_piece_ stores will still point to the correct address.
  std::unique_ptr<Shape> shape_;
};

template <typename NativeT>
absl::Span<const NativeT> LiteralBase::Piece::data() const {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  DCHECK_EQ(subshape().element_type(),
            primitive_util::NativeToPrimitiveType<NativeT>())
      << "Attempting to access "
      << PrimitiveType_Name(primitive_util::NativeToPrimitiveType<NativeT>())
      << " type, but literal element type is "
      << PrimitiveType_Name(subshape().element_type());
  return absl::Span<const NativeT>(reinterpret_cast<const NativeT*>(buffer()),
                                   element_count());
}

template <typename NativeT>
absl::Span<NativeT> LiteralBase::Piece::data() {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  DCHECK_EQ(subshape().element_type(),
            primitive_util::NativeToPrimitiveType<NativeT>())
      << "Attempting to access "
      << PrimitiveType_Name(primitive_util::NativeToPrimitiveType<NativeT>())
      << " type, but literal element type is "
      << PrimitiveType_Name(subshape().element_type());
  return absl::Span<NativeT>(reinterpret_cast<NativeT*>(buffer()),
                             element_count());
}

template <typename NativeT>
NativeT LiteralBase::Piece::Get(absl::Span<const int64_t> multi_index) const {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  return data<NativeT>()[IndexUtil::MultidimensionalIndexToLinearIndex(
      subshape(), multi_index)];
}

template <typename NativeT>
void LiteralBase::Piece::Set(absl::Span<const int64_t> multi_index,
                             NativeT value) {
  DCHECK(LayoutUtil::IsDenseArray(subshape()))
      << __func__ << " is only supported for dense arrays: " << subshape();
  data<NativeT>()[IndexUtil::MultidimensionalIndexToLinearIndex(
      subshape(), multi_index)] = value;
}

template <typename NativeT>
absl::Span<const NativeT> LiteralBase::data(
    const ShapeIndex& shape_index) const {
  return piece(shape_index).data<NativeT>();
}

template <typename NativeT>
absl::Span<NativeT> MutableLiteralBase::data(const ShapeIndex& shape_index) {
  return piece(shape_index).data<NativeT>();
}

template <typename NativeT>
inline NativeT LiteralBase::Get(absl::Span<const int64_t> multi_index,
                                const ShapeIndex& shape_index) const {
  return piece(shape_index).Get<NativeT>(multi_index);
}

template <typename NativeT>
inline NativeT LiteralBase::Get(absl::Span<const int64_t> multi_index) const {
  return root_piece().Get<NativeT>(multi_index);
}

template <typename NativeT>
inline void MutableLiteralBase::Set(absl::Span<const int64_t> multi_index,
                                    const ShapeIndex& shape_index,
                                    NativeT value) {
  return piece(shape_index).Set<NativeT>(multi_index, value);
}

template <typename NativeT>
inline void MutableLiteralBase::Set(absl::Span<const int64_t> multi_index,
                                    NativeT value) {
  return mutable_root_piece().Set<NativeT>(multi_index, value);
}

template <typename NativeT>
NativeT LiteralBase::GetFirstElement() const {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  return data<NativeT>().at(0);
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE void LiteralBase::EachCell(
    absl::FunctionRef<void(absl::Span<const int64_t> indices, NativeT value)>
        per_cell) const {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  if (ShapeUtil::IsZeroElementArray(shape())) {
    return;
  }
  std::vector<int64_t> indices(shape().rank(), 0);

  Shape shape_dynamic = shape();
  for (int64_t i = 0; i < shape_dynamic.rank(); ++i) {
    shape_dynamic.set_dimensions(i, GetDynamicSize(i));
  }
  do {
    per_cell(indices, Get<NativeT>(indices));
  } while (IndexUtil::BumpIndices(shape_dynamic, absl::MakeSpan(indices)));
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE void MutableLiteralBase::MutableEachCell(
    absl::FunctionRef<NativeT(absl::Span<const int64_t> indices, NativeT value)>
        per_cell) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  if (ShapeUtil::IsZeroElementArray(shape())) {
    return;
  }
  std::vector<int64_t> indices(shape().rank(), 0);
  Shape shape_dynamic = shape();
  for (int64_t i = 0; i < shape_dynamic.rank(); ++i) {
    shape_dynamic.set_dimensions(i, GetDynamicSize(i));
  }
  do {
    Set<NativeT>(indices, per_cell(indices, Get<NativeT>(indices)));
  } while (IndexUtil::BumpIndices(shape_dynamic, absl::MakeSpan(indices)));
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE void MutableLiteralBase::PopulateR1(
    absl::Span<const NativeT> values) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  CHECK_EQ(shape().rank(), 1);
  if (shape().is_static()) {
    CHECK_EQ(ShapeUtil::ElementsIn(shape()), values.size());
  } else {
    CHECK_EQ(GetDynamicSize(0), values.size());
  }
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  auto data_span = data<NativeT>();
  std::copy(values.begin(), values.end(), data_span.begin());
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE void MutableLiteralBase::PopulateR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  CHECK_EQ(shape().rank(), 2);
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());

  const int64_t values_dim0_size = values.size();
  const int64_t values_dim1_size = values.begin()->size();
  const int64_t literal_dim0_size = shape().is_dynamic_dimension(0)
                                        ? GetDynamicSize(0)
                                        : shape().dimensions(0);
  const int64_t literal_dim1_size = shape().is_dynamic_dimension(1)
                                        ? GetDynamicSize(1)
                                        : shape().dimensions(1);

  CHECK_EQ(values_dim0_size, literal_dim0_size);
  CHECK_EQ(values_dim1_size, literal_dim1_size);

  int64_t dim0 = 0;
  for (auto inner_list : values) {
    int64_t dim1 = 0;
    for (auto value : inner_list) {
      Set({dim0, dim1}, value);
      ++dim1;
    }
    CHECK_EQ(values_dim1_size, dim1);
    ++dim0;
  }
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE void MutableLiteralBase::PopulateFromArray(
    const Array<NativeT>& values) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  CHECK(shape().IsArray());
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  CHECK_EQ(shape().rank(), values.num_dimensions());
  for (int dim = 0; dim < values.num_dimensions(); ++dim) {
    int64_t shape_size = shape().is_dynamic_dimension(dim)
                             ? GetDynamicSize(dim)
                             : shape().dimensions(dim);
    CHECK_EQ(values.dim(dim), shape_size);
  }
  values.Each([this](absl::Span<const int64_t> indices, NativeT value) {
    this->Set(indices, value);
  });
}

template <typename NativeT>
void MutableLiteralBase::PopulateR2FromArray2D(const Array2D<NativeT>& values) {
  PopulateFromArray(values);
}

template <typename NativeT>
void MutableLiteralBase::PopulateR3FromArray3D(const Array3D<NativeT>& values) {
  PopulateFromArray(values);
}

template <typename NativeT>
void MutableLiteralBase::PopulateR4FromArray4D(const Array4D<NativeT>& values) {
  PopulateFromArray(values);
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE Status MutableLiteralBase::PopulateInternal(
    absl::FunctionRef<NativeT(absl::Span<const int64_t>, int)> generator,
    bool parallel) {
  const Shape& this_shape = shape();
  const int64_t rank = this_shape.rank();
  TF_RET_CHECK(LayoutUtil::IsDenseArray(this_shape));
  TF_RET_CHECK(this_shape.element_type() ==
               primitive_util::NativeToPrimitiveType<NativeT>())
      << "Failing to populate literal with element type "
      << primitive_util::LowercasePrimitiveTypeName(this_shape.element_type())
      << " using data of type "
      << primitive_util::LowercasePrimitiveTypeName(
             primitive_util::NativeToPrimitiveType<NativeT>());
  absl::Span<NativeT> literal_data = data<NativeT>();
  if (rank > 0) {
    StrideConfig stride_config(this_shape, this_shape, this_shape.dimensions());
    int64_t minor_dimension_size =
        ShapeUtil::GetDimension(this_shape, stride_config.minor_dimension);

    auto init_function = [&](absl::Span<const int64_t> indexes,
                             int thread_id) -> StatusOr<bool> {
      DimensionVector minor_scan_indexes(rank, 0);
      const int64_t index =
          IndexUtil::MultidimensionalIndexToLinearIndex(shape(), indexes);
      std::copy(indexes.begin(), indexes.end(), minor_scan_indexes.begin());
      for (int64_t i = 0; i < minor_dimension_size; ++i) {
        minor_scan_indexes[stride_config.minor_dimension] = i;
        literal_data.at(index + i) = generator(minor_scan_indexes, thread_id);
      }
      return true;
    };
    if (parallel) {
      ShapeUtil::ForEachIndexParallel(this_shape, stride_config.base,
                                      stride_config.dimensions,
                                      stride_config.step, init_function);
    } else {
      ShapeUtil::ForEachIndex(
          this_shape, stride_config.base, stride_config.dimensions,
          stride_config.step,
          [&init_function](
              absl::Span<const int64_t> indexes) -> StatusOr<bool> {
            auto result_ignored = init_function(indexes, /*thread_id=*/-1);
            return true;
          });
    }
  } else {
    // For scalars.
    literal_data.at(0) = generator({}, /*thread_id=*/-1);
  }
  return OkStatus();
}

template <typename NativeT>
TF_ATTRIBUTE_NOINLINE Status MutableLiteralBase::Populate(
    absl::FunctionRef<NativeT(absl::Span<const int64_t>)> generator) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  return PopulateInternal<NativeT>(
      [&](absl::Span<const int64_t> indexes, int /*thread_id*/) {
        return generator(indexes);
      },
      /*parallel=*/false);
}
template <typename NativeT>
TF_ATTRIBUTE_NOINLINE Status MutableLiteralBase::PopulateParallel(
    absl::FunctionRef<NativeT(absl::Span<const int64_t>, int)> generator) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  return PopulateInternal<NativeT>(
      [&](absl::Span<const int64_t> indexes, int thread_id) {
        return generator(indexes, thread_id);
      },
      /*parallel=*/data<NativeT>().size() > 32);
}

template <typename NativeT>
void MutableLiteralBase::PopulateWithValue(NativeT value) {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  CHECK(shape().IsArray());
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  for (NativeT& element : data<NativeT>()) {
    element = value;
  }
}

template <typename NativeT>
Literal LiteralBase::Replicate(int64_t times) const {
  CHECK(LayoutUtil::IsDenseArray(shape()))
      << __func__ << " is only supported for dense arrays: " << shape();
  DimensionVector bounds = {times};
  bounds.reserve(shape().dimensions_size() + 1);
  for (int64_t bound : shape().dimensions()) {
    bounds.push_back(bound);
  }
  Literal literal(ShapeUtil::MakeShape(shape().element_type(), bounds));
  int64_t elements = ShapeUtil::ElementsIn(literal.shape());
  if (elements == 0) {
    return literal;
  }

  DimensionVector output_indices(bounds.size(), 0);
  absl::Span<const int64_t> input_indices = output_indices;
  input_indices.remove_prefix(1);

  bool done = false;
  while (!done) {
    const auto element = Get<NativeT>(input_indices);
    literal.Set<NativeT>(output_indices, element);

    done = true;
    for (int n = 0; n < output_indices.size(); ++n) {
      ++output_indices[n];
      if (output_indices[n] < bounds[n]) {
        done = false;
        break;
      }
      output_indices[n] = 0;
    }
  }
  return literal;
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LITERAL_H_
