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

#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/memory/memory.h"
#include "absl/strings/string_view.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/protobuf.h"
#include "tensorflow/core/platform/types.h"

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
  const Shape& shape() const { return root_piece().subshape(); }

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
  int64 size_bytes(const ShapeIndex& shape_index = {}) const;

  // Returns this literal's data as a string. This literal must be a rank-1 U8
  // array.
  string GetR1U8AsString() const;

  // Returns a string representation of the literal value. The Shape of the
  // literal is a prefix of the literal value in the string.

  // Warning: this function can take minutes for multi-million
  // element Literals.
  string ToString() const;

  // Returns a string representation of the literal value which does *not*
  // include the shape string.
  string ToStringWithoutShape() const;

  // Returns a string representation of the literal value which includes the
  // shape string with its layout.does *not* include the shape string.
  string ToStringWithLayout() const;

  // Gets an element in the literal at the given index. The multi_index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  NativeT Get(absl::Span<const int64> multi_index,
              const ShapeIndex& shape_index) const;
  // Overloads of Get for array literals. CHECKs if the literal is not
  // array-shaped and dense.
  template <typename NativeT>
  NativeT Get(absl::Span<const int64> multi_index) const;

  // Returns the element value at index (0, ..., 0), however many zeroes are
  // required for that index.
  template <typename NativeT>
  NativeT GetFirstElement() const;

  // As above but returns any integer type casted to an int64.
  absl::optional<int64> GetFirstInteger() const;

  // As Get(), but determines the correct type and converts the value
  // into text.
  string GetAsString(absl::Span<const int64> multi_index,
                     const ShapeIndex& shape_index = {}) const;

  // Return whether the value at the specified index is equal to the provided
  // generic `value` (T must be an arithmetic type).
  //
  // Precondition: must be an array.
  template <typename T>
  typename std::enable_if<(std::is_arithmetic<T>::value ||
                           std::is_same<T, Eigen::half>::value ||
                           std::is_same<T, bfloat16>::value),
                          bool>::type
  IsEqualAt(absl::Span<const int64> multi_index, T value) const {
    if (auto as_s64 = GetIntegralAsS64(multi_index)) {
      return *as_s64 == value;
    }
    complex128 as_complex128 = *GetAsComplex128(multi_index);
    return as_complex128.imag() == 0 && as_complex128.real() == value;
  }

  bool IsEqualAt(absl::Span<const int64> multi_index, complex128 value) const {
    if (auto as_s64 = GetIntegralAsS64(multi_index)) {
      return *as_s64 == value.real() && value.imag() == 0;
    }
    auto as_complex128 = GetAsComplex128(multi_index);
    return *as_complex128 == value;
  }

  // As Get(), but determines the correct type and converts the value into
  // int64.  This literal must be an array.
  absl::optional<int64> GetIntegralAsS64(
      absl::Span<const int64> multi_index) const;

  // As Get(), but determines the correct type, and converts the value into
  // double. This literal must be an array.
  absl::optional<double> GetAsDouble(absl::Span<const int64> multi_index) const;

  // As Get(), but determines the correct type, and converts the value into
  // complex128. All floating point types can be converted into complex128.
  //
  // This literal must be an array.
  absl::optional<complex128> GetAsComplex128(
      absl::Span<const int64> multi_index) const;

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
      const std::function<void(absl::Span<const int64> indices,
                               const string& value)>& per_cell) const;
  template <typename NativeT>
  void EachCell(
      std::function<void(absl::Span<const int64> indices, NativeT value)>
          per_cell) const;

  // Returns whether every element in this literal is equal to value.
  //
  // value is an int8 because we expect this to be called with small
  // compile-time constants (0, -1, etc.) and so that whatever value you pass
  // can be represented exactly by floating-point types as small as 16 bits.
  //
  // If value doesn't fit in this literal's type, returns false.  Values of 1/0
  // are considered equal to true/false; other values are not considered equal
  // to true. Also if this literal is not array-shaped false is returned.
  bool IsAll(int8 value) const;

  // Like IsAll(const Literal&, int8), except we check whether the literal is
  // equal to a particular floating-point number.
  //
  // If the literal is not a floating-point value, this always returns false.
  //
  // This casts value to the type of literal, then compares using ==.  The usual
  // admonishments about floating-point equality checks apply.  We expect you to
  // use this to check for values that can be expressed precisely as a float,
  // e.g. -0.5.  Also if this literal is not array-shaped false is returned.
  bool IsAllFloat(float value) const;

  // Like IsAll(const Literal&, int8), except we check whether the literal is
  // equal to a particular complex number.
  //
  // If the literal is not a complex value, this always returns false.
  //
  // This casts value to the type of literal, then compares using ==.  The usual
  // admonishments about floating-point equality checks apply.  We expect you to
  // use this to check for complex values that can be expressed precisely as
  // float pairs e.g. (-0.5, 1.0).
  //
  // This literal must have a dense layout.
  bool IsAllComplex(complex64 value) const;

  // Literal consists entirely of the first element of the literal.
  bool IsAllFirst() const;

  // Literal consists entirely of an iota.
  bool IsR1Iota() const;

  // Returns whether this literal is zero at the specified index. This literal
  // must be an array with a dense layout.
  bool IsZero(absl::Span<const int64> indices) const;

  // Returns the count of the elements in the array at the given shape index in
  // this literal.
  int64 element_count(const ShapeIndex& index = {}) const {
    if (index.empty()) {
      // Common case, avoid GetSubshape().
      return ShapeUtil::ElementsIn(shape());
    }
    return ShapeUtil::ElementsIn(ShapeUtil::GetSubshape(shape(), index));
  }

  // Compute a hash for this literal.
  size_t Hash() const;

  // Converts this literal to the given shape. Returns an error is the
  // conversion is not possible.
  StatusOr<Literal> ConvertToShape(const Shape& dest_shape) const;

  // Converts this literal to another primitive type using a bitcast
  // conversion. The to and from primitive types must have the same bit
  // width. Returns an error if the conversion is not possible. This literal
  // must be array-shaped.
  StatusOr<Literal> BitcastConvert(PrimitiveType primitive_dest_type) const;

  // Converts this literal to another primitive type. Returns an error if the
  // conversion is not possible. This literal must be array-shaped.
  StatusOr<Literal> Convert(PrimitiveType primitive_dest_type) const;

  // Clones the underlying buffers into a new Literal.
  Literal Clone() const;

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

  // Creates a new literal by reshaping this literal to have the given
  // dimensions. The total number of elements must not change; The
  // implementation currently only supports monotonic dim0-major layouts.
  // This literal must be an array.
  StatusOr<Literal> Reshape(absl::Span<const int64> dimensions) const;

  // Creates a new literal by broadcasting this literal with `dimensions` to
  // yield a literal of shape `result_shape`.
  StatusOr<Literal> Broadcast(const Shape& result_shape,
                              absl::Span<const int64> dimensions) const;

  // Creates a new literal by reordering the dimensions of this literal.
  // The given `permutation` must be a permutation of the dimension numbers
  // in the original literal, and it specifies the order of the new dimensions
  // in the result literal (i.e., new_order[i] = old_order[permutation[i]]).
  // For example, a transpose call on a literal of shape [3 x 8 x 4] and
  // `permutation` = {2, 0, 1} returns a new literal of shape [4 x 3 x 8].
  // This literal must be an array.
  Literal Transpose(absl::Span<const int64> permutation) const;

  // Creates a sub-array from this literal by extracting the indices
  // [start_index, limit_index) of each dimension. The result literal has the
  // same rank and layout as for the given literal. The number of indices in
  // start_indices and limit_indices must be the rank of the literal, and the
  // indices follow the order of the dimensions.
  // This literal must be an array.
  Literal Slice(absl::Span<const int64> start_indices,
                absl::Span<const int64> limit_indices) const;

  // Creates a literal with a prepended dimension with bound "times"; e.g. a
  // f32[3x2] with times=4 will produce a f32[4x3x2] with the 3x2 from this
  // literal replicated four times.
  // This literal must be an array.
  template <typename NativeT>
  Literal Replicate(int64 times) const;

  // Creates a new Literal object with the shape specified as parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  //
  // Note: It's an antipattern to use this method then immediately call
  // MutableLiteralBase::Populate on the result (since that results in zero
  // initialization, then reinitialization. Consider if a call to
  // absl::make_unique<Literal>(shape), followed by the call to
  // MutableLiteralBase::Populate can be used instead.
  static Literal CreateFromShape(const Shape& shape);

 protected:
  // A data structure representing a subshape at a particular ShapeIndex within
  // the literal. For array-shaped ShapeIndexes, this data structure holds the
  // pointer to the memory allocated for the array data.
  class Piece {
   public:
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
    NativeT Get(absl::Span<const int64> index) const;
    template <typename NativeT>
    void Set(absl::Span<const int64> index, NativeT value);

    // Gets/sets the buffer holding the array data.
    char* buffer() const { return buffer_; }
    void set_buffer(char* buffer) { buffer_ = buffer; }

    // Gets or sets the subshape of this piece. This reference points to a
    // subshape within the shape in the containing Literal (Literal::shape_).
    const Shape& subshape() const { return *subshape_; }
    void set_subshape(const Shape* subshape) { subshape_ = subshape; }

    // Returns the size in bytes of the buffer holding the array data.
    int64 size_bytes() const { return ShapeUtil::ByteSizeOf(subshape()); }

    // Returns the number of elements in this piece's array.
    int64 element_count() const { return ShapeUtil::ElementsIn(subshape()); }

    // Returns the child piece at 'index' of this piece.
    Piece& child(int64 index) { return children_[index]; }

    // Adds a child piece to this piece's children.
    void emplace_back(Piece child_piece) {
      children_.emplace_back(std::move(child_piece));
    }

    // Returns the size of children pieces of this piece.
    int64 children_size() { return children_.size(); }

    // Visitor functions that recursively traverses the piece and calls the
    // given function at each child piece. The function has the type:
    //    void (const ShapeIndex& index, const Piece& piece)
    template <typename Fn>
    void ForEachSubpiece(const Fn& func) const {
      ShapeIndex index;
      return ForEachHelper(
                 [&func](const ShapeIndex& index, const Piece& piece) {
                   func(index, piece);
                   return Status::OK();
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
                   return Status::OK();
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

    // Returns true if this piece and 'other' contain the same data. This piece
    // and 'other' must be array-shaped and compatible.
    bool EqualElements(const Piece& other) const;

    // Writes the shape and data (if array-shaped) into the given proto.
    void WriteToProto(LiteralProto* proto) const;

    // Copy the data from 'src' into this piece's buffer. Shapes of this piece
    // and src must be compatible.
    Status CopyFrom(const Piece& src);

    // Copies the data from the given proto into this piece. The shape of this
    // piece must be equal (not just compatible) to the shape of the proto.
    Status CopyFromProto(const LiteralProto& proto);

   private:
    // Helpers for traversing the piece via ForEachSubpiece rooted at 'index'.
    // The first non-OK (or non-true) value is returned by the function.
    // The callable 'func' has the same signature as described above in
    // ForEachSubpiece*.
    template <typename Fn>
    Status ForEachHelper(const Fn& func, const Piece& piece,
                         ShapeIndex* index) const {
      TF_RETURN_IF_ERROR(func(*index, piece));
      for (int64 i = 0; i < piece.children_.size(); ++i) {
        index->push_back(i);
        TF_RETURN_IF_ERROR(ForEachHelper(func, piece.children_[i], index));
        index->pop_back();
      }
      return Status::OK();
    }
    template <typename Fn>
    bool ForEachHelperBool(const Fn& func, const Piece& piece,
                           ShapeIndex* index) const {
      if (!func(*index, piece)) {
        return false;
      }
      for (int64 i = 0; i < piece.children_.size(); ++i) {
        index->push_back(i);
        if (!ForEachHelperBool(func, piece.children_[i], index)) {
          return false;
        }
        index->pop_back();
      }
      return true;
    }
    template <typename Fn>
    Status ForEachMutableHelper(const Fn& func, Piece* piece,
                                ShapeIndex* index) {
      TF_RETURN_IF_ERROR(func(*index, piece));
      for (int64 i = 0; i < piece->children_.size(); ++i) {
        index->push_back(i);
        TF_RETURN_IF_ERROR(
            ForEachMutableHelper(func, &piece->children_[i], index));
        index->pop_back();
      }
      return Status::OK();
    }

    // Recursive helper for EqualElements.
    template <typename NativeT>
    bool EqualElementsInternal(const Piece& other,
                               std::vector<int64>* multi_index) const;

    // For array-shaped pieces, this is the buffer holding the literal data.
    char* buffer_ = nullptr;

    // The shape of piece. This points into the shape of the containing Literal
    // (Literal::shape_).
    const Shape* subshape_ = nullptr;

    // Children pieces for tuple shaped pieces.
    std::vector<Piece> children_ = {};
  };  // class Piece

  const Piece& piece(const ShapeIndex& shape_index) const {
    Piece* piece = &const_cast<Piece&>(root_piece());
    for (const auto i : shape_index) {
      DCHECK_GE(i, 0);
      DCHECK_LT(i, piece->children_size());
      piece = &piece->child(i);
    }
    return *piece;
  }

  // Returns the piece at the root of the shape.
  virtual const Piece& root_piece() const = 0;

  // LiteralSlice and Literal must access Pieces of other Literals.
  friend class MutableLiteralBase;
  friend class LiteralSlice;
  friend class BorrowingLiteral;

 private:
  template <typename NativeT>
  Literal SliceInternal(const Shape& result_shape,
                        absl::Span<const int64> start_indices) const;
};

// Abstract base class representing a mutable literal in XLA.
class MutableLiteralBase : public LiteralBase {
 public:
  virtual ~MutableLiteralBase() = 0;

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
  Shape* mutable_shape_do_not_use() { return shape_.get(); }

  // Returns a pointer to the underlying buffer holding the array at the given
  // shape index. CHECKs if the subshape of the literal at the given ShapeIndex
  // is not array.
  void* untyped_data(const ShapeIndex& shape_index = {});
  // Unhide const method from parent class.
  using LiteralBase::untyped_data;

  // Copy values from 'src_literal' rooted at 'src_shape_index' into this
  // literal rooted at 'dest_shape_index'. The subshape of this literal rooted
  // at 'dest_shape_index' must be compatible with the subshape of 'src_literal'
  // rooted at 'src_shape_index', but need not be arrays.
  Status CopyFrom(const LiteralSlice& src_literal,
                  const ShapeIndex& dest_shape_index = {},
                  const ShapeIndex& src_shape_index = {});

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
                       absl::Span<const int64> src_base,
                       absl::Span<const int64> dest_base,
                       absl::Span<const int64> copy_size);

  // Copies one element from src_literal[src_index] to (*this)[dest_index].
  Status CopyElementFrom(const LiteralSlice& src_literal,
                         absl::Span<const int64> src_index,
                         absl::Span<const int64> dest_index);

  // Sets an element in the literal at the given index. The multi_index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  void Set(absl::Span<const int64> multi_index, const ShapeIndex& shape_index,
           NativeT value);
  // Overloads of Set for array literals. CHECKs if the literal is not
  // array-shaped and dense.
  template <typename NativeT>
  void Set(absl::Span<const int64> multi_index, NativeT value);

  // As Set(), but truncates `value` to the literal element type before storing.
  // This literal must be an array.
  Status SetIntegralAsS64(absl::Span<const int64> multi_index, int64 value);

  // As Set(), but truncates `value` to the literal element type before storing.
  // This literal must be an array.
  Status SetFromDouble(absl::Span<const int64> multi_index, double value);

  // Populate this literal with the given values. Examples:
  //
  //   // Populate with floats.
  //   Array2D<float> float_values = ...
  //   literal.PopulateR2FromArray2D(values);
  //
  //   // Populate with int32s.
  //   literal.PopulateR2<int32>({{1, 2}, {3, 4}});
  //
  // The shape and element type of this literal must match given values. For
  // example, in the call above to literal.PopulateR2(), 'literal' must be a 2x2
  // array of S32.
  template <typename NativeT>
  void PopulateR1(absl::Span<const NativeT> values);
  void PopulateR1(const tensorflow::core::Bitmap& values);
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
  // NativeT(absl::Span<int64> indexes) or compatible.
  //
  // This literal must have a dense layout.
  template <typename NativeT, typename FnType>
  Status Populate(const FnType& generator);

  // A parallel version of Populate(). This can be used if the generator is
  // thread-safe and the values for the shape's different elements are
  // independent.
  template <typename NativeT, typename FnType>
  Status PopulateParallel(const FnType& generator);

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

  Piece& root_piece() const override { return *root_piece_; };

  // Internal template helper for the Literal::CopySliceFrom(), matching its
  // arguments one by one.
  template <typename NativeT>
  Status CopySliceFromInternal(const LiteralBase& src_literal,
                               absl::Span<const int64> src_base,
                               absl::Span<const int64> dest_base,
                               absl::Span<const int64> copy_size);

  // Utility structure which is used to create the optimal configuration for
  // a ShapeUtil::ForEachIndex() scan across two literals.
  struct StrideConfig {
    StrideConfig(const Shape& source_shape, const Shape& dest_shape,
                 absl::Span<const int64> dimensions);

    // The dimensions of the stride operation. Essentially every dimension
    // will be iterated from base[i] to base[i]+dimensions[i], in step[i]
    // steps.
    absl::Span<const int64> dimensions;
    DimensionVector base;
    DimensionVector step;
    int64 minor_dimension = 0;
    // The size of the strides for source and destination. One of the two
    // (the one looping through its most minor dimension) will be 1, while
    // the other will be the stride size at the dimension matching the other
    // shape most minor dimension being scanned.
    int64 dest_stride = 1;
    int64 source_stride = 1;
    // The size of the inner loop on the most minor dimension.
    int64 minor_loop_size = 1;
  };

  // Literal class always owns the shape. The parent class borrows this shape.
  std::unique_ptr<Shape> shape_;

  Piece* root_piece_ = nullptr;

  // Implementation details shared between Populate() and PopulateParallel()
  template <typename NativeT, typename FnType>
  Status PopulateInternal(const FnType& generator, bool parallel);

  friend class LiteralBase;
  friend class MutableBorrowingLiteral;
};
std::ostream& operator<<(std::ostream& out, const Literal& literal);

// The underlying buffer and shape is always owned by this class.
class Literal : public MutableLiteralBase {
 public:
  Literal() : Literal(ShapeUtil::MakeNil()) {}

  // Create a literal of the given shape. The literal is allocated sufficient
  // memory to hold the shape. Memory is uninitialized.
  explicit Literal(const Shape& shape);
  virtual ~Literal();

  // Literals are moveable, but not copyable. To copy a literal use
  // Literal::Clone or Literal::CloneToUnique. This prevents inadvertent copies
  // of literals which can be expensive.
  Literal(const Literal& other) = delete;
  Literal& operator=(const Literal& other) = delete;
  Literal(Literal&& other);
  // 'allocate_arrays' indicates whether to allocate memory for the arrays in
  // the shape. If false, buffer pointers inside of the Literal::Pieces are set
  // to nullptr.
  Literal(const Shape& shape, bool allocate_arrays);
  Literal& operator=(Literal&& other);

  // Similar to CopyFrom, but with move semantics. The subshape of this literal
  // rooted at 'dest_shape_index' must be *equal* to the shape 'src_literal'
  // (layouts and shapes must match), but need not be arrays. The memory
  // allocated in this literal for the subshape at dest_shape_index is
  // deallocated, and the respective buffers are replaced with those in
  // src_literal. Upon return, src_literal is set to a nil shape (empty tuple).
  virtual Status MoveFrom(Literal&& src_literal,
                          const ShapeIndex& dest_shape_index = {});

  // Returns a vector containing the tuple elements of this Literal as separate
  // Literals. This Literal must be tuple-shaped and can be a nested tuple. The
  // elements are moved into the new Literals; no data is copied. Upon return
  // this Literal is set to a nil shape (empty tuple)
  std::vector<Literal> DecomposeTuple();

 private:
  // Deallocate the buffers held by this literal.
  void DeallocateBuffers();

  // Recursively sets the subshapes and buffers of all subpieces rooted at
  // 'piece'. If 'allocate_array' is true, memory is allocated for the arrays in
  // the shape.
  void SetPiece(const Shape& shape, Piece* piece, bool allocate_arrays);
};

// The underlying buffer is not owned by this class and is always owned by
// others. The shape is not owned by this class and not mutable.
class MutableBorrowingLiteral : public MutableLiteralBase {
 public:
  virtual ~MutableBorrowingLiteral();

  MutableBorrowingLiteral() : MutableLiteralBase() {}

  MutableBorrowingLiteral(const MutableBorrowingLiteral& literal);
  MutableBorrowingLiteral& operator=(const MutableBorrowingLiteral& literal);

  // Implicit conversion constructors.
  MutableBorrowingLiteral(MutableLiteralBase* literal);
  MutableBorrowingLiteral(MutableBorrowingLiteral literal,
                          const ShapeIndex& view_root);
  MutableBorrowingLiteral(const char* src_buf_ptr, const Shape& shape);

  // Create a literal from a list of buffers and a shape.
  // Returns a tuple literal if `shape` is a tuple type.
  MutableBorrowingLiteral(absl::Span<char*> src_buf_ptrs, const Shape& shape);

 private:
  // Recursively copies the subtree from the `src_piece` at the given child
  // index to the `dest_piece`. For buffers only the pointers are copied, but
  // not the content.
  void CopyPieceSubtree(const Shape& shape, Piece* src_piece,
                        Piece* dest_piece);
};

// A read-only view of a Literal. A LiteralSlice contains pointers to shape and
// literal buffers always owned by others.
class LiteralSlice : public LiteralBase {
 public:
  LiteralSlice() : LiteralBase() {}

  // Implicit conversion constructors.
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
  DCHECK(subshape().IsArray()) << ShapeUtil::HumanString(subshape());
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
  DCHECK(subshape().IsArray()) << ShapeUtil::HumanString(subshape());
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
NativeT LiteralBase::Piece::Get(absl::Span<const int64> multi_index) const {
  CHECK(LayoutUtil::IsDenseArray(subshape()));
  return data<NativeT>()[IndexUtil::MultidimensionalIndexToLinearIndex(
      subshape(), multi_index)];
}

template <typename NativeT>
void LiteralBase::Piece::Set(absl::Span<const int64> multi_index,
                             NativeT value) {
  CHECK(LayoutUtil::IsDenseArray(subshape()));
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
inline NativeT LiteralBase::Get(absl::Span<const int64> multi_index,
                                const ShapeIndex& shape_index) const {
  return piece(shape_index).Get<NativeT>(multi_index);
}

template <typename NativeT>
inline NativeT LiteralBase::Get(absl::Span<const int64> multi_index) const {
  return root_piece().Get<NativeT>(multi_index);
}

template <typename NativeT>
inline void MutableLiteralBase::Set(absl::Span<const int64> multi_index,
                                    const ShapeIndex& shape_index,
                                    NativeT value) {
  return piece(shape_index).Set<NativeT>(multi_index, value);
}

template <typename NativeT>
inline void MutableLiteralBase::Set(absl::Span<const int64> multi_index,
                                    NativeT value) {
  return root_piece().Set<NativeT>(multi_index, value);
}

template <typename NativeT>
NativeT LiteralBase::GetFirstElement() const {
  return data<NativeT>().at(0);
}

template <typename NativeT>
void LiteralBase::EachCell(
    std::function<void(absl::Span<const int64> indices, NativeT value)>
        per_cell) const {
  if (ShapeUtil::IsZeroElementArray(shape())) {
    return;
  }
  std::vector<int64> indices(shape().rank(), 0);
  do {
    per_cell(indices, Get<NativeT>(indices));
  } while (IndexUtil::BumpIndices(shape(), absl::MakeSpan(indices)));
}

template <typename NativeT>
inline void MutableLiteralBase::PopulateR1(absl::Span<const NativeT> values) {
  CHECK(shape().IsArray());
  CHECK_EQ(shape().rank(), 1);
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), values.size());
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  auto data_span = data<NativeT>();
  std::copy(values.begin(), values.end(), data_span.begin());
}

template <typename NativeT>
void MutableLiteralBase::PopulateR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  CHECK(shape().IsArray());
  CHECK_EQ(shape().rank(), 2);
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());

  const int64 dim0_size = values.size();
  const int64 dim1_size = values.begin()->size();
  CHECK_EQ(dim0_size, shape().dimensions(0));
  CHECK_EQ(dim1_size, shape().dimensions(1));

  int64 dim0 = 0;
  for (auto inner_list : values) {
    int64 dim1 = 0;
    for (auto value : inner_list) {
      Set({dim0, dim1}, value);
      ++dim1;
    }
    CHECK_EQ(dim1_size, dim1);
    ++dim0;
  }
}

template <typename NativeT>
void MutableLiteralBase::PopulateFromArray(const Array<NativeT>& values) {
  CHECK(shape().IsArray());
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  CHECK_EQ(shape().rank(), values.num_dimensions());
  for (int dim = 0; dim < values.num_dimensions(); ++dim) {
    CHECK_EQ(values.dim(dim), shape().dimensions(dim));
  }
  values.Each([this](absl::Span<const int64> indices, NativeT value) {
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

template <typename NativeT, typename FnType>
Status MutableLiteralBase::PopulateInternal(const FnType& generator,
                                            bool parallel) {
  const Shape& this_shape = shape();
  const int64 rank = this_shape.rank();
  TF_RET_CHECK(LayoutUtil::IsDenseArray(this_shape));
  TF_RET_CHECK(this_shape.element_type() ==
               primitive_util::NativeToPrimitiveType<NativeT>());
  absl::Span<NativeT> literal_data = data<NativeT>();
  if (rank > 0) {
    StrideConfig stride_config(this_shape, this_shape,
                               AsInt64Slice(this_shape.dimensions()));
    int64 minor_dimension_size =
        ShapeUtil::GetDimension(this_shape, stride_config.minor_dimension);

    auto init_function = [&](absl::Span<const int64> indexes) {
      DimensionVector minor_scan_indexes(rank, 0);
      const int64 index =
          IndexUtil::MultidimensionalIndexToLinearIndex(shape(), indexes);
      std::copy(indexes.begin(), indexes.end(), minor_scan_indexes.begin());
      for (int64 i = 0; i < minor_dimension_size; ++i) {
        minor_scan_indexes[stride_config.minor_dimension] = i;
        literal_data.at(index + i) = generator(minor_scan_indexes);
      }
    };
    if (parallel) {
      ShapeUtil::ForEachIndexParallel(this_shape, stride_config.base,
                                      stride_config.dimensions,
                                      stride_config.step, init_function);
    } else {
      ShapeUtil::ForEachIndex(
          this_shape, stride_config.base, stride_config.dimensions,
          stride_config.step,
          [&init_function](absl::Span<const int64> indexes) {
            init_function(indexes);
            return true;
          });
    }
  } else {
    // For scalars.
    literal_data.at(0) = generator({});
  }
  return Status::OK();
}
template <typename NativeT, typename FnType>
Status MutableLiteralBase::Populate(const FnType& generator) {
  return PopulateInternal<NativeT>(generator, /*parallel=*/false);
}

template <typename NativeT, typename FnType>
Status MutableLiteralBase::PopulateParallel(const FnType& generator) {
  return PopulateInternal<NativeT>(generator, /*parallel=*/true);
}

template <typename NativeT>
void MutableLiteralBase::PopulateWithValue(NativeT value) {
  CHECK(shape().IsArray());
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  for (NativeT& element : data<NativeT>()) {
    element = value;
  }
}

template <typename NativeT>
Literal LiteralBase::Replicate(int64 times) const {
  DimensionVector bounds = {times};
  bounds.reserve(shape().dimensions_size() + 1);
  for (int64 bound : shape().dimensions()) {
    bounds.push_back(bound);
  }
  Literal literal(ShapeUtil::MakeShape(shape().element_type(), bounds));
  int64 elements = ShapeUtil::ElementsIn(literal.shape());
  if (elements == 0) {
    return literal;
  }

  DimensionVector output_indices(bounds.size(), 0);
  absl::Span<const int64> input_indices = output_indices;
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
