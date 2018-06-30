/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

// Utilities for dealing with Literal protobufs.

#ifndef TENSORFLOW_COMPILER_XLA_LITERAL_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_LITERAL_UTIL_H_

#include <functional>
#include <initializer_list>
#include <iterator>
#include <memory>
#include <ostream>
#include <string>
#include <type_traits>
#include <vector>

#include "tensorflow/compiler/xla/array2d.h"
#include "tensorflow/compiler/xla/array3d.h"
#include "tensorflow/compiler/xla/array4d.h"
#include "tensorflow/compiler/xla/index_util.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/ptr_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/sparse_index_array.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/bitmap.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
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

  // Returns an ArraySlice of the array for this literal for the given NativeT
  // (e.g., float). CHECKs if the subshape of the literal at the given
  // ShapeIndex is not array. See primitive_util.h for the mapping from XLA type
  // to native type.
  template <typename NativeT>
  tensorflow::gtl::ArraySlice<NativeT> data(
      const ShapeIndex& shape_index = {}) const;

  // Returns a const pointer to the sparse index array. Returns nullptr if the
  // literal is not a sparse array.
  const SparseIndexArray* sparse_indices(
      const ShapeIndex& shape_index = {}) const;

  // Returns a const pointer to (or size of) the underlying buffer holding the
  // array at the given shape index. CHECKs if the subshape of the literal at
  // the given ShapeIndex is not array.
  const void* untyped_data(const ShapeIndex& shape_index = {}) const;
  int64 size_bytes(const ShapeIndex& shape_index = {}) const;

  // Returns this literal's data as a string. This literal must be a rank-1 U8
  // array.
  string GetR1U8AsString() const;

  // Returns a string representation of the literal value.
  // Warning: this function can take minutes for multi-million element Literals.
  string ToString(bool print_layout = false) const;

  // Gets an element in the literal at the given index. The multi_index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  NativeT Get(tensorflow::gtl::ArraySlice<int64> multi_index,
              const ShapeIndex& shape_index) const;
  // Overloads of Get for array literals. CHECKs if the literal is not
  // array-shaped and dense.
  template <typename NativeT>
  NativeT Get(tensorflow::gtl::ArraySlice<int64> multi_index) const;

  // Returns the element value at index (0, ..., 0), however many zeroes are
  // required for that index.
  template <typename NativeT>
  NativeT GetFirstElement() const;

  // As Get(), but determines the correct type and converts the value
  // into text.
  string GetAsString(tensorflow::gtl::ArraySlice<int64> multi_index,
                     const ShapeIndex& shape_index = {}) const;
  // As GetSparseElement(), but determines the correct type and converts the
  // value into text.
  string GetSparseElementAsString(int64 sparse_element_number,
                                  const ShapeIndex& shape_index = {}) const;
  // As Get(), but determines the correct type and converts the value into
  // int64.  This literal must be an array.
  StatusOr<int64> GetIntegralAsS64(
      tensorflow::gtl::ArraySlice<int64> multi_index) const;

  // Returns the multi-index of the element in a sparse literal at the given
  // sparse element number.  The sparse element number is the position with in
  // the sparse array's list of (index, value) pairs, and is checked against the
  // total number of (index, value) pairs in the sparse array.
  tensorflow::gtl::ArraySlice<int64> GetSparseIndex(
      int64 sparse_element_number, const ShapeIndex& shape_index = {}) const;

  // Returns the value of the element in a sparse literal at the given sparse
  // element number.  The sparse element number is the position with in the
  // sparse array's list of (index, value) pairs, and is checked against the
  // total number of (index, value) pairs in the sparse array.
  template <typename NativeT>
  NativeT GetSparseElement(int64 sparse_element_number,
                           const ShapeIndex& shape_index = {}) const;

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
      const std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                               const string& value)>& per_cell) const;
  template <typename NativeT>
  void EachCell(std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                                   NativeT value)>
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

  // Returns whether this literal is zero at the specified index. This literal
  // must be an array with a dense layout.
  bool IsZero(tensorflow::gtl::ArraySlice<int64> indices) const;

  // Returns the count of the elements in the array at the given shape index in
  // this literal.
  int64 element_count(const ShapeIndex& index = {}) const {
    return ShapeUtil::ElementsIn(ShapeUtil::GetSubshape(shape(), index));
  }

  // Returns the count of the elements in the sparse array at the given shape
  // index in this literal, which will be no larger than
  // LayoutUtil::MaxSparseElements(SetSubshape(shape(), index).layout()).
  int64 sparse_element_count() const;

  // Compute a hash for this literal.  This literal must not be a sparse tensor
  // or a tuple containing a sparse tensor.
  size_t Hash() const;

  // Converts this literal to the given shape. Returns an error is the
  // conversion is not possible.
  //
  // round_f32_to_bf16: if true, converting F32 elements to BF16 uses rounding
  // instead of truncation; otherwise, truncation is used.
  //
  // TODO(b/69266521): remove the round_to_bfloat16 flag when rounding becomes
  // the default behavior.
  StatusOr<std::unique_ptr<Literal>> ConvertToShape(
      const Shape& dest_shape, bool round_f32_to_bf16 = false) const;

  // Converts this literal to another primitive type using a bitcast
  // conversion. The to and from primitive types must have the same bit
  // width. Returns an error if the conversion is not possible. This literal
  // must be array-shaped.
  StatusOr<std::unique_ptr<Literal>> BitcastConvert(
      PrimitiveType primitive_dest_type) const;

  // Converts this literal to another primitive type. Returns an error if the
  // conversion is not possible. This literal must be array-shaped.
  StatusOr<std::unique_ptr<Literal>> Convert(
      PrimitiveType primitive_dest_type) const;

  // Returns a literal scalar representing the first element.
  Literal GetFirstScalarLiteral() const;

  // Clones the underlying buffers into a new Literal, or new
  // std::unique_ptr<Literal>.
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
  std::unique_ptr<Literal> Relayout(const Layout& new_layout,
                                    const ShapeIndex& shape_index = {}) const;

  // An overload of Relayout which changes the layout of the entire shape rather
  // than being limited to a single array within the shape.
  std::unique_ptr<Literal> Relayout(const Shape& shape_with_layout) const;

  // Creates a new literal by reshaping this literal to have the given
  // dimensions. The total number of elements must not change; The
  // implementation currently only supports monotonic dim0-major layouts.
  // This literal must be an array.
  StatusOr<std::unique_ptr<Literal>> Reshape(
      tensorflow::gtl::ArraySlice<int64> dimensions) const;

  // Creates a new literal by broadcasting this literal with `dimensions` to
  // yield a literal of shape `result_shape`.
  StatusOr<std::unique_ptr<Literal>> Broadcast(
      const Shape& result_shape,
      tensorflow::gtl::ArraySlice<int64> dimensions) const;

  // Creates a new literal by reordering the dimensions of this literal.
  // The given `permutation` must be a permutation of the dimension numbers
  // in the original literal, and it specifies the order of the new dimensions
  // in the result literal (i.e., new_order[i] = old_order[permutation[i]]).
  // For example, a transpose call on a literal of shape [3 x 8 x 4] and
  // `permutation` = {2, 0, 1} returns a new literal of shape [4 x 3 x 8].
  // This literal must be an array.
  std::unique_ptr<Literal> Transpose(
      tensorflow::gtl::ArraySlice<int64> permutation) const;

  // Creates a sub-array from this literal by extracting the indices
  // [start_index, limit_index) of each dimension. The result literal has the
  // same rank and layout as for the given literal. The number of indices in
  // start_indices and limit_indices must be the rank of the literal, and the
  // indices follow the order of the dimensions.
  // This literal must be an array.
  std::unique_ptr<Literal> Slice(
      tensorflow::gtl::ArraySlice<int64> start_indices,
      tensorflow::gtl::ArraySlice<int64> limit_indices) const;

  // Creates a literal with a prepended dimension with bound "times"; e.g. a
  // f32[3x2] with times=4 will produce a f32[4x3x2] with the 3x2 from this
  // literal replicated four times.
  // This literal must be an array.
  template <typename NativeT>
  std::unique_ptr<Literal> Replicate(int64 times) const;

  // Creates a new Literal object with the shape specified as parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  //
  // Note: It's an antipattern to use this method then immediately call
  // Literal::Populate on the result (since that results in zero initialization,
  // then reinitialization. Conside if a call to MakeUnique<Literal>(shape),
  // followed by the call to Literal::Populate can be used instead.
  static std::unique_ptr<Literal> CreateFromShape(const Shape& shape);

 protected:
  // A data structure representing a subshape at a particular ShapeIndex within
  // the literal. For array-shaped ShapeIndexes, this data structure holds the
  // pointer to the memory allocated for the array data.
  class Piece {
   public:
    // Returns the buffer holding the array data for this piece as an array
    // slice. This piece must be array-shaped.
    template <typename NativeT>
    tensorflow::gtl::ArraySlice<NativeT> data() const;
    template <typename NativeT>
    tensorflow::gtl::MutableArraySlice<NativeT> data();

    // Returns the buffer holding the array data for this piece as a void*. This
    // piece must be array-shaped.
    void* untyped_data();
    const void* untyped_data() const;

    // Gets or sets an element in the array at the given index. The multi_index
    // is CHECKed against the dimension sizes of the array.  This piece must be
    // array-shaped.
    template <typename NativeT>
    NativeT Get(tensorflow::gtl::ArraySlice<int64> index) const;
    template <typename NativeT>
    void Set(tensorflow::gtl::ArraySlice<int64> index, NativeT value);

    // Gets/sets the buffer holding the array data.
    char* buffer() const { return buffer_; }
    void set_buffer(char* buffer) { buffer_ = buffer; }

    // The array of multi-indices that provide the locations of non-zero
    // elements in a sparse array.  Only used if
    // LayoutUtil::IsSparseArray(shape()) is true.
    SparseIndexArray* sparse_indices() const { return sparse_indices_; }
    void set_sparse_indices(SparseIndexArray* sparse_indices) {
      sparse_indices_ = sparse_indices;
    }

    // Gets or sets the subshape of this piece. This reference points to a
    // subshape within the shape in the containing Literal (Literal::shape_).
    const Shape& subshape() const { return *subshape_; }
    void set_subshape(const Shape* subshape) { subshape_ = subshape; }

    // Returns the size in bytes of the buffer holding the array data.
    int64 size_bytes() const { return ShapeUtil::ByteSizeOf(subshape()); }

    // Returns the number of elements in this piece's array.
    int64 element_count() const {
      // If this is a sparse array, use the number of elements represented by
      // the indices in the associated SparseIndexArray.
      return LayoutUtil::IsSparseArray(subshape())
                 ? sparse_indices()->index_count()
                 : ShapeUtil::ElementsIn(subshape());
    }

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

    // Sorts the elements in a sparse array.
    void SortSparseElements();

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

    // Helper for SortSparseElements that has the element type as a template
    // parameter.
    template <typename NativeT>
    void SortSparseElementsInternal();

    // For array-shaped pieces, this is the buffer holding the literal data.
    char* buffer_ = nullptr;

    // For sparse arrays, this is the array of indices.
    SparseIndexArray* sparse_indices_ = nullptr;

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
  friend class Literal;
  friend class LiteralSlice;
  friend class BorrowingLiteral;

 private:
  template <typename NativeT>
  std::unique_ptr<Literal> SliceInternal(
      const Shape& result_shape,
      tensorflow::gtl::ArraySlice<int64> start_indices) const;
};

// Class representing literal values in XLA.
//
// The underlying buffer and shape is always owned by this class.
class Literal : public LiteralBase {
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

  // TODO(b/67651157): Remove this accessor. Literal users should not be able to
  // mutate the shape as this can produce malformed Literals.
  Shape* mutable_shape_do_not_use() { return shape_.get(); }

  // Returns a MutableArraySlice view of the array for this literal for the
  // given NativeT (e.g., float). CHECKs if the subshape of the literal at the
  // given ShapeIndex is not array. See primitive_util.h for the mapping from
  // XLA type to native type.
  template <typename NativeT>
  tensorflow::gtl::MutableArraySlice<NativeT> data(
      const ShapeIndex& shape_index = {});
  // Unhide const method from parent class.
  using LiteralBase::data;

  // Returns a pointer to the sparse index array. Returns nullptr if the literal
  // is not a sparse array.
  SparseIndexArray* sparse_indices(const ShapeIndex& shape_index = {});

  // Returns a pointer to the underlying buffer holding the array at the given
  // shape index. CHECKs if the subshape of the literal at the given ShapeIndex
  // is not array.
  void* untyped_data(const ShapeIndex& shape_index = {});
  // Unhide const method from parent class.
  using LiteralBase::untyped_data;

  // Populates a literal with a sparse layout with the given indices and values.
  // Each index in the indices array is CHECKed against the dimensions in the
  // literal's shape.  If sort is true, then the indices and values will be
  // sorted.  If sort is false, then the indices and values are assumed to
  // already be in sorted order.  See CreateSparse for an example of how data
  // are populated.
  template <typename NativeT>
  void PopulateSparse(SparseIndexArray indices,
                      tensorflow::gtl::ArraySlice<NativeT> values,
                      bool sort = true);

  // Copy values from 'src_literal' rooted at 'src_shape_index' into this
  // literal rooted at 'dest_shape_index'. The subshape of this literal rooted
  // at 'dest_shape_index' must be compatible with the subshape of 'src_literal'
  // rooted at 'src_shape_index', but need not be arrays.
  Status CopyFrom(const LiteralSlice& src_literal,
                  const ShapeIndex& dest_shape_index = {},
                  const ShapeIndex& src_shape_index = {});

  // Similar to CopyFrom, but with move semantincs. The subshape of this literal
  // rooted at 'dest_shape_index' must be *equal* to the shape 'src_literal'
  // (layouts and shapes must match), but need not be arrays. The memory
  // allocated in this literal for the subshape at dest_shape_index is
  // deallocated, and the respective buffers are replaced with those in
  // src_literal. Upon return, src_literal is set to a nil shape (empty tuple).
  Status MoveFrom(Literal&& src_literal,
                  const ShapeIndex& dest_shape_index = {});

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
                       tensorflow::gtl::ArraySlice<int64> src_base,
                       tensorflow::gtl::ArraySlice<int64> dest_base,
                       tensorflow::gtl::ArraySlice<int64> copy_size);

  // Copies one element from src_literal[src_index] to (*this)[dest_index].
  Status CopyElementFrom(const LiteralSlice& src_literal,
                         tensorflow::gtl::ArraySlice<int64> src_index,
                         tensorflow::gtl::ArraySlice<int64> dest_index);

  // Sets an element in the literal at the given index. The multi_index is
  // CHECKed against the dimension sizes.
  template <typename NativeT>
  void Set(tensorflow::gtl::ArraySlice<int64> multi_index,
           const ShapeIndex& shape_index, NativeT value);
  // Overloads of Set for array literals. CHECKs if the literal is not
  // array-shaped and dense.
  template <typename NativeT>
  void Set(tensorflow::gtl::ArraySlice<int64> multi_index, NativeT value);

  // Appends the given element to the literal.  If the elements are not appended
  // in sorted order, then SortSparseElements should be called before calling
  // other methods.  This literal must have a sparse layout.
  template <typename NativeT>
  void AppendSparseElement(tensorflow::gtl::ArraySlice<int64> multi_index,
                           NativeT value, const ShapeIndex& shape_index = {});

  // Sorts the elements in a sparse array.
  void SortSparseElements(const ShapeIndex& shape_index = {});

  // As Set(), but truncates `value` to the literal element type before storing.
  // This literal must be an array.
  Status SetIntegralAsS64(tensorflow::gtl::ArraySlice<int64> multi_index,
                          int64 value);

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
  void PopulateR1(tensorflow::gtl::ArraySlice<NativeT> values);
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
  // NativeT(tensorflow::gtl::ArraySlice<int64> indexes) or compatible.
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

  // Factory methods below.
  //

  // Serialize from a proto.
  static StatusOr<std::unique_ptr<Literal>> CreateFromProto(
      const LiteralProto& proto);

  // Creates a new literal of a given rank. To minimize ambiguity (for users
  // and the compiler) these CreateR[0-2] methods should explicitly specify the
  // native type. For example:
  //
  //  CreateR1<float>({1.0, 42.0});
  //  CreateR2<uint32>({{1, 2}, {3, 4}});
  //
  // The variants not ending with WithLayout use the default XLA layout for the
  // literal's linear representation in memory.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR0(NativeT value);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR1(
      tensorflow::gtl::ArraySlice<NativeT> values);
  static std::unique_ptr<Literal> CreateR1(
      const tensorflow::core::Bitmap& values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2(
      std::initializer_list<std::initializer_list<NativeT>> values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2WithLayout(
      std::initializer_list<std::initializer_list<NativeT>> values,
      const Layout& layout);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3WithLayout(
      std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>
          values,
      const Layout& layout);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4WithLayout(
      std::initializer_list<std::initializer_list<
          std::initializer_list<std::initializer_list<NativeT>>>>
          values,
      const Layout& layout);

  // Creates a literal with a sparse layout and the given indices and values.
  // The shape is initialized from the given dimensions.  The minor dimension of
  // the indices array must equal the rank of the shape (i.e. size of the
  // dimensions array). The major dimension of the indices array must equal the
  // number of elements in the values array. The maximum number of elements in
  // the array is taken from the max_indices() value of the index array.
  //
  // XLA assumes that sparse literals are in sorted order for all operations. If
  // the `sort` argument is true, then the indices and values will be sorted
  // while copying them into the literal. If you have ensured that the indices
  // and values are already sorted, then you may set the `sort` argument to
  // false to skip the sorting step.
  //
  // For example:
  //
  //   CreateSparse(
  //     {12, 12, 12},
  //     SparseIndexArray(10, 3,
  //                      Array2D{
  //                        {0, 1, 2},
  //                        {3, 4, 5},
  //                        {6, 7, 8},
  //                        {9, 10, 11},
  //                      }),
  //     {1.0, 2.0 3.0, 4.0})
  //
  // This creates an array with shape F64[12,12,12]sparse{10}, that has the
  // following non-zero values:
  //
  //     [0,  1,  2]: 1.0
  //     [3,  4,  5]: 2.0
  //     [6,  7,  8]: 3.0
  //     [9, 10, 11]: 4.0
  //
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateSparse(
      tensorflow::gtl::ArraySlice<int64> dimensions, SparseIndexArray indices,
      tensorflow::gtl::ArraySlice<NativeT> values, bool sort = true);

  // Creates a scalar literal value zero of the given primitive type.
  static Literal Zero(PrimitiveType primitive_type);
  // Creates a scalar literal value one of the given primitive type.
  static Literal One(PrimitiveType primitive_type);
  // Creates a scalar literal value containing the minimum value of the given
  // primitive type. For floating-point types, returns -inf.
  static Literal MinValue(PrimitiveType primitive_type);
  // Creates a scalar literal value containing the maximum value of the given
  // primitive type. For floating-point types, returns inf.
  static Literal MaxValue(PrimitiveType primitive_type);
  // Creates a literal of the given shape where each element is `value`.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateFullWithDescendingLayout(
      tensorflow::gtl::ArraySlice<int64> dimensions, NativeT value);

  // Creates a new literal from an Array type. The variants not ending with
  // WithLayout use the default XLA layout for the literal's linear
  // representation in memory.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateFromArray(const Array<NativeT>& values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateFromArrayWithLayout(
      const Array<NativeT>& values, const Layout& layout);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2FromArray2D(
      const Array2D<NativeT>& values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR2FromArray2DWithLayout(
      const Array2D<NativeT>& values, const Layout& layout);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3FromArray3D(
      const Array3D<NativeT>& values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3FromArray3DWithLayout(
      const Array3D<NativeT>& values, const Layout& layout);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4FromArray4D(
      const Array4D<NativeT>& values);
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4FromArray4DWithLayout(
      const Array4D<NativeT>& values, const Layout& layout);

  // Creates a new vector of U8s literal value from a string.
  static std::unique_ptr<Literal> CreateR1U8(tensorflow::StringPiece value);

  // Creates a linspace-populated literal with the given number of rows and
  // columns.
  static std::unique_ptr<Literal> CreateR2F32Linspace(float from, float to,
                                                      int64 rows, int64 cols);

  // Creates a literal that projects the (x, y) dimensions given in values into
  // the z dimension given by "projection".
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR3Projected(
      std::initializer_list<std::initializer_list<NativeT>> values,
      int64 projection);

  // Creates a literal that projects the (x, y) dimensions given in values into
  // the z and p dimensions given.
  template <typename NativeT>
  static std::unique_ptr<Literal> CreateR4Projected(
      std::initializer_list<std::initializer_list<NativeT>> values,
      int64 projection_p, int64 projection_z);

  // Returns an identity matrix (rank 2) with the given row and column count.
  template <typename NativeT>
  static std::unique_ptr<Literal> MakeIdentityR2(int64 size);

  // Returns a tuple literal composed of given literals. Data is copied from the
  // given elements into the returned literal.
  static std::unique_ptr<Literal> MakeTuple(
      tensorflow::gtl::ArraySlice<const Literal*> elements);

  static std::unique_ptr<Literal> MakeTupleFromSlices(
      tensorflow::gtl::ArraySlice<LiteralSlice> elements);

  // As above, but intended to be invoked with move semantics; i.e.
  //
  //  std::vector<std::unique_ptr<Literal>> elements = ...;
  //  auto result = Literal::MakeTupleOwned(std::move(elements));
  //
  // This would have been declared as an overload, but there is ambiguity
  // in invocation between the above signature and this one.
  static std::unique_ptr<Literal> MakeTupleOwned(
      std::vector<std::unique_ptr<Literal>> elements);

  // This overload lets you pass a braced list of unique_ptr<Literal>s to
  // MakeTupleOwned:
  //
  //   Literal::MakeTupleOwned(Literal::CreateR1(...), ...).
  //
  // Simply relying on the MakeTupleOwned(std::vector<unique_ptr<Literal>>)
  // overload doesn't work because std::initializer_list's elements are always
  // const.
  //
  // The arguments to this function must all be unique_ptr<Literal>.
  template <typename... Ts>
  static std::unique_ptr<Literal> MakeTupleOwned(
      std::unique_ptr<Ts>... elements) {
    std::array<std::unique_ptr<Literal>, sizeof...(Ts)> arr{
        std::move(elements)...};
    std::vector<std::unique_ptr<Literal>> v;
    v.insert(v.begin(), std::make_move_iterator(arr.begin()),
             std::make_move_iterator(arr.end()));
    return MakeTupleOwned(std::move(v));
  }

  // Create a constant token literal. Token types have no value.
  static std::unique_ptr<Literal> CreateToken();

  // Returns a vector containing the tuple elements of this Literal as separate
  // Literals. This Literal must be tuple-shaped and can be a nested tuple. The
  // elements are moved into the new Literals; no data is copied. Upon return
  // this Literal is set to a nil shape (empty tuple)
  std::vector<Literal> DecomposeTuple();

  // This operation is the inverse of DecomposeTuple. The given elements are
  // moved into the tuple elements of a new tuple-shaped Literal which is
  // returned. Upon return, each of the Literals in 'elements' is set to a nil
  // shape (empty tuple).
  static Literal MoveIntoTuple(
      tensorflow::gtl::MutableArraySlice<Literal> elements);

  // Creates a new Literal object with its values havings the primitive_type
  // type, and with dimensions defined by the dimensions parameter.
  // The content of the literal values is the default value of the primitive
  // type of literal itself (0 for numeric types, and false for predicates).
  static std::unique_ptr<Literal> CreateFromDimensions(
      PrimitiveType primitive_type,
      tensorflow::gtl::ArraySlice<int64> dimensions);

  // If the given literal's data type is bfloat16, converts it to a float
  // literal; otherwise, returns a copy of it. If the literal is a tuple,
  // recursively converts its elements.
  static std::unique_ptr<Literal> ConvertBF16ToF32(
      const LiteralSlice& bf16_literal);

  // If the given literal's data type is float, converts it to a bfloat16
  // literal; otherwise, returns a copy of it. If the literal is a tuple,
  // recursively converts its elements.
  static std::unique_ptr<Literal> ConvertF32ToBF16(
      const LiteralSlice& f32_literal);

  // Creates a literal with a new shape with the given new dimensions using the
  // data in the given input literal. For reshaping purposes the (flat) data
  // buffer of the input literal is assumed to have the given minor_to_major
  // layout order.
  static std::unique_ptr<Literal> ReshapeSlice(
      tensorflow::gtl::ArraySlice<int64> new_dimensions,
      tensorflow::gtl::ArraySlice<int64> minor_to_major,
      const LiteralSlice& literal);

  // Creates a literal with the supplied shape, and uses the provided value
  // generator to populate the literal's values.
  // Returns the new literal object, or an error Status if failed.
  template <
      PrimitiveType type,
      typename T = typename primitive_util::PrimitiveTypeToNative<type>::type>
  static StatusOr<std::unique_ptr<Literal>> CreateRandomLiteral(
      const Shape& shape,
      const std::function<T(tensorflow::gtl::ArraySlice<int64>)>& generator);

  // Creates a literal with the supplied shape, and initializes the literal
  // values using a normal distribution with given mean and stddev standard
  // deviation, and using the engine as entropy generator.
  // Returns the new literal object, or an error Status if failed.
  template <
      PrimitiveType type, typename E,
      typename T = typename primitive_util::PrimitiveTypeToNative<type>::type>
  static StatusOr<std::unique_ptr<Literal>> CreateRandomLiteral(
      const Shape& shape, E* engine, T mean, T stddev);

  // Creates a literal with the supplied shape, and initializes the literal
  // values using a normal distribution with given mean and stddev standard
  // deviation.
  // Returns the new literal object, or an error Status if failed.
  template <
      PrimitiveType type,
      typename T = typename primitive_util::PrimitiveTypeToNative<type>::type>
  static StatusOr<std::unique_ptr<Literal>> CreateRandomLiteral(
      const Shape& shape, T mean, T stddev);

  //
  // End of factory methods.

  // Returns a multi-dimensional index as a string. For example: '{7, 8}' will
  // be returned for a 2-dimensional index with dimension 0 index equal to 7,
  // dimension 1 equal to 8.
  static string MultiIndexAsString(
      tensorflow::gtl::ArraySlice<int64> multi_index);

 private:
  // Recursively sets the subshapes and buffers of all subpieces rooted at
  // 'piece'. If 'allocate_array' is true, memory is allocated for the arrays in
  // the shape.
  void SetPiece(const Shape& shape, Piece* piece, bool allocate_arrays);

  // Returns the piece at the given ShapeIndex.
  Piece& piece(const ShapeIndex& shape_index) {
    return const_cast<Piece&>(LiteralBase::piece(shape_index));
  }

  Piece& root_piece() const override { return *root_piece_; };

  // Internal template helper for the Literal::CopySliceFrom(), matching its
  // arguments one by one.
  template <typename NativeT>
  Status CopySliceFromInternal(const LiteralBase& src_literal,
                               tensorflow::gtl::ArraySlice<int64> src_base,
                               tensorflow::gtl::ArraySlice<int64> dest_base,
                               tensorflow::gtl::ArraySlice<int64> copy_size);

  // Utility structure which is used to create the optimal configuration for
  // a ShapeUtil::ForEachIndex() scan across two literals.
  struct StrideConfig {
    StrideConfig(const Shape& source_shape, const Shape& dest_shape,
                 tensorflow::gtl::ArraySlice<int64> dimensions);

    // The dimensions of the stride operation. Essentially every dimension
    // will be iterated from base[i] to base[i]+dimensions[i], in step[i]
    // steps.
    tensorflow::gtl::ArraySlice<int64> dimensions;
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

  // Deallocate the buffers held by this literal.
  void DeallocateBuffers();

  friend class LiteralBase;
};
std::ostream& operator<<(std::ostream& out, const Literal& literal);

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
  // lifetime of this class. It points to an appropirately sized buffer with
  // data interpretered as indicated by 'shape'.
  // This constructor is only used for array shapes.
  BorrowingLiteral(const char* src_buf_ptr, const Shape& shape);
  // Similar as above, except to be used for constructing non-nested tuples.
  BorrowingLiteral(tensorflow::gtl::ArraySlice<const char*> src_buf_ptrs,
                   const Shape& shape);
  // TODO(b/79707221): adding constructors for nested tuples as well.

 private:
  // Recursively builds the subtree for the given piece and sets the subshapes
  // of the given piece with the given shape.
  void BuildPieceSubtree(const Shape& shape, Piece* piece);

  // Accessor for the root piece of this literal.
  const Piece& root_piece() const override { return root_piece_; };
  Piece root_piece_;

  // Shape of this literal. Stored as unique_ptr so such that the (default)
  // move construction of this class would be trivially correct: the pointer to
  // Shape root_piece_ stores will still point to the correct address.
  std::unique_ptr<Shape> shape_;
};

template <typename NativeT>
tensorflow::gtl::ArraySlice<NativeT> LiteralBase::Piece::data() const {
  CHECK(ShapeUtil::IsArray(subshape())) << ShapeUtil::HumanString(subshape());
  CHECK_EQ(subshape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>())
      << "Attempting to access "
      << PrimitiveType_Name(primitive_util::NativeToPrimitiveType<NativeT>())
      << " type, but literal element type is "
      << PrimitiveType_Name(subshape().element_type());
  return tensorflow::gtl::ArraySlice<NativeT>(
      reinterpret_cast<const NativeT*>(buffer()), element_count());
}

template <typename NativeT>
tensorflow::gtl::MutableArraySlice<NativeT> LiteralBase::Piece::data() {
  CHECK(ShapeUtil::IsArray(subshape())) << ShapeUtil::HumanString(subshape());
  CHECK_EQ(subshape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>())
      << "Attempting to access "
      << PrimitiveType_Name(primitive_util::NativeToPrimitiveType<NativeT>())
      << " type, but literal element type is "
      << PrimitiveType_Name(subshape().element_type());
  return tensorflow::gtl::MutableArraySlice<NativeT>(
      reinterpret_cast<NativeT*>(buffer()), element_count());
}

template <typename NativeT>
NativeT LiteralBase::Piece::Get(
    tensorflow::gtl::ArraySlice<int64> multi_index) const {
  CHECK(LayoutUtil::IsDenseArray(subshape()));
  return data<NativeT>()[IndexUtil::MultidimensionalIndexToLinearIndex(
      subshape(), multi_index)];
}

template <typename NativeT>
void LiteralBase::Piece::Set(tensorflow::gtl::ArraySlice<int64> multi_index,
                             NativeT value) {
  CHECK(LayoutUtil::IsDenseArray(subshape()));
  data<NativeT>()[IndexUtil::MultidimensionalIndexToLinearIndex(
      subshape(), multi_index)] = value;
}

template <typename NativeT>
tensorflow::gtl::ArraySlice<NativeT> LiteralBase::data(
    const ShapeIndex& shape_index) const {
  return piece(shape_index).data<NativeT>();
}

template <typename NativeT>
tensorflow::gtl::MutableArraySlice<NativeT> Literal::data(
    const ShapeIndex& shape_index) {
  return piece(shape_index).data<NativeT>();
}

template <typename NativeT>
inline NativeT LiteralBase::Get(tensorflow::gtl::ArraySlice<int64> multi_index,
                                const ShapeIndex& shape_index) const {
  return piece(shape_index).Get<NativeT>(multi_index);
}

template <typename NativeT>
inline NativeT LiteralBase::Get(
    tensorflow::gtl::ArraySlice<int64> multi_index) const {
  return root_piece().Get<NativeT>(multi_index);
}

template <typename NativeT>
inline void Literal::Set(tensorflow::gtl::ArraySlice<int64> multi_index,
                         const ShapeIndex& shape_index, NativeT value) {
  return piece(shape_index).Set<NativeT>(multi_index, value);
}

template <typename NativeT>
inline void Literal::Set(tensorflow::gtl::ArraySlice<int64> multi_index,
                         NativeT value) {
  return root_piece().Set<NativeT>(multi_index, value);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR0(NativeT value) {
  auto literal = MakeUnique<Literal>(ShapeUtil::MakeShape(
      primitive_util::NativeToPrimitiveType<NativeT>(), {}));
  literal->Set({}, value);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR1(
    tensorflow::gtl::ArraySlice<NativeT> values) {
  auto literal = MakeUnique<Literal>(
      ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<NativeT>(),
                           {static_cast<int64>(values.size())}));
  literal->PopulateR1(values);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR2WithLayout(
    std::initializer_list<std::initializer_list<NativeT>> values,
    const Layout& layout) {
  auto literal = MakeUnique<Literal>(ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(),
      {static_cast<int64>(values.size()),
       static_cast<int64>(values.begin()->size())},
      AsInt64Slice(layout.minor_to_major())));
  literal->PopulateR2(values);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  return CreateR2WithLayout(values, LayoutUtil::GetDefaultLayoutForR2());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR3WithLayout(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values,
    const Layout& layout) {
  const int64 d0 = values.size();
  const int64 d1 = values.begin()->size();
  const int64 d2 = values.begin()->begin()->size();
  Array3D<NativeT> tmp(d0, d1, d2);
  int64 i0 = 0;
  for (auto d1_values : values) {
    int64 i1 = 0;
    for (auto d2_values : d1_values) {
      int64 i2 = 0;
      for (auto value : d2_values) {
        tmp(i0, i1, i2) = value;
        ++i2;
      }
      ++i1;
    }
    ++i0;
  }
  return CreateR3FromArray3DWithLayout(tmp, layout);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR3(
    std::initializer_list<std::initializer_list<std::initializer_list<NativeT>>>
        values) {
  return CreateR3WithLayout(values, LayoutUtil::GetDefaultLayoutForR3());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR4WithLayout(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        values,
    const Layout& layout) {
  const int64 d0 = values.size();
  const int64 d1 = values.begin()->size();
  const int64 d2 = values.begin()->begin()->size();
  const int64 d3 = values.begin()->begin()->begin()->size();
  Array4D<NativeT> tmp(d0, d1, d2, d3);
  int64 i0 = 0;
  for (auto d1_values : values) {
    int64 i1 = 0;
    for (auto d2_values : d1_values) {
      int64 i2 = 0;
      for (auto d3_values : d2_values) {
        int64 i3 = 0;
        for (auto value : d3_values) {
          tmp(i0, i1, i2, i3) = value;
          ++i3;
        }
        ++i2;
      }
      ++i1;
    }
    ++i0;
  }
  return CreateR4FromArray4DWithLayout(tmp, layout);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateSparse(
    tensorflow::gtl::ArraySlice<int64> dimensions, SparseIndexArray indices,
    tensorflow::gtl::ArraySlice<NativeT> values, bool sort) {
  int64 num_elements = values.size();
  int64 rank = dimensions.size();
  CHECK_EQ(num_elements, indices.index_count());
  CHECK_EQ(rank, indices.rank());
  auto literal = MakeUnique<Literal>(ShapeUtil::MakeShapeWithSparseLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(), dimensions,
      indices.max_indices()));
  literal->PopulateSparse(indices, values, sort);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR4(
    std::initializer_list<std::initializer_list<
        std::initializer_list<std::initializer_list<NativeT>>>>
        values) {
  return CreateR4WithLayout(values, LayoutUtil::GetDefaultLayoutForR4());
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateFromArrayWithLayout(
    const Array<NativeT>& values, const Layout& layout) {
  auto literal = MakeUnique<Literal>(ShapeUtil::MakeShapeWithLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(), values.dimensions(),
      AsInt64Slice(layout.minor_to_major())));
  literal->PopulateFromArray(values);
  return literal;
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateFromArray(
    const Array<NativeT>& values) {
  return CreateFromArrayWithLayout(
      values, LayoutUtil::GetDefaultLayoutForRank(values.num_dimensions()));
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR2FromArray2DWithLayout(
    const Array2D<NativeT>& values, const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR2FromArray2D(
    const Array2D<NativeT>& values) {
  return CreateFromArray(values);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR3FromArray3DWithLayout(
    const Array3D<NativeT>& values, const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR3FromArray3D(
    const Array3D<NativeT>& values) {
  return CreateFromArray(values);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR3Projected(
    std::initializer_list<std::initializer_list<NativeT>> values,
    int64 projection) {
  int64 dim0_size = projection;
  int64 dim1_size = values.size();
  int64 dim2_size = values.begin()->size();

  Array3D<NativeT> array(dim0_size, dim1_size, dim2_size);
  for (int64 dim0 = 0; dim0 < dim0_size; ++dim0) {
    int64 dim1 = 0;
    for (auto inner_list : values) {
      int64 dim2 = 0;
      for (auto value : inner_list) {
        array(dim0, dim1, dim2) = value;
        ++dim2;
      }
      CHECK_EQ(dim2_size, dim2);
      ++dim1;
    }
    CHECK_EQ(dim1_size, dim1);
  }
  return CreateR3FromArray3D(array);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR4Projected(
    std::initializer_list<std::initializer_list<NativeT>> values,
    int64 projection_p, int64 projection_z) {
  int64 dim0_size = projection_p;
  int64 dim1_size = projection_z;
  int64 dim2_size = values.size();
  int64 dim3_size = values.begin()->size();

  Array4D<NativeT> array(dim0_size, dim1_size, dim2_size, dim3_size);
  for (int64 dim0 = 0; dim0 < dim0_size; ++dim0) {
    for (int64 dim1 = 0; dim1 < dim1_size; ++dim1) {
      int64 dim2 = 0;
      for (auto inner_list : values) {
        int64 dim3 = 0;
        for (auto value : inner_list) {
          array(dim0, dim1, dim2, dim3) = value;
          ++dim3;
        }
        CHECK_EQ(dim3_size, dim3);
        ++dim2;
      }
      CHECK_EQ(dim2_size, dim2);
    }
  }
  return CreateR4FromArray4D(array);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR4FromArray4D(
    const Array4D<NativeT>& values) {
  return CreateFromArray(values);
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateR4FromArray4DWithLayout(
    const Array4D<NativeT>& values, const Layout& layout) {
  return CreateFromArrayWithLayout(values, layout);
}

template <typename NativeT>
NativeT LiteralBase::GetFirstElement() const {
  return data<NativeT>().at(0);
}

template <typename NativeT>
NativeT LiteralBase::GetSparseElement(int64 sparse_element_number,
                                      const ShapeIndex& shape_index) const {
  CHECK(
      LayoutUtil::IsSparseArray(ShapeUtil::GetSubshape(shape(), shape_index)));
  return data<NativeT>(shape_index)[sparse_element_number];
}

template <typename NativeT>
void Literal::AppendSparseElement(
    tensorflow::gtl::ArraySlice<int64> multi_index, NativeT value,
    const ShapeIndex& shape_index) {
  Piece& p = piece(shape_index);
  const Shape& subshape = p.subshape();
  CHECK(LayoutUtil::IsSparseArray(subshape));
  int64 rank = ShapeUtil::Rank(subshape);
  CHECK_EQ(multi_index.size(), rank);
  int64 last_element = p.sparse_indices()->index_count();
  CHECK_LT(last_element, LayoutUtil::MaxSparseElements(subshape.layout()));
  p.sparse_indices()->Append(multi_index);
  CHECK_LT(last_element, p.data<NativeT>().size());
  p.data<NativeT>()[last_element] = value;
}

// Returns an identity matrix (rank 2) with the given row and column count.
template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::MakeIdentityR2(int64 size) {
  Array2D<NativeT> array(size, size, 0);
  for (int64 i = 0; i < size; ++i) {
    array(i, i) = 1;
  }
  return CreateR2FromArray2D(array);
}

template <typename NativeT>
void LiteralBase::EachCell(
    std::function<void(tensorflow::gtl::ArraySlice<int64> indices,
                       NativeT value)>
        per_cell) const {
  if (ShapeUtil::IsZeroElementArray(shape())) {
    return;
  }
  std::vector<int64> indices(ShapeUtil::Rank(shape()), 0);
  do {
    per_cell(indices, Get<NativeT>(indices));
  } while (IndexUtil::BumpIndices(shape(), &indices));
}

template <typename NativeT>
inline void Literal::PopulateR1(tensorflow::gtl::ArraySlice<NativeT> values) {
  CHECK(ShapeUtil::IsArray(shape()));
  CHECK_EQ(ShapeUtil::Rank(shape()), 1);
  CHECK_EQ(ShapeUtil::ElementsIn(shape()), values.size());
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  for (int64 i = 0; i < values.size(); ++i) {
    Set({i}, values[i]);
  }
}

template <typename NativeT>
void Literal::PopulateR2(
    std::initializer_list<std::initializer_list<NativeT>> values) {
  CHECK(ShapeUtil::IsArray(shape()));
  CHECK_EQ(ShapeUtil::Rank(shape()), 2);
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
void Literal::PopulateFromArray(const Array<NativeT>& values) {
  CHECK(ShapeUtil::IsArray(shape()));
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  CHECK_EQ(ShapeUtil::Rank(shape()), values.num_dimensions());
  for (int dim = 0; dim < values.num_dimensions(); ++dim) {
    CHECK_EQ(values.dim(dim), shape().dimensions(dim));
  }
  values.Each([this](tensorflow::gtl::ArraySlice<int64> indices,
                     NativeT value) { this->Set(indices, value); });
}

template <typename NativeT>
void Literal::PopulateR2FromArray2D(const Array2D<NativeT>& values) {
  PopulateFromArray(values);
}

template <typename NativeT>
void Literal::PopulateR3FromArray3D(const Array3D<NativeT>& values) {
  PopulateFromArray(values);
}

template <typename NativeT>
void Literal::PopulateR4FromArray4D(const Array4D<NativeT>& values) {
  PopulateFromArray(values);
}

template <typename NativeT>
void Literal::PopulateSparse(SparseIndexArray indices,
                             tensorflow::gtl::ArraySlice<NativeT> values,
                             bool sort) {
  CHECK(LayoutUtil::IsSparseArray(shape()));
  int rank = ShapeUtil::Rank(shape());
  CHECK_EQ(indices.rank(), rank);
  int64 max_elements = LayoutUtil::MaxSparseElements(shape().layout());
  CHECK_LE(indices.max_indices(), max_elements);
  int64 num_elements = values.size();
  CHECK_LE(num_elements, max_elements);
  CHECK_EQ(num_elements, indices.index_count());
  auto root_data = root_piece().data<NativeT>();
  // Piece::data() returns an ArraySlice of size equal to the number of indices
  // in the SparseIndexArray. So there is no need to adjust the size of the data
  // here. It is enough to just copy the incoming values into the data buffer.
  std::copy(values.begin(), values.end(), root_data.begin());
  *this->root_piece().sparse_indices() = std::move(indices);
  if (sort) {
    auto root_data = this->root_piece().data<NativeT>();
    this->root_piece().sparse_indices()->SortWithValues(root_data);
  }
  DCHECK(this->root_piece().sparse_indices()->Validate(shape()));
}

template <typename NativeT, typename FnType>
Status Literal::PopulateInternal(const FnType& generator, bool parallel) {
  const Shape& this_shape = shape();
  const int64 rank = ShapeUtil::Rank(this_shape);
  TF_RET_CHECK(LayoutUtil::IsDenseArray(this_shape));
  TF_RET_CHECK(this_shape.element_type() ==
               primitive_util::NativeToPrimitiveType<NativeT>());
  tensorflow::gtl::MutableArraySlice<NativeT> literal_data = data<NativeT>();
  if (rank > 0) {
    StrideConfig stride_config(this_shape, this_shape,
                               AsInt64Slice(this_shape.dimensions()));
    int64 minor_dimension_size =
        ShapeUtil::GetDimension(this_shape, stride_config.minor_dimension);

    auto init_function = [&](tensorflow::gtl::ArraySlice<int64> indexes) {
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
          [&init_function](tensorflow::gtl::ArraySlice<int64> indexes) {
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
Status Literal::Populate(const FnType& generator) {
  return PopulateInternal<NativeT>(generator, /*parallel=*/false);
}

template <typename NativeT, typename FnType>
Status Literal::PopulateParallel(const FnType& generator) {
  return PopulateInternal<NativeT>(generator, /*parallel=*/true);
}

template <typename NativeT>
void Literal::PopulateWithValue(NativeT value) {
  CHECK(ShapeUtil::IsArray(shape()));
  CHECK_EQ(shape().element_type(),
           primitive_util::NativeToPrimitiveType<NativeT>());
  for (NativeT& element : data<NativeT>()) {
    element = value;
  }
}

template <typename NativeT>
/* static */ std::unique_ptr<Literal> Literal::CreateFullWithDescendingLayout(
    tensorflow::gtl::ArraySlice<int64> dimensions, NativeT value) {
  auto literal = MakeUnique<Literal>(ShapeUtil::MakeShapeWithDescendingLayout(
      primitive_util::NativeToPrimitiveType<NativeT>(), dimensions));
  literal->PopulateWithValue(value);
  return literal;
}

template <typename NativeT>
std::unique_ptr<Literal> LiteralBase::Replicate(int64 times) const {
  DimensionVector bounds = {times};
  bounds.reserve(shape().dimensions_size() + 1);
  for (int64 bound : shape().dimensions()) {
    bounds.push_back(bound);
  }
  auto literal =
      MakeUnique<Literal>(ShapeUtil::MakeShape(shape().element_type(), bounds));
  int64 elements = ShapeUtil::ElementsIn(literal->shape());
  if (elements == 0) {
    return literal;
  }

  DimensionVector output_indices(bounds.size(), 0);
  tensorflow::gtl::ArraySlice<int64> input_indices = output_indices;
  input_indices.remove_prefix(1);

  bool done = false;
  while (!done) {
    const auto element = Get<NativeT>(input_indices);
    literal->Set<NativeT>(output_indices, element);

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

template <PrimitiveType type, typename T>
/* static */ StatusOr<std::unique_ptr<Literal>> Literal::CreateRandomLiteral(
    const Shape& shape,
    const std::function<T(tensorflow::gtl::ArraySlice<int64>)>& generator) {
  using NativeT = typename primitive_util::PrimitiveTypeToNative<type>::type;
  TF_RET_CHECK(shape.element_type() == type);
  auto literal = MakeUnique<Literal>(shape);
  TF_RETURN_IF_ERROR(literal.get()->Populate<NativeT>(
      [&](tensorflow::gtl::ArraySlice<int64> indexes) {
        return generator(indexes);
      }));
  return std::move(literal);
}

template <PrimitiveType type, typename E, typename T>
/* static */ StatusOr<std::unique_ptr<Literal>> Literal::CreateRandomLiteral(
    const Shape& shape, E* engine, T mean, T stddev) {
  using NativeT = typename primitive_util::PrimitiveTypeToNative<type>::type;
  std::normal_distribution<NativeT> generator(mean, stddev);
  return CreateRandomLiteral<type, NativeT>(
      shape, [&](tensorflow::gtl::ArraySlice<int64> /*indexes*/) {
        return generator(*engine);
      });
}

template <PrimitiveType type, typename T>
/* static */ StatusOr<std::unique_ptr<Literal>> Literal::CreateRandomLiteral(
    const Shape& shape, T mean, T stddev) {
  std::minstd_rand0 engine;
  return CreateRandomLiteral<type>(shape, &engine, mean, stddev);
}

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LITERAL_UTIL_H_
