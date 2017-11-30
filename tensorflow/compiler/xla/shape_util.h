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

// Shapes are protobuf messages, so this utility header offers a bunch of
// functionality for querying / poking at them.

#ifndef TENSORFLOW_COMPILER_XLA_SHAPE_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_SHAPE_UTIL_H_

#include <initializer_list>
#include <string>

#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/gtl/array_slice.h"
#include "tensorflow/core/lib/gtl/optional.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// An index for specifying a particular nested subshape within a shape. Used in
// ShapeUtil::GetSubshape and other interfaces. Shapes are recursive data
// structures (trees) and ShapeIndex defines a path through the tree where each
// element of ShapeIndex indexes into a tuple (or nested tuple) within the
// shape. For a non-nested tuple, an index has a single element. For example,
// given a 3-element tuple (a, b, c) containing arrays a, b, and c, the index
// {1} corresponds to array b. For a nested tuple, the index can have more than
// one element. For the nested tuple (a, (b, c, d), e) below are the values
// corresponding to the given indices:
//
//   index {0}    : array a
//   index {1, 2} : array d
//   index {2}    : array e
//   index {0, 0} : invalid index (element at {0} is an array not a tuple)
//
// For indexing into array shapes, the index is always trivially empty, ie {}.
//
// ShapeIndex is a trivial wrapper around std::vector with a minimum number of
// methods implemented.
class ShapeIndex {
 public:
  ShapeIndex() = default;
  ShapeIndex(std::initializer_list<int64> init) : indices_(init) {}

  bool empty() const { return indices_.empty(); }
  size_t size() const { return indices_.size(); }
  void push_back(int64 value) { indices_.push_back(value); }
  void pop_back() { indices_.pop_back(); }

  std::vector<int64>::const_iterator begin() const { return indices_.begin(); }
  std::vector<int64>::const_iterator end() const { return indices_.end(); }
  std::vector<int64>::iterator begin() { return indices_.begin(); }
  std::vector<int64>::iterator end() { return indices_.end(); }

  const int64* data() const { return indices_.data(); }

  int64 back() const { return indices_.back(); }
  int64& back() { return indices_.back(); }

  const int64& operator[](size_t i) const { return indices_[i]; }
  int64& operator[](size_t i) { return indices_[i]; }

  bool operator==(const ShapeIndex& other) const {
    return indices_ == other.indices_;
  }
  bool operator!=(const ShapeIndex& other) const { return !(*this == other); }
  bool operator<(const ShapeIndex& other) const {
    return indices_ < other.indices_;
  }

  string ToString() const;

 private:
  std::vector<int64> indices_;
};

// A view into a ShapeIndex as above, with the cheap/easy ability to consume the
// value at the front of the view.
//
// NB! ShapeIndexView does not own the memory backing the index array.
// The memory backing the index array should be owned by an object
// that lives longer than the ShapeIndexView instances pointing into
// it.
class ShapeIndexView {
 public:
  ShapeIndexView(const ShapeIndex& shape_index, int64 offset = 0)
      : ShapeIndexView(shape_index.data() + offset,
                       shape_index.data() + shape_index.size()) {
    CHECK_LE(offset, shape_index.size());
  }
  ShapeIndexView(std::initializer_list<int64> indices)
      : ShapeIndexView(indices.begin(), indices.end()) {}
  ShapeIndexView(const ShapeIndexView& other) = default;

  using iterator = const int64*;

  iterator begin() const { return begin_; }
  iterator end() const { return end_; }
  int64 size() const { return std::distance(begin_, end_); }
  bool empty() const { return begin_ == end_; }
  int64 front() const {
    CHECK(!empty());
    return *begin_;
  }
  ShapeIndexView ConsumeFront() const {
    CHECK(!empty());
    auto new_begin = begin_;
    ++new_begin;
    return ShapeIndexView(new_begin, end_);
  }

  string ToString() const;

 private:
  ShapeIndexView(iterator begin, iterator end) : begin_(begin), end_(end) {}

  iterator begin_;
  iterator end_;
};

std::ostream& operator<<(std::ostream& out, const ShapeIndex& shape_index);

// Namespaced collection of (static) shape utilities.
//
// These are all effectively convenience functions for testing/tweaking proto
// properties, which do invariant checks before / after the operation.
class ShapeUtil {
 public:
  // Returns the number of elements are contained within the provided shape;
  // e.g. for rank 0 (scalars) the result is always 1.
  // Precondition: !IsTuple(shape)
  static int64 ElementsIn(const Shape& shape);

  // Returns true if 'shape' has zero elements.
  static bool HasZeroElements(const Shape& shape);

  // Returns the number of bytes required for an allocation of shape.  The
  // |pointer_size| parameter is used for calculating the size of tuple
  // shapes. This includes only the size of the top-level buffer. For example, a
  // tuple is stored as an array of pointers to other buffers. In this case,
  // this method only returns the size of the pointer array.
  // Precondition: (!ShapeUtil::IsTuple(shape) || pointer_size > 0) &&
  //               !ShapeUtil::IsOpaque(shape)
  static int64 ByteSizeOf(const Shape& shape, int64 pointer_size = -1);

  // Returns the number of bytes used to store the primitive_type.
  //
  // Precondition: !ShapeUtil::IsOpaque(shape) && !ShapeUtil::IsTuple(shape)
  static int64 ByteSizeOfPrimitiveType(PrimitiveType primitive_type);

  // Returns a human-readable string that represents the given shape, with or
  // without layout. e.g. "f32[42x12] {0, 1}" or "f32[64]".
  static string HumanString(const Shape& shape);
  static string HumanStringWithLayout(const Shape& shape);

  // As above, but for program shapes, returns a string for the form:
  //
  // (param_name: f32[42x12], ...) -> f32[24x42]
  static string HumanString(const ProgramShape& shape);

  // Parses a ShapeUtil::HumanString-format shape string back into a shape
  // object.
  static StatusOr<Shape> ParseShapeString(tensorflow::StringPiece s);

  // Returns whether the LHS and RHS shapes have the same dimensions; note: does
  // not check element type.
  static bool SameDimensions(const Shape& lhs, const Shape& rhs);

  // Returns whether the lhs and rhs shapes have the same element type.
  static bool SameElementType(const Shape& lhs, const Shape& rhs) {
    return lhs.element_type() == rhs.element_type();
  }

  // Returns true if the rank, dimension sizes, and element type are
  // identical. Layout is ignored. Tuple elements are compared recursively for
  // compatibility.
  static bool Compatible(const Shape& lhs, const Shape& rhs);

  // Returns true if the rank and dimension sizes are identical. Element type
  // and layout are ignored. Tuple elements are compared recursively for
  // compatibility.
  static bool CompatibleIgnoringElementType(const Shape& lhs, const Shape& rhs);

  // Returns whether the lhs and rhs shapes are identical protobufs.
  static bool Equal(const Shape& lhs, const Shape& rhs);

  // Returns the rank (number of dimensions) of the given shape.
  // Precondition: !IsTuple(shape)
  static int64 Rank(const Shape& shape);

  // Returns the number of dimensions for which the dimension is not (trivially)
  // 1. e.g., f32[2x1x1] has a true rank of 1D, the other dimensions are just
  // fluff. Note that zero dimensions are included in the true rank, e.g.,
  // f32[3,0,1] has a true rank of 2D.
  static int64 TrueRank(const Shape& shape);

  static ProgramShape MakeProgramShape(std::initializer_list<Shape> parameters,
                                       Shape result);

  ////////////////////
  // Scalar-specific

  static bool IsScalar(const Shape& shape) {
    return !IsTuple(shape) && !IsOpaque(shape) && Rank(shape) == 0;
  }
  static bool IsEffectiveScalar(const Shape& shape) {
    return !IsTuple(shape) && !IsOpaque(shape) && TrueRank(shape) == 0;
  }
  static bool IsScalarF32(const Shape& shape);

  // Extracts the size of the shape's dimension at dimension number
  // GetDimensionNumber(dimension_number).
  static int64 GetDimension(const Shape& shape, int64 dimension_number);

  // Resolves a dimension number, supporting negative indexing.
  //
  // Negative indexing has similar semantics to Python. For an N-dimensional
  // array, dimension -1 is equivalent to dimension N-1, -2 is equivalent to
  // N-2, and so on.
  //
  // This function always returns a positive dimension number for any given
  // dimension_number (which itself can be negative).
  static int64 GetDimensionNumber(const Shape& shape, int64 dimension_number);

  // Returns a shape with the same dimensions as the original, but with the
  // element type changed to type.
  static Shape ChangeElementType(const Shape& original, PrimitiveType type);

  // Creates a tuple shape from a slice of element shapes within the tuple.
  static Shape MakeTupleShape(tensorflow::gtl::ArraySlice<Shape> shapes);

  // Creates an opaque shape. These are generally used for threading a context
  // into a custom operation.
  static Shape MakeOpaqueShape();

  // Appends a shape to the given tuple.
  static void AppendShapeToTuple(const Shape& shape, Shape* tuple_shape);

  // Appends a major dimension to the shape with the given bound.
  static void AppendMajorDimension(int bound, Shape* shape);

  // Returns an empty tuple shape. Can be used to indicate side-effects.
  static Shape MakeNil() { return MakeTupleShape({}); }

  // Constructs a new shape with the given element type and sequence of
  // dimensions.
  static Shape MakeShape(PrimitiveType element_type,
                         tensorflow::gtl::ArraySlice<int64> dimensions);

  // Constructs a new shape with the given minor_to_major order in its Layout.
  // Returns a value shape such that shape.has_layout().
  static Shape MakeShapeWithLayout(
      PrimitiveType element_type, tensorflow::gtl::ArraySlice<int64> dimensions,
      tensorflow::gtl::ArraySlice<int64> minor_to_major);

  // Constructs a new shape with major-first layout.
  static Shape MakeShapeWithMonotonicDim0MajorLayout(
      PrimitiveType element_type,
      tensorflow::gtl::ArraySlice<int64> dimensions);

  // Returns a new shape with major-first layout that has the same layout of
  // elements with a different shape.
  static Shape NormalizeShapeToMonotonicDim0MajorLayout(const Shape& shape);

  // As MakeShape, but the object to write to is passed in.
  static void PopulateShape(PrimitiveType element_type,
                            tensorflow::gtl::ArraySlice<int64> dimensions,
                            Shape* shape);

  // Validates that the provided shape satisfies invariants.
  static Status ValidateShape(const Shape& shape);

  // Validates the provided shape satisfies invariants, except those that
  // pertain to layout.
  //
  // Layout is optional for client-provided shapes, so that the compiler may
  // determine and assign an optimized layout.
  static Status ValidateShapeWithOptionalLayout(const Shape& shape);

  // Returns whether the element type of the shape is integral (signed or
  // unsigned). Note that predicates are not considered integral here, since
  // they are logical values.
  static bool ElementIsIntegral(const Shape& shape);

  // Returns whether the element type of the shape is floating point.
  static bool ElementIsFloating(const Shape& shape);

  // Returns whether the element type of the shape is complex.
  static bool ElementIsComplex(const Shape& shape);

  // Returns whether the element type has the given bit width.
  static bool ElementHasBitWidth(const Shape& shape, int bits);

  // Returns whether the element type of the shape is integral and has
  // the specified number of bits.
  static bool ElementIsIntegralWithBits(const Shape& shape, int bits);

  // Returns whether the element type of the shape is signed. Note
  // that floating point numbers are signed.
  static bool ElementIsSigned(const Shape& shape);

  // Returns whether the shape is a tuple.
  static bool IsTuple(const Shape& shape) {
    return shape.element_type() == TUPLE;
  }

  // Returns whether the shape is an opaque value (i.e. an 'existential' typed
  // value that is passed to CustomCall operations).
  static bool IsOpaque(const Shape& shape) {
    return shape.element_type() == OPAQUE;
  }

  // Returns whether the shape is an array.
  static bool IsArray(const Shape& shape) {
    return !IsTuple(shape) && !IsOpaque(shape);
  }

  // Returns whether the shape is a tuple with at least one element which is
  // also a tuple.
  static bool IsNestedTuple(const Shape& shape);

  // Returns true if shape is an empty tuple.
  static bool IsEmptyTuple(const Shape& shape);

  // Returns true if shape is an empty tuple, or is an array with no elements.
  static bool IsNil(const Shape& shape);

  // Returns the number of elements in the given tuple shape.
  // Precondition: IsTuple(shape)
  static int64 TupleElementCount(const Shape& shape);

  // Returns the tuple element shape at given index.
  // Precondition: IsTuple(shape) && TupleElementCount(shape) > index
  static const Shape& GetTupleElementShape(const Shape& shape, int64 index);

  // Slices tuple elements in the range [start, limit) and returns a new tuple
  // shape. E.g. a tuple like (f32, s32, u32) would slice via 1,3 to (s32, u32).
  static Shape SliceTuple(const Shape& tuple, int64 start, int64 limit);

  // Shorthand for testing whether a shape is of a given element type and
  // sequence of dimensions.
  //
  // DEPRECATED: Use Equal() instead.
  static bool ShapeIs(const Shape& shape, PrimitiveType element_type,
                      std::initializer_list<int64> dimensions);

  // GetSubshape and GetMutableSubshape return a particular nested Shape within
  // the given Shape argument.
  static const Shape& GetSubshape(const Shape& shape, ShapeIndexView index);
  static Shape* GetMutableSubshape(Shape* shape, ShapeIndexView index);

  // Returns whether the given index in the given shape is a leaf element of the
  // shape.
  static bool IsLeafIndex(const Shape& shape, const ShapeIndex& index);

  // Calls the given visitor function for each subshape of the given shape.
  // Subshapes are visited in DFS pre-order starting with the entire shape
  // (index {}).
  using VisitorFunction = std::function<void(const Shape& /*subshape*/,
                                             const ShapeIndex& /*index*/)>;
  static void ForEachSubshape(const Shape& shape, const VisitorFunction& func);
  using MutatingVisitorFunction =
      std::function<void(Shape* /*subshape*/, const ShapeIndex& /*index*/)>;
  static void ForEachMutableSubshape(Shape* shape,
                                     const MutatingVisitorFunction& func);

  // Variants of ForEach(Mutable)Subshape which propagate Status from the
  // visitor function.
  using StatusVisitorFunction = std::function<Status(
      const Shape& /*subshape*/, const ShapeIndex& /*index*/)>;
  static Status ForEachSubshapeWithStatus(const Shape& shape,
                                          const StatusVisitorFunction& func);
  using MutatingStatusVisitorFunction =
      std::function<Status(Shape* /*subshape*/, const ShapeIndex& /*index*/)>;
  static Status ForEachMutableSubshapeWithStatus(
      Shape* shape, const MutatingStatusVisitorFunction& func);

  // Removes all degenerate dimensions (size one) from the given shape. The
  // stripped minor_to_major preserves the relative ordering of non-degenerate
  // dimensions. The stripped shape has the property that the underlying
  // representation (bits in memory) for the stripped shape is the same as the
  // original shape modulo padding. Examples:
  //
  // input shape:    F32 [1, 2, 1], minor_to_major = {0, 1, 2}
  // stripped shape: F32 [2], minor_to_major = {0}
  //
  // input shape:    F32 [6, 1, 5], minor_to_major = {2, 0, 1}
  // stripped shape: F32 [6, 5], minor_to_major = {1, 0}
  //
  // input shape:    F32 [1, 7, 1, 6, 5, 1], minor_to_major = {0, 2, 5, 4, 3, 1}
  // stripped shape: F32 [7, 6, 5], minor_to_major = {0, 2, 1}
  //
  // input shape:    F32 [1, 1], minor_to_major = {0, 1}
  // stripped shape: F32 [], minor_to_major = {}
  // Precondition: !ShapeUtil::IsOpaque(shape) && !ShapeUtil::IsTuple(shape)
  static Shape StripDegenerateDimensions(const Shape& shape);

  // Permutes the dimensions by the given permutation, so
  // return_value.dimensions[permutation[i]] = argument.dimensions[i]
  static Shape PermuteDimensions(tensorflow::gtl::ArraySlice<int64> permutation,
                                 const Shape& shape);

  // If we can go from `shape_pre` to `shape_post` by merely inserting or
  // deleting 1-sized dimensions, return the indices in `shape_pre` of the
  // deleted dimensions and the indices in `dims_post` of the inserted
  // dimensions.
  // For example, if `shape_pre = {a_1, a_2, ..., a_m}` and
  // `shape_post = {b_1, b_2, ..., b_n}` where we can find some sequence of `i`s
  // and some sequence of `j`s so `a_i = 1` for each `i` and `b_j = 1` for each
  // `j` and `a_(k-s) = b_(k-t)` where `s` and `t` are the number of `i`s and
  // `j`s less than `k` for all other `k`, we return the `i`s and `j`s.
  // For another example, if `shape_pre = shape_post = {}`, we return `{}`.
  static std::tuple<bool, std::vector<int64>, std::vector<int64>>
  InsertedOrDeleted1SizedDimensions(const Shape& shape_pre,
                                    const Shape& shape_post);

  // Suppose a reshape transforms input_shape to output shape. Returns a vector
  // of pairs that indicate the input and output dimensions that this reshape
  // doesn't logically (i.e. ignoring the layout) modify. For each pair (I,O) in
  // the returned vector, the reshape transforms any input index whose I-th
  // dimension is x to an output index whose O-th dimension is x too.
  //
  // Post-condition: the returned vector is sorted (by both input and output
  // dimensions because input and output dimensions have the same order).
  //
  // Example:
  //   input  shape = T[a, b, x, y, cd]
  //   output shape = T[ab, x, 1, y, c, d]
  //   return value = {{2, 1}, {3, 3}}
  //
  //   The two pairs represent the input and output dimension of size x and
  //   those of size y.
  static std::vector<std::pair<int64, int64>> DimensionsUnmodifiedByReshape(
      const Shape& input_shape, const Shape& output_shape);

  // Returns whether a transpose from input_shape to output_shape with dimension
  // mapping "dimension_mapping" produces a result which is bit-wise identical
  // to its input and thus may be replaced with a bitcast.
  static bool TransposeIsBitcast(
      const Shape& input_shape, const Shape& output_shape,
      tensorflow::gtl::ArraySlice<int64> dimension_mapping);

  // Returns whether a reshape from "input_shape" to "output_shape" is a
  // bitcast.
  static bool ReshapeIsBitcast(const Shape& input_shape,
                               const Shape& output_shape);

  // Find a physical layout for 'output_shape' such that
  // ShapeUtil::ReshapeIsBitcast(input_shape, output_shape_with_layout) returns
  // true (where 'output_shape_with_layout' is 'output_shape' with the found
  // layout). The layout of 'input_shape' is kept fixed. Returns
  // 'output_shape_with_layout' if such a layout can be found, and an error
  // otherwise.
  static tensorflow::gtl::optional<Shape> AlignLayouts(
      const Shape& input_shape, const Shape& output_shape);

  // Returns a shape with the given dimension deleted.
  // For example:
  // • `DeleteDimension(1, T[m, n, k]) = T[m, k]`
  static Shape DeleteDimension(int64 dim_to_delete, Shape shape);

  // Returns a shape with all the dimensions of the input shape for which `p`
  // returns true.
  // For examples:
  // • `FilterDimensions((< 2), T[m, n, k]) = T[m, n]`
  // • `FilterDimensions(is_even_number, T[m, n, k]) = T[m, k]`
  static Shape FilterDimensions(const std::function<bool(int64)>& p,
                                Shape shape);

  // Iterates through all the shape indexes, in minor to major order, starting
  // from the base indexes, incrementing by the incr steps, up to count
  // (index[i] < base[i] + count[i]), and calls the visitor_function with the
  // current index.
  // The visitor_function visitor function should return true if it wants to
  // continue, or false otherwise.
  //
  // visitor_function must be a callable of type bool(const std::vector<int64>&)
  // or compatible.
  template <typename FnType>
  static void ForEachIndex(const Shape& shape,
                           tensorflow::gtl::ArraySlice<int64> base,
                           tensorflow::gtl::ArraySlice<int64> count,
                           tensorflow::gtl::ArraySlice<int64> incr,
                           const FnType& visitor_function) {
    if (ShapeUtil::HasZeroElements(shape)) {
      return;
    }
    CHECK_EQ(Rank(shape), base.size());
    CHECK_EQ(incr.size(), base.size());
    CHECK_EQ(count.size(), base.size());
    const Layout& layout = shape.layout();
    const int64 rank = layout.minor_to_major_size();
    // Allows handling R0 arrays, such that the visitor function will be called
    // once with the proper empty indexes.
    int64 n = -1;
    std::vector<int64> indexes(base.begin(), base.end());
    while (n < rank && visitor_function(indexes)) {
      // Increments dimensions in minor to major order.
      for (n = 0; n < rank; ++n) {
        int64 dim = layout.minor_to_major(n);
        indexes[dim] += incr[dim];
        if (indexes[dim] < base[dim] + count[dim]) {
          break;
        }
        indexes[dim] = base[dim];
      }
    }
  }

 private:
  // Validates all of the non-layout properties of the shape -- this is a helper
  // used by both the layout-optional and layout-required public method.
  static Status ValidateShapeWithOptionalLayoutInternal(const Shape& shape);

  TF_DISALLOW_COPY_AND_ASSIGN(ShapeUtil);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SHAPE_UTIL_H_
