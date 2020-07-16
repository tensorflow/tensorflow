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

#include "absl/base/macros.h"
#include "absl/container/inlined_vector.h"
#include "absl/types/optional.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/primitive_util.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status_macros.h"
#include "tensorflow/compiler/xla/statusor.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/lib/core/threadpool.h"
#include "tensorflow/core/platform/cpu_info.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

class ShapeIndexView;

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
  template <typename InputIt>
  ShapeIndex(InputIt start, InputIt end) : indices_(start, end) {}

  explicit ShapeIndex(ShapeIndexView v);

  bool empty() const { return indices_.empty(); }
  size_t size() const { return indices_.size(); }
  void push_back(int64 value) { indices_.push_back(value); }
  void pop_back() { indices_.pop_back(); }

  // push_front is O(n), but shapes don't usually have a ton of dimensions.
  void push_front(int64 value) { indices_.insert(indices_.begin(), value); }

  using container_type = absl::InlinedVector<int64, 2>;

  container_type::const_iterator begin() const { return indices_.begin(); }
  container_type::const_iterator end() const { return indices_.end(); }
  container_type::iterator begin() { return indices_.begin(); }
  container_type::iterator end() { return indices_.end(); }

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

  template <typename H>
  friend H AbslHashValue(H h, const ShapeIndex& index) {
    return H::combine(std::move(h), index.indices_);
  }

 private:
  container_type indices_;
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
      : indices_(shape_index.data() + offset, shape_index.size() - offset) {
    CHECK_LE(offset, shape_index.size());
  }
  ShapeIndexView(std::initializer_list<int64> indices) : indices_(indices) {}
  ShapeIndexView(const ShapeIndexView& other) = default;

  using iterator = const int64*;

  iterator begin() const { return indices_.begin(); }
  iterator end() const { return indices_.end(); }
  int64 size() const { return indices_.size(); }
  bool empty() const { return indices_.empty(); }
  int64 front() const {
    CHECK(!empty());
    return indices_.front();
  }
  int64 back() const {
    CHECK(!empty());
    return indices_.back();
  }
  ShapeIndexView ConsumeFront() const {
    ShapeIndexView result = *this;
    result.indices_.remove_prefix(1);
    return result;
  }
  ShapeIndexView ConsumeBack() const {
    ShapeIndexView result = *this;
    result.indices_.remove_suffix(1);
    return result;
  }
  ShapeIndex ToShapeIndex() const { return ShapeIndex(begin(), end()); }

  bool operator==(const ShapeIndexView& other) const;
  bool operator!=(const ShapeIndexView& other) const;

  string ToString() const;

  // Returns true if this shape index starts with 'prefix'.
  bool StartsWith(ShapeIndexView prefix) const;

 private:
  absl::Span<const int64> indices_;
};

inline ShapeIndex::ShapeIndex(ShapeIndexView v)
    : ShapeIndex(v.begin(), v.end()) {}

std::ostream& operator<<(std::ostream& out, const ShapeIndex& shape_index);
std::ostream& operator<<(std::ostream& out, const ShapeIndexView& shape_index);

// Namespaced collection of (static) shape utilities.
//
// These are all effectively convenience functions for testing/tweaking proto
// properties, which do invariant checks before / after the operation.
class ShapeUtil {
 public:
  // Data structure which describes the coordinates and the shape, of a tuple
  // shaped sub-shape.
  struct IndexedShape {
    IndexedShape() = default;
    IndexedShape(ShapeIndex index, Shape shape)
        : index(std::move(index)), shape(std::move(shape)) {}
    ShapeIndex index;
    Shape shape;
  };

  // Returns the number of elements are contained within the provided shape;
  // e.g. for rank 0 (scalars) the result is always 1.
  // Precondition: shape.IsArray()
  static int64 ElementsIn(const Shape& shape);

  // As ElementsIn(), but recurses through tuples.
  static int64 ElementsInRecursive(const Shape& shape);

  // Returns true if shape has the primitive type, recurses through tuples.
  static bool HasPrimitiveType(const Shape& shape,
                               PrimitiveType primitive_type);

  // Returns true if 'shape' is an array with zero elements.
  static bool IsZeroElementArray(const Shape& shape);

  // Returns the number of bytes required for an allocation of shape.  The
  // |pointer_size| parameter is used for calculating the size of tuple
  // shapes. This includes only the size of the top-level buffer. For example, a
  // tuple is stored as an array of pointers to other buffers. In this case,
  // this method only returns the size of the pointer array.
  static int64 ByteSizeOf(const Shape& shape, int64 pointer_size = -1);

  // Returns the number of bytes used to store the primitive_type.
  //
  // Precondition: shape.IsArray()
  static int64 ByteSizeOfPrimitiveType(PrimitiveType primitive_type);

  // Returns the number of bytes required to store the tuple member pointers for
  // a allocation of shape. The `shape` must be a TUPLE shape, and
  // `pointer_size` must be larger than zero.
  static int64 ByteSizeOfTupleIndexTable(const Shape& shape,
                                         int64 pointer_size);

  // Returns the number of bytes required for the elements in an allocation of
  // `shape`, which must be an array shape. Shapes use a separate
  // memory location for each element, and so for these shapes,
  // `ByteSizeOf(shape) == ByteSizeOfElements(shape)`. This
  // size also includes padding if present in the layout.
  static int64 ByteSizeOfElements(const Shape& shape);

  // Returns a human-readable string that represents the given shape, with or
  // without layout. e.g. "f32[42x12] {0, 1}" or "f32[64]".
  static string HumanString(const Shape& shape);
  static string HumanStringWithLayout(const Shape& shape);

  // As above, but for program shapes, returns a string for the form:
  //
  // (param_name: f32[42x12], ...) -> f32[24x42]
  static string HumanString(const ProgramShape& program_shape);

  // Returns whether the LHS and RHS shapes have the same dimensions; note: does
  // not check element type.
  // Precondition: IsArray(lhs) && IsArray(rhs)
  static bool SameDimensions(const Shape& lhs, const Shape& rhs);

  // Returns whether the lhs and rhs shapes have the same element type.
  static bool SameElementType(const Shape& lhs, const Shape& rhs) {
    return lhs.element_type() == rhs.element_type();
  }

  // As SameElementType, but allows floating point types to have different
  // precisions.
  static bool SameElementTypeIgnoringFpPrecision(const Shape& a,
                                                 const Shape& b) {
    if (ElementIsFloating(a) && ElementIsFloating(b)) {
      return true;
    }
    return ShapeUtil::SameElementType(a, b);
  }

  // Returns the higher-precision element type if a and b are both floating
  // point types; otherwise, checks that they have the same element type
  // and returns it.
  static PrimitiveType HigherPrecisionElementType(const Shape& a,
                                                  const Shape& b) {
    if (SameElementType(a, b)) {
      return a.element_type();
    }
    return primitive_util::BitWidth(a.element_type()) <
                   primitive_util::BitWidth(b.element_type())
               ? b.element_type()
               : a.element_type();
  }

  // Returns true if the rank, dimension sizes, and element type are
  // identical. Layout is ignored. Tuple elements are compared recursively for
  // compatibility.
  static bool Compatible(const Shape& lhs, const Shape& rhs);

  // Returns true if the rank and dimension sizes are identical. Element type
  // and layout are ignored. Tuple elements are compared recursively for
  // compatibility.
  static bool CompatibleIgnoringElementType(const Shape& lhs, const Shape& rhs);

  // As Compatible, but allow one of lhs and rhs to be BF16 while the other
  // being F32. Tuple elements are compared recursively for compatibility.
  static bool CompatibleIgnoringFpPrecision(const Shape& lhs, const Shape& rhs);

  // Returns whether the lhs and rhs shapes are identical.
  static bool Equal(const Shape& lhs, const Shape& rhs);

  // As Equal, but does not compare the element type.
  static bool EqualIgnoringElementType(const Shape& lhs, const Shape& rhs);

  // As Equal, but allow one of lhs and rhs to be F16 while the other is F32.
  static bool EqualIgnoringFpPrecision(const Shape& lhs, const Shape& rhs);

  // Two shapes have same structure if all subshape indices of lhs are presented
  // on rhs and vice versa.
  // A nested tuple shape of (F32, (S32[2], F32[2, 2])) is structurally equal to
  // (S32, (F32[3], S32[2])) as their structures are both (,(,))
  //
  // In contrast, (F32, (F32, F32)) is structurally different from
  // ((F32, F32), F32) as the former has structure (,(,)) while the latter has
  // ((,),)
  static bool EqualStructure(const Shape& lhs, const Shape& rhs);

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
    return shape.IsArray() && shape.rank() == 0;
  }
  static bool IsEffectiveScalar(const Shape& shape) {
    return shape.IsArray() && TrueRank(shape) == 0;
  }

  // Returns whether "shape" is a scalar (array) with the given element_type.
  static bool IsScalarWithElementType(const Shape& shape,
                                      PrimitiveType element_type);

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

  // Retursn a shape with same dimensions but with all dimensions set to static.
  static Shape MakeStaticShape(const Shape& original);

  // Creates a tuple shape from a slice of element shapes within the tuple.
  static Shape MakeTupleShape(absl::Span<const Shape> shapes);

  // Creates an opaque shape. These are generally used for threading a context
  // into a custom operation.
  static Shape MakeOpaqueShape();

  // Creates a token shape. Values of this shape are used for ordering
  // side-effecting operations.
  static Shape MakeTokenShape();

  // Appends a shape to the given tuple.
  static void AppendShapeToTuple(const Shape& shape, Shape* tuple_shape);

  // Update a subshape of a tuple.
  static void UpdateTupleShape(const Shape& shape, int64 index,
                               Shape* tuple_shape);

  // Update the dynamic dimension for a shape. This shape can be a nested tuple.
  static void UpdateDynamicDimension(Shape* shape, ShapeIndexView index,
                                     int64 dim, bool is_dynamic);

  // Appends a major dimension to the shape with the given bound.
  static void AppendMajorDimension(int bound, Shape* shape);

  // Returns an empty tuple shape. Can be used as a sentinel Shape value.
  static Shape MakeNil() { return MakeTupleShape({}); }

  // Checks whether the shape is initialized.
  static bool IsInitialized(const Shape& shape) {
    return shape.element_type() != PRIMITIVE_TYPE_INVALID;
  }

  // Constructs a new shape with the given element type and sequence of
  // dimensions.
  static Shape MakeShape(PrimitiveType element_type,
                         absl::Span<const int64> dimensions);

  // Make a scalar shape with given primitive type.
  static Shape MakeScalarShape(PrimitiveType element_type);

  // Constructs a new shape with the given element type and sequence of
  // potentially dynamic dimensions. The argument 'dynamic_dimensions' indicates
  // with a true value that the respective dimension is dynamic. If the
  // dimension is dynamic then the respective value in 'dimension' is an upper
  // bound on the dimension size. 'dimensions' and 'dynamic_dimensions' must be
  // the same size.
  static Shape MakeShape(PrimitiveType element_type,
                         absl::Span<const int64> dimensions,
                         const std::vector<bool>& dynamic_dimensions);

  // Constructs a new shape with the given element type and sequence of
  // dimensions. Method checks if the element type is valid and the shape's
  // size fits in std::numeric_limits<int64>::max().
  static StatusOr<Shape> MakeValidatedShape(PrimitiveType element_type,
                                            absl::Span<const int64> dimensions);
  static StatusOr<Shape> MakeValidatedShape(
      PrimitiveType element_type, absl::Span<const int64> dimensions,
      const std::vector<bool>& dynamic_dimensions);

  // Creates a Shape with element type corresponding to T and the given
  // dimensions
  template <typename T>
  static Shape MakeShapeWithType(absl::Span<const int64> dimensions) {
    return ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<T>(),
                                dimensions);
  }

  // Constructs a new shape with the given minor_to_major order in its Layout.
  // Returns a value shape such that shape.has_layout().
  static Shape MakeShapeWithLayout(PrimitiveType element_type,
                                   absl::Span<const int64> dimensions,
                                   absl::Span<const int64> minor_to_major,
                                   absl::Span<const Tile> tiles = {},
                                   int64 element_size_in_bits = 0,
                                   int64 memory_space = 0);

  // Returns the same shape except with all dimensions set to be static.
  static Shape MakeShapeWithStaticDimensions(const Shape& shape);

  // Constructs a new shape with major-first layout (i.e. {n, n-1, ..., 0}).
  static Shape MakeShapeWithDescendingLayout(
      PrimitiveType element_type, absl::Span<const int64> dimensions);

  // Returns a new Shape based on the given Shape with low-dimension-major
  // layout (i.e. {n, n-1, ..., 0}, like Fortran), and with the dimensions
  // rearranged so that it has the same in-memory layout as the given shape.
  //
  // For example, transforms f32[B,H,W,C]{0,3,2,1} to f32[H,W,C,B]{3,2,1,0}.
  static Shape MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
      const Shape& shape);

  // As MakeShape, but the object to write to is passed in.
  static Status PopulateShape(PrimitiveType element_type,
                              absl::Span<const int64> dimensions, Shape* shape);

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

  // Returns whether the given primitive type corresponds to an array shape.
  static bool IsArrayPrimitiveType(PrimitiveType primitive_type);

  // Returns whether the shape is a tuple with at least one element which is
  // also a tuple.
  static bool IsNestedTuple(const Shape& shape);

  // Returns true if shape is an empty tuple.
  static bool IsEmptyTuple(const Shape& shape);

  // Returns the number of elements in the given tuple shape.
  // Precondition: IsTuple(shape)
  static int64 TupleElementCount(const Shape& shape);

  // Returns the tuple element shape at given index.
  // Precondition: IsTuple(shape) && TupleElementCount(shape) > index
  static const Shape& GetTupleElementShape(const Shape& shape, int64 index);

  // Returns the number of elements, recursively, in the given shape.
  static int64 SubshapeCount(const Shape& shape);

  // Slices tuple elements in the range [start, limit) and returns a new tuple
  // shape. E.g. a tuple like (f32, s32, u32) would slice via 1,3 to (s32, u32).
  static Shape SliceTuple(const Shape& tuple, int64 start, int64 limit);

  // Returns the shape of the real/imaginary components of the given complex
  // shape.
  static Shape ComplexComponentShape(const Shape& complex_shape);

  // Returns true if the given shape has a subshape at the given index.
  static bool IndexIsValid(const Shape& shape, ShapeIndexView index);

  // GetSubshape and GetMutableSubshape return a particular nested Shape within
  // the given Shape argument. The non-Try variants check fail if index is
  // invalid.
  static const Shape& GetSubshape(const Shape& shape, ShapeIndexView index);
  static StatusOr<const Shape*> TryGetSubshape(const Shape& shape,
                                               ShapeIndexView index);
  static Shape* GetMutableSubshape(Shape* shape, ShapeIndexView index);

  // Returns whether the given index in the given shape is a leaf element of the
  // shape.
  static bool IsLeafIndex(const Shape& shape, const ShapeIndex& index);

  // Returns the number of leaves in the shape.
  static int64 GetLeafCount(const Shape& shape);

  // Retrieves all the leaf shapes and their indexes, in the order walked by
  // the ForEachSubshape() API.
  static std::vector<IndexedShape> GetLeafShapes(const Shape& shape);

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

  // Returns true if `shape` (which must be an array) with degenerate dimensions
  // (dimensions with bound 1).
  static bool HasDegenerateDimensions(const Shape& shape);

  // Drops any degenerate dimensions (i.e. dimensions of size 1)
  static Shape DropDegenerateDimensions(const Shape& shape);

  // Permutes the dimensions by the given permutation, so
  // return_value.dimensions[permutation[i]] = argument.dimensions[i].
  //
  // Postcondition: For any valid permutation,
  //
  //   !HasLayout(shape) ||
  //   TransposeIsBitcast(shape, PermuteDimensions(permutation, shape),
  //                      InversePermutation(permutation)).
  static Shape PermuteDimensions(absl::Span<const int64> permutation,
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

  // Return whether the given reshape instruction leaves the dimensions at the
  // given input indices unmodified, and returns their output indices.
  //
  // Example:
  //   input_dim_indices = {2, 3}
  //   input  shape = T[a, b, x, y, cd]
  //   output shape = T[ab, x, 1, y, c, d]
  //   return value = {1, 3}
  //
  // Precondition: input_dim_indices is sorted.
  static absl::optional<std::vector<int64>> ReshapeLeavesDimensionsUnmodified(
      const Shape& from_shape, const Shape& to_shape,
      absl::Span<const int64> input_dim_indices);

  // Returns whether a transpose from input_shape to output_shape with dimension
  // mapping "dimension_mapping" produces a result which is bit-wise identical
  // to its input and thus may be replaced with a bitcast.
  //
  // Precondition: Both input_shape and output_shape have explicit layouts.
  static bool TransposeIsBitcast(const Shape& input_shape,
                                 const Shape& output_shape,
                                 absl::Span<const int64> dimension_mapping);

  // Returns whether a reshape from "input_shape" to "output_shape" is a
  // bitcast.
  //
  // Precondition: Both input_shape and output_shape have explicit layouts.
  static bool ReshapeIsBitcast(const Shape& input_shape,
                               const Shape& output_shape);

  // Find a physical layout for 'output_shape' such that
  // ShapeUtil::ReshapeIsBitcast(input_shape, output_shape_with_layout) returns
  // true (where 'output_shape_with_layout' is 'output_shape' with the found
  // layout). The layout of 'input_shape' is kept fixed. Returns
  // 'output_shape_with_layout' if such a layout can be found, and an error
  // otherwise.
  static absl::optional<Shape> AlignLayouts(const Shape& input_shape,
                                            const Shape& output_shape);

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

  // Returns true if `dynamic_shape` has dimensions that are less-equal to the
  // "bounded_shape". Shapes must be arrays.
  static bool DynamicArrayShapeIsCompatible(const xla::Shape& dynamic_shape,
                                            const xla::Shape& bounded_shape);

  // Same as DynamicArrayShapeIsCompatible() but supports tuples.
  static bool DynamicShapeIsCompatible(const xla::Shape& dynamic_shape,
                                       const xla::Shape& bounded_shape);

  // Iterates through all the shape indexes, in minor to major order,
  // starting from the base indexes, incrementing by the incr steps, up to
  // count (index[i] < base[i] + count[i]), and calls the visitor_function
  // with the current index. The visitor_function visitor function should
  // return true if it wants to continue, or false otherwise.
  //
  // visitor_function must be a callable of type
  // StatusOr<bool>(absl::Span<int64>) or compatible.
  template <typename FnType>
  static Status ForEachIndexWithStatus(const Shape& shape,
                                       absl::Span<const int64> base,
                                       absl::Span<const int64> count,
                                       absl::Span<const int64> incr,
                                       const FnType& visitor_function) {
    return ForEachIndexInternal(shape, base, count, incr, visitor_function);
  }

  // Simple ergonomic wrapper around ShapeUtil::ForEachIndexWithStatus.
  struct IndexIterationSpace {
    std::vector<int64> index_base;
    std::vector<int64> index_count;
    std::vector<int64> index_incr;
  };

  template <typename FnTy>
  static Status ForEachIndexWithStatus(
      const Shape& shape, const IndexIterationSpace& iteration_space,
      FnTy&& function) {
    return ShapeUtil::ForEachIndexWithStatus(
        shape, iteration_space.index_base, iteration_space.index_count,
        iteration_space.index_incr, std::forward<FnTy>(function));
  }

  template <typename FnType>
  static void ForEachIndex(const Shape& shape, absl::Span<const int64> base,
                           absl::Span<const int64> count,
                           absl::Span<const int64> incr,
                           const FnType& visitor_function) {
    ForEachIndexWithStatus(shape, base, count, incr,
                           [&](absl::Span<const int64> indices) {
                             return StatusOr<bool>(visitor_function(indices));
                           })
        .IgnoreError();
  }

  // These convenience wrappers don't take `base`, `count` and `incr`
  // explicitly, but iterate over every element in `shape` instead.

  template <typename FnType>
  static Status ForEachIndexWithStatus(const Shape& shape,
                                       const FnType& visitor_function) {
    std::vector<int64> base(shape.dimensions_size());
    std::vector<int64> incr(shape.dimensions_size(), 1);
    return ForEachIndexWithStatus(shape, base,
                                  /*count=*/AsInt64Slice(shape.dimensions()),
                                  incr, visitor_function);
  }

  template <typename FnType>
  static void ForEachIndex(const Shape& shape, const FnType& visitor_function) {
    ForEachIndexWithStatus(shape, [&](absl::Span<const int64> indices) {
      return StatusOr<bool>(visitor_function(indices));
    }).IgnoreError();
  }

  // A parallel version of ForEachIndex(WithStatus). This can only be used if
  // the visitor_function is thread-safe and the order of iteration does not
  // matter.
  //
  // visitor_function must be a callable of type
  // void(Span<int64>) or compatible.
  template <typename FnType>
  static void ForEachIndexParallel(const Shape& shape,
                                   absl::Span<const int64> base,
                                   absl::Span<const int64> count,
                                   absl::Span<const int64> incr,
                                   const FnType& visitor_function) {
    // The parallel version of ForEachIndexInternal can never fail.
    CHECK(ForEachIndexInternal(
              shape, base, count, incr,
              [&visitor_function](
                  absl::Span<const int64> indexes) -> StatusOr<bool> {
                visitor_function(indexes);
                return true;
              },
              /*parallel=*/true)
              .ok());
  }

  // Compute a hash for `shape`.
  static size_t Hash(const Shape& shape);

  // About 0-2-1 transpose:
  //
  // If a shape can be viewed as three logical components 0-1-2 in the order of
  // major to minor, a 0-2-1-transpose changes the order of such logical
  // components to 0-2-1. We call the shape being transposed the input shape and
  // the transposed shape the output shape. The logical view of the input/output
  // shapes for the transpose are called the 0-1-2/0-2-1 shapes or the
  // normalized shapes. The original input/output shapes are called unnormalized
  // shapes.
  //
  // If `b` is a 0-2-1 transpose of `a` in 0-1-2, return the dimensions for the
  // normalized shape of `b` or the 0-2-1 shape.
  static absl::optional<std::vector<int64>> FindTranspose021(const Shape& a,
                                                             const Shape& b);

 private:
  // Validates the shape size is sane. This makes sure it's safe to do
  // calculations in int64 without overflowing.
  static Status ValidateShapeSize(const Shape& shape);

  // Validates all of the non-layout properties of the shape -- this is a helper
  // used by both the layout-optional and layout-required public method.
  static Status ValidateShapeWithOptionalLayoutInternal(const Shape& shape);

  template <typename FnType>
  static Status ForEachIndexInternal(const Shape& shape,
                                     absl::Span<const int64> base,
                                     absl::Span<const int64> count,
                                     absl::Span<const int64> incr,
                                     const FnType& visitor_function,
                                     bool parallel = false) {
    if (ShapeUtil::IsZeroElementArray(shape)) {
      return Status::OK();
    }
    CHECK_EQ(shape.rank(), base.size());
    CHECK_EQ(incr.size(), base.size());
    CHECK_EQ(count.size(), base.size());
    const int64 rank = LayoutUtil::MinorToMajor(shape).size();
    // Allows handling R0 arrays, such that the visitor function will be called
    // once with the proper empty indexes.
    int64 n = -1;
    std::vector<int64> indexes(base.begin(), base.end());
    const int kNumThreads = tensorflow::port::MaxParallelism();
    absl::optional<tensorflow::thread::ThreadPool> pool;
    if (parallel) {
      pool.emplace(tensorflow::Env::Default(), "foreach", kNumThreads);
    }

    tensorflow::mutex mu;
    Status status;  // Guarded by mu

    while (n < rank) {
      if (pool != absl::nullopt) {
        pool->Schedule([indexes, &visitor_function, &mu, &status] {
          StatusOr<bool> result = visitor_function(indexes);
          if (!result.ok()) {
            tensorflow::mutex_lock lock(mu);
            status = status.ok() ? result.status() : status;
          }
        });
      } else {
        TF_ASSIGN_OR_RETURN(bool should_continue, visitor_function(indexes));
        if (!should_continue) {
          break;
        }
      }
      // Increments dimensions in minor to major order.
      for (n = 0; n < rank; ++n) {
        int64 dim = LayoutUtil::Minor(shape.layout(), n);
        indexes[dim] += incr[dim];
        if (indexes[dim] < base[dim] + count[dim]) {
          break;
        }
        indexes[dim] = base[dim];
      }
    }

    // Waits for the scheduled work to complete.
    pool.reset();
    return status;
  }

  TF_DISALLOW_COPY_AND_ASSIGN(ShapeUtil);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_SHAPE_UTIL_H_
