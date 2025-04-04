/* Copyright 2017 The OpenXLA Authors.

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

#ifndef XLA_SHAPE_UTIL_H_
#define XLA_SHAPE_UTIL_H_

#include <cstdint>
#include <functional>
#include <initializer_list>
#include <iterator>
#include <numeric>
#include <optional>
#include <ostream>
#include <string>
#include <tuple>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"
#include "absl/functional/function_ref.h"
#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/layout_util.h"
#include "xla/overflow_util.h"
#include "xla/primitive_util.h"
#include "xla/printer.h"
#include "xla/shape.h"
#include "xla/tsl/platform/errors.h"
#include "xla/tsl/platform/logging.h"  // IWYU pragma: keep
#include "xla/tsl/platform/macros.h"
#include "xla/xla_data.pb.h"

namespace xla {

// A view into a ShapeIndex below, with the cheap/easy ability to consume the
// value at the front of the view.
//
// NB! ShapeIndexView does not own the memory backing the index array.
// The memory backing the index array should be owned by an object
// that lives longer than the ShapeIndexView instances pointing into
// it.
using ShapeIndexView = absl::Span<const int64_t>;

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
struct ShapeIndex : public absl::InlinedVector<int64_t, 2> {
  using InlinedVector::InlinedVector;
  TF_ATTRIBUTE_NOINLINE ShapeIndex() = default;

  explicit ShapeIndex(ShapeIndexView view)
      : ShapeIndex(view.begin(), view.end()) {}

  // push_front is O(n), but shapes don't usually have a ton of dimensions.
  void push_front(int64_t value) { insert(begin(), value); }
  void pop_front() { erase(begin()); }

  std::string ToString() const;
};

std::ostream& operator<<(std::ostream& out, const ShapeIndex& shape_index);

// Namespaced collection of (static) shape utilities.
//
// These are all effectively convenience functions for testing/tweaking proto
// properties, which do invariant checks before / after the operation.
class ShapeUtil {
 public:
  using DynamicSizeType = int32_t;

  // Data structure which describes the coordinates and the shape, of a tuple
  // shaped sub-shape.
  struct IndexedShape {
    IndexedShape() = default;
    IndexedShape(ShapeIndex index, Shape shape)
        : index(std::move(index)), shape(std::move(shape)) {}
    ShapeIndex index;
    Shape shape;
  };

  // Returns the product of the statically bound dimensions.
  template <bool kBoundedDynamicOk>
  static inline std::pair<int64_t, bool> ExtentProduct(const Shape& shape) {
    DCHECK(shape.IsArray()) << ShapeUtil::HumanString(shape);
    int64_t product = 1;
    bool any_overflows = false;
    for (int dim = 0; dim < shape.dimensions().size(); ++dim) {
      if constexpr (kBoundedDynamicOk) {
        if (shape.is_unbounded_dynamic_dimension(dim)) {
          continue;
        }
      } else {
        DCHECK(!shape.is_unbounded_dynamic_dimension(dim));
      }
      bool overflow;
      std::tie(product, overflow) =
          OverflowSafeMultiply(product, shape.dimensions(dim));
      any_overflows |= overflow;
    }
    return {product, any_overflows};
  }

  // Returns the product of the statically bound dimensions.
  static inline int64_t StaticExtentProduct(const Shape& shape) {
    auto [product, overflow] = ExtentProduct</*kBoundedDynamicOk=*/true>(shape);
    DCHECK(!overflow);
    return product;
  }

  // Returns the number of elements contained within the provided shape;
  // e.g. for rank 0 (scalars) the result is always 1.
  // Precondition: shape.IsArray()
  static inline int64_t ElementsIn(const Shape& shape) {
    auto [product, overflow] =
        ExtentProduct</*kBoundedDynamicOk=*/false>(shape);
    DCHECK(!overflow);
    return product;
  }

  // As ElementsIn(), but recurses through tuples.
  static int64_t ElementsInRecursive(const Shape& shape);

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
  static int64_t ByteSizeOf(const Shape& shape, int64_t pointer_size = -1);

  // Returns the number of bytes used to store the primitive_type.
  //
  // Precondition: primitive_type is an array type (otherwise crashes)
  static int64_t ByteSizeOfPrimitiveType(PrimitiveType primitive_type);

  // Returns the number of bytes required to store the tuple member pointers for
  // a allocation of shape. The `shape` must be a TUPLE shape, and
  // `pointer_size` must be larger than zero.
  static int64_t ByteSizeOfTupleIndexTable(const Shape& shape,
                                           int64_t pointer_size);

  // Returns the number of bytes required for the elements in an allocation of
  // `shape`, which must be an array shape. Shapes use a separate
  // memory location for each element, and so for these shapes,
  // `ByteSizeOf(shape) == ByteSizeOfElements(shape)`. This
  // size also includes padding if present in the layout.
  static int64_t ByteSizeOfElements(const Shape& shape);

  // Returns the size in bytes for the serialized form of this shape.
  // This serialized size includes the header of the serialized format, and so
  // should not be used for subshapes.  Use SerializedSizeOfData for that
  // purpose.
  static absl::StatusOr<int64_t> SerializedSize(const Shape& shape);

  // As above, but assumes the given ShapeProto is the result of
  // shape.ToProto().  This can be used to avoid converting the shape to a
  // protobuf multiple times.
  static absl::StatusOr<int64_t> SerializedSizeWithProto(
      const Shape& shape, const ShapeProto& proto);

  // Prints a human-readable string that represents the given shape, with or
  // without layout. e.g. "f32[42x12] {0, 1}" or "f32[64]".
  static void PrintHumanString(xla::Printer* printer, const Shape& shape);
  static void PrintHumanStringWithLayout(xla::Printer* printer,
                                         const Shape& shape);

  // As above, but for program shapes, prints a string for the form:
  //
  // (param_name: f32[42x12], ...) -> f32[24x42]
  static void PrintHumanString(xla::Printer* printer,
                               const ProgramShape& program_shape);

  // Returns a human-readable string that represents the given shape, with or
  // without layout. e.g. "f32[42x12] {0, 1}" or "f32[64]".
  static std::string HumanString(const Shape& shape);
  static std::string HumanStringWithLayout(const Shape& shape);

  // As above, but for program shapes, returns a string for the form:
  //
  // (param_name: f32[42x12], ...) -> f32[24x42]
  static std::string HumanString(const ProgramShape& program_shape);

  // Returns whether the LHS and RHS shapes have the same dimensions, ignoring
  // the unbounded dimension sizes; note: does not check element type.
  // Precondition: IsArray(lhs) && IsArray(rhs)
  static bool SameDimensions(const Shape& lhs, const Shape& rhs);

  // Returns whether the LHS and RHS shapes have the same rank; note: does
  // not check element type.
  // Precondition: IsArray(lhs) && IsArray(rhs)
  static bool SameRank(const Shape& lhs, const Shape& rhs);

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
    return primitive_util::HigherPrecisionType(a.element_type(),
                                               b.element_type());
  }

  // Returns true if the rank, dimension sizes, and element type are
  // identical. Layout is ignored. Tuple elements are compared recursively for
  // compatibility.
  static bool Compatible(const Shape& lhs, const Shape& rhs);

  // Returns true if the rank and dimension sizes are identical. Element type
  // and layout are ignored. Tuple elements are compared recursively for
  // compatibility.
  static bool CompatibleIgnoringElementType(const Shape& lhs, const Shape& rhs);

  // Returns true if the tuple tree shapes and leaf ranks are identical.
  // Leaf dimensions, element type, and layout are ignored. Tuple elements are
  // compared recursively for compatibility.
  static bool CompatibleKind(const Shape& lhs, const Shape& rhs);

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
  // 1. e.g., f32[2x1x1] has a true dimensionality of 1D, the other dimensions
  // are just fluff. Note that zero dimensions are included in the true
  // dimensionality, e.g., f32[3,0,1] has a true dimensionality of 2D.
  // Precondition: array_shape.IsArray().
  static int64_t TrueNumDimensions(const Shape& array_shape);

  static ProgramShape MakeProgramShape(std::initializer_list<Shape> parameters,
                                       Shape result);

  ////////////////////
  // Scalar-specific

  static bool IsScalar(const Shape& shape) {
    return shape.IsArray() && shape.dimensions().size() == 0;
  }
  static bool IsEffectiveScalar(const Shape& shape) {
    return shape.IsArray() && TrueNumDimensions(shape) == 0;
  }

  // Returns whether "shape" is a scalar (array) with the given element_type.
  static bool IsScalarWithElementType(const Shape& shape,
                                      PrimitiveType element_type);

  // Creates a `DimensionVector` by copying dimensions from a given shape.
  static DimensionVector CreateDimensionVectorFromShape(const Shape& shape);

  // Extracts the size of the shape's dimension at dimension number
  // GetDimensionNumber(dimension_number).
  static int64_t GetDimension(const Shape& shape, int64_t dimension_number);

  // Resolves a dimension number, supporting negative indexing.
  //
  // Negative indexing has similar semantics to Python. For an N-dimensional
  // array, dimension -1 is equivalent to dimension N-1, -2 is equivalent to
  // N-2, and so on.
  //
  // This function always returns a positive dimension number for any given
  // dimension_number (which itself can be negative).
  static int64_t GetDimensionNumber(const Shape& shape,
                                    int64_t dimension_number);

  // Returns a shape with the same dimensions as the original, but with the
  // element type changed to type.
  static Shape ChangeElementType(const Shape& original, PrimitiveType type);

  // Returns a shape with same dimensions but with all dimensions set to static.
  // If the shape has a layout, its dynamic_shape_metadata_prefix_bytes will be
  // set to zero.
  static Shape MakeStaticShape(const Shape& original);

  // Creates a tuple shape from a slice of element shapes within the tuple.
  static Shape MakeTupleShape(absl::Span<const Shape> shapes);
  static Shape MakeTupleShapeWithPtrs(absl::Span<const Shape* const> shapes);

  // Creates a tuple shape from a slice of element shapes within the tuple. If
  // only one shape is passed, returns that.
  static Shape MakeMaybeTupleShape(absl::Span<const Shape> shapes);

  // Creates an opaque shape. These are generally used for threading a context
  // into a custom operation.
  static Shape MakeOpaqueShape();

  // Creates a token shape. Values of this shape are used for ordering
  // side-effecting operations.
  static Shape MakeTokenShape();

  // Appends a shape to the given tuple.
  static void AppendShapeToTuple(const Shape& shape, Shape* tuple_shape);

  // Update a subshape of a tuple.
  static void UpdateTupleShape(const Shape& shape, int64_t index,
                               Shape* tuple_shape);

  // Update the dynamic dimension for a shape. This shape can be a nested tuple.
  static void UpdateDynamicDimension(Shape* shape, ShapeIndexView index,
                                     int64_t dim, bool is_dynamic);

  // Appends a major dimension to the shape with the given bound.
  static void AppendMajorDimension(int bound, Shape* shape);

  // Prepends a major dimension sized `bound` to the shape.
  static Shape PrependMajorDimension(int64_t bound, Shape shape);

  // Appends a minor dimension to the shape with the given bound.
  static void AppendMinorDimension(int bound, Shape* shape);

  // Copy the dynamic dimensions property from one shape to another.
  static void CopyDynamicDimensions(Shape* to, const Shape& from);

  // Returns true if the given dimension is effectively the most major dimension
  // of the shape taking into account any unit dimensions. Requires that the
  // shape has a layout.
  static bool IsEffectivelyMostMajorDimension(const Shape& shape,
                                              int64_t dimension);

  // Returns an empty tuple shape. Can be used as a sentinel Shape value.
  static Shape MakeNil() { return MakeTupleShape({}); }

  // Checks whether the shape is initialized.
  static bool IsInitialized(const Shape& shape) {
    return shape.element_type() != PRIMITIVE_TYPE_INVALID;
  }

  // Constructs a new shape with the given element type and sequence of
  // dimensions.
  static Shape MakeShape(PrimitiveType element_type,
                         absl::Span<const int64_t> dimensions);

  // Make a scalar shape with given primitive type.
  static Shape MakeScalarShape(PrimitiveType element_type);

  // Constructs a new shape with the given element type and sequence of
  // potentially dynamic dimensions. The argument 'dynamic_dimensions' indicates
  // with a true value that the respective dimension is dynamic. If the
  // dimension is dynamic then the respective value in 'dimension' is an upper
  // bound on the dimension size. 'dimensions' and 'dynamic_dimensions' must be
  // the same size.
  static Shape MakeShape(PrimitiveType element_type,
                         absl::Span<const int64_t> dimensions,
                         const std::vector<bool>& dynamic_dimensions);

  // Constructs a new shape with the given element type and sequence of
  // dimensions. Method checks if the element type is valid, the shape's
  // size fits in std::numeric_limits<int64_t>::max(), and dynamic size is not
  // marked static.
  static absl::StatusOr<Shape> MakeValidatedShape(
      PrimitiveType element_type, absl::Span<const int64_t> dimensions);
  static absl::StatusOr<Shape> MakeValidatedShape(
      PrimitiveType element_type, absl::Span<const int64_t> dimensions,
      const std::vector<bool>& dynamic_dimensions);

  // Creates a Shape with element type corresponding to T and the given
  // dimensions
  template <typename T>
  static Shape MakeShapeWithType(absl::Span<const int64_t> dimensions) {
    return ShapeUtil::MakeShape(primitive_util::NativeToPrimitiveType<T>(),
                                dimensions);
  }

  // Constructs a new dense array shape with the given minor_to_major order in
  // its Layout. Returns a value shape such that shape.has_layout().
  static Shape MakeShapeWithDenseLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dimensions,
      absl::Span<const int64_t> minor_to_major,
      absl::Span<const Tile> tiles = {},
      int64_t tail_padding_alignment_in_elements = 1,
      int64_t element_size_in_bits = 0, int64_t memory_space = 0,
      absl::Span<const SplitConfig> split_configs = {});

  // Constructs a new sparse array shape with the given minor_to_major order and
  // dim_level_types in its Layout. Returns a value shape such that
  // shape.has_layout().
  static Shape MakeShapeWithSparseLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dimensions,
      absl::Span<const int64_t> minor_to_major,
      absl::Span<const DimLevelType> dim_level_types,
      absl::Span<const bool> dim_unique = {},
      absl::Span<const bool> dim_ordered = {},
      PrimitiveType index_primitive_type = PRIMITIVE_TYPE_INVALID,
      PrimitiveType pointer_primitive_type = PRIMITIVE_TYPE_INVALID,
      int64_t tail_padding_alignment_in_elements = 1,
      int64_t element_size_in_bits = 0, int64_t memory_space = 0,
      std::optional<Shape> physical_shape = std::nullopt);

  // Constructs a new shape with the given dimension `dim` as the most major
  // dimension in the layout. If the shape does not have a layout, assumes a
  // default layout. If the shape is a tuple, apply this to all the leaf shapes
  // of the tuple.
  static Shape MoveDimToMajor(const Shape& shape, int64_t dim);

  // Returns the same shape except with all dimensions set to be static.
  static Shape MakeShapeWithStaticDimensions(const Shape& shape);

  // Constructs a new shape with major-first layout (i.e. {n, n-1, ..., 0}).
  static Shape MakeShapeWithDescendingLayout(
      PrimitiveType element_type, absl::Span<const int64_t> dimensions);

  // Returns a new Shape based on the given Shape with low-dimension-major
  // layout (i.e. {n, n-1, ..., 0}, like Fortran), and with the dimensions
  // rearranged so that it has the same in-memory layout as the given shape.
  //
  // For example, transforms f32[B,H,W,C]{0,3,2,1} to f32[H,W,C,B]{3,2,1,0}.
  static Shape MakeShapeWithDescendingLayoutAndSamePhysicalLayout(
      const Shape& shape);

  // As MakeShape, but the object to write to is passed in.
  // Precondition:
  //   - if element_type is a non-array type, dimensions must be empty.
  //   - shape must not be null.
  static absl::Status PopulateShape(PrimitiveType element_type,
                                    absl::Span<const int64_t> dimensions,
                                    Shape* shape);

  // Validates that the provided shape satisfies invariants.
  static absl::Status ValidateShape(const Shape& shape);

  // Validates the provided shape satisfies invariants, except those that
  // pertain to layout.
  //
  // Layout is optional for client-provided shapes, so that the compiler may
  // determine and assign an optimized layout.
  static absl::Status ValidateShapeWithOptionalLayout(const Shape& shape);

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

  // Returns whether the shape is a tuple with at least one element which is
  // also a tuple.
  static bool IsNestedTuple(const Shape& shape);

  // Returns true if shape is an empty tuple.
  static bool IsEmptyTuple(const Shape& shape);

  // Returns the number of elements in the given tuple shape.
  // Precondition: IsTuple(shape)
  static int64_t TupleElementCount(const Shape& shape);

  // Returns the tuple element shape at given index.
  // Precondition: IsTuple(shape) && TupleElementCount(shape) > index
  static const Shape& GetTupleElementShape(const Shape& shape, int64_t index);

  // Returns the number of elements, recursively, in the given shape.
  static int64_t SubshapeCount(const Shape& shape);

  // Slices tuple elements in the range [start, limit) and returns a new tuple
  // shape. E.g. a tuple like (f32, s32, u32) would slice via 1,3 to (s32, u32).
  static Shape SliceTuple(const Shape& tuple, int64_t start, int64_t limit);

  // Returns the shape of the real/imaginary components of the given complex
  // shape.
  static Shape ComplexComponentShape(const Shape& complex_shape);

  // Returns true if the given shape has a subshape at the given index.
  static bool IndexIsValid(const Shape& shape, ShapeIndexView index);

  // GetSubshape and GetMutableSubshape return a particular nested Shape within
  // the given Shape argument. The non-Try variants check fail if index is
  // invalid.
  static const Shape& GetSubshape(const Shape& shape, ShapeIndexView index);

  // Faster version for one index.
  static const Shape& GetSubshapeOneIndex(const Shape& shape, int64_t index);

  static absl::StatusOr<const Shape*> TryGetSubshape(const Shape& shape,
                                                     ShapeIndexView index);
  static Shape* GetMutableSubshape(Shape* shape, ShapeIndexView index);

  // Returns whether the given index in the given shape is a leaf element of the
  // shape.
  static bool IsLeafIndex(const Shape& shape, const ShapeIndex& index);

  // Returns the number of leaves in the shape.
  static int64_t GetLeafCount(const Shape& shape);
  static int64_t GetLeafCountTuple(const Shape& shape);

  // Retrieves all the leaf shapes and their indexes, in the order walked by
  // the ForEachSubshape() API.
  static std::vector<IndexedShape> GetLeafShapes(const Shape& shape);

  // Calls the given visitor function for each subshape of the given shape.
  // Subshapes are visited in DFS pre-order starting with the entire shape
  // (index {}).
  //
  // The visitor function must have the signature
  //
  //   void fn(const Shape& subshape, const ShapeIndex& index), or
  //   void fn(Shape* subshape, const ShapeIndex& index) (mutable version)
  template <typename Fn>
  static void ForEachSubshape(const Shape& shape, Fn&& fn) {
    ForEachSubshapeWithStatus(shape, [&](const Shape& subshape,
                                         const ShapeIndex& index) {
      fn(subshape, index);
      return absl::OkStatus();
    }).IgnoreError();
  }
  template <typename Fn>
  static void ForEachMutableSubshape(Shape* shape, Fn&& fn) {
    ForEachMutableSubshapeWithStatus(shape, [&](Shape* subshape,
                                                const ShapeIndex& index) {
      fn(subshape, index);
      return absl::OkStatus();
    }).IgnoreError();
  }

  // Calls the given visitor function for each leaf subshape of the given shape.
  // Subshapes are visited in DFS pre-order starting with the entire shape
  // (index {}).
  //
  // The visitor function must have the signature
  //
  //   absl::Status fn(const Shape& subshape, const ShapeIndex& index)
  //   void fn(Shape* subshape, const ShapeIndex& index) (mutable version)
  template <typename Fn>
  static absl::Status ForEachLeafShapeWithStatus(const Shape& shape, Fn&& fn) {
    return ForEachSubshapeWithStatus(
        shape, [&](const Shape& subshape, const ShapeIndex& index) {
          if (IsLeafIndex(shape, index)) {
            TF_RETURN_IF_ERROR(fn(subshape, index));
          }
          return absl::OkStatus();
        });
  }
  template <typename Fn>
  static absl::Status ForEachMutableLeafShapeWithStatus(Shape* shape, Fn&& fn) {
    return ForEachMutableSubshapeWithStatus(
        shape, [&](Shape* subshape, const ShapeIndex& index) {
          if (IsLeafIndex(*shape, index)) {
            TF_RETURN_IF_ERROR(fn(subshape, index));
          }
          return absl::OkStatus();
        });
  }

  // Calls the given visitor function for each leaf subshape of the given shape.
  // Subshapes are visited in DFS pre-order starting with the entire shape
  // (index {}).
  //
  // The visitor function must have the signature
  //   void fn(const Shape& subshape, const ShapeIndex& index)
  //   void fn(Shape* subshape, const ShapeIndex& index) (mutable version)
  template <typename Fn>
  static void ForEachLeafShape(const Shape& shape, Fn&& fn) {
    ForEachLeafShapeWithStatus(shape, [&](const Shape& subshape,
                                          const ShapeIndex& index) {
      fn(subshape, index);
      return absl::OkStatus();
    }).IgnoreError();
  }
  template <typename Fn>
  static void ForEachMutableLeafShape(Shape* shape, Fn&& fn) {
    ForEachMutableLeafShapeWithStatus(shape, [&](Shape* subshape,
                                                 const ShapeIndex& index) {
      fn(subshape, index);
      return absl::OkStatus();
    }).IgnoreError();
  }

  // Variants of ForEach(Mutable)Subshape which propagate absl::Status from the
  // visitor function.
  //
  // Visitor function must have the signature
  //
  //   absl::Status fn(const Shape& subshape, const ShapeIndex& index), or
  //   absl::Status fn(Shape* subshape, const ShapeIndex& index) (mutable
  //   version)
  //
  template <typename Fn>
  static absl::Status ForEachSubshapeWithStatus(const Shape& shape, Fn&& fn) {
    return ForEachMutableSubshapeWithStatus(
        const_cast<Shape*>(&shape),
        [&](Shape* subshape, const ShapeIndex& index) -> absl::Status {
          return fn(*const_cast<const Shape*>(subshape), index);
        });
  }
  template <typename Fn>
  static absl::Status ForEachMutableSubshapeWithStatus(Shape* shape, Fn&& fn) {
    ShapeIndex index;
    return ForEachMutableSubshapeWithStatusHelper(shape, fn, &index);
  }

  // Calls the given visitor function for each subshape of the given shape.
  // Subshapes are visited in DFS post-order starting with the entire shape
  // (index {}).
  //
  // The visitor function must have the signature
  //
  //   void fn(const Shape& subshape, const ShapeIndex& index), or
  //   void fn(Shape* subshape, const ShapeIndex& index) (mutable version)
  template <typename Fn>
  static void ForEachSubshapePostOrder(const Shape& shape, Fn&& fn) {
    ForEachSubshapePostOrderWithStatus(shape, [&](const Shape& subshape,
                                                  const ShapeIndex& index) {
      fn(subshape, index);
      return absl::OkStatus();
    }).IgnoreError();
  }
  template <typename Fn>
  static void ForEachMutableSubshapePostOrder(Shape* shape, Fn&& fn) {
    ForEachMutableSubshapePostOrderWithStatus(
        shape,
        [&](Shape* subshape, const ShapeIndex& index) {
          fn(subshape, index);
          return absl::OkStatus();
        })
        .IgnoreError();
  }

  // Variants of ForEach(Mutable)SubshapePostOrder which propagate absl::Status
  // from the visitor function.
  //
  // Visitor function must have the signature
  //
  //   absl::Status fn(const Shape& subshape, const ShapeIndex& index), or
  //   absl::Status fn(Shape* subshape, const ShapeIndex& index) (mutable
  //   version)
  //
  template <typename Fn>
  static absl::Status ForEachSubshapePostOrderWithStatus(const Shape& shape,
                                                         Fn&& fn) {
    return ForEachMutableSubshapePostOrderWithStatus(
        const_cast<Shape*>(&shape),
        [&](Shape* subshape, const ShapeIndex& index) -> absl::Status {
          return fn(*const_cast<const Shape*>(subshape), index);
        });
  }
  template <typename Fn>
  static absl::Status ForEachMutableSubshapePostOrderWithStatus(Shape* shape,
                                                                Fn&& fn) {
    ShapeIndex index;
    return ForEachMutableSubshapePostOrderWithStatusHelper(shape, fn, &index);
  }

  // Returns true if `shape` (which must be an array) with degenerate dimensions
  // (dimensions with bound 1).
  static bool HasDegenerateDimensions(const Shape& shape);

  // Extracts the packing factor for a 1D interleaved array based on the layout.
  // For example, bf16[1024]{0:T(1024)(128)(2,1)} -> 2
  static absl::StatusOr<int64_t> PackedFactorFor1DInterleavedArray(
      const Shape& shape);

  // Drops any degenerate dimensions (i.e. dimensions of size 1)
  static Shape DropDegenerateDimensions(const Shape& shape);

  // Permutes the dimensions by the given permutation, so
  // return_value.dimensions[i] = argument.dimensions[permutation[i]].
  //
  // Postcondition: For any valid permutation,
  //
  //   !HasLayout(shape) ||
  //   TransposeIsBitcast(shape, PermuteDimensions(permutation, shape),
  //                      permutation).
  static Shape PermuteDimensions(absl::Span<const int64_t> permutation,
                                 const Shape& shape);

  // Describes how we can go from shape A to shape B by inserting degenerate
  // 1-sized dimensions in `added_dimensions` and removing degenerate 1-sized
  // dimensions from B in `removed_dimensions`.
  //
  // Only exists if shapes A and B only differ by degenerate dimensions.
  struct ShapeEqualityDescriptor {
    std::vector<int64_t> deleted_dimensions;
    std::vector<int64_t> inserted_dimensions;
  };

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
  static std::optional<ShapeEqualityDescriptor>
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
  static std::vector<std::pair<int64_t, int64_t>> DimensionsUnmodifiedByReshape(
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
  static std::optional<std::vector<int64_t>> ReshapeLeavesDimensionsUnmodified(
      const Shape& from_shape, const Shape& to_shape,
      absl::Span<const int64_t> input_dim_indices);

  // Returns whether a transpose from input_shape to output_shape with dimension
  // mapping "dimension_mapping" produces a result which is bit-wise identical
  // to its input and thus may be replaced with a bitcast.
  //
  // Precondition: Both input_shape and output_shape have explicit layouts.
  static bool TransposeIsBitcast(const Shape& input_shape,
                                 const Shape& output_shape,
                                 absl::Span<const int64_t> dimension_mapping,
                                 bool ignore_element_type = false);

  // Returns whether a reshape from `input_shape` to `output_shape` is a
  // bitcast, when minor_to_major in layout is considered.
  //
  // Precondition: Both input_shape and output_shape have explicit layouts.
  static bool ReshapeIsBitcast(const Shape& input_shape,
                               const Shape& output_shape,
                               bool ignore_element_type = false);

  // Returns whether there is a bitcasting reshape or transpose from `a` to `b`.
  //
  // Precondition: Both input_shape and output_shape have explicit layouts.
  static bool IsReshapeOrTransposeBitcast(const Shape& a, const Shape& b,
                                          bool ignore_element_type = false);

  // If the given bitcast is a transpose, deduce and return `dimensions`
  // attribute of such a transpose. Otherwise, return std::nullopt.
  static std::optional<std::vector<int64_t>>
  DeduceTransposeDimensionsForBitcast(const Shape& input_shape,
                                      const Shape& output_shape);

  // This means that the bitcast can be decomposed to a single reshape.
  struct BitcastDecompositionReshape {};

  // This means that the bitcast can be decomposed to a single transpose.
  struct BitcastDecompositionTranspose {
    std::vector<int64_t> transpose_dims;
  };

  // Every bitcast from A to B can be represented as a sequence of:
  // 1) Transpose to a normalized layout of A
  // 2) Reshape to a normalized layout of B
  // 3) Transpose from (2) to B
  //
  // All members are always set, even if they correspond to an identity
  // operation.
  //
  // Note: Some bitcasts can be converted to a single transpose or reshape,
  // using other methods.
  struct BitcastDecompositionTrt {
    std::vector<int64_t> transpose1_dims;
    // Has a normalized layout.
    Shape transpose1_shape;
    // Has a normalized layout.
    Shape reshape_shape;
    std::vector<int64_t> transpose2_dims;

    bool IsTranspose1Identity() const;
    bool IsTranspose2Identity() const;
  };

  // A variant type holding one of the possible bitcast decompositions.
  using BitcastDecomposition =
      std::variant<BitcastDecompositionReshape, BitcastDecompositionTranspose,
                   BitcastDecompositionTrt>;

  // Decomposes a bitcast to a sequence of transpose, reshape, transpose.
  //
  // See the comment on BitcastDecompositionTrt.
  static BitcastDecompositionTrt DecomposeBitcastToTrt(
      const Shape& input_shape, const Shape& output_shape);

  // Decomposes a bitcast to one of the possible decompositions.
  static BitcastDecomposition DecomposeBitcast(const Shape& input_shape,
                                               const Shape& output_shape);

  // Find a physical layout for 'output_shape' such that
  // ShapeUtil::ReshapeIsBitcast(input_shape, output_shape_with_layout) returns
  // true (where 'output_shape_with_layout' is 'output_shape' with the found
  // layout). The layout of 'input_shape' is kept fixed. Returns
  // 'output_shape_with_layout' if such a layout can be found, and an error
  // otherwise.
  static std::optional<Shape> AlignLayouts(const Shape& input_shape,
                                           const Shape& output_shape);

  // Returns a shape with the given logical dimensions reordered, updating the
  // layout so that physical dimensions are preserved.
  static Shape ReorderLogicalDimensions(const Shape& shape,
                                        absl::Span<const int64_t> permutation);

  // Returns a shape with the given dimension deleted.
  // For example:
  // • `DeleteDimension(1, T[m, n, k]) = T[m, k]`
  static Shape DeleteDimension(int64_t dim_to_delete, Shape shape);

  // Returns a shape with dimensions in `to_drop` dropped.
  static Shape DeleteDimensions(absl::Span<int64_t const> dims_to_delete,
                                Shape shape);

  // Returns a shape with all the dimensions of the input shape for which `p`
  // returns true.
  // For examples:
  // • `FilterDimensions((< 2), T[m, n, k]) = T[m, n]`
  // • `FilterDimensions(is_even_number, T[m, n, k]) = T[m, k]`
  static Shape FilterDimensions(absl::FunctionRef<bool(int64_t)> p,
                                Shape shape);

  // Returns true if `dynamic_shape` has dimensions that are less-equal to the
  // "bounded_shape". Shapes must be arrays.
  static bool DynamicArrayShapeIsCompatible(const xla::Shape& dynamic_shape,
                                            const xla::Shape& bounded_shape);

  // Same as DynamicArrayShapeIsCompatible() but supports tuples.
  static bool DynamicShapeIsCompatible(const xla::Shape& dynamic_shape,
                                       const xla::Shape& bounded_shape);

  using ForEachVisitorFunction =
      absl::FunctionRef<absl::StatusOr<bool>(absl::Span<const int64_t>)>;

  using ForEachVisitorFunctionNoStatus =
      absl::FunctionRef<bool(absl::Span<const int64_t>)>;

  // Iterates through all the shape indexes, in minor to major order,
  // starting from the base indexes, incrementing by the incr steps, up to
  // count (index[i] < base[i] + count[i]), and calls the visitor_function
  // with the current index. The visitor_function visitor function should
  // return true if it wants to continue, or false otherwise.
  static absl::Status ForEachIndexWithStatus(
      const Shape& shape, absl::Span<const int64_t> base,
      absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
      const ForEachVisitorFunction& visitor_function);

  // Simple ergonomic wrapper around ShapeUtil::ForEachIndexWithStatus.
  struct IndexIterationSpace {
    std::vector<int64_t> index_base;
    std::vector<int64_t> index_count;
    std::vector<int64_t> index_incr;
  };

  template <typename FnTy>
  static absl::Status ForEachIndexWithStatus(
      const Shape& shape, const IndexIterationSpace& iteration_space,
      FnTy&& function) {
    return ShapeUtil::ForEachIndexWithStatus(
        shape, iteration_space.index_base, iteration_space.index_count,
        iteration_space.index_incr, std::forward<FnTy>(function));
  }

  static void ForEachIndex(const Shape& shape, absl::Span<const int64_t> base,
                           absl::Span<const int64_t> count,
                           absl::Span<const int64_t> incr,
                           const ForEachVisitorFunction& visitor_function);

  static void ForEachIndexNoStatus(
      const Shape& shape, absl::Span<const int64_t> base,
      absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
      const ForEachVisitorFunctionNoStatus& visitor_function);

  // These convenience wrappers don't take `base`, `count` and `incr`
  // explicitly, but iterate over every element in `shape` instead.

  static absl::Status ForEachIndexWithStatus(
      const Shape& shape, const ForEachVisitorFunction& visitor_function) {
    std::vector<int64_t> base(shape.dimensions().size());
    std::vector<int64_t> incr(shape.dimensions().size(), 1);
    return ForEachIndexWithStatus(shape, base,
                                  /*count=*/shape.dimensions(), incr,
                                  visitor_function);
  }

  static void ForEachIndexNoStatus(
      const Shape& shape,
      const ForEachVisitorFunctionNoStatus& visitor_function) {
    std::vector<int64_t> base(shape.dimensions().size());
    std::vector<int64_t> incr(shape.dimensions().size(), 1);
    ForEachIndexNoStatus(shape, base,
                         /*count=*/shape.dimensions(), incr, visitor_function);
  }

  static void ForEachIndex(const Shape& shape,
                           const ForEachVisitorFunction& visitor_function) {
    ForEachIndexWithStatus(shape, [&](absl::Span<const int64_t> indices) {
      return absl::StatusOr<bool>(visitor_function(indices));
    }).IgnoreError();
  }

  using ForEachParallelVisitorFunction =
      absl::FunctionRef<absl::StatusOr<bool>(absl::Span<const int64_t>, int)>;

  // A parallel version of ForEachIndex(WithStatus). This can only be used if
  // the visitor_function is thread-safe and the order of iteration does not
  // matter.
  //
  // Please use GetForEachIndexParallelThreadCount() to get the number of
  // threads in the threadpool of ForEachIndexParallel*. This will not change
  // during the runtime of the process. Please DO NOT use
  // tsl::port::MaxParallelism() for this purpose, as it may change.
  static void ForEachIndexParallel(
      const Shape& shape, absl::Span<const int64_t> base,
      absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
      const ForEachParallelVisitorFunction& visitor_function);

  // Returns the number of threads in the threadpool of ForEachIndexParallel*.
  static int GetForEachIndexParallelThreadCount();

  static absl::Status ForEachIndexParallelWithStatus(
      const Shape& shape, absl::Span<const int64_t> base,
      absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
      const ForEachParallelVisitorFunction& visitor_function);

  // Convenience wrapper which doesn't take `base`, `count` and `incr`
  // explicitly, but iterates over every element in `shape` instead.
  static void ForEachIndexParallel(
      const Shape& shape,
      const ForEachParallelVisitorFunction& visitor_function);

  static absl::Status ForEachIndexParallelWithStatus(
      const Shape& shape,
      const ForEachParallelVisitorFunction& visitor_function);

  // Strips device-specific information, namely tiling and memory-space
  // information, from a shape.
  static Shape DeviceShapeToHostShape(Shape s);

  // Returns true iff element type of shape `from` can be safely upcasted to
  // element type of shape `to`.
  static bool ElementCanUpcast(const Shape& from, const Shape& to);

  // Computes byte strides of an array shape `shape`. `shape` must have a
  // layout. Ignores tiling. `strides` must have size equal to the number of
  // dimensions of `shape`.
  static absl::Status ByteStrides(const Shape& shape,
                                  absl::Span<int64_t> strides);
  // Same as above but returns the stride array, or std::nullopt if error.
  static std::optional<absl::InlinedVector<int64_t, 4>> ByteStrides(
      const Shape& shape);

  // Returns the array size in bytes (layout/tiling required), all paddings are
  // included.
  static int64_t ArraySize(const Shape& shape);

  // Returns the size of array data in bytes, ignoring the trailing padding
  // due to the tiling requirement.
  static int64_t ArrayDataSize(const Shape& shape);

  // Updates element_size_in_bits on each subshape's layout. If
  // 'pack_subbyte_types' is true, sets the element size to the dtype bitwidth
  // for subbyte types (S4, U4, etc) and 0 for non-subbyte types, which
  // indicates that for arrays of subbyte types, multiple elements are packed in
  // a single byte. If 'pack_subbyte_types' is false, sets the element size to 0
  // for all types.
  static void UpdateElementSizeInBits(Shape* s, bool pack_subbyte_types);

  // Recursively flattens a tuple shape into a vector of subshapes.
  static void FlattenTupleShape(const Shape& shape,
                                std::vector<const Shape*>& flattened);
  static std::vector<const Shape*> FlattenTupleShape(const Shape& shape);

 private:
  // Fills *shape ignoring dynamic dimensions. Returns true on success.
  // This populates the following fields in the shape:
  // - sets shape->element_type to element_type,
  // - sets shape->dimensions to dimensions,
  // - sets shape->layout.minor_to_major to [ndims - 1, ndims - 2, ..., 0]
  //   where ndims is the size of dimensions.
  // REQUIRES: *shape is empty.
  [[nodiscard]] static bool FillNewShape(PrimitiveType element_type,
                                         absl::Span<const int64_t> dimensions,
                                         Shape* shape);

  // Helper for ForEachSubshape which visits the subshapes of the given shape in
  // DFS pre-order starting with the index.
  template <typename Fn>
  static absl::Status ForEachMutableSubshapeWithStatusHelper(
      Shape* shape, Fn&& fn, ShapeIndex* index) {
    TF_RETURN_IF_ERROR(fn(shape, *index));
    if (shape->IsTuple()) {
      for (int64_t i = 0; i < ShapeUtil::TupleElementCount(*shape); ++i) {
        index->push_back(i);
        TF_RETURN_IF_ERROR(ForEachMutableSubshapeWithStatusHelper(
            shape->mutable_tuple_shapes(i), fn, index));
        index->pop_back();
      }
    }
    return absl::OkStatus();
  }

  // Helper for ForEachSubshapePost which visits the subshapes of the given
  // shape in DFS post-order.
  template <typename Fn>
  static absl::Status ForEachMutableSubshapePostOrderWithStatusHelper(
      Shape* shape, Fn&& fn, ShapeIndex* index) {
    if (shape->IsTuple()) {
      for (int64_t i = 0; i < ShapeUtil::TupleElementCount(*shape); ++i) {
        index->push_back(i);
        TF_RETURN_IF_ERROR(ForEachMutableSubshapePostOrderWithStatusHelper(
            shape->mutable_tuple_shapes(i), fn, index));
        index->pop_back();
      }
    }
    TF_RETURN_IF_ERROR(fn(shape, *index));
    return absl::OkStatus();
  }

  // Keeps track of the iteration state for the ForEach...Internal routines
  struct ForEachState {
    ForEachState(const Shape& s, absl::Span<const int64_t> b,
                 absl::Span<const int64_t> c, absl::Span<const int64_t> i);
    inline ~ForEachState() = default;

    const Shape& shape;
    // Pointers to arrays of the passed-in spans
    const int64_t* const base;
    const int64_t* const count;
    const int64_t* const incr;

    const int64_t* const minor_to_major;  // Base of s's minor_to_major array
    const int64_t rank;

    std::vector<int64_t> indexes;  // The mutable set of indices we go through
    int64_t* indexes_ptr;          // Points into "indexes"
    absl::Span<const int64_t> indexes_span;  // Pre-formed span of "indexes"

    int64_t IncrementDim();
    bool IsZeroElementArray() const;

    // Returns the number of visited elements assuming that the iteration will
    // not be interrupted.
    int64_t CalculateNumSteps() const;
  };

  static absl::Status ForEachIndexInternal(
      const Shape& shape, absl::Span<const int64_t> base,
      absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
      const ForEachVisitorFunction& visitor_function);

  static void ForEachIndexInternalNoStatus(
      const Shape& shape, absl::Span<const int64_t> base,
      absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
      const ForEachVisitorFunctionNoStatus& visitor_function);

  static absl::Status ForEachIndexInternalParallel(
      const Shape& shape, absl::Span<const int64_t> base,
      absl::Span<const int64_t> count, absl::Span<const int64_t> incr,
      const ForEachParallelVisitorFunction& visitor_function);

  ShapeUtil(const ShapeUtil&) = delete;
  ShapeUtil& operator=(const ShapeUtil&) = delete;
};

inline ShapeUtil::ForEachState::ForEachState(const Shape& s,
                                             absl::Span<const int64_t> b,
                                             absl::Span<const int64_t> c,
                                             absl::Span<const int64_t> i)
    : shape(s),
      base(b.data()),
      count(c.data()),
      incr(i.data()),
      minor_to_major(shape.layout().minor_to_major().data()),
      rank(LayoutUtil::MinorToMajor(shape).size()),
      indexes(b.begin(), b.end()),
      indexes_ptr((rank == 0) ? nullptr : indexes.data()),
      indexes_span(indexes) {
  CHECK_EQ(shape.dimensions().size(), b.size());
  CHECK_EQ(i.size(), b.size());
  CHECK_EQ(c.size(), b.size());
}

inline int64_t ShapeUtil::ForEachState::IncrementDim() {
  int64_t n;
  for (n = 0; n < rank; ++n) {
    int64_t dim = minor_to_major[n];
    indexes_ptr[dim] += incr[dim];
    if (indexes_ptr[dim] < base[dim] + count[dim]) {
      break;
    }
    indexes_ptr[dim] = base[dim];
  }
  return n;
}

inline bool ShapeUtil::ForEachState::IsZeroElementArray() const {
  return ShapeUtil::IsZeroElementArray(shape);
}

}  // namespace xla

#endif  // XLA_SHAPE_UTIL_H_
