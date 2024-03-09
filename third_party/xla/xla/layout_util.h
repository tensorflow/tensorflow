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

// Utility functions related to layouts of Shapes.

#ifndef XLA_LAYOUT_UTIL_H_
#define XLA_LAYOUT_UTIL_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/types/span.h"
#include "xla/layout.h"
#include "xla/printer.h"
#include "xla/shape.h"
#include "xla/status.h"
#include "xla/xla_data.pb.h"
#include "tsl/platform/logging.h"  // IWYU pragma: keep

namespace xla {

// Namespaced collection of (static) Layout utilities.
class LayoutUtil {
 public:
  LayoutUtil(const LayoutUtil&) = delete;
  LayoutUtil& operator=(const LayoutUtil&) = delete;

  // Creates a layout with the given minor-to-major dimension order. (This is a
  // convenience function for protobuf construction.)
  static Layout MakeLayout(
      absl::Span<const int64_t> minor_to_major,
      absl::Span<const DimLevelType> dim_level_types = {},
      absl::Span<const bool> dim_unique = {},
      absl::Span<const bool> dim_ordered = {},
      absl::Span<const Tile> tiles = {},
      int64_t tail_padding_alignment_in_elements = 1,
      PrimitiveType index_primitive_type = PRIMITIVE_TYPE_INVALID,
      PrimitiveType pointer_primitive_type = PRIMITIVE_TYPE_INVALID,
      int64_t element_size_in_bits = 0, int64_t memory_space = 0,
      absl::Span<const SplitConfig> split_configs = {},
      std::optional<Shape> physical_shape = std::nullopt,
      int64_t dynamic_shape_metadata_prefix_bytes = 0);

  // Similar to MakeLayout, but take indices in reverse order.
  static Layout MakeLayoutFromMajorToMinor(
      absl::Span<const int64_t> major_to_minor);

  // Returns a layout with descending ((i.e. {n-1, n-2, ... 0}) minor-to-major
  // dimensions.
  static Layout MakeDescendingLayout(int64_t rank);

  // Returns a layout with ascending ((i.e. {0, 1, ... n-1}) minor-to-major
  // dimensions.
  static Layout MakeAscendingLayout(int64_t rank);

  // Returns default layout for the given shape.
  static Layout GetDefaultLayoutForShape(const Shape& shape);

  // Helper functions that create default layouts for various ranks.
  static Layout GetDefaultLayoutForRank(int64_t rank);
  static Layout GetDefaultLayoutForR2();
  static Layout GetDefaultLayoutForR3();
  static Layout GetDefaultLayoutForR4();

  // Sets the default layout on the Shape.
  static void SetToDefaultLayout(Shape* shape);

  // Returns a shape with the same dimensions as `shape` but with the default
  // layout.
  static Shape GetWithDefaultLayout(const Shape& shape);

  // Sets the layouts of all Shapes within the given ProgramShape to the
  // default.
  static void SetToDefaultLayout(ProgramShape* program_shape);

  // Validates that the layout within the given shape is correct. The check
  // is performed for all subshapes as well. If missing layouts are allowed
  // the check does not fail on array shapes without layouts.
  static Status ValidateLayoutInShape(const Shape& shape,
                                      bool allow_missing_layouts = false);

  // Validates that the provided layout satisfies invariants for the given
  // shape.
  static Status ValidateLayoutForShape(const Layout& layout,
                                       const Shape& shape);

  // Clears the layout in the given Shape. After this function is called,
  // HasLayout will return false for the shape.
  static void ClearLayout(Shape* shape);

  // Clears the layout on all Shapes within the given ProgramShape.
  static void ClearLayout(ProgramShape* program_shape);

  // Clears the tiling fields from the shape and/or all of its subshapes.
  static void ClearTiles(Shape* shape);

  // Returns whether the given Shape is an array and has a dense in-memory
  // representation.
  static bool IsDenseArray(const Shape& shape);

  // Returns whether the given Shape is an array and has a sparse in-memory
  // representation.
  static bool IsSparseArray(const Shape& shape);

  // Returns whether the given Shape is a sparse array and has a COO (coordinate
  // matrix) in-memory representation.
  static bool IsCOOArray(const Shape& shape);

  // Returns whether the given Shape is a sparse array and has a CSR (compressed
  // sparse row) in-memory representation.
  static bool IsCSRArray(const Shape& shape);

  // Returns whether the given Shape is a sparse array and has a CSR (compressed
  // sparse row) in-memory representation.
  static bool IsCSCArray(const Shape& shape);

  // Returns whether the given Layout has a dense in-memory representation.
  static bool IsDense(const Layout& layout);

  // Returns whether the given Layout has a sparse in-memory representation.
  static bool IsSparse(const Layout& layout);

  // Returns whether the given Layout represents a COO (coordinate matrix)
  // sparse array.
  static bool IsCOO(const Layout& layout);

  // Returns whether the given Layout represents a CSC (compressed sparse
  // column) array.
  static bool IsCSR(const Layout& layout);

  // Returns whether the given Layout represents a CSC (compressed sparse
  // column) array.
  static bool IsCSC(const Layout& layout);

  // Returns whether the layout is monotonic and dim 0 is minor in the layout.
  // * R0 and R1: this is always trivially true.
  // * R2+: equivalent to column-major. Dimension 0 is the minor, dimension 1 is
  //        more major, and so on until dimension N-1 which is the major.
  static bool IsMonotonicWithDim0Minor(const Layout& layout);

  // Returns whether the layout is monotonic and dim 0 is major in the layout.
  // * R0 and R1: this is always trivially true.
  // * R2+: equivalent to row-major. Dimension 0 is the major, dimension 1 is
  //        more minor, and so on until dimension N-1 which is the minor.
  //
  // Returns `true` for "default", major-to-minor layouts (e.g. {3,2,1,0}).
  static bool IsMonotonicWithDim0Major(const Layout& layout);

  // Returns whether the given shape has a layout. For tuple shapes, true is
  // returned only if all elements have layouts.
  static bool HasLayout(const Shape& shape);

  // Returns whether all Shapes within the given ProgramShape have layouts.
  static bool HasLayout(const ProgramShape& program_shape);

  // Returns whether any subshapes of the shape have custom (!= 0)
  // element_size_in_bits.
  static bool HasCustomElementSizeInBits(const Shape& shape);

  // Returns whether lhs and rhs are identical.
  static bool Equal(const Layout& lhs, const Layout& rhs);

  // Returns the minor_to_major array for the given Shape.  Requires that the
  // shape is an array.
  static inline absl::Span<const int64_t> MinorToMajor(const Shape& shape) {
    DCHECK(shape.IsArray());
    return shape.layout().minor_to_major();
  }

  static inline absl::Span<const int64_t> MinorToMajor(const Layout& layout) {
    return layout.minor_to_major();
  }

  // Major(0) is the most major logical dimension number, Major(1) is the
  // second-most-major logical dimension number and so on.
  //
  // This can be used to translate physical dimension numbers to logical
  // dimension numbers. Assume that we are numbering the physical dimensions so
  // that the most major physical dimension has physical dimension number 0 and
  // so on. Then a physical dimension number p corresponds to the logical
  // dimension number Major(p). So this function could also be called
  // PhysicalToLogical().
  //
  // As an example, consider physical dimension number 0, which by definition is
  // the most major. Then Major(0) is the most major logical dimension, so Major
  // maps the physical dimension number 0 to the most major logical dimension
  // number Major(0).
  static int64_t Major(const Layout& layout,
                       int64_t physical_dimension_number) {
    DCHECK_LE(0, physical_dimension_number);
    DCHECK_LT(physical_dimension_number, layout.minor_to_major_size());
    return Minor(layout,
                 layout.minor_to_major_size() - 1 - physical_dimension_number);
  }

  // Minor(0) is the most minor logical dimension number, minor(1) is the
  // second-most-minor logical dimension number and so on.
  static inline int64_t Minor(const Layout& layout,
                              int64_t physical_dimension_number) {
    DCHECK_LE(0, physical_dimension_number);
    DCHECK_LT(physical_dimension_number, layout.minor_to_major_size());
    return layout.minor_to_major(physical_dimension_number);
  }

  // Returns the inverse mapping of the Major() function. More precisely, return
  // a vector v such that if l == Major(p), then v[l] == p.
  //
  // This can be used to translate logical dimension numbers into physical
  // dimension numbers. Assume that we are numbering the physical dimensions so
  // that the most major physical dimension has physical dimension number 0 and
  // so on. Then a logical dimension number l corresponds to the physical
  // dimension number MakeLogicalToPhysical(layout)[l].
  //
  // In the returned vector, the first element represents the most major logical
  // dimension. The element whose contents are 0 represents the most major
  // physical dimension, and the element with contents (rank - 1) represents
  // the most minor physical dimension.
  static std::vector<int64_t> MakeLogicalToPhysical(const Layout& layout);

  // Prints a human-readable string that represents the given layout.
  static void PrintHumanString(Printer* printer, const Layout& layout);

  // Returns a human-readable string that represents the given layout.
  static std::string HumanString(const Layout& layout);

  // Copies the layout from 'src' to 'dst'. Recursively copies layouts of
  // tuples.  'src' and 'dst' need not be compatible but the two shapes must
  // have the same tuple structure (if any) and arrays must have the same
  // rank. within the shapes must have the same number of dimensions.
  static Status CopyLayoutBetweenShapes(const Shape& src, Shape* dst);

  // Returns true if the layouts of lhs and rhs are equal, false
  // otherwise. Recursively compares layouts of tuples.
  //
  // lhs and rhs need not be compatible to have the same layout but the two
  // shapes must have the same tuple structure (if any) and arrays must have the
  // same rank. Element type is ignored.
  static bool LayoutsInShapesEqual(const Shape& lhs, const Shape& rhs);

  // Returns whether the given dimensions are consecutive in the given layout,
  // not necessarily in the order given.
  static bool AreDimensionsConsecutive(const Layout& layout,
                                       absl::Span<const int64_t> dims);

  // Constructs a new layout by making the given dimension `dim` in the given
  // layout `layout` as the most major dimension.
  static Layout MoveDimToMajor(const Layout& layout, int64_t dim);

  // Returns the linearized index of the cell at the given indices. The unit
  // of the offset is in elements of the shape.
  //
  // NOTE: this method only uses the top-level tile and disregards the sub-tile
  // in the layout. This method is also performance critical.
  static int64_t LinearIndex(const Shape& shape,
                             absl::Span<const int64_t> indices);

  // If the shape has a layout, returns the contained memory space.  Otherwise,
  // returns Layout::kDefaultMemorySpace.
  static int64_t MemorySpace(const Shape& shape);

  static xla::DimLevelType GetDimLevelType(const Layout& layout, int64_t dim);
  static bool DimUnique(const Layout& layout, int64_t dim);
  static bool DimOrdered(const Layout& layout, int64_t dim);

  // Return true iff the given DimLevelType and dim_unique/dim_ordered values
  // represent a valid encoding.
  static bool ValidateDimLevel(xla::DimLevelType dim_level_type,
                               bool dim_unique, bool dim_ordered);

  // Returns true if `byte_strides` is major to minor order, i.e. the strides
  // form a cumulative product of the byte size and dimensions in reverse order
  // and the smallest stride is the byte size for `element_type`.
  static bool ByteStridesIsMajorToMinor(absl::Span<const int64_t> byte_strides,
                                        absl::Span<const int64_t> dims,
                                        PrimitiveType element_type);

  // The max size of the split in the given dimension. If the layout doesn't
  // have a split config in the given dimension, the value returned from this
  // function is equal to the Shape::dimensions(). If there is a split config in
  // the given dimension, we then find the size of the largest split in that
  // dimension.
  static int64_t MaxSplitSize(const Shape& shape, int64_t dim);

  // This function is analogous to ShapeUtil::ElementsIn, except we use the max
  // split sizes for each dimension to calculate the max number of elements
  // stored in a particular split. This can be useful for calculating how much
  // memory to allocate in each of the memories.
  static int64_t MaxElementsInPerSplit(const Shape& shape);
};

}  // namespace xla

#endif  // XLA_LAYOUT_UTIL_H_
