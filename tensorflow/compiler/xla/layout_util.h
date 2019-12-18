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

// Utility functions related to layouts of Shapes.

#ifndef TENSORFLOW_COMPILER_XLA_LAYOUT_UTIL_H_
#define TENSORFLOW_COMPILER_XLA_LAYOUT_UTIL_H_

#include <string>

#include "absl/types/span.h"
#include "tensorflow/compiler/xla/layout.h"
#include "tensorflow/compiler/xla/shape.h"
#include "tensorflow/compiler/xla/status.h"
#include "tensorflow/compiler/xla/types.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/macros.h"
#include "tensorflow/core/platform/types.h"

namespace xla {

// Namespaced collection of (static) Layout utilities.
class LayoutUtil {
 public:
  // Creates a layout with the given minor-to-major dimension order. (This is a
  // convenience function for protobuf construction.)
  static Layout MakeLayout(absl::Span<const int64> minor_to_major,
                           absl::Span<const Tile> tiles = {},
                           int64 element_size_in_bits = 0,
                           int64 memory_space = 0);

  // Similar to MakeLayout, but take indices in reverse order.
  static Layout MakeLayoutFromMajorToMinor(
      absl::Span<const int64> major_to_minor);

  // Returns a layout with descending ((i.e. {n, n-1, ..., 0}) minor-to-major
  // dimensions.
  static Layout MakeDescendingLayout(int64 rank);

  // Creates a sparse layout with the given maximum number of elements. (This is
  // a convenience function for protobuf construction.)
  static Layout MakeSparseLayout(int64 max_sparse_elements);

  // Returns default layout for the given shape.
  static Layout GetDefaultLayoutForShape(const Shape& shape);

  // Helper functions that create default layouts for various ranks.
  static Layout GetDefaultLayoutForRank(int64 rank);
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

  // Returns whether the given Shape is an array and has a dense format layout.
  static bool IsDenseArray(const Shape& shape);

  // Returns whether the given Layout has a dense format.
  static bool IsDense(const Layout& layout);

  // Returns whether the layout is monotonic and dim 0 is minor in the layout.
  // * R0 and R1: this is always trivially true.
  // * R2+: equivalent to column-major. Dimension 0 is the minor, dimension 1 is
  //        more major, and so on until dimension N-1 which is the major.
  static bool IsMonotonicWithDim0Minor(const Layout& layout);

  // Returns whether the layout is monotonic and dim 0 is major in the layout.
  // * R0 and R1: this is always trivially true.
  // * R2+: equivalent to row-major. Dimension 0 is the major, dimension 1 is
  //        more minor, and so on until dimension N-1 which is the minor.
  static bool IsMonotonicWithDim0Major(const Layout& layout);

  // Returns whether the given Shape is an array (i.e. not a tuple) and has a
  // sparse format layout.
  static bool IsSparseArray(const Shape& shape);

  // Returns whether the given Layout has a sparse format.
  static bool IsSparse(const Layout& layout);

  // Returns the maximum number of elements that can be stored in a sparse
  // layout.
  static int64 MaxSparseElements(const Layout& layout);

  // Returns whether the given shape has a layout. For tuple shapes, true is
  // returned only if all elements have layouts.
  static bool HasLayout(const Shape& shape);

  // Returns whether all Shapes within the given ProgramShape have layouts.
  static bool HasLayout(const ProgramShape& program_shape);

  // Returns whether lhs and rhs are identical.
  static bool Equal(const Layout& lhs, const Layout& rhs);

  // Returns the minor_to_major array for the given Shape.  Requires that the
  // shape is an array and has a dense layout.
  static absl::Span<const int64> MinorToMajor(const Shape& shape);
  static absl::Span<const int64> MinorToMajor(const Layout& layout);

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
  static int64 Major(const Layout& layout, int64 physical_dimension_number);

  // Minor(0) is the most minor logical dimension number, minor(1) is the
  // second-most-minor logical dimension number and so on.
  static int64 Minor(const Layout& layout, int64 physical_dimension_number);

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
  static std::vector<int64> MakeLogicalToPhysical(const Layout& layout);

  // Returns a human-readable string that represents the given layout.
  static string HumanString(const Layout& layout);

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
                                       absl::Span<const int64> dims);

  // Compute a hash for `layout`.
  static size_t Hash(const Layout& layout);

 private:
  TF_DISALLOW_COPY_AND_ASSIGN(LayoutUtil);
};

}  // namespace xla

#endif  // TENSORFLOW_COMPILER_XLA_LAYOUT_UTIL_H_
