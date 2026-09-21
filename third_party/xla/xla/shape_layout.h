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

#ifndef XLA_SHAPE_LAYOUT_H_
#define XLA_SHAPE_LAYOUT_H_

#include <string>

#include "absl/status/status.h"
#include "xla/layout.h"
#include "xla/printer.h"
#include "xla/shape.h"
#include "xla/shape_util.h"

namespace xla {

// A ShapeLayout object encapsulates the layout of a particular shape (including
// tuples). This differs from the Layout proto which describes the layout of a
// single array. ShapeLayout contains a Layout proto for each array in the shape
// (a tuple can have more than one array). For array shapes, this object
// trivially holds a single Layout. Logically, ShapeLayout holds a nonmutable
// shape with mutable layouts.
class ShapeLayout {
 public:
  // Constructs a ShapeLayout of the given shape. Layouts are copied from the
  // shape parameter.
  explicit ShapeLayout(const Shape& shape) : shape_(shape) {}

  // Assigns the layouts in this ShapeLayout to the Layout fields of the given
  // shape. 'to_shape' and the shape of the ShapeLayout object must be
  // compatible.
  absl::Status AssignLayoutToShape(Shape* to_shape) const;

  // Returns true if the Layouts in this ShapeLayout match the layouts in the
  // given shape. Returns false otherwise. If the given shape is not compatible
  // with the ShapeLayout's shape, then false is returned. If
  // `ignore_fully_empty_tiling` is true, tiling info is ignored if one of the
  // shapes has no tiling at all in all its subshapes.
  bool MatchesLayoutInShape(const Shape& shape,
                            bool minor_to_major_only = false,
                            bool ignore_fully_empty_tiling = false) const;

  // Copies the layout from the given shape into this ShapeLayout. 'other_shape'
  // must be compatible with the ShapeLayout's shape.
  absl::Status CopyLayoutFromShape(const Shape& other_shape);

  // Clears (Layout::Clear) all the Layouts stored in this object.
  void Clear();
  void Clear(ShapeIndexView shape_index);
  // Clears all the tiles in the layouts stored in this object.
  void ClearTiles();
  void ClearTiles(ShapeIndexView shape_index);

  // Sets all Layouts stored in this object to the default layout.
  void SetToDefaultLayout();

  // Returns the shape (with layouts).
  const Shape& shape() const { return shape_; }

  // Clear dynamic dimensions of this module. Pretending the module creates
  // static results. Useful in inspecting full outputs when testing.
  void ClearDynamicShape() { shape_.clear_dynamic_dimensions(); }

  // Checks that a layout is set for the shape, and returns a reference to the
  // layout directly on the shape. Shape must not be a tuple.
  const Layout& layout() const;

  // Returns true if all layouts have been set for this ShapeLayout object. That
  // is, every array has a layout.
  bool LayoutIsSet() const;
  bool AnyLayoutIsSet() const;

  // Resets the layout on the shape to the provided layout. Shape must not be a
  // tuple.
  void ResetLayout(const Layout& layout);

  // Resets the layout on the shape at the provided ShapeIndex to the provided
  // layout. Shape must be a tuple.
  void ResetLayout(const Layout& layout, ShapeIndexView shape_index);

  // Returns a string representation of this object.
  void Print(Printer* printer) const { shape_.Print(printer, true); }

  // Returns a string representation of this object.
  std::string ToString() const { return shape_.ToString(true); }

  // Tests for equality of both shape and layout (ShapeUtil::Equal).
  bool operator==(const ShapeLayout& other) const;
  bool operator!=(const ShapeLayout& other) const;

 private:
  Shape shape_;
};

}  // namespace xla

#endif  // XLA_SHAPE_LAYOUT_H_
