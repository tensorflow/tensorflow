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

#include "tensorflow/compiler/xla/shape_layout.h"

#include "tensorflow/compiler/xla/layout_util.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/core/platform/logging.h"

namespace xla {

Status ShapeLayout::CopyLayoutFromShape(const Shape& other_shape) {
  if (!ShapeUtil::Compatible(other_shape, shape_)) {
    return InvalidArgument("Shape %s is not compatible with shape %s",
                           ShapeUtil::HumanString(other_shape),
                           ShapeUtil::HumanString(shape()));
  }
  shape_ = other_shape;
  return Status::OK();
}

Status ShapeLayout::AssignLayoutToShape(Shape* to_shape) const {
  if (!ShapeUtil::Compatible(*to_shape, shape_)) {
    return InvalidArgument("Shape %s is not compatible with shape %s",
                           ShapeUtil::HumanString(*to_shape),
                           ShapeUtil::HumanString(shape()));
  }
  *to_shape = shape_;
  return Status::OK();
}

void ShapeLayout::SetToDefaultLayout() {
  LayoutUtil::SetToDefaultLayout(&shape_);
}

bool ShapeLayout::MatchesLayoutInShape(const Shape& shape,
                                       bool minor_to_major_only,
                                       bool ignore_fully_empty_tiling) const {
  auto equal = Shape::Equal().IgnoreDynamicDimension();
  if (ignore_fully_empty_tiling) {
    bool fully_empty_tiling = true;
    auto check_tiling = [&fully_empty_tiling](const Shape& subshape,
                                              const xla::ShapeIndex& index) {
      if (!fully_empty_tiling) {
        return;
      }
      if (subshape.IsArray() && !subshape.layout().tiles().empty()) {
        fully_empty_tiling = false;
      }
    };
    ShapeUtil::ForEachSubshape(shape, check_tiling);
    if (fully_empty_tiling) {
      equal.MinorToMajorOnlyInLayout();
    } else {
      fully_empty_tiling = true;
      // Check the other shape.
      ShapeUtil::ForEachSubshape(shape_, check_tiling);
      if (fully_empty_tiling) {
        equal.MinorToMajorOnlyInLayout();
      }
    }
  }
  if (minor_to_major_only) {
    equal.MinorToMajorOnlyInLayout();
  }
  return equal(shape, shape_);
}

const Layout& ShapeLayout::layout() const {
  CHECK(LayoutIsSet());
  CHECK(!shape_.IsTuple());
  return shape_.layout();
}

void ShapeLayout::Clear() { LayoutUtil::ClearLayout(&shape_); }

bool ShapeLayout::LayoutIsSet() const { return LayoutUtil::HasLayout(shape_); }

void ShapeLayout::ResetLayout(const Layout& layout) {
  CHECK(!shape_.IsTuple());
  CHECK(!shape_.IsOpaque());
  *shape_.mutable_layout() = layout;
  TF_CHECK_OK(ShapeUtil::ValidateShape(shape_));
}

void ShapeLayout::ResetLayout(const Layout& layout,
                              ShapeIndexView shape_index) {
  *ShapeUtil::GetMutableSubshape(&shape_, shape_index)->mutable_layout() =
      layout;
  TF_CHECK_OK(ShapeUtil::ValidateShape(shape_));
}

bool ShapeLayout::operator==(const ShapeLayout& other) const {
  return Shape::Equal().IgnoreDynamicDimension()(shape_, other.shape_);
}

bool ShapeLayout::operator!=(const ShapeLayout& other) const {
  return !Shape::Equal().IgnoreDynamicDimension()(shape_, other.shape_);
}

}  // namespace xla
