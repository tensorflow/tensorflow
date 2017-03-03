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

tensorflow::Status ShapeLayout::CopyLayoutFromShape(const Shape& other_shape) {
  if (!ShapeUtil::Compatible(other_shape, shape_)) {
    return InvalidArgument("Shape %s is not compatible with shape %s",
                           ShapeUtil::HumanString(other_shape).c_str(),
                           ShapeUtil::HumanString(shape()).c_str());
  }
  shape_ = other_shape;
  return tensorflow::Status::OK();
}

tensorflow::Status ShapeLayout::AssignLayoutToShape(Shape* other_shape) const {
  if (!ShapeUtil::Compatible(*other_shape, shape_)) {
    return InvalidArgument("Shape %s is not compatible with shape %s",
                           ShapeUtil::HumanString(*other_shape).c_str(),
                           ShapeUtil::HumanString(shape()).c_str());
  }
  *other_shape = shape_;
  return tensorflow::Status::OK();
}

void ShapeLayout::SetToDefaultLayout() {
  LayoutUtil::SetToDefaultLayout(&shape_);
}

bool ShapeLayout::MatchesLayoutInShape(const Shape& shape) const {
  return ShapeUtil::Equal(shape, shape_);
}

const Layout& ShapeLayout::layout() const {
  CHECK(LayoutIsSet());
  CHECK(!ShapeUtil::IsTuple(shape_));
  return shape_.layout();
}

void ShapeLayout::Clear() { LayoutUtil::ClearLayout(&shape_); }

bool ShapeLayout::LayoutIsSet() const { return LayoutUtil::HasLayout(shape_); }

void ShapeLayout::ResetLayout(const Layout& layout) {
  CHECK(!ShapeUtil::IsTuple(shape_));
  CHECK(!ShapeUtil::IsOpaque(shape_));
  *shape_.mutable_layout() = layout;
  TF_CHECK_OK(ShapeUtil::ValidateShape(shape_));
}

bool ShapeLayout::operator==(const ShapeLayout& other) const {
  return ShapeUtil::Equal(shape_, other.shape_);
}

bool ShapeLayout::operator!=(const ShapeLayout& other) const {
  return !ShapeUtil::Equal(shape_, other.shape_);
}

}  // namespace xla
