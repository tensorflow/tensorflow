/* Copyright 2022 The OpenXLA Authors.

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

#include "xla/python/ifrt/shape.h"

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>
#include <variant>

#include "absl/container/inlined_vector.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/strings/string_view.h"
#include "xla/python/ifrt/shape.pb.h"
#include "xla/util.h"
#include "tsl/platform/errors.h"
#include "tsl/platform/statusor.h"

namespace xla {
namespace ifrt {

namespace {

// Helper type for the visitor.
template <class... Ts>
struct overloaded : Ts... {
  using Ts::operator()...;
};

// Explicit deduction guide.
template <class... Ts>
overloaded(Ts...) -> overloaded<Ts...>;

}  // namespace

absl::StatusOr<Shape> Shape::FromProto(const ShapeProto& proto) {
  Shape::Dimensions dims;
  dims.reserve(proto.dims_size());
  for (int64_t dim : proto.dims()) {
    if (dim < 0) {
      return InvalidArgument(
          "Shape expects non-negative dimension sizes, but got %d", dim);
    }
    dims.push_back(dim);
  }
  return Shape(std::move(dims));
}

ShapeProto Shape::ToProto() const {
  ShapeProto proto;
  proto.mutable_dims()->Reserve(dims().size());
  for (int64_t dim : dims()) {
    proto.mutable_dims()->AddAlreadyReserved(dim);
  }
  return proto;
}

int64_t Shape::num_elements() const {
  int64_t count = 1;
  for (int64_t d : dims_) {
    count *= d;
  }
  return count;
}

std::string Shape::DebugString() const {
  return absl::StrCat("[", absl::StrJoin(dims_, ","), "]");
}

absl::StatusOr<BoundedDynamicShapeTag> BoundedDynamicShapeTag::FromProto(
    const BoundedDynamicShapeTagProto& proto) {
  BoundedDynamicShapeTag::DynamicDimensions dynamic_dims;
  dynamic_dims.reserve(proto.is_dynamic_dims_size());
  for (bool dynamic_dim : proto.is_dynamic_dims()) {
    dynamic_dims.push_back(dynamic_dim);
  }
  return BoundedDynamicShapeTag(std::move(dynamic_dims));
}

BoundedDynamicShapeTagProto BoundedDynamicShapeTag::ToProto() const {
  BoundedDynamicShapeTagProto proto;
  proto.mutable_is_dynamic_dims()->Reserve(dynamic_dims_.size());
  for (bool dynamic_dim : dynamic_dims_) {
    proto.mutable_is_dynamic_dims()->AddAlreadyReserved(dynamic_dim);
  }
  return proto;
}

absl::StatusOr<DynamicShape> DynamicShape::Create(Shape shape,
                                                  DynamicShapeTag tag) {
  TF_RETURN_IF_ERROR(std::visit(
      overloaded{
          [&](const BoundedDynamicShapeTag& tag) -> absl::Status {
            if (tag.DynamicDims().size() != shape.dims().size()) {
              return InvalidArgument(
                  "Shape and tag must have the same number of dimensions.");
            }
            return absl::OkStatus();
          },
      },
      tag));
  return DynamicShape(std::move(shape), std::move(tag));
}

absl::StatusOr<Shape> DynamicShape::GetPaddedShape() const {
  return std::visit(
      overloaded{
          [this](BoundedDynamicShapeTag tag) { return shape_; },
      },
      tag_);
}

bool DynamicShape::IsDynamicDim(int dimension) const {
  return std::visit(
      overloaded{
          [dimension](BoundedDynamicShapeTag tag) {
            return tag.DynamicDims().at(dimension);
          },
      },
      tag_);
}

absl::StatusOr<DynamicShape> DynamicShape::FromProto(
    const DynamicShapeProto& proto) {
  TF_ASSIGN_OR_RETURN(Shape shape, Shape::FromProto(proto.shape()));
  if (proto.has_bounded_dynamic_shape_tag()) {
    TF_ASSIGN_OR_RETURN(
        BoundedDynamicShapeTag tag,
        BoundedDynamicShapeTag::FromProto(proto.bounded_dynamic_shape_tag()));
    return DynamicShape::Create(std::move(shape), std::move(tag));
  }
  return InvalidArgument("Only support bounded dynamic shape.");
}

DynamicShapeProto DynamicShape::ToProto() const {
  DynamicShapeProto proto;
  *proto.mutable_shape() = shape_.ToProto();
  std::visit(
      overloaded{
          [&proto](BoundedDynamicShapeTag tag) {
            *proto.mutable_bounded_dynamic_shape_tag() = tag.ToProto();
          },
      },
      tag_);
  return proto;
}

std::string DynamicShape::DebugString() const {
  return std::visit(
      overloaded{[this](BoundedDynamicShapeTag tag) {
        absl::InlinedVector<std::string, Shape::kInlineDimensionSize> dim_reps;
        dim_reps.reserve(shape_.dims().size());
        for (int i = 0; i < shape_.dims().size(); ++i) {
          absl::string_view prefix = tag.DynamicDims()[i] ? "<=" : "";
          dim_reps.push_back(absl::StrCat(prefix, shape_.dims()[i]));
        }
        return absl::StrCat("[", absl::StrJoin(dim_reps, ","), "]");
      }},
      tag_);
}

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
  return os << shape.DebugString();
}

std::ostream& operator<<(std::ostream& os, const DynamicShape& dynamic_shape) {
  return os << dynamic_shape.DebugString();
}

}  // namespace ifrt
}  // namespace xla
