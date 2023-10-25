/* Copyright 2022 The TensorFlow Authors. All Rights Reserved.

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

#include <ostream>
#include <string>
#include <utility>

#include "absl/strings/str_join.h"
#include "xla/python/ifrt/types.pb.h"
#include "xla/util.h"

namespace xla {
namespace ifrt {

StatusOr<Shape> Shape::FromProto(const ShapeProto& proto) {
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

std::ostream& operator<<(std::ostream& os, const Shape& shape) {
  return os << shape.DebugString();
}

}  // namespace ifrt
}  // namespace xla
