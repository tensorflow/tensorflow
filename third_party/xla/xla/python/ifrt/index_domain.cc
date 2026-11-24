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

#include "xla/python/ifrt/index_domain.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <ostream>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/index_domain.pb.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {

namespace {

// Validates that `origin` and `shape` have matching numbers of dimensions and
// that the cumulative product of non-zero upper-bound dimensions
// (`origin.elements()[i] + shape.dims()[i]`) does not exceed
// `Shape::kMaxNumElements`.
absl::Status ValidateOriginAndShape(const Index& origin, const Shape& shape) {
  absl::Span<const int64_t> origin_elements = origin.elements();
  absl::Span<const int64_t> shape_dims = shape.dims();
  if (origin_elements.size() != shape_dims.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "IndexDomain requires origin and shape to have the same number of "
        "dimensions, but got origin=",
        origin, " vs. shape=", shape));
  }
  int64_t upper_bound_product = 1;
  for (size_t i = 0; i < shape_dims.size(); ++i) {
    // Safe from overflow: both operands are in `[0, Shape::kMaxNumElements]`.
    const int64_t upper_bound = origin_elements[i] + shape_dims[i];
    if (upper_bound == 0) {
      continue;
    }
    if (__builtin_mul_overflow(upper_bound_product, upper_bound,
                               &upper_bound_product) ||
        upper_bound_product > Shape::kMaxNumElements) {
      return absl::InvalidArgumentError(absl::StrCat(
          "IndexDomain upper-bound non-zero dimension product exceeds "
          "Shape::kMaxNumElements (",
          Shape::kMaxNumElements, ") for origin=", origin,
          " and shape=", shape));
    }
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<IndexDomain> IndexDomain::Create(Index origin, Shape shape) {
  ABSL_RETURN_IF_ERROR(ValidateOriginAndShape(origin, shape));
  return IndexDomain(UncheckedTag{}, std::move(origin), std::move(shape));
}

IndexDomain::IndexDomain(Index origin, Shape shape)
    : origin_(std::move(origin)), shape_(std::move(shape)) {
  CHECK_OK(ValidateOriginAndShape(origin_, shape_));
}

absl::StatusOr<IndexDomain> IndexDomain::FromProto(
    const IndexDomainProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for IndexDomain deserialization"));
  }

  ABSL_ASSIGN_OR_RETURN(Index origin_val, Index::FromProto(proto.origin()));
  ABSL_ASSIGN_OR_RETURN(Shape shape_val, Shape::FromProto(proto.shape()));

  return Create(std::move(origin_val), std::move(shape_val));
}

void IndexDomain::ToProto(IndexDomainProto& proto,
                          SerDesVersion version) const {
  CHECK_GE(version.version_number(), SerDesVersionNumber(0))
      << "Unsupported " << version.version_number()
      << " for IndexDomain serialization";

  proto.Clear();
  proto.set_version_number(SerDesVersionNumber(0).value());

  origin_.ToProto(*proto.mutable_origin(), version);
  shape_.ToProto(*proto.mutable_shape(), version);
}

IndexDomain& IndexDomain::operator+=(const Index& offset) {
  origin_ += offset;
  CHECK_OK(ValidateOriginAndShape(origin_, shape_));
  return *this;
}

absl::StatusOr<IndexDomain> IndexDomain::Intersect(
    const IndexDomain& other) const {
  absl::Span<const int64_t> a_origin = origin_.elements();
  absl::Span<const int64_t> a_shape = shape_.dims();
  absl::Span<const int64_t> b_origin = other.origin_.elements();
  absl::Span<const int64_t> b_shape = other.shape_.dims();
  if (a_shape.size() != b_shape.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot intersect IndexDomains with mismatched numbers of dimensions: ",
        *this, " vs. ", other));
  }

  const int num_dims = a_shape.size();
  Index::Elements origin(num_dims);
  Shape::Dimensions shape(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    const int64_t low = std::max(a_origin[i], b_origin[i]);
    const int64_t high =
        std::min(a_origin[i] + a_shape[i], b_origin[i] + b_shape[i]);
    if (low >= high) {
      return IndexDomain(Shape(Shape::Dimensions(num_dims, 0)));
    }
    origin[i] = low;
    shape[i] = high - low;
  }
  // For every non-empty intersection, `0 <= low < high <=
  // a_origin[i] + a_shape[i]`, so the non-zero products of `origin`, `shape`,
  // and `origin + shape` are all bounded by `a`'s upper-bound product
  // (`<= Shape::kMaxNumElements`).
  return IndexDomain(UncheckedTag{}, Index(origin), Shape(shape));
}

absl::StatusOr<bool> IndexDomain::Contains(const IndexDomain& other) const {
  absl::Span<const int64_t> a_origin = origin_.elements();
  absl::Span<const int64_t> a_shape = shape_.dims();
  absl::Span<const int64_t> b_origin = other.origin_.elements();
  absl::Span<const int64_t> b_shape = other.shape_.dims();
  if (a_shape.size() != b_shape.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot check containment of IndexDomains with mismatched numbers of "
        "dimensions: ",
        *this, " vs. ", other));
  }

  bool contains = true;
  for (int i = 0; i < a_shape.size(); ++i) {
    if (b_origin[i] < a_origin[i] ||
        b_origin[i] + b_shape[i] > a_origin[i] + a_shape[i]) {
      contains = false;
    }
  }
  return contains;
}

std::ostream& operator<<(std::ostream& os, const IndexDomain& index_domain) {
  return os << absl::StrCat(index_domain);
}

}  // namespace ifrt
}  // namespace xla
