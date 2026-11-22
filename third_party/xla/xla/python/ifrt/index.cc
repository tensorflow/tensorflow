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

#include "xla/python/ifrt/index.h"

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <utility>

#include "absl/log/check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/index.pb.h"
#include "xla/python/ifrt/serdes_version.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {

namespace {

// Returns whether the cumulative product of non-zero `elements` fits in
// `Shape::kMaxNumElements`. All elements must be non-negative.
bool NonZeroProductWithinLimit(absl::Span<const int64_t> elements) {
  int64_t product = 1;
  for (int64_t element : elements) {
    if (element == 0) {
      continue;
    }
    if (__builtin_mul_overflow(product, element, &product) ||
        product > Shape::kMaxNumElements) {
      return false;
    }
  }
  return true;
}

}  // namespace

absl::StatusOr<Index> Index::Create(absl::Span<const int64_t> elements) {
  for (int64_t element : elements) {
    if (element < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Index expects non-negative element values, but got ",
                       element, " in [", absl::StrJoin(elements, ","), "]"));
    }
  }
  if (!NonZeroProductWithinLimit(elements)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Product of non-zero elements in Index [", absl::StrJoin(elements, ","),
        "] exceeds Shape::kMaxNumElements (", Shape::kMaxNumElements, ")"));
  }
  return Index(UncheckedTag{}, Elements(elements.begin(), elements.end()));
}

Index::Index(absl::Span<const int64_t> elements)
    : Index([](absl::Span<const int64_t> e) {
        absl::StatusOr<Index> index = Create(e);
        CHECK_OK(index);
        return *std::move(index);
      }(elements)) {}

absl::StatusOr<Index> Index::FromProto(const IndexProto& proto) {
  const SerDesVersionNumber version_number(proto.version_number());
  if (version_number != SerDesVersionNumber(0)) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Unsupported ", version_number, " for Index deserialization"));
  }
  return Create(proto.elements());
}

void Index::ToProto(IndexProto& proto, SerDesVersion version) const {
  CHECK_GE(version.version_number(), SerDesVersionNumber(0))
      << "Unsupported " << version.version_number()
      << " for Index serialization";

  proto.Clear();
  proto.set_version_number(SerDesVersionNumber(0).value());

  proto.mutable_elements()->Reserve(elements().size());
  for (int64_t element : elements()) {
    proto.mutable_elements()->AddAlreadyReserved(element);
  }
}

Index Index::operator+(const Index& offset) const {
  CHECK_EQ(elements_.size(), offset.elements_.size());
  Index result = *this;
  for (size_t i = 0; i < elements_.size(); ++i) {
    // Safe from overflow: both operands are in `[0, Shape::kMaxNumElements]`.
    result.elements_[i] += offset.elements_[i];
  }
  CHECK(NonZeroProductWithinLimit(result.elements_));
  return result;
}

Index Index::operator-(const Index& offset) const {
  CHECK_EQ(elements_.size(), offset.elements_.size());
  Index result = *this;
  for (size_t i = 0; i < elements_.size(); ++i) {
    CHECK_GE(result.elements_[i], offset.elements_[i]);
    result.elements_[i] -= offset.elements_[i];
  }
  return result;
}

Index Index::operator*(absl::Span<const int64_t> multiplier) const {
  CHECK_EQ(elements_.size(), multiplier.size());
  Index result = *this;
  for (size_t i = 0; i < elements_.size(); ++i) {
    CHECK_GE(multiplier[i], 0);
    CHECK(!__builtin_mul_overflow(result.elements_[i], multiplier[i],
                                  &result.elements_[i]));
  }
  CHECK(NonZeroProductWithinLimit(result.elements_));
  return result;
}

std::ostream& operator<<(std::ostream& os, const Index& index) {
  return os << absl::StrCat(index);
}

}  // namespace ifrt
}  // namespace xla
