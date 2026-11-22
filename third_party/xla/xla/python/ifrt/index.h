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

#ifndef XLA_PYTHON_IFRT_INDEX_H_
#define XLA_PYTHON_IFRT_INDEX_H_

#include <cstdint>
#include <ostream>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/log/check.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/str_join.h"
#include "absl/types/span.h"
#include "xla/python/ifrt/index.pb.h"
#include "xla/python/ifrt/serdes_default_version_accessor.h"
#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {

// Multi-dimensional index. Every element must be non-negative, and the
// cumulative product of non-zero elements must not exceed
// `Shape::kMaxNumElements`.
class Index {
 public:
  // Maximum elements to inline.
  static constexpr int kInlineElementSize = 6;

  using Elements = absl::InlinedVector<int64_t, kInlineElementSize>;

  // Constructs an `Index` from `elements`. Returns `InvalidArgumentError` if
  // any element value is negative or if the cumulative product of non-zero
  // elements exceeds `Shape::kMaxNumElements`.
  static absl::StatusOr<Index> Create(absl::Span<const int64_t> elements);

  // Constructs an `Index` from `elements`. `CHECK`-fails if `Create(elements)`
  // returns an error.
  explicit Index(absl::Span<const int64_t> elements);

  // Constructs an `Index` with `num_elements` zero elements. `CHECK`-fails if
  // `num_elements < 0`.
  static Index Zeros(int num_elements) {
    CHECK_GE(num_elements, 0);
    return Index(UncheckedTag{}, Elements(/*n=*/num_elements));
  }

  Index(const Index&) = default;
  Index(Index&&) = default;
  Index& operator=(const Index&) = default;
  Index& operator=(Index&&) = default;

  // Constructs `Index` from `IndexProto`.
  static absl::StatusOr<Index> FromProto(const IndexProto& proto);

  // Converts the index to a protobuf.
  void ToProto(
      IndexProto& proto,
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const;

  // Returns a `IndexProto` representation.
  IndexProto ToProto(
      SerDesVersion version = SerDesDefaultVersionAccessor::Get()) const {
    IndexProto proto;
    ToProto(proto, version);
    return proto;
  }

  absl::Span<const int64_t> elements() const { return elements_; }

  bool operator==(const Index& other) const {
    return elements_ == other.elements_;
  }
  bool operator!=(const Index& other) const {
    return elements_ != other.elements_;
  }

  template <typename H>
  friend H AbslHashValue(H h, const Index& index);

  // Element-wise addition. `CHECK`-fails if `this` and `offset` have different
  // numbers of dimensions or if the cumulative product of non-zero resulting
  // elements exceeds `Shape::kMaxNumElements`.
  Index operator+(const Index& offset) const;

  // Element-wise subtraction. `CHECK`-fails if `this` and `offset` have
  // different numbers of dimensions or if any resulting element would be
  // negative.
  Index operator-(const Index& offset) const;

  // Element-wise multiplication. `CHECK`-fails if `this` and `multiplier` have
  // different numbers of dimensions, if any `multiplier[i] < 0`, or if the
  // cumulative product of non-zero resulting elements exceeds
  // `Shape::kMaxNumElements`.
  Index operator*(absl::Span<const int64_t> multiplier) const;

  // Compound element-wise arithmetic operators with the same `CHECK`
  // preconditions as `operator+`, `operator-`, and `operator*`.
  Index& operator+=(const Index& offset) { return *this = *this + offset; }
  Index& operator-=(const Index& offset) { return *this = *this - offset; }
  Index& operator*=(absl::Span<const int64_t> multiplier) {
    return *this = *this * multiplier;
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Index& index) {
    sink.Append(absl::StrCat("[", absl::StrJoin(index.elements_, ","), "]"));
  }

 private:
  struct UncheckedTag {};
  Index(UncheckedTag, Elements elements) : elements_(std::move(elements)) {}

  Elements elements_;
};

std::ostream& operator<<(std::ostream& os, const Index& index);

template <typename H>
H AbslHashValue(H h, const Index& index) {
  return H::combine(std::move(h), index.elements_);
}

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_INDEX_H_
