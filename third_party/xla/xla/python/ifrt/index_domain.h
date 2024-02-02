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

#ifndef XLA_PYTHON_IFRT_INDEX_DOMAIN_H_
#define XLA_PYTHON_IFRT_INDEX_DOMAIN_H_

#include <cstdint>
#include <ostream>
#include <string>
#include <utility>

#include "xla/python/ifrt/index.h"
#include "xla/python/ifrt/shape.h"

namespace xla {
namespace ifrt {

// Domain of a multi-dimensional index space. Informally, it represents a slice
// that is defined by the origin (lower inclusive bound) of the slice and the
// shape of the slice.
class IndexDomain {
 public:
  // General `IndexDomain` construction.
  IndexDomain(Index origin, Shape shape)
      : origin_(std::move(origin)), shape_(std::move(shape)) {}

  // `IndexDomain` construction with a zeros origin.
  explicit IndexDomain(Shape shape)
      : origin_(Index::Zeros(shape.dims().size())), shape_(std::move(shape)) {}

  IndexDomain(const IndexDomain&) = default;
  IndexDomain(IndexDomain&&) = default;
  IndexDomain& operator=(const IndexDomain&) = default;
  IndexDomain& operator=(IndexDomain&&) = default;

  const Index& origin() const { return origin_; }
  const Shape& shape() const { return shape_; }

  bool operator==(const IndexDomain& other) const {
    return origin_ == other.origin_ && shape_ == other.shape_;
  }
  bool operator!=(const IndexDomain& other) const {
    return origin_ != other.origin_ || shape_ != other.shape_;
  }
  IndexDomain operator+(const Index& offset) const {
    return IndexDomain(origin_ + offset, shape_);
  }
  IndexDomain operator-(const Index& offset) const {
    return IndexDomain(origin_ - offset, shape_);
  }
  IndexDomain& operator+=(const Index& offset) {
    origin_ += offset;
    return *this;
  }
  IndexDomain& operator-=(const Index& offset) {
    origin_ -= offset;
    return *this;
  }
  std::string DebugString() const;

 private:
  Index origin_;
  Shape shape_;
};

std::ostream& operator<<(std::ostream& os, const IndexDomain& index_domain);

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_INDEX_DOMAIN_H_
