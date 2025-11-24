/* Copyright 2025 The OpenXLA Authors.

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

#ifndef XLA_PJRT_PJRT_DEVICE_DIMENSIONS_H_
#define XLA_PJRT_PJRT_DEVICE_DIMENSIONS_H_

#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <ostream>
#include <string>
#include <utility>

#include "absl/container/inlined_vector.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "xla/pjrt/proto/pjrt_device_dimensions.pb.h"

namespace xla {

// Represents device dimensions (e.g., mesh bounds or chip coordinates).
class PjRtDeviceDimensions {
 public:
  using DimensionsContainer = absl::InlinedVector<int32_t, 4>;
  using iterator = DimensionsContainer::iterator;
  using const_iterator = DimensionsContainer::const_iterator;

  PjRtDeviceDimensions() = default;
  PjRtDeviceDimensions(std::initializer_list<int32_t> dims)
      : dimensions_(dims) {}
  explicit PjRtDeviceDimensions(absl::Span<const int32_t> dims)
      : dimensions_(dims.begin(), dims.end()) {}

  int32_t& operator[](size_t i) { return dimensions_[i]; }

  iterator begin() { return dimensions_.begin(); }
  const_iterator begin() const { return dimensions_.begin(); }

  iterator end() { return dimensions_.end(); }
  const_iterator end() const { return dimensions_.end(); }

  const int32_t& operator[](size_t i) const { return dimensions_[i]; }

  const int32_t* data() const { return dimensions_.data(); }
  size_t size() const { return dimensions_.size(); }

  friend bool operator==(const PjRtDeviceDimensions& a,
                         const PjRtDeviceDimensions& b) {
    return a.dimensions_ == b.dimensions_;
  }

  friend bool operator!=(const PjRtDeviceDimensions& a,
                         const PjRtDeviceDimensions& b) {
    return !(a == b);
  }

  friend bool operator<(const PjRtDeviceDimensions& a,
                        const PjRtDeviceDimensions& b) {
    return a.dimensions_ < b.dimensions_;
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const PjRtDeviceDimensions& d) {
    return os << d.ToString();
  }

  template <typename H>
  friend H AbslHashValue(H h, const PjRtDeviceDimensions& c) {
    return H::combine(std::move(h), c.dimensions_);
  }

  static absl::StatusOr<PjRtDeviceDimensions> FromProto(
      const PjRtDeviceDimensionsProto& proto) {
    return PjRtDeviceDimensions(proto.dimensions());
  }

  PjRtDeviceDimensionsProto ToProto() const;

  std::string ToString(absl::string_view sep = ",") const;

  static absl::StatusOr<PjRtDeviceDimensions> FromString(
      absl::string_view text);

 private:
  DimensionsContainer dimensions_;
};

// Support for absl flags.
bool AbslParseFlag(absl::string_view text, PjRtDeviceDimensions* bounds,
                   std::string* err);
std::string AbslUnparseFlag(PjRtDeviceDimensions bounds);

}  // namespace xla

#endif  // XLA_PJRT_PJRT_DEVICE_DIMENSIONS_H_
