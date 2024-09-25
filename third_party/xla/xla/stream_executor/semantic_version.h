/* Copyright 2024 The OpenXLA Authors.

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

#ifndef XLA_STREAM_EXECUTOR_SEMANTIC_VERSION_H_
#define XLA_STREAM_EXECUTOR_SEMANTIC_VERSION_H_

#include <array>
#include <iosfwd>
#include <string>
#include <tuple>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace stream_executor {

// `SemanticVersion` represents a version number of the form X.Y.Z with
// - X being called the major version,
// - Y being called the minor version,
// - Z being called the patch version.
//
// The type is lexicographically ordered and supports printing and parsing.
class SemanticVersion {
 public:
  constexpr SemanticVersion(unsigned major, unsigned minor, unsigned patch)
      : major_(major), minor_(minor), patch_(patch) {}
  explicit SemanticVersion(std::array<unsigned, 3> other)
      : major_(other[0]), minor_(other[1]), patch_(other[2]) {}

  static absl::StatusOr<SemanticVersion> ParseFromString(absl::string_view str);

  unsigned& major() { return major_; }
  unsigned major() const { return major_; }

  unsigned& minor() { return minor_; }
  unsigned minor() const { return minor_; }

  unsigned& patch() { return patch_; }
  unsigned patch() const { return patch_; }

  std::string ToString() const;

  friend bool operator==(const SemanticVersion& lhs,
                         const SemanticVersion& rhs) {
    return std::tie(lhs.major_, lhs.minor_, lhs.patch_) ==
           std::tie(rhs.major_, rhs.minor_, rhs.patch_);
  }
  friend bool operator<(const SemanticVersion& lhs,
                        const SemanticVersion& rhs) {
    return std::tie(lhs.major_, lhs.minor_, lhs.patch_) <
           std::tie(rhs.major_, rhs.minor_, rhs.patch_);
  }
  friend bool operator!=(const SemanticVersion& lhs,
                         const SemanticVersion& rhs) {
    return !(lhs == rhs);
  }
  friend bool operator>(const SemanticVersion& lhs,
                        const SemanticVersion& rhs) {
    return rhs < lhs;
  }
  friend bool operator>=(const SemanticVersion& lhs,
                         const SemanticVersion& rhs) {
    return !(lhs < rhs);
  }
  friend bool operator<=(const SemanticVersion& lhs,
                         const SemanticVersion& rhs) {
    return !(lhs > rhs);
  }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SemanticVersion& version) {
    sink.Append(version.ToString());
  }

  template <typename H>
  friend H AbslHashValue(H h, const SemanticVersion& v) {
    return H::combine(std::move(h), v.major_, v.minor_, v.patch_);
  }

  friend std::ostream& operator<<(std::ostream& os,
                                  const SemanticVersion& version) {
    return os << version.ToString();
  }

 private:
  unsigned major_;
  unsigned minor_;
  unsigned patch_;
};

}  // namespace stream_executor

#endif  // XLA_STREAM_EXECUTOR_SEMANTIC_VERSION_H_
