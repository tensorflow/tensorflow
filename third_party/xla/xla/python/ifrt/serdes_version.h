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

#ifndef XLA_PYTHON_IFRT_SERDES_VERSION_H_
#define XLA_PYTHON_IFRT_SERDES_VERSION_H_

#include "absl/strings/str_cat.h"
#include "xla/tsl/lib/gtl/int_type.h"

namespace xla {
namespace ifrt {

// Literal type that encodes a SerDes version number. Used in implementation
// details of `SerDesVersion` and serialization/deserialization logics. Public
// APIs are expected to use `SerDesVersion` only.
TSL_LIB_GTL_DEFINE_INT_TYPE(SerDesVersionNumber, int);

template <typename Sink>
void AbslStringify(Sink& sink, SerDesVersionNumber version_number) {
  sink.Append(absl::StrCat("IFRT SerDes version ", version_number.value()));
}

// Represents a version of the IFRT serialization format.
class SerDesVersion {
 public:
  // List of SerDes version literals and introduction dates.
  //
  // 0: 2025-05-20, initial version.

  // Returns the current version.
  static SerDesVersion current() {
    return SerDesVersion(SerDesVersionNumber(0));
  }

  SerDesVersion(const SerDesVersion& other) = default;
  SerDesVersion& operator=(const SerDesVersion& other) = default;

  SerDesVersionNumber version_number() const { return version_number_; }

  bool operator==(const SerDesVersion& other) const {
    return version_number_ == other.version_number_;
  }
  bool operator!=(const SerDesVersion& other) const {
    return version_number_ != other.version_number_;
  }

 private:
  // Private constructor to restrict using old versions. Old versions are
  // necessary for realizing specific version compatibility.
  explicit SerDesVersion(SerDesVersionNumber version_number)
      : version_number_(version_number) {}

  // Returns the minimum supported version.
  static SerDesVersion minimum() {
    return SerDesVersion(SerDesVersionNumber(0));
  }

  // Returns a version that was introduced at least 4 weeks ago.
  static SerDesVersion week_4_old() {
    return SerDesVersion(SerDesVersionNumber(0));
  }

  // Visibility-controlled accessors that can use an old version.
  friend class SerDesAnyVersionAccessor;
  friend class SerDesWeek4OldVersionAccessor;

  SerDesVersionNumber version_number_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SERDES_VERSION_H_
