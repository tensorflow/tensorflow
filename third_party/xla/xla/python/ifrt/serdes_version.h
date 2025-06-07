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

namespace xla {
namespace ifrt {

// Represents a version of the IFRT serialization format.
class SerDesVersion {
 public:
  // List of SerDes versions and introduction dates.
  enum {
    kV0Initial = 0,  // 2025-05-20, initial version.
  };

  // Returns the current version.
  static SerDesVersion current() { return SerDesVersion(kV0Initial); }

  SerDesVersion(const SerDesVersion& other) = default;
  SerDesVersion& operator=(const SerDesVersion& other) = default;

  int version() const { return version_; }

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const SerDesVersion& version) {
    sink.Append(absl::StrCat("IFRT SerDes version ", version.version()));
  }

 private:
  // Private constructor to restrict using old versions. Old versions are
  // necessary for realizing specific version compatibility.
  explicit SerDesVersion(int version) : version_(version) {}

  // Returns the minimum supported version.
  static SerDesVersion minimum() { return SerDesVersion(kV0Initial); }

  // Returns a version that was introduced at least 4 weeks ago.
  static SerDesVersion week_4_old() { return SerDesVersion(kV0Initial); }

  // Visibility-controlled accessors that can use an old version.
  friend class SerDesAnyVersionAccessor;
  friend class SerDesWeek4OldVersionAccessor;

  int version_;
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SERDES_VERSION_H_
