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

#ifndef XLA_PYTHON_IFRT_SERDES_DEFAULT_VERSION_ACCESSOR_H_
#define XLA_PYTHON_IFRT_SERDES_DEFAULT_VERSION_ACCESSOR_H_

#include "absl/log/check.h"  // IWYU pragma: keep (used when IFRT_TESTING_BAD_DEFAULT_SERDES_VERSION is defined)
#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {

// Accessor for `SerDesVersion` that picks the default version for
// `SerializeOptions` and `ToProto` methods.
//
// When `IFRT_TESTING_BAD_DEFAULT_SERDES_VERSION` macro is defined, it uses a
// bad `SerDesVersion`. Any code that uses the default version will fail. This
// is useful for testing the code that always explicitly specifies the version,
// e.g., in IFRT Proxy, to find any missing plumbing of the SerDes version.
class SerDesDefaultVersionAccessor {
 public:
#ifdef IFRT_TESTING_BAD_DEFAULT_SERDES_VERSION
  static SerDesVersion Get() {
    CHECK(false) << "Using a default `SerDesVersion` is disallowed when "
                    "`IFRT_TESTING_BAD_DEFAULT_SERDES_VERSION` is defined.";
  }
#else
  static SerDesVersion Get() { return SerDesVersion::current(); }
#endif
};

}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SERDES_DEFAULT_VERSION_ACCESSOR_H_
