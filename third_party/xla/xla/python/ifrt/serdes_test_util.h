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

#ifndef XLA_PYTHON_IFRT_SERDES_TEST_UTIL_H_
#define XLA_PYTHON_IFRT_SERDES_TEST_UTIL_H_

#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {
namespace test_util {

// A mixin that provides access to the SerDes version for parameterized tests.
class SerDesVersionMixin {
 public:
  explicit SerDesVersionMixin(int version) : version_(SerDesVersion(version)) {}

  SerDesVersion version() const { return version_; }

 private:
  SerDesVersion version_;
};

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SERDES_TEST_UTIL_H_
