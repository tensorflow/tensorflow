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

#include <vector>

#include "xla/python/ifrt/serdes_version.h"

namespace xla {
namespace ifrt {
namespace test_util {

// Returns all supported SerDes versions. Expected to be used to construct the
// test parameters for parameterized tests.
std::vector<SerDesVersion> AllSupportedSerDesVersions();

// Returns all supported SerDes versions that are no more than 4 weeks old.
// Expected to be used to construct the test parameters for parameterized tests
// where serialization only supports up to 4 weeks old formats.
std::vector<SerDesVersion> Week4OldOrLaterSerDesVersions();

}  // namespace test_util
}  // namespace ifrt
}  // namespace xla

#endif  // XLA_PYTHON_IFRT_SERDES_TEST_UTIL_H_
