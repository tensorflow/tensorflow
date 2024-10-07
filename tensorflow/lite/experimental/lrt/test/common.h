// Copyright 2024 Google LLC.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LRT_TEST_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LRT_TEST_COMMON_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"

namespace lrt {
namespace testing {

std::string GetTestFilePath(absl::string_view filename);
absl::StatusOr<std::vector<char>> LoadBinaryFile(absl::string_view file_name);

}  // namespace testing
}  // namespace lrt

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LRT_TEST_COMMON_H_
