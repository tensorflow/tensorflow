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

#ifndef TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_COMMON_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_COMMON_H_

#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/core/litert_model_init.h"

#define _ASSERT_RESULT_OK_ASSIGN(decl, expr, result) \
  auto result = (expr);                              \
  ASSERT_TRUE(result.HasValue());                    \
  decl = result.Value();

#define ASSERT_RESULT_OK_ASSIGN(decl, expr) \
  _ASSERT_RESULT_OK_ASSIGN(decl, expr, _CONCAT_NAME(_result, __COUNTER__))

#define _ASSERT_RESULT_OK_MOVE(decl, expr, result) \
  auto result = (expr);                            \
  ASSERT_TRUE(result.HasValue());                  \
  decl = std::move(result.Value());

#define ASSERT_RESULT_OK_MOVE(decl, expr) \
  _ASSERT_RESULT_OK_MOVE(decl, expr, _CONCAT_NAME(_result, __COUNTER__))

#define ASSERT_STATUS_HAS_CODE(expr, code) \
  {                                        \
    LiteRtStatus status = (expr);          \
    ASSERT_EQ(status, code);               \
  }

#define ASSERT_STATUS_OK(expr) ASSERT_STATUS_HAS_CODE(expr, kLiteRtStatusOk);

namespace litert {
namespace testing {

std::string GetTestFilePath(absl::string_view filename);

absl::StatusOr<std::vector<char>> LoadBinaryFile(absl::string_view filename);

internal::UniqueLiteRtModel LoadTestFileModel(absl::string_view filename);

void TouchTestFile(absl::string_view filename, absl::string_view dir);

}  // namespace testing
}  // namespace litert

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_COMMON_H_
