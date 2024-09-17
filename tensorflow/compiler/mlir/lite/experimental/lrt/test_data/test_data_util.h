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

#ifndef TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_TEST_DATA_TEST_DATA_UTIL_H_
#define TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_TEST_DATA_TEST_DATA_UTIL_H_

// NOLINTNEXTLINE
#include <filesystem>
#include <string>
#include <string_view>

#include "absl/log/check.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/c/lite_rt_model.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/cc/lite_rt_support.h"
#include "tensorflow/compiler/mlir/lite/experimental/lrt/core/lite_rt_model_init.h"
#include "tsl/platform/platform.h"

#define _ASSERT_RESULT_OK_ASSIGN(decl, expr, result) \
  auto result = (expr);                              \
  ASSERT_TRUE(result.HasValue());                    \
  decl = result.Value();

#define ASSERT_RESULT_OK_ASSIGN(decl, expr) \
  _ASSERT_RESULT_OK_ASSIGN(decl, expr, _CONCAT_NAME(_result, __COUNTER__))

#define ASSERT_STATUS_HAS_CODE(expr, code) \
  {                                        \
    auto stat = (expr);                    \
    auto code_ = GetStatusCode(stat);      \
    StatusDestroy(stat);                   \
    ASSERT_EQ(code_, code);                \
  }

#define ASSERT_STATUS_OK(expr) ASSERT_STATUS_HAS_CODE(expr, kLrtStatusOk);

inline std::string GetTestFilePath(std::string_view filename) {
  static constexpr std::string_view kTestDataDir =
      "tensorflow/compiler/mlir/lite/experimental/lrt/"
      "test_data/";

  std::filesystem::path result_path;
  if constexpr (!tsl::kIsOpenSource) {
    result_path.append("third_party");
  }

  result_path.append(kTestDataDir);
  result_path.append(filename);

  return result_path.generic_string();
}

inline UniqueLrtModel LoadTestFileModel(std::string_view filename) {
  LrtModel model = nullptr;
  LRT_CHECK_STATUS_OK(
      LoadModelFromFile(GetTestFilePath(filename).c_str(), &model));
  CHECK_NE(model, nullptr);
  return UniqueLrtModel(model);
}

#endif  // TENSORFLOW_COMPILER_MLIR_LITE_EXPERIMENTAL_LRT_TEST_DATA_TEST_DATA_UTIL_H_
