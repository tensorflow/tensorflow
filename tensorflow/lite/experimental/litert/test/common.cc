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

#include "tensorflow/lite/experimental/litert/test/common.h"

// NOLINTNEXTLINE
#include <filesystem>
#include <fstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tsl/platform/platform.h"

namespace litert {
namespace testing {

std::string GetTestFilePath(absl::string_view filename) {
  static constexpr std::string_view kTestDataDir =
      "tensorflow/lite/experimental/litert/"
      "test/testdata/";

  std::filesystem::path result_path;
  if constexpr (!tsl::kIsOpenSource) {
    result_path.append("third_party");
  }

  result_path.append(kTestDataDir);
  result_path.append(filename.data());

  return result_path.generic_string();
}

absl::StatusOr<std::vector<char>> LoadBinaryFile(absl::string_view filename) {
  std::string model_path = GetTestFilePath(filename);
  ABSL_CHECK(std::filesystem::exists(model_path));
  auto size = std::filesystem::file_size(model_path);
  std::vector<char> buffer(size);
  std::ifstream f(model_path, std::ifstream::binary);
  if (!f) {
    return absl::InternalError("Failed to open file");
  }
  f.read(buffer.data(), buffer.size());
  if (!f) {
    return absl::InternalError("Failed to read file");
  }
  f.close();
  return buffer;
}

Model LoadTestFileModel(absl::string_view filename) {
  auto model_result = internal::LoadModelFromFile(filename);

  auto model = internal::LoadModelFromFile(GetTestFilePath(filename).data());
  ABSL_CHECK_EQ(model.Status(), kLiteRtStatusOk);
  return std::move(model.Value());
}

void TouchTestFile(absl::string_view filename, absl::string_view dir) {
  std::filesystem::path path(dir.data());
  path.append(filename.data());
  std::ofstream f(path);
}

}  // namespace testing
}  // namespace litert
