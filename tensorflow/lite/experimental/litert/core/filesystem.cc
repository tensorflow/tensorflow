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

#include "tensorflow/lite/experimental/litert/core/filesystem.h"

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <fstream>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_detail.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_macros.h"

namespace litert::internal {

namespace {

using StdPath = std::filesystem::path;

StdPath MakeStdPath(absl::string_view path) {
  return StdPath(std::string(path.begin(), path.end()));
}

bool StdExists(const StdPath& std_path) {
  return std::filesystem::exists(std_path);
}

size_t StdSize(const StdPath& std_path) {
  return std::filesystem::file_size(std_path);
}

LiteRtStatus StdIFRead(const StdPath& std_path, char* data, size_t size) {
  std::ifstream in_file_stream(std_path, std::ifstream::binary);
  if (!in_file_stream) {
    return kLiteRtStatusErrorFileIO;
  }

  in_file_stream.read(data, size);
  if (!in_file_stream) {
    return kLiteRtStatusErrorFileIO;
  }

  in_file_stream.close();
  return kLiteRtStatusOk;
}

}  // namespace

void Touch(absl::string_view path) { std::ofstream(MakeStdPath(path)); }

std::string Join(const std::vector<absl::string_view>& paths) {
  StdPath std_path;
  for (auto subpath : paths) {
    std_path /= MakeStdPath(subpath);
  }
  return std_path.generic_string();
}

bool Exists(absl::string_view path) { return StdExists(MakeStdPath(path)); }

Expected<size_t> Size(absl::string_view path) {
  auto std_path = MakeStdPath(path);
  if (!StdExists(std_path)) {
    return Error(kLiteRtStatusErrorNotFound, "File not found");
  }
  return StdSize(std_path);
}

Expected<OwningBufferRef<uint8_t>> LoadBinaryFile(absl::string_view path) {
  auto std_path = MakeStdPath(path);

  if (!StdExists(std_path)) {
    return Error(kLiteRtStatusErrorFileIO, "File not found");
  }

  OwningBufferRef<uint8_t> buf(StdSize(std_path));
  LITERT_EXPECT_OK(StdIFRead(std_path, buf.StrData(), buf.Size()));

  return buf;
}

}  // namespace litert::internal
