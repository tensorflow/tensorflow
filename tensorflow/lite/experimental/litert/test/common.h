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
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/core/model/model_buffer.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/interpreter.h"

namespace litert::testing {

// A x-platform compatible replacement for testing::UniqueTestDirectory.
class UniqueTestDirectory {
 public:
  static Expected<UniqueTestDirectory> Create();
  ~UniqueTestDirectory();

  UniqueTestDirectory(const UniqueTestDirectory&) = delete;
  UniqueTestDirectory(UniqueTestDirectory&&) = default;
  UniqueTestDirectory& operator=(const UniqueTestDirectory&) = delete;
  UniqueTestDirectory& operator=(UniqueTestDirectory&&) = default;

  absl::string_view Str() const { return tmpdir_; }

 private:
  explicit UniqueTestDirectory(std::string&& tmpdir)
      : tmpdir_(std::move(tmpdir)) {}
  std::string tmpdir_;
};

// Gets the path to the given filename in the testdata directory.
std::string GetTestFilePath(absl::string_view filename);

// Gets a path to the given filename in the tflite directory.
std::string GetTfliteFilePath(absl::string_view filename);

// Gets a full path given a path relative to the litert directory.
std::string GetLiteRtPath(absl::string_view rel_path);

Model LoadTestFileModel(absl::string_view filename);

class TflRuntime {
 public:
  using Ptr = std::unique_ptr<TflRuntime>;

  static Expected<Ptr> CreateFromFlatBuffer(
      internal::FlatbufferWrapper::Ptr flatbuffer);

  ::tflite::Interpreter& Interpreter() { return *interpreter_; }

  const internal::FlatbufferWrapper& Flatbuffer() const { return *flatbuffer_; }

 private:
  TflRuntime(internal::FlatbufferWrapper::Ptr flatbuffer,
             ::tflite::Interpreter::Ptr interpreter)
      : flatbuffer_(std::move(flatbuffer)),
        interpreter_(std::move(interpreter)) {}

  internal::FlatbufferWrapper::Ptr flatbuffer_;
  ::tflite::Interpreter::Ptr interpreter_;
};

inline Expected<TflRuntime::Ptr> MakeRuntimeFromTestFile(
    absl::string_view filename) {
  auto flatbuffer =
      internal::FlatbufferWrapper::CreateFromTflFile(GetTestFilePath(filename));
  if (!flatbuffer) {
    return flatbuffer.Error();
  }
  return TflRuntime::CreateFromFlatBuffer(std::move(*flatbuffer));
}

inline Expected<TflRuntime::Ptr> MakeRuntimeFromTestFileWithNpuModel(
    absl::string_view filename, absl::string_view npu_filename) {
  auto buf = internal::GetModelBufWithByteCode(GetTestFilePath(filename),
                                               GetTestFilePath(npu_filename));
  if (!buf) {
    return buf.Error();
  }
  auto flatbuffer =
      internal::FlatbufferWrapper::CreateFromBuffer(std::move(*buf));
  if (!flatbuffer) {
    return flatbuffer.Error();
  }
  return TflRuntime::CreateFromFlatBuffer(std::move(*flatbuffer));
}

}  // namespace litert::testing

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_LITERT_TEST_COMMON_H_
