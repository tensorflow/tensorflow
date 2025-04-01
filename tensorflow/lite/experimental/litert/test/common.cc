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

#include <cstddef>
#include <cstdint>
#include <filesystem>  // NOLINT
#include <ios>
#include <memory>
#include <random>
#include <sstream>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/attributes.h"
#include "absl/base/const_init.h"
#include "absl/strings/string_view.h"
#include "absl/synchronization/mutex.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/c/litert_logging.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model_predicates.h"
#include "tensorflow/lite/experimental/litert/core/filesystem.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tsl/platform/platform.h"

namespace litert::testing {

Expected<UniqueTestDirectory> UniqueTestDirectory::Create() {
  constexpr size_t kMaxTries = 1000;
  ABSL_CONST_INIT static absl::Mutex mutex(absl::kConstInit);

  // We don't want multiple threads to create the same directory.
  absl::MutexLock l(&mutex);

  auto tmp_dir = std::filesystem::temp_directory_path();
  std::random_device dev;
  std::mt19937 prng(dev());
  std::uniform_int_distribution<uint64_t> rand(0);
  std::stringstream ss;

  for (auto i = 0; i < kMaxTries; ++i) {
    ss.clear();
    ss << std::hex << rand(prng);
    auto path = tmp_dir / ss.str();
    if (std::filesystem::create_directory(path)) {
      LITERT_LOG(LITERT_INFO, "Created unique temporary directory %s",
                 path.c_str());
      return UniqueTestDirectory(path);
    }
  }

  return Error(kLiteRtStatusErrorRuntimeFailure,
               "Could not create a unique temporary directory");
}

UniqueTestDirectory::~UniqueTestDirectory() {
  std::filesystem::remove_all(tmpdir_);
}

std::string GetTestFilePath(absl::string_view filename) {
  static constexpr absl::string_view kTestDataDir =
      "tensorflow/lite/experimental/litert/"
      "test/testdata/";

  if constexpr (!tsl::kIsOpenSource) {
    return internal::Join({"third_party", kTestDataDir, filename});
  } else {
    return internal::Join({kTestDataDir, filename});
  }
}

std::string GetTfliteFilePath(absl::string_view filename) {
  static constexpr absl::string_view kTestDataDir = "tensorflow/lite/";

  if constexpr (!tsl::kIsOpenSource) {
    return internal::Join({"third_party", kTestDataDir, filename});
  } else {
    return internal::Join({kTestDataDir, filename});
  }
}

std::string GetLiteRtPath(absl::string_view rel_path) {
  static constexpr absl::string_view kLiteRtRoot =
      "tensorflow/lite/experimental/litert/";

  if constexpr (!tsl::kIsOpenSource) {
    return internal::Join({"third_party", kLiteRtRoot, rel_path});
  } else {
    return internal::Join({kLiteRtRoot, rel_path});
  }
}

Model LoadTestFileModel(absl::string_view filename) {
  return *Model::CreateFromFile(GetTestFilePath(filename));
}

Expected<TflRuntime::Ptr> TflRuntime::CreateFromFlatBuffer(
    internal::FlatbufferWrapper::Ptr flatbuffer) {
  ::tflite::Interpreter::Ptr interp;
  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(flatbuffer->FlatbufferModel(), resolver)(&interp);
  if (interp == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }
  return TflRuntime::Ptr(
      new TflRuntime(std::move(flatbuffer), std::move(interp)));
}

}  // namespace litert::testing
