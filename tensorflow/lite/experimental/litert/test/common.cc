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

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model_predicates.h"
#include "tensorflow/lite/experimental/litert/core/filesystem.h"
#include "tensorflow/lite/experimental/litert/core/util/flatbuffer_tools.h"
#include "tensorflow/lite/interpreter.h"
#include "tensorflow/lite/kernels/register.h"
#include "tsl/platform/platform.h"

namespace litert {
namespace testing {

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

}  // namespace testing
}  // namespace litert
