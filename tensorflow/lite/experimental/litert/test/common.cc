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

#include <cstdint>
#include <cstring>
#include <filesystem>  // NOLINT
#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/absl_check.h"
#include "absl/strings/string_view.h"
#include "tensorflow/compiler/mlir/lite/allocation.h"
#include "tensorflow/lite/experimental/litert/c/litert_common.h"
#include "tensorflow/lite/experimental/litert/cc/litert_buffer_ref.h"
#include "tensorflow/lite/experimental/litert/cc/litert_expected.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model.h"
#include "tensorflow/lite/experimental/litert/cc/litert_model_predicates.h"
#include "tensorflow/lite/experimental/litert/core/model/model_buffer.h"
#include "tensorflow/lite/experimental/litert/core/model/model_load.h"
#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/stderr_reporter.h"
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

Expected<std::vector<char>> LoadBinaryFile(absl::string_view filename) {
  std::string model_path = GetTestFilePath(filename);
  ABSL_CHECK(std::filesystem::exists(model_path));
  auto size = std::filesystem::file_size(model_path);
  std::vector<char> buffer(size);
  std::ifstream f(model_path, std::ifstream::binary);
  if (!f) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to open file");
  }
  f.read(buffer.data(), buffer.size());
  if (!f) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure, "Failed to read file");
  }
  f.close();
  return buffer;
}

Model LoadTestFileModel(absl::string_view filename) {
  auto model = internal::LoadModelFromFile(GetTestFilePath(filename).data());
  return std::move(model.Value());
}

void TouchTestFile(absl::string_view filename, absl::string_view dir) {
  std::filesystem::path path(dir.data());
  path.append(filename.data());
  std::ofstream f(path);
}

bool ValidateTopology(const std::vector<Op>& ops) {
  for (const auto& op : ops) {
    const auto inputs = op.Inputs();
    for (int i = 0; i < inputs.size(); ++i) {
      if (!MatchUse(inputs.at(i), UseInfo{op.Code(), i})) {
        return false;
      }
    }
    const auto outputs = op.Outputs();
    for (int i = 0; i < outputs.size(); ++i) {
      const auto defining_op = outputs.at(i).DefiningOp();
      if (!defining_op.has_value()) {
        return false;
      }
      if (defining_op->op != op.Get() || defining_op->op_output_index != i) {
        return false;
      }
    }
  }
  return true;
}

Expected<OwningBufferRef<uint8_t>> GetModelBufWithByteCode(
    absl::string_view tfl_file, absl::string_view npu_file) {
  return internal::GetModelBufWithByteCode(GetTestFilePath(tfl_file),
                                           GetTestFilePath(npu_file));
}

Expected<TflRuntime::Ptr> TflRuntime::CreateFromTflFile(
    absl::string_view filename) {
  auto runtime = std::make_unique<TflRuntime>();

  {
    auto alloc = tflite::GetAllocationFromFile(filename.data(),
                                               tflite::DefaultErrorReporter());
    if (alloc == nullptr) {
      return Unexpected(kLiteRtStatusErrorFileIO);
    }
    runtime->alloc_ = std::move(alloc);
  }

  runtime->fb_model_ = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(runtime->alloc_->base()),
      runtime->alloc_->bytes());
  if (runtime->fb_model_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorFileIO);
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*runtime->fb_model_, resolver)(&runtime->interp_);
  if (runtime->interp_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }

  return runtime;
}

Expected<TflRuntime::Ptr> TflRuntime::CreateFromTflFileWithByteCode(
    absl::string_view tfl_filename, absl::string_view npu_filename) {
  auto runtime = std::make_unique<TflRuntime>();

  {
    auto model_with_byte_code =
        GetModelBufWithByteCode(tfl_filename, npu_filename);
    if (!model_with_byte_code) {
      return model_with_byte_code.Error();
    }
    runtime->model_buf_ = std::move(*model_with_byte_code);
  }

  runtime->alloc_ = std::make_unique<tflite::MemoryAllocation>(
      runtime->model_buf_.Data(), runtime->model_buf_.Size(),
      tflite::DefaultErrorReporter());

  runtime->fb_model_ = tflite::FlatBufferModel::BuildFromBuffer(
      reinterpret_cast<const char*>(runtime->alloc_->base()),
      runtime->alloc_->bytes());
  if (runtime->fb_model_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorFileIO);
  }

  tflite::ops::builtin::BuiltinOpResolver resolver;
  tflite::InterpreterBuilder(*runtime->fb_model_, resolver)(&runtime->interp_);
  if (runtime->interp_ == nullptr) {
    return Unexpected(kLiteRtStatusErrorRuntimeFailure);
  }

  return runtime;
}

}  // namespace testing
}  // namespace litert
