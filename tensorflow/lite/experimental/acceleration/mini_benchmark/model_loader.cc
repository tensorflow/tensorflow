/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/model_loader.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/lite/allocation.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/model_builder.h"
#include "tensorflow/lite/stderr_reporter.h"

namespace tflite {
namespace acceleration {

std::unique_ptr<ModelLoader> ModelLoader::CreateFromFdOrPath(
    absl::string_view fd_or_path) {
  if (!absl::StartsWith(fd_or_path, "fd:")) {
    return std::make_unique<ModelLoader>(fd_or_path);
  }

  std::vector<std::string> parts = absl::StrSplit(fd_or_path, ':');
  int model_fd;
  size_t model_offset, model_size;
  if (parts.size() != 4 || !absl::SimpleAtoi(parts[1], &model_fd) ||
      !absl::SimpleAtoi(parts[2], &model_offset) ||
      !absl::SimpleAtoi(parts[3], &model_size)) {
    return nullptr;
  }
  return std::make_unique<ModelLoader>(model_fd, model_offset, model_size);
}

MinibenchmarkStatus ModelLoader::Init() {
  if (model_) {
    // Already done.
    return kMinibenchmarkSuccess;
  }
  if (model_path_.empty() && model_fd_ <= 0) {
    return kMinibenchmarkPreconditionNotMet;
  }
  if (!model_path_.empty()) {
    model_ = FlatBufferModel::VerifyAndBuildFromFile(model_path_.c_str());
  } else if (MMAPAllocation::IsSupported()) {
    auto allocation = std::make_unique<MMAPAllocation>(
        model_fd_, model_offset_, model_size_, tflite::DefaultErrorReporter());
    if (!allocation->valid()) {
      return kMinibenchmarkModelReadFailed;
    }
    model_ =
        FlatBufferModel::VerifyAndBuildFromAllocation(std::move(allocation));
  } else {
    return kMinibenchmarkUnsupportedPlatform;
  }
  if (!model_) {
    return kMinibenchmarkModelBuildFailed;
  }
  return kMinibenchmarkSuccess;
}

}  // namespace acceleration
}  // namespace tflite
