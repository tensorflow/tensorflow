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

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/strings/match.h"
#include "absl/strings/numbers.h"
#include "absl/strings/str_split.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/minimal_logging.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {
namespace acceleration {

MinibenchmarkStatus ModelLoader::Init() {
  if (model_) {
    // Already done.
    return kMinibenchmarkSuccess;
  }
  MinibenchmarkStatus status = InitInternal();
  if (status != kMinibenchmarkSuccess) {
    return status;
  }
  if (!model_) {
    return kMinibenchmarkModelBuildFailed;
  }
  return kMinibenchmarkSuccess;
}

MinibenchmarkStatus PathModelLoader::InitInternal() {
  if (model_path_.empty()) {
    return kMinibenchmarkPreconditionNotMet;
  }
  model_ = FlatBufferModel::VerifyAndBuildFromFile(model_path_.c_str());
  return kMinibenchmarkSuccess;
}

#ifndef _WIN32

MinibenchmarkStatus MmapModelLoader::InitInternal() {
  if (model_fd_ < 0 || model_offset_ < 0 || model_size_ < 0) {
    return kMinibenchmarkModelReadFailed;
  }
  if (!MMAPAllocation::IsSupported()) {
    return kMinibenchmarkUnsupportedPlatform;
  }
  auto allocation = std::make_unique<MMAPAllocation>(
      model_fd_, model_offset_, model_size_, tflite::DefaultErrorReporter());
  if (!allocation->valid()) {
    return kMinibenchmarkModelReadFailed;
  }
  model_ = FlatBufferModel::VerifyAndBuildFromAllocation(std::move(allocation));
  return kMinibenchmarkSuccess;
}

MinibenchmarkStatus PipeModelLoader::InitInternal() {
  if (pipe_fd_ < 0) {
    return kMinibenchmarkModelReadFailed;
  }

  std::free(model_buffer_);
  model_buffer_ = reinterpret_cast<uint8_t*>(std::malloc(model_size_));

  int read_bytes = 0;
  int remaining_bytes = model_size_;
  uint8_t* buffer = model_buffer_;
  while (remaining_bytes > 0 &&
         (read_bytes = read(pipe_fd_, buffer, remaining_bytes)) > 0) {
    remaining_bytes -= read_bytes;
    buffer += read_bytes;
  }
  // Close the read pipe.
  close(pipe_fd_);
  if (read_bytes < 0 || remaining_bytes != 0) {
    TFLITE_LOG_PROD(TFLITE_LOG_INFO,
                    "Read Model from pipe failed: %s. Expect to read %d bytes, "
                    "%d bytes missing.",
                    std::strerror(errno), model_size_, remaining_bytes);
    // If read() failed with -1, or read partial or too much data.
    return kMinibenchmarkModelReadFailed;
  }

  model_ = FlatBufferModel::BuildFromModel(tflite::GetModel(model_buffer_));
  return kMinibenchmarkSuccess;
}

#endif  // !_WIN32

std::unique_ptr<ModelLoader> CreateModelLoaderFromPath(absl::string_view path) {
  if (absl::StartsWith(path, "fd:")) {
#ifdef _WIN32
    return kMinibenchmarkUnsupportedPlatform;
#endif  // _WIN32
    std::vector<std::string> parts = absl::StrSplit(path, ':');
    int model_fd;
    size_t model_offset, model_size;
    if (parts.size() != 4 || !absl::SimpleAtoi(parts[1], &model_fd) ||
        !absl::SimpleAtoi(parts[2], &model_offset) ||
        !absl::SimpleAtoi(parts[3], &model_size)) {
      return nullptr;
    }
    return std::make_unique<MmapModelLoader>(model_fd, model_offset,
                                             model_size);
  }
  if (absl::StartsWith(path, "pipe:")) {
#ifdef _WIN32
    return kMinibenchmarkUnsupportedPlatform;
#endif  // _WIN32
    std::vector<std::string> parts = absl::StrSplit(path, ':');
    int read_fd, write_fd;
    size_t model_size;
    if (parts.size() != 4 || !absl::SimpleAtoi(parts[1], &read_fd) ||
        !absl::SimpleAtoi(parts[2], &write_fd) ||
        !absl::SimpleAtoi(parts[3], &model_size)) {
      return nullptr;
    }
    // If set, close the write pipe for the read process / thread.
    if (write_fd >= 0) {
      close(write_fd);
    }
    return std::make_unique<PipeModelLoader>(read_fd, model_size);
  }
  return std::make_unique<PathModelLoader>(path);
}

}  // namespace acceleration
}  // namespace tflite
