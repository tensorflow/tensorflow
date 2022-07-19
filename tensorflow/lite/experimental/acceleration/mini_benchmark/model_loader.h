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
#ifndef TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_LOADER_H_
#define TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_LOADER_H_

#include <stddef.h>
#include <unistd.h>

#include <memory>
#include <string>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {
namespace acceleration {

// Class to load the Model.
class ModelLoader {
 public:
  // Create the model loader from a model_path or a file descriptor. File
  // descriptor path must be in the format of
  // "fd:%model_fd%:%model_offset%:%model_size%". Return nullptr if the path
  // starts with "fd:" but cannot be parsed with the given format.
  static std::unique_ptr<ModelLoader> CreateFromFdOrPath(
      absl::string_view fd_or_path);

  // Create the model loader from model_path.
  explicit ModelLoader(absl::string_view model_path)
      : model_path_(model_path) {}

#ifndef _WIN32
  // Create the model loader from file descriptor. The model_fd only has to be
  // valid for the duration of the constructor (it's dup'ed inside). This
  // constructor is not available on Windows.
  ModelLoader(int model_fd, size_t model_offset, size_t model_size)
      : model_fd_(dup(model_fd)),
        model_offset_(model_offset),
        model_size_(model_size) {}
#endif  // !_WIN32

  ~ModelLoader() {
    if (model_fd_ >= 0) {
      close(model_fd_);
    }
  }

  // Return whether the model is loaded successfully.
  MinibenchmarkStatus Init();

  const FlatBufferModel* GetModel() const { return model_.get(); }

 private:
  const std::string model_path_;
  const int model_fd_ = -1;
  const size_t model_offset_ = 0;
  const size_t model_size_ = 0;
  std::unique_ptr<FlatBufferModel> model_;
};

}  // namespace acceleration

}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_LOADER_H_
