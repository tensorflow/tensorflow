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

#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/experimental/acceleration/mini_benchmark/status_codes.h"
#include "tensorflow/lite/model_builder.h"

namespace tflite {
namespace acceleration {

// Class to load the Model.
class ModelLoader {
 public:
  virtual ~ModelLoader() {}

  // Return whether the model is loaded successfully.
  virtual MinibenchmarkStatus Init();

  const FlatBufferModel* GetModel() const { return model_.get(); }

 protected:
  // ModelLoader() = default;

  // Interface for subclass to create model_. If failed, Init() will return the
  // error status; If succeeded but model_ is null, Init() function will return
  // ModelBuildFailed.
  virtual MinibenchmarkStatus InitInternal() = 0;

  std::unique_ptr<FlatBufferModel> model_;
};

// Load the Model from a file path.
class PathModelLoader : public ModelLoader {
 public:
  explicit PathModelLoader(absl::string_view model_path)
      : ModelLoader(), model_path_(model_path) {}

 protected:
  MinibenchmarkStatus InitInternal() override;

 private:
  const std::string model_path_;
};

#ifndef _WIN32
// Load the Model from a file descriptor. This class is not available on
// Windows.
class MmapModelLoader : public ModelLoader {
 public:
  // Create the model loader from file descriptor. The model_fd only has to be
  // valid for the duration of the constructor (it's dup'ed inside).
  MmapModelLoader(int model_fd, size_t model_offset, size_t model_size)
      : ModelLoader(),
        model_fd_(dup(model_fd)),
        model_offset_(model_offset),
        model_size_(model_size) {}

  ~MmapModelLoader() override {
    if (model_fd_ >= 0) {
      close(model_fd_);
    }
  }

 protected:
  MinibenchmarkStatus InitInternal() override;

 private:
  const int model_fd_ = -1;
  const size_t model_offset_ = 0;
  const size_t model_size_ = 0;
};

// Load the Model from a pipe file descriptor.
// IMPORTANT: This class tries to read the model from a pipe file descriptor,
// and the caller needs to ensure that this pipe should be read from in a
// different process / thread than written to. It may block when running in the
// same process / thread.
class PipeModelLoader : public ModelLoader {
 public:
  PipeModelLoader(int pipe_fd, size_t model_size)
      : ModelLoader(), pipe_fd_(pipe_fd), model_size_(model_size) {}

  // Move only.
  PipeModelLoader(PipeModelLoader&&) = default;
  PipeModelLoader& operator=(PipeModelLoader&&) = default;

  ~PipeModelLoader() override { std::free(model_buffer_); }

 protected:
  // Read the serialized Model from read_pipe_fd. Return ModelReadFailed if the
  // readin bytes is less than read_size. This function also closes the
  // read_pipe_fd and write_pipe_fd.
  MinibenchmarkStatus InitInternal() override;

 private:
  const int pipe_fd_ = -1;
  const size_t model_size_ = 0;
  uint8_t* model_buffer_ = nullptr;
};

#endif  // !_WIN32

// Create the model loader from a string path. Path can be one of the following:
// 1) File descriptor path: path must be in the format of
// "fd:%model_fd%:%model_offset%:%model_size%". Returns null if path cannot be
// parsed.
// 2) Pipe descriptor path: path must be in the format of
// "pipe:%read_pipe%:%write_pipe%:%model_size%". This function also closes the
// write_pipe for the caller, so it should be called at the read thread /
// process. Returns null if path cannot be parsed.
// 3) File path: Always return a PathModelLoader.
// NOTE: This helper function is designed for creating the ModelLoader from
// command line parameters. Prefer to use the ModelLoader constructors directly
// when possible.
std::unique_ptr<ModelLoader> CreateModelLoaderFromPath(absl::string_view path);

}  // namespace acceleration
}  // namespace tflite

#endif  // TENSORFLOW_LITE_EXPERIMENTAL_ACCELERATION_MINI_BENCHMARK_MODEL_LOADER_H_
