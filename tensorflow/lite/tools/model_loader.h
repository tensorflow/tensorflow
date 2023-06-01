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
#ifndef TENSORFLOW_LITE_TOOLS_MODEL_LOADER_H_
#define TENSORFLOW_LITE_TOOLS_MODEL_LOADER_H_

#ifndef _WIN32
#include <unistd.h>
#endif  // !_WIN32

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <string>
#include <vector>

#include "absl/strings/string_view.h"
#include "tensorflow/lite/core/model_builder.h"

namespace tflite {
namespace tools {

// Class to load the Model.
class ModelLoader {
 public:
  virtual ~ModelLoader() = default;

  // Return whether the model is loaded successfully.
  virtual bool Init();

  const FlatBufferModel* GetModel() const { return model_.get(); }

  // Return whether the FlatBufferModel is created from FlatbufferBuilder
  // directly.
  virtual bool IsLoadedFromFlatbufferBuilder() = 0;

 protected:
  // Interface for subclass to create model_. Init() calls InitInternal(). If
  // InitInternal() returns false, or if it returns true but model_ remains
  // null, then Init() will return false.
  virtual bool InitInternal() = 0;

  std::unique_ptr<FlatBufferModel> model_;
};

// Load the Model from a file path.
class PathModelLoader : public ModelLoader {
 public:
  explicit PathModelLoader(absl::string_view model_path)
      : ModelLoader(), model_path_(model_path) {}

  bool IsLoadedFromFlatbufferBuilder() override { return false; }

 protected:
  bool InitInternal() override;

 private:
  const std::string model_path_;
};

// Load the Model from buffer. The buffer is owned by the caller.
class BufferModelLoader : public ModelLoader {
 public:
  BufferModelLoader(const char* caller_owned_buffer, size_t model_size)
      : caller_owned_buffer_(caller_owned_buffer), model_size_(model_size) {}

  // Move only.
  BufferModelLoader(BufferModelLoader&&) = default;
  BufferModelLoader& operator=(BufferModelLoader&&) = default;

  ~BufferModelLoader() override = default;

  bool IsLoadedFromFlatbufferBuilder() override { return false; }

 protected:
  bool InitInternal() override;

 private:
  const char* caller_owned_buffer_ = nullptr;
  size_t model_size_ = 0;
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

  bool IsLoadedFromFlatbufferBuilder() override { return false; }

 protected:
  bool InitInternal() override;

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

  bool IsLoadedFromFlatbufferBuilder() override { return true; }

 protected:
  // Reads the serialized Model from read_pipe_fd. Returns false if the number
  // of bytes read in is less than read_size. This function also closes the
  // read_pipe_fd and write_pipe_fd.
  bool InitInternal() override;

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
// write_pipe when write_pipe >= 0, so it should be called at the read thread /
// process. Returns null if path cannot be parsed.
// 3) File path: Always return a PathModelLoader.
// 4) Buffer path: path must be in the format of
// "buffer:%buffer_handle%:%buffer_size%". This model loader does not own the
// buffer_handle, and the caller needs to ensure the buffer_handle out-lives the
// model loader.
std::unique_ptr<ModelLoader> CreateModelLoaderFromPath(const std::string& path);

}  // namespace tools
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_MODEL_LOADER_H_
