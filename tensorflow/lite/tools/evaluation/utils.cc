/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/evaluation/utils.h"

#include <dirent.h>
#include <sys/stat.h>

#include <algorithm>
#include <fstream>
#include <memory>
#include <string>

#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/gl_delegate.h"
#endif

namespace tflite {
namespace evaluation {

std::string StripTrailingSlashes(const std::string& path) {
  int end = path.size();
  while (end > 0 && path[end - 1] == '/') {
    end--;
  }
  return path.substr(0, end);
}

bool ReadFileLines(const std::string& file_path,
                   std::vector<std::string>* lines_output) {
  if (!lines_output) {
    return false;
  }
  std::ifstream stream(file_path.c_str());
  if (!stream) {
    return false;
  }
  std::string line;
  while (std::getline(stream, line)) {
    lines_output->push_back(line);
  }
  return true;
}

TfLiteStatus GetSortedFileNames(const std::string& directory,
                                std::vector<std::string>* result) {
  DIR* dir;
  struct dirent* ent;
  if (result == nullptr) {
    return kTfLiteError;
  }
  result->clear();
  std::string dir_path = StripTrailingSlashes(directory);
  if ((dir = opendir(dir_path.c_str())) != nullptr) {
    while ((ent = readdir(dir)) != nullptr) {
      std::string filename(std::string(ent->d_name));
      if (filename.size() <= 2) continue;
      result->emplace_back(dir_path + "/" + filename);
    }
    closedir(dir);
  } else {
    return kTfLiteError;
  }
  std::sort(result->begin(), result->end());
  return kTfLiteOk;
}

Interpreter::TfLiteDelegatePtr CreateNNAPIDelegate() {
#if defined(__ANDROID__)
  return Interpreter::TfLiteDelegatePtr(
      NnApiDelegate(),
      // NnApiDelegate() returns a singleton, so provide a no-op deleter.
      [](TfLiteDelegate*) {});
#else
  return Interpreter::TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
#endif  // defined(__ANDROID__)
}

Interpreter::TfLiteDelegatePtr CreateGPUDelegate(
    tflite::FlatBufferModel* model) {
#if defined(__ANDROID__)
  TfLiteGpuDelegateOptions options;
  options.metadata = TfLiteGpuDelegateGetModelMetadata(model->GetModel());
  options.compile_options.precision_loss_allowed = 1;
  options.compile_options.preferred_gl_object_type =
      TFLITE_GL_OBJECT_TYPE_FASTEST;
  options.compile_options.dynamic_batch_enabled = 0;
  return Interpreter::TfLiteDelegatePtr(TfLiteGpuDelegateCreate(&options),
                                        &TfLiteGpuDelegateDelete);
#else
  return Interpreter::TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
#endif  // defined(__ANDROID__)
}

}  // namespace evaluation
}  // namespace tflite
