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

#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_UTILS_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_UTILS_H_

#include <string>
#include <unordered_set>
#include <vector>

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#include "tensorflow/lite/context.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#include "tensorflow/lite/model.h"

namespace tflite {
namespace evaluation {
std::string StripTrailingSlashes(const std::string& path);

bool ReadFileLines(const std::string& file_path,
                   std::vector<std::string>* lines_output);

// If extension set is empty, all files will be listed. The strings in
// extension set are expected to be in lowercase and include the dot.
TfLiteStatus GetSortedFileNames(
    const std::string& directory, std::vector<std::string>* result,
    const std::unordered_set<std::string>& extensions);

inline TfLiteStatus GetSortedFileNames(const std::string& directory,
                                       std::vector<std::string>* result) {
  return GetSortedFileNames(directory, result,
                            std::unordered_set<std::string>());
}

Interpreter::TfLiteDelegatePtr CreateNNAPIDelegate();

Interpreter::TfLiteDelegatePtr CreateNNAPIDelegate(
    StatefulNnApiDelegate::Options options);

Interpreter::TfLiteDelegatePtr CreateGPUDelegate();
#if defined(__ANDROID__)
Interpreter::TfLiteDelegatePtr CreateGPUDelegate(
    TfLiteGpuDelegateOptionsV2* options);
#endif

}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_UTILS_H_
