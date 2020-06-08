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

#include <memory>
#include <string>
#include <unordered_set>
#include <vector>

#if defined(__ANDROID__)
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#if (defined(__arm__) || defined(__aarch64__))
#include "tensorflow/lite/experimental/delegates/hexagon/hexagon_delegate.h"
#endif
#endif

// TODO(b/149248802): include XNNPACK delegate when the issue is resolved.
#if !defined(__Fuchsia__) || defined(TFLITE_WITHOUT_XNNPACK)
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace evaluation {

// Same w/ Interpreter::TfLiteDelegatePtr to avoid pulling
// tensorflow/lite/interpreter.h dependency
using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>;

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

TfLiteDelegatePtr CreateNNAPIDelegate();
#if defined(__ANDROID__)
TfLiteDelegatePtr CreateNNAPIDelegate(StatefulNnApiDelegate::Options options);
#endif

TfLiteDelegatePtr CreateGPUDelegate();
#if defined(__ANDROID__)
TfLiteDelegatePtr CreateGPUDelegate(TfLiteGpuDelegateOptionsV2* options);
#endif

TfLiteDelegatePtr CreateHexagonDelegate(
    const std::string& library_directory_path, bool profiling);
#if defined(__ANDROID__) && (defined(__arm__) || defined(__aarch64__))
TfLiteDelegatePtr CreateHexagonDelegate(
    const TfLiteHexagonDelegateOptions* options,
    const std::string& library_directory_path);
#endif

// TODO(b/149248802): include XNNPACK delegate when the issue is resolved.
#if !defined(__Fuchsia__) || defined(TFLITE_WITHOUT_XNNPACK)
TfLiteDelegatePtr CreateXNNPACKDelegate();
TfLiteDelegatePtr CreateXNNPACKDelegate(
    const TfLiteXNNPackDelegateOptions* options);
#endif
TfLiteDelegatePtr CreateXNNPACKDelegate(int num_threads);
}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_UTILS_H_
