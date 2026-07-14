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

#if !TFLITE_WITH_STABLE_ABI
// TODO(b/240438534): enable nnapi.
#if defined(__ANDROID__)
#define TFLITE_SUPPORTS_NNAPI_DELEGATE 1
#define TFLITE_SUPPORTS_GPU_DELEGATE 1
#elif defined(CL_DELEGATE_NO_GL)
#define TFLITE_SUPPORTS_GPU_DELEGATE 1
#endif  // defined(__ANDROID__)
#endif  // TFLITE_WITH_STABLE_ABI

// XNNPACK does not support s390x
// (see <https://github.com/tensorflow/tensorflow/pull/51655>).
#ifdef __s390x__
#define TFLITE_WITHOUT_XNNPACK 1
#endif

#if TFLITE_SUPPORTS_GPU_DELEGATE
#include "tensorflow/lite/delegates/gpu/delegate.h"
#endif

#if TFLITE_ENABLE_HEXAGON
#include "tensorflow/lite/delegates/hexagon/hexagon_delegate.h"
#endif

#if TFLITE_SUPPORTS_NNAPI_DELEGATE
#include "tensorflow/lite/delegates/nnapi/nnapi_delegate.h"
#endif  // TFLITE_SUPPORTS_NNAPI_DELEGATE

#ifndef TFLITE_WITHOUT_XNNPACK
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"
#endif  // !defined(TFLITE_WITHOUT_XNNPACK)

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace evaluation {

// Same as Interpreter::TfLiteDelegatePtr, defined here to avoid pulling
// in tensorflow/lite/interpreter.h dependency.
using TfLiteDelegatePtr =
    std::unique_ptr<TfLiteOpaqueDelegate, void (*)(TfLiteOpaqueDelegate*)>;

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

// Returns nullptr on error, e.g. if NNAPI isn't supported on this platform.
TfLiteDelegatePtr CreateNNAPIDelegate();
#if TFLITE_SUPPORTS_NNAPI_DELEGATE
TfLiteDelegatePtr CreateNNAPIDelegate(StatefulNnApiDelegate::Options options);
#endif  // TFLITE_SUPPORTS_NNAPI_DELEGATE

TfLiteDelegatePtr CreateGPUDelegate();
#if TFLITE_SUPPORTS_GPU_DELEGATE
TfLiteDelegatePtr CreateGPUDelegate(TfLiteGpuDelegateOptionsV2* options);
#endif  // TFLITE_SUPPORTS_GPU_DELEGATE

TfLiteDelegatePtr CreateHexagonDelegate(
    const std::string& library_directory_path, bool profiling);
#if TFLITE_ENABLE_HEXAGON
TfLiteDelegatePtr CreateHexagonDelegate(
    const TfLiteHexagonDelegateOptions* options,
    const std::string& library_directory_path);
#endif

#ifndef TFLITE_WITHOUT_XNNPACK
TfLiteXNNPackDelegateOptions XNNPackDelegateOptionsDefault();
TfLiteDelegatePtr CreateXNNPACKDelegate();
TfLiteDelegatePtr CreateXNNPACKDelegate(
    const TfLiteXNNPackDelegateOptions* options);
#endif  // !defined(TFLITE_WITHOUT_XNNPACK)
TfLiteDelegatePtr CreateXNNPACKDelegate(
    int num_threads, bool force_fp16,
    const char* weight_cache_file_path = nullptr);

TfLiteDelegatePtr CreateCoreMlDelegate();
}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_UTILS_H_
