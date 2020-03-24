/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/lite/tools/evaluation/evaluation_delegate_provider.h"

namespace tflite {
namespace evaluation {
namespace {
constexpr char kNnapiDelegate[] = "nnapi";
constexpr char kGpuDelegate[] = "gpu";
constexpr char kHexagonDelegate[] = "hexagon";
constexpr char kXnnpackDelegate[] = "xnnpack";
}  // namespace

TfliteInferenceParams::Delegate ParseStringToDelegateType(
    const std::string& val) {
  if (val == kNnapiDelegate) return TfliteInferenceParams::NNAPI;
  if (val == kGpuDelegate) return TfliteInferenceParams::GPU;
  if (val == kHexagonDelegate) return TfliteInferenceParams::HEXAGON;
  if (val == kXnnpackDelegate) return TfliteInferenceParams::XNNPACK;
  return TfliteInferenceParams::NONE;
}

TfLiteDelegatePtr CreateTfLiteDelegate(const TfliteInferenceParams& params,
                                       std::string* error_msg) {
  const auto type = params.delegate();

  switch (type) {
    case TfliteInferenceParams::NNAPI: {
      auto p = CreateNNAPIDelegate();
      if (!p && error_msg) *error_msg = "NNAPI not supported";
      return p;
    }
    case TfliteInferenceParams::GPU: {
      auto p = CreateGPUDelegate();
      if (!p && error_msg) *error_msg = "GPU delegate not supported.";
      return p;
    }
    case TfliteInferenceParams::HEXAGON: {
      auto p = CreateHexagonDelegate(/*library_directory_path=*/"",
                                     /*profiling=*/false);
      if (!p && error_msg) {
        *error_msg =
            "Hexagon delegate is not supported on the platform or required "
            "libraries are missing.";
      }
      return p;
    }
    case TfliteInferenceParams::XNNPACK: {
      auto p = CreateXNNPACKDelegate(params.num_threads());
      if (!p && error_msg) *error_msg = "XNNPACK delegate not supported.";
      return p;
    }
    case TfliteInferenceParams::NONE:
      if (error_msg) *error_msg = "No delegate type is specified.";
      return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
    default:
      if (error_msg) {
        *error_msg = "Creation of delegate type: " +
                     TfliteInferenceParams::Delegate_Name(type) +
                     " not supported yet.";
      }
      return TfLiteDelegatePtr(nullptr, [](TfLiteDelegate*) {});
  }
}

}  // namespace evaluation
}  // namespace tflite
