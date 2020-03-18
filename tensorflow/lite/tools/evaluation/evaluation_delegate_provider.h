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

#ifndef TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_DELEGATE_PROVIDER_H_
#define TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_DELEGATE_PROVIDER_H_

#include "tensorflow/lite/tools/evaluation/proto/evaluation_stages.pb.h"
#include "tensorflow/lite/tools/evaluation/utils.h"

namespace tflite {
namespace evaluation {

// Parse a string 'val' to the corresponding delegate type defined by
// TfliteInferenceParams::Delegate.
TfliteInferenceParams::Delegate ParseStringToDelegateType(
    const std::string& val);

// Create a TfLite delegate based on the given TfliteInferenceParams 'params'.
// If there's an error during the creation, an error message will be recorded to
// 'error_msg' if provided.
TfLiteDelegatePtr CreateTfLiteDelegate(const TfliteInferenceParams& params,
                                       std::string* error_msg = nullptr);
}  // namespace evaluation
}  // namespace tflite

#endif  // TENSORFLOW_LITE_TOOLS_EVALUATION_EVALUATION_DELEGATE_PROVIDER_H_
