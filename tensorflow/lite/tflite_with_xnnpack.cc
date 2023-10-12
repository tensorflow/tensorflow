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
#include <memory>

#include "tensorflow/lite/core/c/common.h"
#include "tensorflow/lite/delegates/xnnpack/xnnpack_delegate.h"

namespace tflite {
// Corresponding weak declaration found in lite/tflite_with_xnnpack_optional.cc
// when TFLITE_BUILD_WITH_XNNPACK_DELEGATE macro isn't defined.
std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>
AcquireXNNPACKDelegate() {
  auto opts = TfLiteXNNPackDelegateOptionsDefault();
  return std::unique_ptr<TfLiteDelegate, void (*)(TfLiteDelegate*)>(
      TfLiteXNNPackDelegateCreate(&opts), TfLiteXNNPackDelegateDelete);
}
}  // namespace tflite
