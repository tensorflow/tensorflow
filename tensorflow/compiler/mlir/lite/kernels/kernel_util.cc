/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#include "tensorflow/compiler/mlir/lite/kernels/kernel_util.h"

#include "tensorflow/compiler/mlir/lite/context_util.h"

#if defined(__APPLE__)
#include "TargetConditionals.h"
#endif

namespace tflite {

bool HasUnspecifiedDimension(const TfLiteTensor* tensor) {
#ifndef TF_LITE_STATIC_MEMORY
  if (tensor->dims_signature) {
    for (int i : TfLiteIntArrayView(tensor->dims_signature)) {
      if (i == -1) return true;
    }
  }
#endif  // TF_LITE_STATIC_MEMORY
  return false;
}

}  // namespace tflite

// LINT.ThenChange(//tensorflow/lite/kernels/kernel_util.cc)
