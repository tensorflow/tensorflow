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

#include "tensorflow/lite/micro/kernels/kernel_util.h"

#include "tensorflow/lite/c/common.h"

namespace tflite {
namespace micro {

bool HaveSameShapes(const TfLiteEvalTensor* input1,
                    const TfLiteEvalTensor* input2) {
  TFLITE_DCHECK(input1 != nullptr);
  TFLITE_DCHECK(input2 != nullptr);
  return TfLiteIntArrayEqual(input1->dims, input2->dims);
}

}  // namespace micro
}  // namespace tflite
