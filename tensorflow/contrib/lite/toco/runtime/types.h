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
#ifndef THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_RUNTIME_TYPES_H_
#define THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_RUNTIME_TYPES_H_

#include "tensorflow/contrib/lite/kernels/internal/common.h"
#include "tensorflow/contrib/lite/kernels/internal/compatibility.h"
#include "tensorflow/contrib/lite/kernels/internal/types.h"

namespace toco {

// TODO(ahentz): These are just stopgaps for now, untils we move all
// the code over to tflite.
using tflite::Dims;
using tflite::FusedActivationFunctionType;
using tflite::RequiredBufferSizeForDims;

}  // namespace toco

#endif  // THIRD_PARTY_TENSORFLOW_CONTRIB_LITE_TOCO_RUNTIME_TYPES_H_
