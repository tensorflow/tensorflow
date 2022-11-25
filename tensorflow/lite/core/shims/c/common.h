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
#ifndef TENSORFLOW_LITE_CORE_SHIMS_C_COMMON_H_
#define TENSORFLOW_LITE_CORE_SHIMS_C_COMMON_H_

#include "tensorflow/lite/core/c/common.h"

// TfLiteOpaqueDelegate: allows delegation of nodes to alternative backends.
// TfLiteOpaqueDelegate is an abstract type that is intended to have the same
// role as TfLiteDelegate, but without necessarily exposing the implementation
// details of how delegates are implemented.
typedef TfLiteDelegate TfLiteOpaqueDelegate;

#endif  // TENSORFLOW_LITE_CORE_SHIMS_C_COMMON_H_
