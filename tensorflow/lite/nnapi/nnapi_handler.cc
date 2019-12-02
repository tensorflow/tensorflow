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
#include "tensorflow/lite/nnapi/nnapi_handler.h"

#include <cstdio>

#include "tensorflow/lite/nnapi/nnapi_implementation.h"

namespace tflite {
namespace nnapi {

const NnApi* NnApiPassthroughInstance() {
  static const NnApi orig_nnapi_copy = *NnApiImplementation();
  return &orig_nnapi_copy;
}

// static
NnApiHandler* NnApiHandler::Instance() {
  // Ensuring that the original copy of nnapi is saved before we return
  // access to NnApiHandler
  NnApiPassthroughInstance();
  static NnApiHandler handler{const_cast<NnApi*>(NnApiImplementation())};
  return &handler;
}

void NnApiHandler::Reset() {
  // Restores global NNAPI to original value
  *nnapi_ = *NnApiPassthroughInstance();
}

}  // namespace nnapi
}  // namespace tflite
