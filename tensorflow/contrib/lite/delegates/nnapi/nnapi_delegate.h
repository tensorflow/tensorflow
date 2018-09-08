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
#ifndef TENSORFLOW_CONTRIB_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
#define TENSORFLOW_CONTRIB_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_

#include "tensorflow/contrib/lite/c/c_api_internal.h"

namespace tflite {

// Return a delegate that can be used to use the NN API.
// e.g.
//   NnApiDelegate* delegate = NnApiDelegate();
//   interpreter->ModifyGraphWithDelegate(&delegate);
// NnApiDelegate() returns a singleton, so you should not free this
// pointer or worry about its lifetime.
TfLiteDelegate* NnApiDelegate();
}  // namespace tflite

#endif  // TENSORFLOW_CONTRIB_LITE_DELEGATES_NNAPI_NNAPI_DELEGATE_H_
