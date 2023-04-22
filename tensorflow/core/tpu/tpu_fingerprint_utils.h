/* Copyright 2021 The TensorFlow Authors. All Rights Reserved.

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

#ifndef TENSORFLOW_CORE_TPU_TPU_FINGERPRINT_UTILS_H_
#define TENSORFLOW_CORE_TPU_TPU_FINGERPRINT_UTILS_H_

#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/lib/core/status.h"

namespace tensorflow {
// Computes a fingerprint of the contents of `library`.
Status FingerprintFunctionLibrary(const FunctionLibraryDefinition& library,
                                  uint64* fingerprint);
}  // namespace tensorflow
#endif  // TENSORFLOW_CORE_TPU_TPU_FINGERPRINT_UTILS_H_
